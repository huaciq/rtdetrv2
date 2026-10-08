"""One fixed-checkpoint, final-detection analysis. No training or query probes.

Export: torchrun --nproc_per_node=2 tools/baseline_error_analysis.py --config ...
Offline: python tools/baseline_error_analysis.py --predictions ... --annotations ...
"""
import argparse
import copy
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import json
import io
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
# Capture canonical matching before src's faster-coco-eval dependency aliases
# pycocotools in sys.modules. The alias lacks Python-readable evalImgs.
from pycocotools.coco import COCO as PythonCOCO
from pycocotools.cocoeval import COCOeval as PythonCOCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ERRORS = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10",
           "AR100", "ARs", "ARm", "ARl")
HIGH_SCORE = 0.5


def check_dependencies():
    """Fail before inference/output writes, not after the complete GPU export."""
    versions = {}
    for name in ("torch", "faster-coco-eval", "tidecv", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"Analysis dependency '{name}' has no installation metadata in {sys.executable}. "
                "Activate rtdetr_zxy and install the analysis dependencies using this interpreter "
                "(see BASELINE_ERROR_ANALYSIS.md). If predictions.json already exists, "
                "recover with --predictions; GPU inference does not need to be repeated."
            ) from error
    try:
        importlib.import_module("tidecv")
        importlib.import_module("tidecv.data")
        importlib.import_module("faster_coco_eval")
    except ImportError as error:
        raise RuntimeError(
            f"An analysis dependency cannot be imported in {sys.executable}: {error}. "
            "Fix the analysis installation before running inference; see BASELINE_ERROR_ANALYSIS.md."
        ) from error
    return versions


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               indent=2), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size(area):
    # Exclusive bins for counts. COCO AP itself retains official inclusive bins.
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def ious(pred, gt):
    p = np.asarray([a["bbox"] for a in pred], dtype=float).reshape(-1, 4)
    g = np.asarray([a["bbox"] for a in gt], dtype=float).reshape(-1, 4)
    p[:, 2:] += p[:, :2]
    g[:, 2:] += g[:, :2]
    wh = np.maximum(0, np.minimum(p[:, None, 2:], g[None, :, 2:]) -
                    np.maximum(p[:, None, :2], g[None, :, :2]))
    intersection = wh.prod(-1)
    union = (p[:, 2:] - p[:, :2]).prod(-1)[:, None] + \
        (g[:, 2:] - g[:, :2]).prod(-1)[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def preflight(gt, predictions, image_root):
    images = {a["id"]: a for a in gt["images"]}
    cats = {a["id"]: a["name"] for a in gt["categories"]}
    if len(images) != len(gt["images"]) or not images or not cats:
        raise ValueError("Empty or duplicate image/category IDs")
    for image in images.values():
        path = (Path(image_root) / image["file_name"]).resolve()
        if not path.is_relative_to(Path(image_root).resolve()) or not path.is_file():
            raise FileNotFoundError(f"Validation image missing/outside image root: {path}")
    for item in gt["annotations"] + predictions:
        box = np.asarray(item["bbox"], dtype=float)
        if (item["image_id"] not in images or item["category_id"] not in cats
                or box.shape != (4,) or not np.isfinite(box).all()
                or (box[2:] <= 0).any()):
            raise ValueError(f"Invalid COCO record: {item}")
    ids = [a["id"] for a in gt["annotations"]]
    if len(ids) != len(set(ids)) or any(a <= 0 for a in ids):
        raise ValueError("COCO annotation IDs must be unique and positive for matching")
    if any(not np.isfinite(p["score"]) or not 0 <= p["score"] <= 1 for p in predictions):
        raise ValueError("Invalid prediction confidence")
    return images, cats


def export(args):
    import torch
    from torch.utils.data import DataLoader
    from src.core import YAMLConfig

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    cfg = YAMLConfig(args.config)
    dec = cfg.yaml_cfg.get("RTDETRTransformerv2", {})
    for flag in ("sar_quality", "decoder_quality", "final_quality_probe", "umqr", "sber",
                 "class_conditioned_final_quality_probe", "multi_threshold_class_conditioned_quality_probe"):
        if dec.get(flag, False):
            raise ValueError(f"Baseline-only analysis rejects {flag}")
    if dec.get("query_select_method", "default") != "default":
        raise ValueError("Baseline-only analysis requires original query selection")
    val = cfg.yaml_cfg["val_dataloader"]["dataset"]
    if args.annotations:
        val["ann_file"] = args.annotations
    if args.images:
        val["img_folder"] = args.images
    args.annotations, args.images = val["ann_file"], val["img_folder"]
    gt = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    preflight(gt, [], args.images)
    # Every tensor is subsequently loaded strictly; avoid an unnecessary download.
    cfg.yaml_cfg["PResNet"]["pretrained"] = False
    post = cfg.postprocessor.to(device).eval()
    if post.final_score_method != "default" or any(getattr(post, k) != 0 for k in (
            "final_quality_gamma", "oracle_final_iou_gamma", "class_aware_oracle_final_iou_gamma",
            "pairwise_class_aware_oracle_beta")):
        raise ValueError("Baseline-only analysis rejects quality/oracle reranking")
    model = cfg.model.to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    key = "ema" if cfg.yaml_cfg.get("use_ema", False) else "model"
    weights = state[key]["module"] if key == "ema" else state[key]
    model.load_state_dict(weights, strict=True)
    original = cfg.val_dataloader
    # No padding, duplicates, drop_last, DDP forward collectives or query hooks.
    indices = list(range(rank, len(original.dataset), world))
    loader = DataLoader(original.dataset, batch_size=original.batch_size,
                        sampler=indices, collate_fn=original.collate_fn,
                        num_workers=original.num_workers, drop_last=False)
    out = Path(args.output)
    shard = out / f"predictions.rank{rank}.jsonl"
    with torch.inference_mode(), shard.open("x", encoding="utf-8") as stream:
        for batch, (samples, targets) in enumerate(loader):
            output = model(samples.to(device))  # FP32, as original evaluate().
            orig = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = post(output, orig)
            for target, result in zip(targets, results):
                image_id = int(target["image_id"].item())
                # Match FasterCocoEvaluator's original dtype/rounding in xyxy->xywh.
                boxes = result["boxes"].cpu().clone()
                boxes[:, 2:] -= boxes[:, :2]
                records = [{"image_id": image_id, "category_id": int(label),
                            "bbox": box, "score": float(score)}
                           for box, label, score in zip(boxes.tolist(), result["labels"].tolist(),
                                                        result["scores"].tolist())]
                stream.write(json.dumps({"image_id": image_id, "predictions": records},
                                        allow_nan=False) + "\n")
            if batch % 20 == 0:
                print(f"rank={rank} validation batch={batch}/{len(loader)}", flush=True)
    if world > 1:
        torch.distributed.barrier()
        # Other ranks exit before the CPU analysis, avoiding long NCCL timeouts.
        torch.distributed.destroy_process_group()
    if rank != 0:
        return None
    merged = {}
    for r in range(world):
        with (out / f"predictions.rank{r}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["image_id"] in merged:
                    raise ValueError("Duplicate validation image in export")
                merged[record["image_id"]] = record["predictions"]
    if set(merged) != {a["id"] for a in gt["images"]}:
        raise ValueError("Export does not cover the complete original validation split")
    predictions = [p for image_id in sorted(merged) for p in merged[image_id]]
    dump(out / "predictions.json", predictions)
    metadata = {"config": args.config, "resolved_config": cfg.yaml_cfg,
                "checkpoint": args.checkpoint, "checkpoint_sha256": sha(args.checkpoint),
                "weights": key, "last_epoch": state.get("last_epoch"), "seed": 0,
                "gpu_process_count": world, "precision": "FP32 original evaluation",
                "strict_load": True, "image_count": len(merged),
                "predictions_per_image": dict(Counter(len(v) for v in merged.values()))}
    dump(out / "export_metadata.json", metadata)
    return predictions, metadata


def coco_evaluate(gt_path, predictions, out):
    from faster_coco_eval import COCO, COCOeval_faster
    coco = COCO(str(gt_path))
    if predictions:
        detections = coco.loadRes(copy.deepcopy(predictions))
    else:
        detections = COCO()
        detections.dataset = {"images": coco.dataset["images"],
                              "categories": coco.dataset["categories"], "annotations": []}
        detections.createIndex()
    evaluator = COCOeval_faster(coco, detections, "bbox", separate_eval=True)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = dict(zip(METRICS, map(float, evaluator.stats)))
    recall = evaluator.eval["recall"][np.isclose(evaluator.params.iouThrs, .75), :, 0, -1]
    metrics["AR75"] = float(recall[recall >= 0].mean()) if (recall >= 0).any() else -1
    dump(out / "coco_metrics.json", metrics)
    # The faster evaluator intentionally doesn't expose evalImgs in Python.
    # Reuse canonical pycocotools matching for the two fixed diagnostic thresholds.
    # This reads the same predictions; it does not perform another model pass.
    with contextlib.redirect_stdout(io.StringIO()):
        python_gt = PythonCOCO(str(gt_path))
        if predictions:
            python_dt = python_gt.loadRes(copy.deepcopy(predictions))
        else:
            python_dt = PythonCOCO()
            python_dt.dataset = copy.deepcopy(detections.dataset)
            python_dt.createIndex()
        matching = PythonCOCOeval(python_gt, python_dt, "bbox")
        matching.params.iouThrs = np.array([.5, .75])
        matching.params.areaRng = [matching.params.areaRng[0]]
        matching.params.maxDets = [100]
        matching.evaluate()
    # Standard COCO score-ordered one-to-one assignments, all areas, maxDets=100/category.
    matches, missed = {}, {t: set() for t in (.5, .75)}
    for result in matching.evalImgs:
        if result is None:
            continue
        for threshold in (.5, .75):
            t = np.flatnonzero(np.isclose(matching.params.iouThrs, threshold))[0]
            for did, match, ignore in zip(result["dtIds"], result["dtMatches"][t], result["dtIgnore"][t]):
                matches[(int(did) - 1, threshold)] = (int(match), bool(ignore))
            for gid, match, ignore in zip(result["gtIds"], result["gtMatches"][t], result["gtIgnore"]):
                if not ignore and not match:
                    missed[threshold].add(int(gid))
    return metrics, matches, missed


def tide_evaluate(gt, predictions, out):
    from tidecv import TIDE
    from tidecv.data import Data
    truth, detections = Data("validation", max_dets=100), Data("baseline")
    gt_lookup = {}
    for c in gt["categories"]:
        truth.add_class(c["id"], c["name"])
    for image in gt["images"]:
        truth.add_image(image["id"], image["file_name"])
    for a in gt["annotations"]:
        gt_lookup[len(truth.annotations)] = a
        if a.get("iscrowd", 0):
            truth.add_ignore_region(a["image_id"], a["category_id"], box=a["bbox"])
        else:
            truth.add_ground_truth(a["image_id"], a["category_id"], box=a["bbox"])
    for p in predictions:
        detections.add_detection(p["image_id"], p["category_id"], p["score"], box=p["bbox"])
    tide = TIDE(pos_threshold=.5, background_threshold=.1)
    run = tide.evaluate(truth, detections, mode=TIDE.BOX, name="baseline")
    contributions, undefined = {}, []
    for error in TIDE._error_types:
        try:
            contributions[error.short_name] = run.fix_main_errors(error_types=[error])[error]
        except ZeroDivisionError:
            # Removing every remaining GT leaves the Miss oracle undefined.
            # Preserve that fact instead of inventing a zero contribution.
            contributions[error.short_name] = None
            undefined.append(error.short_name)
    counts = {e.short_name: n for e, n in run.count_errors().items()}
    detail = []
    for e in run.errors:
        a, p = getattr(e, "gt", None), getattr(e, "pred", None)
        annotation = gt_lookup[a["_id"]] if a is not None else None
        detail.append({"error": e.short_name, "image_id": (p or a)["image"],
                       "prediction_id": p["_id"] if p is not None else "",
                       "gt_id": annotation["id"] if annotation else "",
                       "category_id": (p or a)["class"],
                       "size": size(annotation.get("area", np.prod(annotation["bbox"][2:])))
                       if annotation else size(np.prod(p["bbox"][2:])),
                       "size_basis": "GT" if annotation else "prediction",
                       "score": p["score"] if p else "",
                       "ignored_prediction": p is not None and p.get("used") is None})
    write_csv(out / "tide_errors.csv", detail, ["error", "image_id", "prediction_id", "gt_id",
              "category_id", "size", "size_basis", "score", "ignored_prediction"])
    result = {"AP50_percent": run.ap, "dAP_percent": contributions, "counts": counts,
              "max_dets": "official TIDE: 100 per image, across categories",
              "thresholds": {"foreground_iou": .5, "background_iou": .1},
              "non_additive": True, "undefined_oracles": undefined,
              "source": "https://github.com/dbolya/tide",
              "version": importlib.metadata.version("tidecv")}
    dump(out / "tide_summary.json", result)
    return result


def details(gt, predictions, matches, missed, out):
    by_gt, by_pred = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        if not a.get("iscrowd", 0):
            by_gt[a["image_id"]].append(a)
    for index, p in enumerate(predictions):
        by_pred[p["image_id"]].append((index, p))
    pred_rows, gt_rows = [], []
    for image in gt["images"]:
        anns = by_gt[image["id"]]
        indexed = by_pred[image["id"]]
        preds = [p for _, p in indexed]
        matrix = ious(preds, anns)
        for row, (index, p) in enumerate(indexed):
            if p["score"] < HIGH_SCORE:
                continue
            match, ignored = matches.get((index, .5), (0, False))
            evaluated = (index, .5) in matches
            same = [j for j, a in enumerate(anns) if a["category_id"] == p["category_id"]]
            any_iou = float(matrix[row].max()) if anns else 0.
            same_iou = float(matrix[row, same].max()) if same else 0.
            nearest = anns[int(matrix[row].argmax())] if any_iou > 0 else None
            status = "Ignored" if ignored else "TP" if match else "FP"
            # Supplemental transparent label; official TIDE labels are in tide_errors.csv.
            kind = ("Ignored" if ignored else "TP" if match else "Dupe" if same_iou >= .5
                    else "Loc" if same_iou >= .1 else "Cls" if any_iou >= .5
                    else "Bkg" if any_iou < .1 else "Both")
            associated = (anns[max(same, key=lambda j: matrix[row, j])]
                          if same and kind in ("Loc", "Dupe") else nearest)
            match75 = matches.get((index, .75), (0, False))[0]
            def matched_iou(gt_id):
                column = next((j for j, a in enumerate(anns) if a["id"] == gt_id), None)
                return float(matrix[row, column]) if column is not None else ""
            pred_rows.append({"prediction_id": index, "image_id": p["image_id"],
                "category_id": p["category_id"], "score": p["score"], "bbox": json.dumps(p["bbox"]),
                "status50": status if evaluated else "OutsideCOCOTop100",
                "error50": kind if evaluated else "OutsideCOCOTop100", "max_iou_any": any_iou,
                "max_iou_same": same_iou, "nearest_gt_id": nearest["id"] if nearest else "",
                "nearest_gt_category": nearest["category_id"] if nearest else "",
                "matched_gt50": match, "matched_gt75": match75,
                "matched_iou50": matched_iou(match), "matched_iou75": matched_iou(match75),
                "predicted_size": size(np.prod(p["bbox"][2:])),
                "error_target_gt_id": associated["id"] if associated else "",
                "target_size": size(associated.get("area", np.prod(associated["bbox"][2:]))) if associated else "",
                "high_confidence": True})
        for col, a in enumerate(anns):
            same = [j for j, p in enumerate(preds) if p["category_id"] == a["category_id"]]
            high_same = [j for j in same if preds[j]["score"] >= HIGH_SCORE]
            best = max(same, key=lambda j: matrix[j, col]) if same else None
            best_iou = float(matrix[best, col]) if best is not None else 0.
            high_iou = float(matrix[high_same, col].max()) if high_same else 0.
            matched_high = {t: any(matches.get((i, t), (0, True)) == (a["id"], False)
                                   and p["score"] >= HIGH_SCORE for i, p in indexed) for t in (.5, .75)}
            gt_rows.append({"gt_id": a["id"], "image_id": a["image_id"],
                "category_id": a["category_id"], "bbox": json.dumps(a["bbox"]),
                "size": size(a.get("area", np.prod(a["bbox"][2:]))),
                "miss50": a["id"] in missed[.5], "miss75": a["id"] in missed[.75],
                "miss50_high_conf": not matched_high[.5], "miss75_high_conf": not matched_high[.75],
                "best_iou_same_all_exported": best_iou,
                "best_iou_same_high_conf": high_iou,
                "best_iou_any_all_exported": float(matrix[:, col].max()) if preds else 0.,
                "best_same_prediction_id": indexed[best][0] if best is not None else "",
                "best_same_score": preds[best]["score"] if best is not None else 0.,
                "coverage_bin": ">=0.75" if best_iou >= .75 else "0.50-0.75" if best_iou >= .5
                                else "0.10-0.50" if best_iou >= .1 else "<0.10"})
    pfields = ["prediction_id", "image_id", "category_id", "score", "bbox", "status50", "error50",
               "max_iou_any", "max_iou_same", "nearest_gt_id", "nearest_gt_category", "matched_gt50",
               "matched_gt75", "matched_iou50", "matched_iou75", "predicted_size", "error_target_gt_id", "target_size", "high_confidence"]
    gfields = ["gt_id", "image_id", "category_id", "bbox", "size", "miss50", "miss75", "miss50_high_conf",
               "miss75_high_conf", "best_iou_same_all_exported", "best_iou_same_high_conf",
               "best_iou_any_all_exported", "best_same_prediction_id", "best_same_score", "coverage_bin"]
    write_csv(out / "high_confidence_detections.csv", pred_rows, pfields)
    write_csv(out / "gt_localization.csv", gt_rows, gfields)
    return pred_rows, gt_rows


def summaries(cats, pred_rows, gt_rows, tide, metrics, out):
    rows = []
    def add(section, cat, scale, metric, value, denominator="", basis=""):
        rows.append(dict(section=section, category_id=cat, category_name=cats.get(cat, "all"),
                         size=scale, metric=metric, value=value, denominator=denominator, basis=basis))
    for name, value in metrics.items():
        add("COCO", "all", "all", name, value * 100 if value >= 0 else None,
            basis="percent; original COCO protocol; empty value means unavailable")
    for name in ERRORS:
        add("TIDE", "all", "all", name + "_dAP", tide["dAP_percent"][name], basis="independent AP50 oracle; not additive")
        add("TIDE", "all", "all", name + "_count", tide["counts"][name], basis="TIDE top100/image")
    for cat in ["all"] + sorted(cats):
        for scale in ("all", "small", "medium", "large"):
            gs = [r for r in gt_rows if (cat == "all" or r["category_id"] == cat) and (scale == "all" or r["size"] == scale)]
            ps = [r for r in pred_rows if (cat == "all" or r["category_id"] == cat)]
            add("GT", cat, scale, "count", len(gs), basis="GT annotation area")
            for metric in ("miss50", "miss75", "miss50_high_conf", "miss75_high_conf"):
                add("GT", cat, scale, metric, sum(r[metric] for r in gs), len(gs), "COCO one-to-one; GT area")
            for bin_name in ("<0.10", "0.10-0.50", "0.50-0.75", ">=0.75"):
                add("coverage", cat, scale, bin_name, sum(r["coverage_bin"] == bin_name for r in gs), len(gs),
                    "best same-class IoU; all exported scores; not one-to-one recall")
            for error in ("Cls", "Loc", "Both", "Dupe", "Bkg"):
                relevant = [r for r in ps if r["error50"] == error and
                            (scale == "all" or (r["predicted_size"] if error == "Bkg" else r["target_size"]) == scale)]
                add("high_conf_FP", cat, scale, error, len(relevant), basis=
                    "score>=0.5; COCO top100/category; prediction area for Bkg, same-class GT for Loc/Dupe, nearest GT for Cls/Both")
    write_csv(out / "error_summary.csv", rows)
    return rows


def select_examples(rows, group_key):
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    selected, seen = [], set()
    # Category/scale/error round robin, one example per image, fixed deterministic order.
    while len(selected) < 20 and groups:
        for key in sorted(list(groups), key=str):
            while groups[key] and groups[key][0]["image_id"] in seen:
                groups[key].pop(0)
            if groups[key]:
                row = groups[key].pop(0)
                selected.append(row)
                seen.add(row["image_id"])
            if not groups[key]:
                del groups[key]
            if len(selected) == 20:
                break
    return selected


def visualize(images, cats, gt, predictions, pred_rows, gt_rows, image_root, out):
    from PIL import Image, ImageDraw
    by_gt = defaultdict(list)
    for a in gt["annotations"]:
        by_gt[a["image_id"]].append(a)
    pools = {"TP": [r for r in pred_rows if r["status50"] == "TP"],
             "FP": [r for r in pred_rows if r["status50"] == "FP"],
             "FN": [r for r in gt_rows if r["miss75"]]}
    index = []
    for kind, pool in pools.items():
        pool.sort(key=lambda r: (-r.get("score", 0), r["image_id"], r.get("prediction_id", r.get("gt_id"))))
        examples = select_examples(pool, lambda r: (r["category_id"], r.get("size", r.get("target_size")),
                                                    r.get("error50", r.get("coverage_bin"))))
        folder = out / "visualizations" / kind
        folder.mkdir(parents=True, exist_ok=True)
        for number, record in enumerate(examples, 1):
            image = images[record["image_id"]]
            canvas = Image.open(Path(image_root) / image["file_name"]).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            def box(b, color, label):
                x, y, w, h = b
                draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
                # IDs keep labels independent of platform font/category encoding.
                draw.text((max(0, x), max(0, y-12)), label, fill=color, stroke_width=1, stroke_fill="black")
            for a in by_gt[record["image_id"]]:
                box(a["bbox"], "lime", f"GT {a['id']} cat={a['category_id']}")
            if kind == "FN":
                focus = json.loads(record["bbox"])
                box(focus, "orange", f"FN@.75 GT={record['gt_id']}")
                pid = record["best_same_prediction_id"]
                if pid != "":
                    p = predictions[pid]
                    box(p["bbox"], "red", f"nearest score={p['score']:.3f} IoU={record['best_iou_same_all_exported']:.3f}")
            else:
                p = predictions[record["prediction_id"]]
                focus = p["bbox"]
                box(focus, "cyan" if kind == "TP" else "red",
                    f"{record['error50']} cat={p['category_id']} s={p['score']:.3f} IoU={record['matched_iou50'] if kind == 'TP' else round(record['max_iou_same'], 3)}")
            # Export both full context and a magnified crop for tiny SAR targets.
            name = f"{number:02d}_image{image['id']}.png"
            canvas.save(folder / name)
            x, y, w, h = focus
            pad = max(w, h, 32)
            bounds = (max(0, int(x-pad)), max(0, int(y-pad)),
                      min(canvas.width, int(x+w+pad)), min(canvas.height, int(y+h+pad)))
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                bounds = (0, 0, canvas.width, canvas.height)
            canvas.crop(bounds).resize((512, 512)).save(folder / name.replace(".png", "_crop.png"))
            index.append({"kind": kind, "image_id": image["id"], "category_id": record["category_id"],
                          "category_name": cats[record["category_id"]], "file": str(Path("visualizations")/kind/name),
                          "record": json.dumps(record, ensure_ascii=False)})
    write_csv(out / "visualization_index.csv", index, ["kind", "image_id", "category_id", "category_name", "file", "record"])
    return dict(Counter(r["kind"] for r in index))


def report(metrics, tide, rows, gt_rows, counts, manifest, out):
    d = tide["dAP_percent"]
    small = [r for r in gt_rows if r["size"] == "small"]
    n = len(small)
    loc = sum(.5 <= r["best_iou_same_all_exported"] < .75 for r in small)
    missing = sum(r["best_iou_same_all_exported"] < .1 for r in small)
    cls = sum(r["best_iou_same_all_exported"] < .5 and r["best_iou_any_all_exported"] >= .5 for r in small)
    # A transparent evidence ranking, not a causal claim or architecture search.
    if any(v is None for v in d.values()):
        recommendation = "存在未定义的 TIDE oracle；保留判断，不能据此给出可靠选题排序。"
    elif d["Loc"] >= max(d["Bkg"], d["Cls"], d["Miss"], d["Both"]) and d["Loc"] > 0:
        recommendation = "优先研究定位/表征：Loc 是这四项独立 oracle 中的最大项。"
    elif max(d["Bkg"], d["Cls"]) > max(d["Loc"], d["Miss"], d["Both"]):
        recommendation = "优先研究目标—杂波/类别区分：Bkg 或 Cls 的独立 oracle 贡献最大。"
    else:
        recommendation = "优先研究漏检相关表征与召回；当前证据不足以在定位、目标—杂波区分和监督/知识迁移之间作强结论。"
    fmt = lambda v: "N/A" if v is None else f"{v:.3f}"
    lines = ["# ERROR_ANALYSIS", "", "## 范围与证据", "",
             "一次 baseline best checkpoint 的最终输出分析；无训练、模型修改、参数 sweep 或 encoder/query probe。",
             f"来源：`{manifest.get('checkpoint', manifest.get('predictions_source'))}`；权重：`{manifest.get('weights', 'external predictions; unverified')}`。",
             f"Git：`{manifest['git_commit']}`。完整运行协议、路径、SHA256、版本见 `analysis_manifest.json`。", "",
             "## 原 COCO 指标（%）", "", "| AP50:95 | AP50 | AP75 | APs | APm | APl | AR100 | AR75 |",
             "|---|---|---|---|---|---|---|---|",
             "| " + " | ".join(f"{metrics[k]*100:.3f}" if metrics[k] >= 0 else "N/A"
                                for k in ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR100", "AR75")) + " |", "",
             "## 六类错误贡献", "", "| 错误 | TIDE ΔAP@0.50（百分点） | 数量 |", "|---|---:|---:|"]
    lines += [f"| {k} | {fmt(d[k])} | {tide['counts'][k]} |" for k in ERRORS]
    lines += ["", "使用 [官方 TIDE](https://github.com/dbolya/tide)，foreground IoU=0.5、background IoU=0.1、top100/图（跨类别）。",
              "这些 ΔAP 是独立反事实修正贡献，不能相加，也不是 AP50:95 的贡献。原 COCO 评估保留 top100/图/类别；两者截断口径不同。",
              "Cls=类别错误；Loc=同类定位不足；Both=类别与定位同时错误；Dupe=重复检测；Bkg=背景虚警；Miss=无法被其他错误修正的漏检。", "",
              "## 小目标定位与混淆", "",
              f"小目标 GT={n}；所有导出预测中同类 best IoU∈[0.5,0.75) 的 GT={loc}；同类 best IoU<0.1 的 GT={missing}；同类 best IoU<0.5 但异类/任意类别 best IoU≥0.5 的 GT={cls}。",
              f"AP50−AP75={100*(metrics['AP50']-metrics['AP75']):.3f} 个百分点。AP 差值本身不能证明定位机制是原因。",
              "上述 best-IoU 是最终检测覆盖上限，允许一个预测覆盖多个 GT，不能称为 COCO recall，也不能证明 encoder 已找到候选。",
              "低分框具有高 IoU 只说明输出中存在该框；需要结合 score≥0.5 的覆盖和 COCO 一对一漏检判断是否是置信度/杂波区分问题。", "",
              "## small/medium/large 分布", "", "| 尺度 | GT | FN@.50 | FN@.75 | 高置信 Bkg | 高置信 Loc |", "|---|---:|---:|---:|---:|---:|"]
    def value(section, scale, metric):
        return next(r["value"] for r in rows if r["section"] == section and r["category_id"] == "all" and r["size"] == scale and r["metric"] == metric)
    for scale in ("small", "medium", "large"):
        lines.append(f"| {scale} | {value('GT', scale, 'count')} | {value('GT', scale, 'miss50')} | {value('GT', scale, 'miss75')} | {value('high_conf_FP', scale, 'Bkg')} | {value('high_conf_FP', scale, 'Loc')} |")
    lines += ["", "FN 使用原 COCO 所有分数的一对一匹配。高置信 FP 固定 score≥0.5；Bkg 的尺度来自预测面积，Loc 来自最高 IoU 的同类 GT 面积。计数采用互斥尺度区间；COCO AP 沿用官方面积边界。",
              "逐类别各类高置信 FP 数量见 `error_summary.csv`；每个高置信检测的 score、同类/任意类别 IoU、最近 GT 类别见 `high_confidence_detections.csv`。", "",
              "## 选题判断", "", recommendation, "",
              f"- 定位/表征证据：Loc ΔAP={fmt(d['Loc'])}；small 有 {loc}/{n} 个同类框已达到 .5 而未达到 .75。再结合 `gt_localization.csv` 的高置信覆盖与 FN75 计数。",
              f"- 目标—杂波区分证据：Bkg ΔAP={fmt(d['Bkg'])}、Cls ΔAP={fmt(d['Cls'])}、Both ΔAP={fmt(d['Both'])}；small 存在 {cls} 个异类覆盖但同类定位失败的 GT。Bkg 仅表明与已标注 GT 不重叠；需看图排除漏标后才能称为真实杂波。",
              f"- 监督与知识迁移证据：Miss ΔAP={fmt(d['Miss'])}；small 完全缺少同类重叠框（IoU<.1）的 GT={missing}。这支持检查漏检/表征，不能单凭一个 checkpoint 证明换监督或引入 teacher 是最优方案；本次没有标签质量、teacher 或监督对照证据。",
              "- 自动推荐仅按独立 oracle 最大项给出保守优先级；Both/Miss 若占主导或证据接近，应保留判断不确定性。查看下面典型图后再确定视觉机制，不能从 IoU 推断 SAR 散射样式。", "",
              "## 典型图与交付物", "",
              f"实际保存 TP={counts.get('TP', 0)}、FP={counts.get('FP', 0)}、FN={counts.get('FN', 0)} 例，目标各20，按类别/尺度/错误类型轮转，每组同一图只取一例；不足20时不复制补足。",
              "TP/FP 为 score≥0.5 且 IoU=.5 的 COCO 匹配；FN 为 IoU=.75 下的漏检（可能在 .5 下已检出）。绿色=GT，红/青色=预测，橙色=FN75；每例附512×512放大裁剪。",
              "`visualization_index.csv` 保存图与原始记录的对应关系；`visualizations/{TP,FP,FN}/` 保存图片。",
              "`predictions.json` 为未按分数过滤的完整 COCO 最终预测；`validation_gt.json` 为原 GT 副本；`coco_metrics.json`、`tide_summary.json`、`tide_errors.csv` 和其余 CSV 保留数值证据。",
              "本报告的视觉机制结论尚需人工核对典型图片；未声称完成标签审计或因果机制验证。"]
    (out / "ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args, predictions, metadata):
    versions = check_dependencies()
    out = Path(args.output)
    gt = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    images, cats = preflight(gt, predictions, args.images)
    dump(out / "validation_gt.json", gt)
    if not (out / "predictions.json").exists():
        dump(out / "predictions.json", predictions)
    manifest = dict(metadata, annotations_source=args.annotations, images_root=args.images,
                    annotations_sha256=sha(args.annotations), predictions_sha256=sha(out / "predictions.json"),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    utc=datetime.now(timezone.utc).isoformat(), high_confidence_score=HIGH_SCORE,
                    versions=versions)
    dump(out / "analysis_manifest.json", manifest)
    metrics, matches, missed = coco_evaluate(out / "validation_gt.json", predictions, out)
    tide = tide_evaluate(gt, predictions, out)
    pred_rows, gt_rows = details(gt, predictions, matches, missed, out)
    rows = summaries(cats, pred_rows, gt_rows, tide, metrics, out)
    counts = visualize(images, cats, gt, predictions, pred_rows, gt_rows, args.images, out)
    report(metrics, tide, rows, gt_rows, counts, manifest, out)
    dump(out / "COMPLETE.json", {"metrics": metrics, "visualizations": counts, "status": "complete"})
    print(f"Analysis complete: {out / 'ERROR_ANALYSIS.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--annotations")
    parser.add_argument("--images")
    parser.add_argument("--predictions", help="Existing full COCO predictions; skips model inference")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.predictions and (not args.annotations or not args.images):
        parser.error("Offline analysis requires --annotations and --images")
    if not args.predictions and (not args.config or not args.checkpoint):
        parser.error("Export requires --config and --checkpoint")
    rank = int(os.environ.get("RANK", 0))
    if args.predictions and int(os.environ.get("WORLD_SIZE", 1)) != 1:
        parser.error("Offline CPU analysis must be launched with python, not torchrun")
    check_dependencies()
    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        # Shell console/PID files are allowed; never overwrite previous analysis evidence.
        if any(p.name not in ("console.log", "launcher.pid") for p in out.iterdir()):
            raise FileExistsError("Use a new output directory; existing analysis will not be overwritten")
    if args.predictions:
        analyze(args, json.loads(Path(args.predictions).read_text(encoding="utf-8")),
                {"predictions_source": args.predictions, "provenance_verified": False})
    else:
        # Rank0 creates the shared directory before any shard writes.
        # mkdir(exist_ok=True) is safe on all ranks; rank0 still rejects old evidence.
        out.mkdir(parents=True, exist_ok=True)
        try:
            result = export(args)
        finally:
            # Avoid the repository's atexit barrier after a rank-local failure.
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        if result is not None:
            analyze(args, *result)


if __name__ == "__main__":
    main()
