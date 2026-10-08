"""P1: bounded, prediction-only SAR localization evidence. No model inference."""
import argparse
import copy
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
from collections import Counter, defaultdict

import numpy as np
import yaml
from PIL import Image, ImageDraw

import baseline_error_analysis as base

CLASSES = ("Bridge", "Harbor", "Storage Tank")
BINS = ("<8", "8-16", "16-32", ">=32")
AXES = ("center", "area_scale", "aspect_ratio")


def canonical_category(name):
    key = "".join(c for c in name.casefold() if c.isalnum())
    mapping = {"".join(c for c in n.casefold() if c.isalnum()): n for n in CLASSES}
    if key not in mapping:
        raise ValueError(f"Unrecognized OGSOD class name: {name}; do not guess IDs")
    return mapping[key]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolved_config(path):
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = copy.deepcopy(v)
        return a
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    result = {}
    for included in value.get("__include__", []):
        merge(result, resolved_config(Path(path).parent / included))
    return merge(result, {k: v for k, v in value.items() if k != "__include__"})


def short_bin(value):
    return BINS[0] if value < 8 else BINS[1] if value < 16 else BINS[2] if value < 32 else BINS[3]


def input_box(bbox, image, hw, clip=False):
    x, y, w, h = bbox
    edges = np.array([x, y, x+w, y+h], dtype=float)
    if clip:
        edges[[0, 2]] = edges[[0, 2]].clip(0, image["width"])
        edges[[1, 3]] = edges[[1, 3]].clip(0, image["height"])
    return edges * np.array([hw[1]/image["width"], hw[0]/image["height"]]*2)


def edge_iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(intersection / max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection, 1e-12))


def geometry_error(pred, gt):
    pc, gc = (pred[:2]+pred[2:])/2, (gt[:2]+gt[2:])/2
    pw, gw = pred[2:]-pred[:2], gt[2:]-gt[:2]
    ratios = pw/gw
    log = np.log(ratios)
    original = edge_iou(pred, gt)
    def centered(center, wh):
        return np.concatenate((center-wh/2, center+wh/2))
    corrected = {"center": centered(gc, pw),
                 "area_scale": centered(pc, pw*np.sqrt(gw.prod()/pw.prod())),
                 "aspect_ratio": centered(pc, pw*np.array([np.sqrt((gw[0]/gw[1])/(pw[0]/pw[1])),
                                                            np.sqrt((pw[0]/pw[1])/(gw[0]/gw[1]))])),
                 "shape": centered(pc, gw)}
    gains = {k: edge_iou(v, gt)-original for k, v in corrected.items()}
    top = max(gains[k] for k in AXES)
    winners = [k for k in AXES if abs(gains[k]-top) <= 1e-12]
    dominant = winners[0] if len(winners) == 1 and top > 0 else "mixed_or_tied"
    result = {"iou": original, "center_dx_px": float(pc[0]-gc[0]), "center_dy_px": float(pc[1]-gc[1]),
              "center_l2_px": float(np.linalg.norm(pc-gc)),
              "center_relative_l2": float(np.linalg.norm((pc-gc)/gw)),
              "center_dx_relative": float((pc[0]-gc[0])/gw[0]), "center_dy_relative": float((pc[1]-gc[1])/gw[1]),
              "width_relative_error": float(ratios[0]-1), "height_relative_error": float(ratios[1]-1),
              "log_width_ratio": float(log[0]), "log_height_ratio": float(log[1]),
              "log_linear_scale_ratio": float(log.mean()), "log_aspect_ratio_error": float(log[0]-log[1]),
              "mean_abs_boundary_px": float(np.abs(pred-gt).mean()), "dominant_oracle": dominant}
    for key, value in zip(("left", "top", "right", "bottom"), pred-gt):
        result[f"{key}_error_px"] = float(value)
    for key, value in corrected.items():
        result[f"iou_fix_{key}"] = edge_iou(value, gt)
        result[f"gain_fix_{key}"] = gains[key]
    return result


def verify_provenance(args):
    source, analyzed = Path(args.export_dir), Path(args.analysis_dir)
    metadata, manifest = read(source/"export_metadata.json"), read(analyzed/"analysis_manifest.json")
    checks = []
    def check(name, actual, expected):
        checks.append({"check": name, "passed": actual == expected, "actual": actual, "expected": expected})
        if actual != expected:
            raise ValueError(f"Provenance mismatch {name}: {repr(actual)[:500]} != {repr(expected)[:500]}")
    check("checkpoint_bytes", base.sha(args.checkpoint), metadata["checkpoint_sha256"])
    check("best_filename", Path(args.checkpoint).name, "best.pth")
    check("recorded_checkpoint_filename", Path(metadata["checkpoint"]).name, "best.pth")
    check("strict_loading", metadata["strict_load"], True)
    check("EMA", metadata["weights"], "ema")
    cfg = resolved_config(args.config)
    recorded = copy.deepcopy(metadata["resolved_config"])
    recorded.pop("__include__", None)
    # Export disabled downloads after resolving paths; strictly loaded checkpoint
    # tensors replace initialization. This is the sole allowed model-config difference.
    expected = copy.deepcopy(cfg)
    expected["PResNet"]["pretrained"] = False
    check("complete_resolved_config", recorded, expected)
    check("analysis_prediction_bytes", base.sha(analyzed/"predictions.json"), manifest["predictions_sha256"])
    check("export_to_analysis_prediction_bytes", base.sha(source/"predictions.json"), base.sha(analyzed/"predictions.json"))
    gt_path = Path(cfg["val_dataloader"]["dataset"]["ann_file"])
    gt = read(gt_path)
    check("original_validation_GT", read(analyzed/"validation_gt.json"), gt)
    check("manifest_GT_bytes", base.sha(Path(manifest["annotations_source"])), manifest["annotations_sha256"])
    check("manifest_GT_content", read(manifest["annotations_source"]), gt)
    check("analysis_complete", read(analyzed/"COMPLETE.json")["status"], "complete")
    if not (analyzed/"ERROR_ANALYSIS.md").is_file() or not (analyzed/"error_summary.csv").is_file():
        raise FileNotFoundError("Previous error report/summary missing")
    # Check checkpoint structure and epoch without constructing a model/optimizer.
    import torch
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    check("checkpoint_epoch", state.get("last_epoch"), metadata["last_epoch"])
    names = list(state["ema"]["module"])
    forbidden = ("umqr", "sber", "quality_head", "quality_probe_head")
    check("no_experiment_head_in_checkpoint", any(any(f in key for f in forbidden) for key in names), False)
    del state
    train = read(args.train_annotations)
    check("train_split_matches_config", str(Path(args.train_annotations).resolve()),
          str(Path(cfg["train_dataloader"]["dataset"]["ann_file"]).resolve()))
    train_names = {i["file_name"] for i in train["images"]}
    check("train_val_filename_overlap", sorted(train_names & {i["file_name"] for i in gt["images"]}), [])
    check("train_val_categories", train["categories"], gt["categories"])
    # Image IDs can be independently re-numbered across splits: filenames identify images.
    hw = cfg["eval_spatial_size"]
    ops = cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
    check("exact_validation_transform", ops, [{"type": "Resize", "size": hw},
                                              {"type": "ConvertPILImage", "dtype": "float32", "scale": True}])
    check("no_validation_multiscale", cfg["val_dataloader"].get("collate_fn", {}).get("scales"), None)
    prediction_path = analyzed/"predictions.json"
    predictions = read(prediction_path)
    images, cats = base.preflight(gt, predictions, args.images)
    check("original_image_root", str(Path(args.images).resolve()),
          str(Path(cfg["val_dataloader"]["dataset"]["img_folder"]).resolve()))
    check("export_image_count", metadata["image_count"], len(images))
    counts = Counter(p["image_id"] for p in predictions)
    check("complete_prediction_image_set", sorted(counts), sorted(images))
    check("full_original_top300", sorted(set(counts.values())), [cfg["RTDETRPostProcessor"]["num_top_queries"]])
    # Reconstruct existing shards as an extra content link; no new inference.
    shards = [source/f"predictions.rank{r}.jsonl" for r in range(metadata["gpu_process_count"])]
    shard_verified = all(p.is_file() for p in shards)
    if shard_verified:
        indexed = {}
        for path in shards:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["image_id"] in indexed:
                        raise ValueError("Duplicate image in export shards")
                    indexed[row["image_id"]] = row["predictions"]
        check("shard_image_set", sorted(indexed), sorted(images))
        check("shard_prediction_content", [p for i in sorted(indexed) for p in indexed[i]], predictions)
    try:
        code_exists = subprocess.run(["git", "cat-file", "-e", manifest["git_commit"]+"^{commit}"],
                                     cwd=base.ROOT, capture_output=True).returncode == 0
    except OSError:
        code_exists = False
    result = {"status": "verified_data_and_checkpoint_link; export_git_commit_requires_review",
              "checks": [{"check": c["check"], "passed": c["passed"]} for c in checks],
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": metadata["checkpoint_sha256"],
              "checkpoint_epoch": metadata["last_epoch"], "config": args.config,
              "prediction_sha256": base.sha(prediction_path), "validation_sha256": base.sha(gt_path),
              "existing_report_sha256": base.sha(analyzed/"ERROR_ANALYSIS.md"),
              "existing_summary_sha256": base.sha(analyzed/"error_summary.csv"),
              "train_sha256": base.sha(args.train_annotations), "analysis_git_commit": manifest["git_commit"],
              "diagnostic_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.ROOT, text=True).strip(),
              "diagnostic_script_sha256": base.sha(__file__), "config_file_sha256": base.sha(args.config),
              "analysis_git_available": code_exists, "export_git_commit": metadata.get("export_git_commit"),
              "export_shards_verified": shard_verified,
              "limitations": ["Old export_metadata did not record export Git commit; CPU recovery commit cannot substitute for it.",
                              "Disjoint filenames do not rule out spatially adjacent scenes or duplicate image content."]}
    return gt, predictions, images, cats, cfg, result


def verify_existing_coco(analyzed, evaluator):
    """Independently recompute saved COCO metrics and CSV links from these predictions."""
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    metrics = dict(zip(base.METRICS, map(float, evaluator.stats)))
    t75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .75))[0])
    values = evaluator.eval["recall"][t75, :, 0, -1]
    metrics["AR75"] = float(values[values >= 0].mean()) if (values >= 0).any() else -1.
    saved = read(analyzed/"coco_metrics.json")
    with (analyzed/"error_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_values = {r["metric"]: r["value"] for r in csv.DictReader(stream) if r["section"] == "COCO"}
    for key, value in metrics.items():
        if key not in saved or not np.isclose(saved[key], value, atol=1e-5, rtol=0):
            raise ValueError(f"Saved COCO metric differs from original predictions: {key}, {saved.get(key)} vs {value}")
        expected = value*100 if value >= 0 else None
        actual = float(csv_values[key]) if csv_values.get(key) else None
        if (expected is None) != (actual is None) or (actual is not None and not np.isclose(actual, expected, atol=.001, rtol=0)):
            raise ValueError(f"error_summary.csv COCO metric differs: {key}")
    return {"saved_COCO_recomputed": True, "error_summary_COCO_values_verified": True,
            "absolute_fraction_tolerance": 1e-5, "metrics": metrics}


def group_metrics(gt, predictions, images, hw, group=None):
    """COCO area-ignore matching adapted to a fixed post-resize short-side bin.

    Keep outside-group GT as ignored, and ignore unmatched outside-group predictions.
    Never delete other-size GT and turn its valid detections into false positives.
    """
    truth = base.PythonCOCO()
    truth.dataset = copy.deepcopy(gt)
    eligible = Counter()
    if group is not None:
        for a in truth.dataset["annotations"]:
            b = input_box(a["bbox"], images[a["image_id"]], hw, clip=True)
            valid = bool((b[2:]-b[:2] > 0).all())
            inside = valid and short_bin(float((b[2:]-b[:2]).min())) == group
            a["area"] = 0. if inside else 2.
            if inside and not a.get("iscrowd", 0):
                eligible[a["category_id"]] += 1
    else:
        eligible.update(a["category_id"] for a in gt["annotations"] if not a.get("iscrowd", 0))
    with contextlib.redirect_stdout(io.StringIO()):
        truth.createIndex()
        detected = truth.loadRes(copy.deepcopy(predictions))
        if group is not None:
            for p in detected.dataset["annotations"]:
                b = input_box(p["bbox"], images[p["image_id"]], hw, clip=False)
                p["area"] = 0. if short_bin(float((b[2:]-b[:2]).min())) == group else 2.
        evaluator = base.PythonCOCOeval(truth, detected, "bbox")
        if group is not None:
            evaluator.params.areaRng = [[0., 0.]]
            evaluator.params.areaRngLbl = [group]
        evaluator.evaluate()
        evaluator.accumulate()
    t75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .75))[0])
    rows = []
    for k, cat in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][t75, :, k, 0, -1]
        recall = evaluator.eval["recall"][t75, k, 0, -1]
        rows.append({"category_id": int(cat), "size_group": group or "all", "gt_count": eligible[cat],
                     "AP75": float(precision[precision >= 0].mean()) if (precision >= 0).any() else None,
                     "AR75": float(recall) if recall >= 0 else None})
    return rows, evaluator


def geometry_rows(gt, predictions, images, cats, hw, evaluator):
    matched, matched75 = {}, set()
    t50 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .5))[0])
    t75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .75))[0])
    eligible_pred_ids = set()
    for entry in evaluator.evalImgs:
        if entry is None or list(entry["aRng"]) != list(evaluator.params.areaRng[0]):
            continue
        for did, gid, ignored in zip(entry["dtIds"], entry["dtMatches"][t50], entry["dtIgnore"][t50]):
            if not ignored:
                eligible_pred_ids.add(int(did)-1)
            if gid and not ignored:
                matched[int(gid)] = int(did)-1
        matched75.update(int(g) for g, d, ignored in zip(entry["gtIds"], entry["gtMatches"][t75], entry["gtIgnore"])
                         if d and not ignored)
    pred_by_image = defaultdict(list)
    for i in sorted(eligible_pred_ids):
        p = predictions[i]
        pred_by_image[p["image_id"]].append((i, p))
    rows, errors = [], []
    by_image = defaultdict(list)
    for a in gt["annotations"]:
        if not a.get("iscrowd", 0):
            by_image[a["image_id"]].append(a)
    for iid, anns in by_image.items():
        indexed = pred_by_image[iid]
        ps = [p for _, p in indexed]
        matrix = base.ious(ps, anns)
        for col, a in enumerate(anns):
            image = images[iid]
            raw = input_box(a["bbox"], image, hw)
            clipped = input_box(a["bbox"], image, hw, clip=True)
            wh = clipped[2:]-clipped[:2]
            same = [j for j, p in enumerate(ps) if p["category_id"] == a["category_id"]]
            best = max(same, key=lambda j: matrix[j, col]) if same else None
            row = {"gt_id": a["id"], "image_id": iid, "category_id": a["category_id"],
                   "category_name": cats[a["category_id"]], "bbox": json.dumps(a["bbox"]),
                   "original_width": a["bbox"][2], "original_height": a["bbox"][3],
                   "original_aspect_ratio": a["bbox"][2]/a["bbox"][3],
                   "input_width": float(wh[0]), "input_height": float(wh[1]),
                   "input_short_side": float(wh.min()), "input_aspect_ratio": float(wh[0]/wh[1]) if wh[1]>0 else None,
                   "size_group": short_bin(float(wh.min())) if (wh>0).all() else "invalid_after_clip",
                   "clipped_at_image_border": bool(not np.allclose(raw, clipped)),
                   "miss50": a["id"] not in matched, "miss75": a["id"] not in matched75,
                   "best_same_iou_top100": float(matrix[best, col]) if best is not None else 0.,
                   "best_same_score": ps[best]["score"] if best is not None else 0.,
                   "one_pixel_short_axis_shift_iou": float(max(wh.min()-1, 0)/(wh.min()+1)),
                   "P3_short_side_cells": float(wh.min()/8)}
            if a["id"] in matched:
                pid = matched[a["id"]]
                p = predictions[pid]
                error = geometry_error(input_box(p["bbox"], image, hw), raw)
                errors.append(dict(row, prediction_id=pid, score=p["score"], **error))
                row["matched50_iou"] = error["iou"]
            else:
                row["matched50_iou"] = None
            rows.append(row)
    return rows, errors


def summarize(rows, errors, metrics, cats):
    summary = []
    for metric in metrics:
        cat, group = metric["category_id"], metric["size_group"]
        gs = [r for r in rows if r["category_id"] == cat and (group == "all" or r["size_group"] == group)]
        es = [r for r in errors if r["category_id"] == cat and (group == "all" or r["size_group"] == group)]
        low = [r for r in es if .5 <= r["iou"] < .75]
        result = dict(metric, category_name=cats[cat], matched50_count=len(es), borderline50_75_count=len(low),
                      miss50_count=sum(r["miss50"] for r in gs), miss75_count=sum(r["miss75"] for r in gs),
                      miss75_fraction=sum(r["miss75"] for r in gs)/len(gs) if gs else None)
        for field in ("input_short_side", "input_aspect_ratio", "original_width", "original_height", "original_aspect_ratio"):
            values = [r[field] for r in gs if r[field] is not None]
            for name, q in (("median", .5), ("P90", .9)):
                result[f"{field}_{name}"] = float(np.quantile(values, q)) if values else None
        for field in ("iou", "center_l2_px", "center_relative_l2", "mean_abs_boundary_px", "width_relative_error", "height_relative_error",
                      "log_linear_scale_ratio", "log_aspect_ratio_error"):
            values = [abs(r[field]) for r in es]
            result[f"{field}_abs_median_TP50"] = float(np.median(values)) if values else None
            result[f"{field}_abs_P90_TP50"] = float(np.quantile(values, .9)) if values else None
        for field in ("center_dx_px", "center_dy_px", "log_width_ratio", "log_height_ratio"):
            result[f"{field}_signed_mean_TP50"] = float(np.mean([r[field] for r in es])) if es else None
        for axis in AXES:
            result[f"dominant_{axis}_count_50_75"] = sum(r["dominant_oracle"] == axis for r in low)
            result[f"recover75_by_{axis}_count_50_75"] = sum(r[f"iou_fix_{axis}"] >= .75 for r in low)
            result[f"median_gain_{axis}_50_75"] = float(np.median([r[f"gain_fix_{axis}"] for r in low])) if low else None
        summary.append(result)
    return summary


def charts(rows, errors, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in CLASSES:
        gt_rows = [r for r in rows if r["category_name"] == name and r["input_short_side"] > 0]
        es = [r for r in errors if r["category_name"] == name]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter([r["input_short_side"] for r in gt_rows], [r["best_same_iou_top100"] for r in gt_rows], s=8, alpha=.35)
        axes[0].set(xscale="log", xlabel="GT short side after resize (px)", ylabel="Best same-class IoU (coverage only)")
        axes[0].axhline(.75, color="red", linestyle="--")
        selected = [next(r for r in summary if r["category_name"] == name and r["size_group"] == g) for g in BINS]
        x = np.arange(4)
        axes[1].bar(x-.18, [r["AP75"] or 0 for r in selected], width=.36, label="AP75")
        axes[1].bar(x+.18, [r["AR75"] or 0 for r in selected], width=.36, label="AR75")
        axes[1].set(xticks=x, xticklabels=[f"{g}\nn={r['gt_count']}" for g, r in zip(BINS, selected)], ylim=(0, 1), xlabel="Post-resize short-side bin")
        axes[1].legend()
        for j, r in enumerate(selected):
            if r["AP75"] is None:
                axes[1].text(j, .02, "N/A", ha="center")
        axes[2].hist([r["iou"] for r in es], bins=np.linspace(.5, 1, 21))
        axes[2].set(xlabel="One-to-one TP50 IoU", ylabel="Count")
        fig.suptitle(name + " — fixed E0 predictions; no model changes")
        fig.tight_layout()
        filename = name.lower().replace(" ", "_") + "_size_iou"
        fig.savefig(out/(filename+".png"), dpi=170)
        fig.savefig(out/(filename+".svg"))
        plt.close(fig)


def visuals(rows, errors, gt, predictions, images, image_root, analyzed, out):
    candidates = [r for r in errors if .5 <= r["iou"] < .75]
    candidates.sort(key=lambda r: (r["category_name"] not in ("Bridge", "Storage Tank"), r["iou"], r["gt_id"]))
    selected = base.select_examples(candidates, lambda r: (r["category_name"], r["size_group"], r["dominant_oracle"]))
    index = []
    folder = out/"examples"
    folder.mkdir()
    for n, row in enumerate(selected, 1):
        image = images[row["image_id"]]
        canvas = Image.open(Path(image_root)/image["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        p = predictions[row["prediction_id"]]
        gt_box = json.loads(row["bbox"])
        for b, color in ((gt_box, "lime"), (p["bbox"], "red")):
            x, y, w, h = b
            draw.rectangle((x, y, x+w, y+h), outline=color, width=2)
        draw.text((5, 5), f"{row['category_name']} IoU={row['iou']:.3f} score={row['score']:.3f} GT={row['gt_id']}", fill="white", stroke_fill="black", stroke_width=1)
        name = f"{n:02d}_{row['category_name'].replace(' ', '_')}_GT{row['gt_id']}.png"
        canvas.save(folder/name)
        x, y, w, h = gt_box
        pad = max(w, h, 24)
        bounds = (max(0, int(x-pad)), max(0, int(y-pad)), min(canvas.width, int(x+w+pad)), min(canvas.height, int(y+h+pad)))
        crop = canvas.crop(bounds)
        crop.thumbnail((700, 700))
        # Upscale preserving aspect ratio; do not invent visible SAR structure.
        factor = 700/max(crop.size)
        crop.resize((max(1, round(crop.width*factor)), max(1, round(crop.height*factor))), Image.Resampling.NEAREST).save(folder/name.replace(".png", "_crop.png"))
        index.append({"sample_id": f"new_{n:02d}", "image_id": row["image_id"], "gt_id": row["gt_id"],
                      "category_name": row["category_name"], "image_path": str(folder/name), "source": "new TP50 with IoU<.75"})
    old_index = analyzed/"visualization_index.csv"
    if old_index.exists():
        with old_index.open(encoding="utf-8-sig", newline="") as stream:
            for r in csv.DictReader(stream):
                if r["kind"] == "FN":
                    name = canonical_category(r["category_name"])
                    entry = json.loads(r["record"])
                    path = (analyzed/r["file"]).resolve()
                    if not path.is_relative_to(analyzed.resolve()) or not path.is_file():
                        raise FileNotFoundError("Existing FN75 visualization missing/outside analysis directory")
                    index.append({"sample_id": f"old_FN75_{entry['gt_id']}", "image_id": int(r["image_id"]), "gt_id": entry["gt_id"],
                                  "category_name": name, "image_path": str(path), "source": "existing FN75"})
    fields = ["sample_id", "image_id", "gt_id", "category_name", "image_path", "source"]
    base.write_csv(out/"example_index.csv", index, fields)
    reviews = [dict(r, review_status="pending", boundary_blur="unknown", scattering_shift="unknown",
                    label_quality_issue="unknown", evidence_notes="", reviewer="") for r in index]
    base.write_csv(out/"manual_review.csv", reviews, fields+["review_status", "boundary_blur", "scattering_shift", "label_quality_issue", "evidence_notes", "reviewer"])
    return index


def report(summary, provenance, index, out):
    lines = ["# LOCALIZATION_DIAGNOSIS", "", "## 来源核对", "",
             f"checkpoint：`{provenance['checkpoint']}`；epoch={provenance['checkpoint_epoch']}；SHA256=`{provenance['checkpoint_sha256']}`。",
             "已核对 checkpoint 字节、EMA/strict load记录、完整配置、原 val GT、预测哈希及已有报告完成标记。详细结果见 `provenance.json`。",
             f"原导出 shards 内容核对：{provenance['export_shards_verified']}。CPU 分析 Git commit=`{provenance['analysis_git_commit']}`。",
             "旧版导出未记录 GPU 推理 Git commit，因此该项待核实；不能将 CPU 恢复分析的 commit 当作推理版本。文件名无交集不等于场景级无泄漏。", "",
             "## 指标与实际输入尺寸", "",
             "验证协议为原 Resize(640,640)，不是等比例 letterbox。实际 GT 尺寸先按原数据加载器裁到图像范围，再按两个轴分别缩放；CSV 同时保留原始长宽比和输入长宽比。",
             "短边区间为 <8、[8,16)、[16,32)、≥32 px。bin AP75/AR75 保留全部预测、忽略其它 bin GT，并忽略未匹配且自身尺寸在 bin 外的预测；不删除其它尺寸 GT 将其检测误算 FP。每类别 maxDets=100。",
             "bin AP/AR 按尺寸 ignore 规则重新匹配；FN75比例来自全验证集匹配，因此 AR75 与 FN75比例可能不互补。",
             "几何误差统计来自 COCO 一对一 TP50，完全漏检不赋予虚构框误差；全部 GT best-IoU 只是覆盖上限，不是 recall。", "",
             "| 类别 | 输入短边 | GT | AP75% | AR75% | FN75% | IoU∈[.5,.75) TP50 |", "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "N/A" if x is None else f"{100*x:.2f}"
    for r in summary:
        lines.append(f"| {r['category_name']} | {r['size_group']} | {r['gt_count']} | {fmt(r['AP75'])} | {fmt(r['AR75'])} | {fmt(r['miss75_fraction'])} | {r['borderline50_75_count']} |")
    lines += ["", "## 误差分解与五个选题问题", "",
              "固定单因素 oracle：平移到 GT 中心但保留预测宽高；匹配 GT 面积但保留预测长宽比和中心；匹配 GT 长宽比但保留面积和中心。IoU 增益相互独立、不可相加，反映数学误差成分，不证明 SAR 散射机制。边界偏移由中心和宽高共同决定，不能当作第四个独立自由度。"]
    for name in ("Bridge", "Storage Tank", "Harbor"):
        r = next(r for r in summary if r["category_name"] == name and r["size_group"] == "all")
        gains = {a: r[f"median_gain_{a}_50_75"] for a in AXES}
        valid = {a: v for a, v in gains.items() if v is not None}
        priority = max(valid, key=valid.get) if valid else "待验证（无合格样本）"
        lines += ["", f"### {name}", "",
                  f"[.5,.75) 框的中位单因素 IoU 增益：{gains}；按该数学指标最高项为 `{priority}`。",
                  "逐框中心、宽高误差、四边偏移和 oracle 恢复至 .75 的比例见 `matched_box_errors.csv` 与 `class_size_statistics.csv`。"]
        if name in ("Bridge", "Storage Tank"):
            low = next(s for s in summary if s["category_name"] == name and s["size_group"] == "<8")
            high = next(s for s in summary if s["category_name"] == name and s["size_group"] == ">=32")
            lines.append(f"极小 vs ≥32px 的 FN75 比例为 {fmt(low['miss75_fraction'])} vs {fmt(high['miss75_fraction'])}，GT 数为 {low['gt_count']} vs {high['gt_count']}；同时查看 AP75 与中间两组，避免仅由样本占比判断集中性。")
    lines += ["", "1. 主要框误差：以上 oracle 和有符号中心/边界分布可提供数值证据；散射偏移、模糊边界与标签质量必须人工核对。",
              "2. 是否集中于极小目标：依据各 bin 的 FN75 条件比例和 AP75，而非只比较错误总数；空组/少样本保留不确定性。",
              "3. 高分辨率特征：小短边、P3 cell 数与1px偏移敏感性可支持研究假设，但这次没有高分辨率干预对照，不能证明有效；decoder 连续采样也不能直接归结为 stride 量化。",
              "4. D-FINE 优势：P1 无 D-FINE 预测，待同预算 P2 结果验证；不能拿官方 COCO 或旧80e结果作 OGSOD 优势证据。",
              "5. 创新切入点：优先围绕 Bridge/Storage Tank 中可恢复的最大几何误差项形成假设；若人工复核显示标签问题，应先解决监督可靠性。当前不以未验证的物理解释确定新模块。", "",
              "## 人工复核状态", "",
              f"索引包含 {len(index)} 个新定位错误/已有 FN75 样本。`manual_review.csv` 全部初始为 pending/unknown；生成图像不等于完成人工复核。",
              "对原图和原 GT 核对 boundary_blur、scattering_shift、label_quality_issue，并填写证据、审阅者。无法确认保留 unknown；本自动报告没有声称观察到任何 SAR 视觉原因。", "",
              "## 输出", "",
              "`gt_geometry.csv`、`matched_box_errors.csv`、`class_size_statistics.csv`、`provenance.json`、各类别 size_iou PNG/SVG、`examples/`、`example_index.csv`、`manual_review.csv`。",
              "没有训练、模型推理、网络模块或参数 sweep。"]
    (out/"LOCALIZATION_DIAGNOSIS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(p.name not in ("console.log", "launcher.pid") for p in out.iterdir()):
        raise FileExistsError("Use a new localization output directory")
    gt, predictions, images, cats, cfg, provenance = verify_provenance(args)
    provenance["original_categories"] = cats
    cats = {k: canonical_category(v) for k, v in cats.items()}
    if set(cats.values()) != set(CLASSES) or len(cats) != 3:
        raise ValueError(f"Expected actual category names {CLASSES}; found {cats}. Do not guess category IDs.")
    for image in images.values():
        with Image.open(Path(args.images)/image["file_name"]) as im:
            if im.size != (image["width"], image["height"]):
                raise ValueError(f"Image/GT dimensions differ: image {image['id']}")
    metrics, evaluator = group_metrics(gt, predictions, images, cfg["eval_spatial_size"])
    provenance["existing_COCO_validation"] = verify_existing_coco(Path(args.analysis_dir), evaluator)
    base.dump(out/"provenance.json", provenance)
    rows, errors = geometry_rows(gt, predictions, images, cats, cfg["eval_spatial_size"], evaluator)
    for group in BINS:
        extra, _ = group_metrics(gt, predictions, images, cfg["eval_spatial_size"], group)
        metrics += extra
    summary = summarize(rows, errors, metrics, cats)
    base.write_csv(out/"gt_geometry.csv", rows)
    base.write_csv(out/"matched_box_errors.csv", errors, list(errors[0]) if errors else ["gt_id", "iou"])
    base.write_csv(out/"class_size_statistics.csv", summary)
    charts(rows, errors, summary, out)
    index = visuals(rows, errors, gt, predictions, images, args.images, Path(args.analysis_dir), out)
    report(summary, provenance, index, out)
    base.dump(out/"COMPLETE.json", {"status": "numerical_analysis_complete", "manual_review": "pending"})
    print(f"P1 numerical analysis complete; visual review pending: {out}", flush=True)


if __name__ == "__main__":
    main()
