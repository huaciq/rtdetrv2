"""Evidence-only OGSOD audit. No optimizer steps or detector source changes.

Discover evidence: python tools/sar_audit.py --discover --output NEW_DIRECTORY
Synthetic checks: python tools/sar_audit.py --self-test --output NEW_DIRECTORY
Server audit: torchrun --standalone --nproc_per_node=2 tools/sar_audit.py \
    --spec run_spec.json --output NEW_DIRECTORY
"""
import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIGS = {
    "baseline": "configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml",
    "qaqs": "configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_qaqs_gbs64_gpu2_zxy.yml",
    "relative_failed": "configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_qaqs_rel_small_gbs64_gpu2_zxy.yml",
}
PROBE_CONFIGS = {
    "old_scalar_probe": "configs/rtdetrv2/rtdetrv2_r18vd_ogsod_qaqs_final_quality_probe_e15_gbs64_gpu2_zxy.yml",
    "old_class_probe": "configs/rtdetrv2/rtdetrv2_r18vd_ogsod_qaqs_class_conditioned_quality_probe_e15_gbs64_gpu2_zxy.yml",
    "ranking_lambda05": "configs/rtdetrv2/rtdetrv2_r18vd_ogsod_qaqs_class_conditioned_quality_ranking_lambda05_e15_gbs64_gpu2_zxy.yml",
    "ranking_lambda10": "configs/rtdetrv2/rtdetrv2_r18vd_ogsod_qaqs_class_conditioned_quality_ranking_lambda10_e15_gbs64_gpu2_zxy.yml",
}
OLD_REF = "f102bb3^"
CRITERION_PATH = "src/zoo/rtdetr/rtdetrv2_criterion.py"
SCALES = {"all": (0, 1e10), "small": (0, 32**2),
          "medium": (32**2, 96**2), "large": (96**2, 1e10)}
THRESHOLDS = (.5, .75, .9)
GRAD_FIELDS = ["run", "batch", "branch", "group", "term", "raw_loss",
               "weighted_loss", "grad_norm", "base_reg_grad_norm", "ratio",
               "base_near_zero", "cosine", "finite", "global_gt", "small_640",
               "small_input", "medium_input", "large_input", "full_pre_clip_norm",
               "clip_coefficient", "group_post_clip_norm", "normalization",
               "precision", "evidence"]
LOC_FIELDS = ["run", "stage", "candidate_set", "category_id", "category_name",
              "scale", "gt_count", "zero_iou_gt", "QR50", "QR75", "QR90",
              "unique_QR50", "unique_QR75", "unique_QR90", "AP", "AP50", "AP75",
              "center_x_px_P50", "center_x_px_P90", "center_y_px_P50", "center_y_px_P90",
              "relative_center_x_P50", "relative_center_x_P90",
              "relative_center_y_P50", "relative_center_y_P90",
              "log_width_ratio_P50", "log_width_ratio_P90",
              "log_height_ratio_P50", "log_height_ratio_P90"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                   encoding="utf-8").strip()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically; a completed stage is never used until its marker exists.
    temp = path.with_suffix(path.suffix + ".partial")
    def convert(item):
        if isinstance(item,Path):
            return str(item)
        if hasattr(item,"tolist"):
            return item.tolist()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert),
                    encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def config_diff(left, right, prefix=""):
    result = {}
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(a, dict) and isinstance(b, dict):
            result.update(config_diff(a, b, name))
        elif a != b:
            result[name] = {"before": a, "after": b}
    return result


def read_yaml(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    result = {}
    for include in value.get("__include__", []):
        result = merge(result, read_yaml(Path(path).parent / include))
    value.pop("__include__", None)
    return merge(result, value)


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def evidence_directory(path):
    path = Path(path)
    if not path.is_dir():
        return {"path": str(path), "exists": False}
    files = []
    for name in ("*.pth", "*.json", "*.yml", "*.yaml", "log.txt", "console.log", "resume.log"):
        for item in sorted(path.glob(name)):
            files.append({"path": str(item), "bytes": item.stat().st_size,
                          "sha256": sha(item) if item.suffix != ".pth" else None})
    logs = []
    for name in ("console.log", "resume.log"):
        item = path / name
        if item.exists():
            with item.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if any(term in line.lower() for term in ("checkpoint", "tuning", "resume", "missing", "unexpected", "unmatched", "seed", "load ", "git ")):
                        logs.append(line.rstrip()[:4000])
    return {"path": str(path), "exists": True, "files": files, "loading_log_lines": logs,
            "checkpoint_selection": "requires explicit run_spec checkpoint; no automatic best selection"}


def static_audit(out):
    configs = {name: read_yaml(ROOT / path) for name, path in CONFIGS.items()}
    manifest = {"status": "static_only_missing_checkpoints_and_server_data",
                "audit_git_commit": git("rev-parse", "HEAD"), "audit_git_branch": git("branch", "--show-current"),
                "worktree_status": git("status", "--short"), "time_utc": datetime.now(timezone.utc).isoformat(),
                "old_criterion_ref": OLD_REF, "runs": {},
                "config_differences": {"baseline_to_qaqs": config_diff(configs["baseline"], configs["qaqs"]),
                                       "qaqs_to_relative": config_diff(configs["qaqs"], configs["relative_failed"])},
                "reported_probe_counts": {"old": 366800, "ranking": 366700, "raw_image_visits_if_100_per_image": [3668, 3667],
                    "conclusion": "not resolved by source alone; N=3667 at world_size=2 pads to 3668, world_size=1 visits 3667; alternative sample-set changes remain possible"},
                "missing": ["explicit checkpoint for each run", "actual run git commit and launch/parent checkpoint evidence", "server annotation files, images, logs and checkpoint tensors"]}
    spec = {"weights": "ema", "gradient_weights":"model", "seed": 0, "audit_epoch": 0,
            "gradient_precision": "fp32", "gradient_batches": 16, "runs": [], "probes": []}
    for name, cfg in configs.items():
        dump(out / f"{name}_resolved_config.json", cfg)
        manifest["runs"][name] = {"config": CONFIGS[name], "config_sha256": sha(ROOT / CONFIGS[name]),
                                  "resolved_config": cfg, "evidence": evidence_directory(cfg["output_dir"]),
                                  "run_git_commit": None, "checkpoint": None, "checkpoint_sha256": None,
                                  "loaded_weights": None, "missing_keys": None, "unexpected_keys": None}
        spec["runs"].append({"name": name, "config": CONFIGS[name], "checkpoint": None,
                              "run_dir": cfg["output_dir"], "run_git_commit": None,
                              "parent_checkpoint": None, "launch_command": None})
    paths = (CRITERION_PATH, "src/zoo/rtdetr/rtdetrv2_decoder.py", "src/zoo/rtdetr/rtdetr_postprocessor.py",
             "src/solver/query_stats.py", "src/solver/det_engine.py", "src/misc/dist_utils.py", "src/core/yaml_utils.py")
    manifest["source_hashes"] = {p: sha(ROOT / p) for p in paths}
    manifest["historical_source_changes"] = git("diff","--name-only",OLD_REF,"--","src").splitlines()
    manifest["probe_evidence_candidates"] = {}
    for name,path in PROBE_CONFIGS.items():
        cfg = read_yaml(ROOT/path)
        manifest["probe_evidence_candidates"][name] = {"config":path,"weights":"model",
                    "evidence":evidence_directory(cfg["output_dir"])}
    dump(out / "eval_manifest.json", manifest)
    dump(out / "run_spec.json", spec)
    write_csv(out / "loss_gradient_audit.csv", [], GRAD_FIELDS)
    write_csv(out / "localization_by_class_scale.csv", [], LOC_FIELDS)
    dump(out / "decoder_refinement_summary.json", {"status": "pending_real_checkpoints_and_data", "runs": {}})
    dump(out / "probe_count_audit.json", {"status":"pending_explicit_historical_probe_checkpoints",
          "reported_counts":[366800,366700],"hypothesis":"DDP padding; verify image visits and code/runtime/config versions"})
    summary(out, manifest)
    return manifest


def summary(out, manifest):
    content = f"""# OGSOD audit and diagnosis

Status: **{manifest['status']}**. Audit source: `{manifest['audit_git_commit']}`.

No optimizer step, new detector module or complete training is performed by this tool.
Config paths and historical output directory names are evidence candidates, not proof of the actual run configuration.

## Confirmed source findings

- Boxes use normalized cxcywh. Small mask is sqrt(w*h)*640 < 32, with a 4/640 lower bound on [w,h,w,h]. The loss is Smooth-L1 (beta=1), not relative L1.
- Reduction sums selected coordinates and divides by the DDP mean number of all GT (clamped to 1), not the number of small GT. Ordinary DDP gradient averaging yields a global GT normalization when unclamped.
- loss_rel_small weight=0.5 is applied at the final decoder and both ordinary auxiliary layers (R18 has 3 layers). It is absent from encoder and denoising losses. Each auxiliary layer rematches by default.
- The inherited train collate scales are null and Resize is 640x640. Thus the reference size matches this config; this must also hold in the failed run's effective launch configuration.
- The inherited global batch is 64, LR=0.0004, epochs=80, EMA=true, AMP=true, clip_max_norm=0.1. Original COCO area and training input area are distinct.
- Validation DistributedSampler(shuffle=False, drop_last=False) pads to ceil(N/world_size)*world_size. Probe arrays are concatenated without image IDs or deduplication. The installed faster-coco-eval merge source must be recorded separately; it normally removes duplicate image IDs.
- 366800/366700 counts alone do not identify a missing image. If N=3667, two ranks produce 3668 visits and one rank 3667. Historical logs and unique IDs are required to decide.
- remap_mscoco_category=false passes category_id directly to the 3-class model. Categories must be 0,1,2 unless the actual data/config uses another mapping. category2label property alone does not perform remapping.
- Resume loads model and EMA separately; evaluation uses EMA when enabled. A checkpoint without EMA can leave evaluation using an independently initialized EMA. This audit requires the requested weight key and refuses silent fallback.
- YAML load_config has a mutable default dict. This tool passes a fresh dict for every construction to prevent cross-config contamination.

## Evidence status

Missing or unresolved evidence: {json.dumps(manifest.get('missing', []), ensure_ascii=False)}

See eval_manifest.json for resolved configs, differences, hashes, dataset protocol and exact load keys. Empty CSV files are pending evidence, not zero-valued results.

## Metric definitions

COCO AP retains the repository postprocessor, original annotation area/ignore and evaluator maxDets. Class-by-scale AP is sliced from the same COCO evaluation tensor; absent strata are null.
Geometric QR is class-blind max IoU per noncrowd, nonignored valid GT over all regular queries; unique QR uses maximum bipartite cardinality at the threshold, independently within each reported stratum. It does not maximize total IoU.
COCO area ranges include their boundaries, so targets at 32^2 or 96^2 can appear in two scale strata. Geometry uses clipped GT boxes, official scale uses original annotation area. Zero-IoU and uncovered GT remain in the denominator.
Geometry is also computed for query IDs retained by the actual class-pair TopK output and class-pair Top100. Fixed-query trajectories choose the encoder best query per GT once; layer-best trajectories reselect at each layer. Query indices are never paired across models.
Center errors are absolute x/y pixels and absolute error divided by GT width/height; width/height errors are signed log ratios. P50/P90 include all eligible GT, including missed targets.
Gradient norms are vector norms, with parameter gradients averaged across ranks before measuring. Prediction gradients are concatenated across independent samples and divided by world size. Raw losses are recovered from known weights and recorded per branch. Ratios with near-zero baseline gradients are null. Post-clip subgroup norms use the coefficient from the complete parameter gradient, not independent subgroup clipping.
Gradient inputs are saved per rank in fixed_batches/ and reused across runs. Model BN buffers are restored before each batch; training mode and configured augmentation are used at a fixed audit epoch. No EMA update is made. Default gradient precision is FP32; --spec may request amp, matching the training autocast/FP32-criterion split without loss scaling.

## Reproduction

Read tools/SAR_AUDIT.md. --discover writes a run_spec with null checkpoints; fill exact paths and run provenance, then launch two ranks. --resume-dir resumes only incomplete audit stages, and requires unchanged spec, arguments, dataset/checkpoint/source hashes and GPU count.
"""
    if manifest.get("local_verification"):
        verification = manifest["local_verification"]
        content += "\n## Local verification\n\n"+json.dumps(verification,ensure_ascii=False,indent=2)+"\n"
    evidence = []
    for name in CONFIGS:
        path = out/name/"eval_complete.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            decoder = sorted(k for k in result["metrics"] if k.startswith("decoder_"))[-1]
            stats = result["metrics"][decoder]["stats"]
            evidence.append(f"| {name} | {result['sample_set']} | {result['unique_images']} | "+
                            " | ".join(f"{100*v:.4f}" if v>=0 else "n/a" for v in stats[:6])+" |")
    if evidence:
        content += "\n## Completed evaluation evidence\n\nAP values below are percentages; limited smoke rows are not full dataset results.\n\n"
        content += "| Run | Sample set | Unique images | AP | AP50 | AP75 | APs | APm | APl |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        content += "\n".join(evidence)+"\n"
    path = out/"relative_failed"/"loss_gradient_distributions.json"
    if path.exists():
        values = json.loads(path.read_text(encoding="utf-8"))["gradient_strata"]
        content += "\n## Relative-loss gradient evidence\n\n"
        for group in ("bbox_head","decoder_shared","prediction_boxes"):
            record = values.get(f"all:{group}:relative",{})
            content += f"- {group}: ratio {json.dumps(record.get('ratio',{}))}; cosine {json.dumps(record.get('cosine',{}))}.\n"
        content += "\nMeasured ratios and conflict do not establish the cause of AP degradation without the fixed initialization/budget comparison.\n"
    (out / "audit_summary.md").write_text(content, encoding="utf-8")


def runtime():
    global torch, np, YAMLConfig, RTDETRCriterionv2, box_iou, box_cxcywh_to_xyxy
    import numpy as np
    import torch
    from src.core import YAMLConfig
    from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2
    from src.zoo.rtdetr.box_ops import box_iou, box_cxcywh_to_xyxy


def fresh_config(path):
    from src.core.yaml_utils import load_config
    from src.core._config import BaseConfig
    # Reproduce YAMLConfig.__init__ while avoiding the loader's mutable default.
    cfg = YAMLConfig.__new__(YAMLConfig)
    BaseConfig.__init__(cfg)
    clean = load_config(path, cfg={})
    for key in list(cfg.__dict__):
        if not key.startswith("_") and key in clean:
            cfg.__dict__[key] = clean[key]
    cfg.yaml_cfg = copy.deepcopy(clean)
    return cfg


def old_criterion(current):
    source = git("show", f"{OLD_REF}:{CRITERION_PATH}")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RTDETRCriterionv2":
            node.decorator_list = []
    namespace = {"__name__": "src.zoo.rtdetr._audit_historical_criterion", "__package__": "src.zoo.rtdetr"}
    exec(compile(tree, f"git:{OLD_REF}:{CRITERION_PATH}", "exec"), namespace)
    cls = namespace["RTDETRCriterionv2"]
    arguments = {}
    for name, arg in inspect.signature(cls).parameters.items():
        if hasattr(current, name):
            arguments[name] = copy.deepcopy(getattr(current, name))
        elif arg.default is inspect.Parameter.empty:
            raise ValueError(f"cannot reconstruct historical criterion argument {name}")
    arguments["weight_dict"].pop("loss_rel_small", None)
    return cls(**arguments)


def annotation_audit(path, image_root, num_classes):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [x["id"] for x in value["images"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate image IDs in {path}")
    categories = value["categories"]
    if {x["id"] for x in categories} != set(range(num_classes)):
        raise ValueError("remap=false requires direct category IDs 0..num_classes-1")
    image_map = {x["id"]: x for x in value["images"]}
    counts = Counter()
    missing_images, area_diff, invalid = [], [], []
    for image in value["images"]:
        if not (Path(image_root) / image["file_name"]).is_file():
            missing_images.append(image["id"])
    ann_ids = [a["id"] for a in value["annotations"]]
    if len(ann_ids) != len(set(ann_ids)):
        raise ValueError(f"duplicate annotation IDs in {path}")
    for ann in value["annotations"]:
        if ann["image_id"] not in image_map or ann["category_id"] not in range(num_classes):
            raise ValueError(f"invalid annotation foreign key: {ann['id']}")
        area = float(ann["area"])
        x, y, w, h = ann["bbox"]
        if not all(math.isfinite(float(v)) for v in (area, x, y, w, h)) or area < 0:
            raise ValueError(f"nonfinite/negative annotation: {ann['id']}")
        if not math.isclose(area, w*h, rel_tol=1e-5, abs_tol=1e-5):
            area_diff.append(ann["id"])
        image = image_map[ann["image_id"]]
        if min(image["width"], x+w) <= max(0, x) or min(image["height"], y+h) <= max(0, y):
            invalid.append(ann["id"])
        for scale, (lo, hi) in SCALES.items():
            if lo <= area <= hi:
                counts[f"{ann['category_id']}:{scale}:total"] += 1
                if not ann.get("iscrowd", 0) and not ann.get("ignore", 0) and ann["id"] not in invalid:
                    counts[f"{ann['category_id']}:{scale}:geometry_eligible"] += 1
    result = {"path": str(path), "sha256": sha(path), "categories": categories,
              "unique_image_count": len(ids), "image_ids": sorted(ids), "GT_count": len(ann_ids),
              "iscrowd_count": sum(bool(a.get("iscrowd", 0)) for a in value["annotations"]),
              "ignore_count": sum(bool(a.get("ignore", 0)) for a in value["annotations"]),
              "area_not_bbox_area_ids": area_diff, "invalid_box_ids": invalid,
              "area_boundary_counts": {str(t): sum(float(a["area"]) == t for a in value["annotations"]) for t in (1024, 9216)},
              "class_scale_counts": dict(counts), "missing_image_ids": missing_images,
              "DDP_padding": {str(n): {"visits": math.ceil(len(ids)/n)*n,
                                       "padding": math.ceil(len(ids)/n)*n-len(ids)} for n in (1, 2)}}
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} missing images under {image_root}")
    return result, value


class LayerCapture:
    """Read-only hooks on eval forwards; never switches decoder to train mode."""
    def __init__(self, model):
        from src.zoo.rtdetr.utils import inverse_sigmoid
        self.inverse = inverse_sigmoid
        self.decoder = model.decoder
        self.handles = []
        self.clear()
        for i, layer in enumerate(self.decoder.decoder.layers):
            self.handles.append(layer.register_forward_hook(self.layer_hook(i)))
            self.handles.append(self.decoder.dec_bbox_head[i].register_forward_hook(self.box_hook(i)))

    def clear(self):
        self.refs, self.features, self.boxes = {}, {}, {}

    def layer_hook(self, index):
        def hook(module, args, output):
            self.refs[index] = args[1].squeeze(2).detach()
            self.features[index] = output.detach()
        return hook

    def box_hook(self, index):
        def hook(module, args, output):
            self.boxes[index] = (output.detach() + self.inverse(self.refs[index])).sigmoid()
        return hook

    def collect(self, outputs):
        stages = [("encoder", outputs["enc_topk_boxes"], outputs["enc_topk_logits"])]
        for i in sorted(self.boxes):
            logits = self.decoder.dec_score_head[i](self.features[i])
            stages.append((f"decoder_{i}", self.boxes[i], logits))
        torch.testing.assert_close(stages[-1][1], outputs["pred_boxes"], rtol=0, atol=0)
        torch.testing.assert_close(stages[-1][2], outputs["pred_logits"], rtol=0, atol=0)
        if stages[-1][1].shape[1] != self.decoder.num_queries:
            raise ValueError("unexpected DN/extra queries in eval capture")
        return stages

    def close(self):
        for handle in self.handles:
            handle.remove()


def cardinality(matrix, threshold):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if not matrix.shape[0] or not matrix.shape[1]:
        return 0
    matches = maximum_bipartite_matching(csr_matrix(matrix >= threshold), perm_type="column")
    return int((matches >= 0).sum())


def geometries(stages, image, annotations, postprocessor, output_path):
    eligible = [a for a in annotations if not a.get("iscrowd", 0) and not a.get("ignore", 0)]
    gts, labels, areas, gt_ids = [], [], [], []
    for a in eligible:
        x, y, w, h = a["bbox"]
        box = [max(0, x), max(0, y), min(image["width"], x+w), min(image["height"], y+h)]
        if box[2] > box[0] and box[3] > box[1]:
            gts.append(box); labels.append(a["category_id"]); areas.append(a["area"]); gt_ids.append(a["id"])
    gt = torch.tensor(gts, dtype=torch.float32).reshape(-1, 4)
    scale = torch.tensor([image["width"], image["height"]]*2)
    records, ious, arrays, stage_predictions = [], [], {}, {}
    labels, areas = np.array(labels), np.array(areas)
    for stage, boxes, logits in stages:
        pred = box_cxcywh_to_xyxy(boxes.cpu().float()) * scale
        iou = box_iou(pred, gt)[0].numpy()
        ious.append(iou)
        result = postprocessor({"pred_boxes": boxes[None], "pred_logits": logits[None]}, scale[:2][None].to(boxes.device))[0]
        stage_predictions[stage] = {k: v.cpu() for k, v in result.items()}
        # All three selected configs use the original class-only pair TopK.
        pair_ids = logits.sigmoid().flatten().topk(min(postprocessor.num_top_queries, logits.numel())).indices.cpu().numpy()
        output_queries = np.unique(pair_ids // logits.shape[-1])
        top100_pairs = logits.sigmoid().flatten().topk(min(100, logits.numel())).indices.cpu().numpy()
        arrays[f"{stage}_boxes_cxcywh"] = boxes.cpu().numpy()
        arrays[f"{stage}_logits"] = logits.cpu().numpy()
        arrays[f"{stage}_actual_output_query_ids"] = output_queries
        arrays[f"{stage}_iou"] = iou
        for candidate_set, query_ids in (("all_regular_queries", np.arange(len(pred))),
                                         ("actual_output", output_queries),
                                         ("classification_top100_pairs", np.unique(top100_pairs // logits.shape[-1]))):
            selected = iou[query_ids]
            for category in [-1] + sorted(set(labels.tolist())):
                for size, (lo, hi) in SCALES.items():
                    mask = (areas >= lo) & (areas <= hi)
                    if category != -1:
                        mask &= labels == category
                    count = int(mask.sum())
                    if not count:
                        continue
                    sub = selected[:, mask]
                    maxima, best = sub.max(axis=0), sub.argmax(axis=0)
                    chosen, target = pred[query_ids[best]].numpy(), gt.numpy()[mask]
                    wh = target[:, 2:] - target[:, :2]
                    offset = np.abs((chosen[:, :2]+chosen[:, 2:])/2 - (target[:, :2]+target[:, 2:])/2)
                    log_wh = np.log(np.maximum(chosen[:, 2:]-chosen[:, :2], 1e-12) / wh)
                    errors = np.column_stack((offset, offset/wh, log_wh))
                    records.append({"stage": stage, "candidate_set": candidate_set, "category_id": category,
                        "scale": size, "gt_count": count, "zero_iou_gt": int((maxima == 0).sum()),
                        "covered": [int((maxima >= t).sum()) for t in THRESHOLDS],
                        "unique": [cardinality(sub, t) for t in THRESHOLDS], "errors": errors.tolist()})
    fixed, best_traces = [], []
    if len(gt):
        fixed_queries = ious[0].argmax(axis=0)
        fixed = [matrix[fixed_queries, np.arange(len(gt))].tolist() for matrix in ious]
        best_traces = [matrix.max(axis=0).tolist() for matrix in ious]
        arrays["fixed_encoder_best_query_ids"] = fixed_queries
    arrays.update({"GT_boxes_xyxy": gt.numpy(), "GT_annotation_ids": np.array(gt_ids),
                   "GT_category_ids": labels, "GT_original_area": areas,
                   "original_image_wh":np.array([image["width"],image["height"]]),
                   "regular_query_ids": np.arange(stages[0][1].shape[0]),
                   "stage_names": np.array([s[0] for s in stages])})
    np.savez_compressed(output_path, **arrays)
    trajectories = []
    for category in [-1] + sorted(set(labels.tolist())):
        for size, (lo, hi) in SCALES.items():
            mask = (areas >= lo) & (areas <= hi)
            if category != -1:
                mask &= labels == category
            if not mask.any():
                continue
            for kind, traces in (("fixed_encoder_best_query", fixed), ("layer_best_query", best_traces)):
                for i in range(1, len(traces)):
                    delta = np.asarray(traces[i])[mask] - np.asarray(traces[i-1])[mask]
                    trajectories.append({"from": stages[i-1][0], "to": stages[i][0], "kind": kind,
                                         "category_id": category, "scale": size, "deltas": delta.tolist()})
    return records, trajectories, stage_predictions


def reduce_localization(run_name, per_image, categories):
    buckets, trajectories = {}, defaultdict(list)
    for item in per_image:
        for row in item["geometry"]:
            key = (row["stage"], row["candidate_set"], row["category_id"], row["scale"])
            if key not in buckets:
                buckets[key] = {"gt_count": 0, "zero_iou_gt": 0, "covered": np.zeros(3), "unique": np.zeros(3), "errors": []}
            bucket = buckets[key]
            for field in ("gt_count", "zero_iou_gt"):
                bucket[field] += row[field]
            for field in ("covered", "unique"):
                bucket[field] += row[field]
            bucket["errors"].extend(row["errors"])
        for row in item["trajectories"]:
            key = (row["from"], row["to"], row["kind"], row["category_id"], row["scale"])
            trajectories[key].extend(row["deltas"])
    names = {x["id"]: x["name"] for x in categories}
    error_names = ["center_x_px", "center_y_px", "relative_center_x", "relative_center_y", "log_width_ratio", "log_height_ratio"]
    rows = []
    for key, bucket in sorted(buckets.items()):
        stage, candidate_set, category, size = key
        row = {"run": run_name, "stage": stage, "candidate_set": candidate_set, "category_id": category,
               "category_name": names.get(category, "all"), "scale": size,
               "gt_count": bucket["gt_count"], "zero_iou_gt": bucket["zero_iou_gt"]}
        for i, suffix in enumerate(("50", "75", "90")):
            row[f"QR{suffix}"] = float(bucket["covered"][i]/bucket["gt_count"])
            row[f"unique_QR{suffix}"] = float(bucket["unique"][i]/bucket["gt_count"])
        errors = np.array(bucket["errors"])
        for i, name in enumerate(error_names):
            row[f"{name}_P50"], row[f"{name}_P90"] = np.quantile(errors[:, i], [.5, .9]).tolist()
        rows.append(row)
    refined = []
    for key, values in sorted(trajectories.items()):
        a = np.array(values)
        refined.append(dict(zip(("from", "to", "kind", "category_id", "scale"), key),
                            GT_count=len(a), improved_fraction=float((a>1e-7).mean()),
                            worsened_fraction=float((a < -1e-7).mean()),
                            unchanged_fraction=float((np.abs(a) <= 1e-7).mean()),
                            delta_P50=float(np.quantile(a, .5)), delta_P90=float(np.quantile(a, .9))))
    return rows, refined


def mean_valid(array):
    valid = array[array > -1]
    return float(valid.mean()) if valid.size else None


def coco_rows(name, stage, evaluator, annotations):
    ev = evaluator.coco_eval["bbox"]
    precision = ev.eval["precision"]
    rows = []
    for category in [-1] + list(ev.params.catIds):
        category_indices = list(range(len(ev.params.catIds))) if category == -1 else [list(ev.params.catIds).index(category)]
        for ai, size in enumerate(ev.params.areaRngLbl):
            selected = precision[:, :, category_indices, ai, list(ev.params.maxDets).index(100)]
            row = {"run": name, "stage": stage, "candidate_set": "COCO_actual_output", "category_id": category,
                   "category_name": next((c["name"] for c in annotations["categories"] if c["id"] == category), "all"), "scale": size,
                   "AP": mean_valid(selected)}
            for threshold, metric in ((.5, "AP50"), (.75, "AP75")):
                idx = int(np.argmin(np.abs(ev.params.iouThrs-threshold)))
                row[metric] = mean_valid(selected[idx])
            rows.append(row)
    return rows


def weight_load(model, path, key):
    signature = inspect.signature(torch.load)
    kwargs = {"map_location": "cpu"}
    if "weights_only" in signature.parameters:
        kwargs["weights_only"] = False
    state = torch.load(path, **kwargs)
    if key == "ema":
        if not isinstance(state.get("ema"), dict) or "module" not in state["ema"]:
            raise ValueError(f"EMA explicitly requested but missing: {path}")
        weights = state["ema"]["module"]
    else:
        if "model" not in state:
            raise ValueError(f"model key missing: {path}")
        weights = state["model"]
    expected = model.state_dict()
    missing = sorted(set(expected)-set(weights))
    unexpected = sorted(set(weights)-set(expected))
    mismatched = {k: [list(expected[k].shape), list(weights[k].shape)] for k in set(expected)&set(weights) if expected[k].shape != weights[k].shape}
    result = {"checkpoint": str(Path(path).resolve()), "checkpoint_sha256": sha(path),
              "loaded_weights": key, "checkpoint_keys": sorted(state),
              "last_epoch": state.get("last_epoch"), "missing_keys": missing,
              "unexpected_keys": unexpected, "mismatched_shapes": mismatched,
              "ema_present": "ema" in state, "model_present": "model" in state,
              "strict_load_passed": not (missing or unexpected or mismatched)}
    if result["strict_load_passed"]:
        model.load_state_dict(weights, strict=True)
    return result


def distributed_vector(loss, tensors, parameter_vector):
    grads = torch.autograd.grad(loss, tensors, allow_unused=True, retain_graph=True)
    vector = torch.cat([(torch.zeros_like(t) if g is None else g).detach().float().flatten() for t, g in zip(tensors, grads)])
    if torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        if parameter_vector:
            torch.distributed.all_reduce(vector)
            vector /= world
        else:
            chunks = [torch.empty_like(vector) for _ in range(world)]
            torch.distributed.all_gather(chunks, vector)
            vector = torch.cat(chunks) / world
    return vector


def zero_equivalence(model, outputs, targets, current, old):
    previous = current.weight_dict.get("loss_rel_small")
    current.weight_dict["loss_rel_small"] = 0.0
    try:
        new_losses, old_losses = current(outputs, targets), old(outputs, targets)
        base = {k: v for k, v in new_losses.items() if not k.startswith("loss_rel_small")}
        if set(base) != set(old_losses):
            raise AssertionError("historical/new base loss keys differ")
        loss_error = 0.0
        for key in base:
            torch.testing.assert_close(base[key], old_losses[key], rtol=1e-6, atol=1e-7)
            loss_error = max(loss_error, float((base[key]-old_losses[key]).abs()))
        params = [p for p in model.parameters() if p.requires_grad]
        a = distributed_vector(sum(new_losses.values()), params, True)
        b = distributed_vector(sum(old_losses.values()), params, True)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        unchanged = not any(p!=CRITERION_PATH for p in git("diff","--name-only",OLD_REF,"--","src").splitlines())
        return {"passed": True, "loss_max_abs_error": loss_error,
                "gradient_max_abs_error": float((a-b).abs().max()),
                "loss_tolerance": {"rtol": 1e-6, "atol": 1e-7},
                "gradient_tolerance": {"rtol": 1e-5, "atol": 1e-7},
                "prediction_equivalence": "identical forward graph; historical detector/data source unchanged" if unchanged else "same outputs used; historical detector changes require separate replay",
                "historical_detector_source_unchanged":unchanged,
                "historical_ref": OLD_REF}
    finally:
        if previous is None:
            current.weight_dict.pop("loss_rel_small", None)
        else:
            current.weight_dict["loss_rel_small"] = previous


def target_losses(name, batch, outputs, targets, criterion, input_hw, rank, world):
    """Per-target regression contributions; they are scalars, not additive gradient shares."""
    from src.zoo.rtdetr.box_ops import generalized_box_iou
    num = torch.tensor(float(sum(len(t["labels"]) for t in targets)),device=outputs["pred_boxes"].device)
    if world>1:
        torch.distributed.all_reduce(num)
    denominator = max(float(num)/world,1.0)
    branches = [("final",outputs)]
    for family in ("aux","enc","dn"):
        branches.extend((f"{family}_{i}",b) for i,b in enumerate(outputs.get(f"{family}_aux_outputs" if family != "aux" else "aux_outputs",[])))
    rows = []
    for branch,output in branches:
        if branch.startswith("dn_"):
            indices = criterion.get_cdn_matched_indices(outputs["dn_meta"],targets)
            denom = denominator*outputs["dn_meta"]["dn_num_group"]
        else:
            indices = criterion.matcher(output,targets)["indices"]
            denom = denominator
        for image_index,(queries,matched) in enumerate(indices):
            t = targets[image_index]
            boxes = output["pred_boxes"][image_index,queries].detach()
            gt = t["boxes"][matched].as_subclass(torch.Tensor)
            labels = t["labels"][matched]
            raw_bbox = (boxes-gt).abs().sum(-1)
            raw_giou = 1-torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(boxes),box_cxcywh_to_xyxy(gt)))
            ref_size = gt[:,2:].prod(-1).sqrt()*criterion.relative_small_reference_size
            small = ref_size<criterion.relative_small_threshold_px
            ref_error = (boxes-gt)/gt[:,[2,3,2,3]].clamp_min(criterion.relative_small_min_scale_px/criterion.relative_small_reference_size)
            raw_relative = torch.nn.functional.smooth_l1_loss(ref_error,torch.zeros_like(ref_error),reduction="none").sum(-1)*small
            active = criterion.relative_small_loss and (branch=="final" or branch.startswith("aux_"))
            for i in range(len(matched)):
                wh = gt[i,2:]*gt.new_tensor([input_hw[1],input_hw[0]])
                area = float(wh.prod())
                rows.append({"run":name,"batch":batch,"rank":rank,"branch":branch,
                    "image_id":int(t["image_id"]),"query_id":int(queries[i]),"target_index":int(matched[i]),
                    "category_id":int(labels[i]),"training_area_px":area,
                    "training_scale":"small" if area<1024 else "medium" if area<9216 else "large",
                    "small_reference_mask":bool(small[i]),"denominator":denom,
                    "bbox_raw":float(raw_bbox[i]),"bbox_contribution":float(raw_bbox[i])*criterion.weight_dict["loss_bbox"]/denom,
                    "giou_raw":float(raw_giou[i]),"giou_contribution":float(raw_giou[i])*criterion.weight_dict["loss_giou"]/denom,
                    "relative_raw_if_active":float(raw_relative[i]) if active else 0.,
                    "relative_contribution":float(raw_relative[i])*criterion.weight_dict.get("loss_rel_small",0)/denom if active else 0.})
    return rows


def distribution(values):
    values = np.array([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if not len(values):
        return {"count":0}
    return {"count":len(values),"mean":float(values.mean()),"P50":float(np.quantile(values,.5)),
            "P90":float(np.quantile(values,.9)),"P99":float(np.quantile(values,.99)),"max":float(values.max())}


def gradient_rows(name, batch_index, model, outputs, targets, criterion, device, input_hw, precision, clip):
    losses = criterion(outputs, targets)
    zero = outputs["pred_boxes"].sum()*0
    branches = {"final": outputs}
    for family, output_key in (("aux", "aux_outputs"), ("dn", "dn_aux_outputs"), ("enc", "enc_aux_outputs")):
        for i, branch in enumerate(outputs.get(output_key, [])):
            branches[f"{family}_{i}"] = branch
    groups = {"bbox_head": [p for n,p in model.named_parameters() if p.requires_grad and ".dec_bbox_head." in n],
              "decoder_shared": [p for n,p in model.named_parameters() if p.requires_grad and (".decoder.layers." in n or ".query_pos_head." in n)]}
    counts = torch.tensor([sum(len(t["labels"]) for t in targets), 0, 0, 0, 0], device=device, dtype=torch.float64)
    for t in targets:
        wh = t["boxes"][:, 2:].as_subclass(torch.Tensor)
        ref_size = (wh.prod(-1).clamp_min(0)).sqrt()*640
        area = wh.prod(-1)*input_hw[0]*input_hw[1]
        counts[1] += (ref_size < 32).sum()
        counts[2] += (area < 1024).sum()
        counts[3] += ((area >= 1024)&(area < 9216)).sum()
        counts[4] += (area >= 9216).sum()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(counts)
    full_parameters = [p for p in model.parameters() if p.requires_grad]
    total_vector = distributed_vector(sum(losses.values()), full_parameters, True)
    pre_norm = float(total_vector.norm())
    coefficient = min(1.0, clip/(pre_norm+1e-6)) if clip > 0 else 1.0
    if not torch.isfinite(total_vector).all():
        raise FloatingPointError("nonfinite full gradient")
    rows = []
    all_groups = dict(groups, prediction_boxes=[b["pred_boxes"] for b in branches.values()])
    # Compute all requested groups in one backward traversal per scalar term.
    # Form the aggregate vectors by adding branch vectors, rather than repeating backward.
    flattened = [t for tensors in all_groups.values() for t in tensors]
    lengths = [sum(t.numel() for t in tensors) for tensors in all_groups.values()]
    widths = [len(tensors) for tensors in all_groups.values()]
    aggregate = {term:{g:torch.zeros(n,device=device) for g,n in zip(all_groups,lengths)}
                 for term in ("bbox","giou","relative","classification","quality")}
    aggregate_values = {term:zero for term in aggregate}
    branch_data = {}
    for branch in branches:
        def select(prefix):
            return sum((v for k,v in losses.items() if k.startswith(prefix) and
                        k == prefix + ("" if branch == "final" else "_"+branch)), zero)
        terms = {"bbox": select("loss_bbox"), "giou": select("loss_giou"),
                 "relative": select("loss_rel_small"), "classification": select("loss_vfl")}
        terms["quality"] = select("loss_quality")
        vectors = {}
        for term,value in terms.items():
            grads = torch.autograd.grad(value,flattened,retain_graph=True,allow_unused=True)
            start = 0
            vectors[term] = {}
            for (group,tensors),width in zip(all_groups.items(),widths):
                vector = torch.cat([(torch.zeros_like(t) if g is None else g).detach().float().flatten()
                                    for t,g in zip(tensors,grads[start:start+width])])
                start += width
                if torch.distributed.is_initialized():
                    if group == "prediction_boxes":
                        chunks = [torch.empty_like(vector) for _ in range(torch.distributed.get_world_size())]
                        torch.distributed.all_gather(chunks,vector)
                        vector = torch.cat(chunks)/torch.distributed.get_world_size()
                    else:
                        torch.distributed.all_reduce(vector)
                        vector /= torch.distributed.get_world_size()
                vectors[term][group] = vector
                if aggregate[term][group].shape != vector.shape:
                    aggregate[term][group] = torch.zeros_like(vector)
                aggregate[term][group] += vector
            aggregate_values[term] = aggregate_values[term]+value
        branch_data[branch] = (terms,vectors)
    branch_data["all"] = (aggregate_values,aggregate)
    for branch,(terms,vectors) in branch_data.items():
        terms["base_regression"] = terms["bbox"]+terms["giou"]
        vectors["base_regression"] = {g:vectors["bbox"][g]+vectors["giou"][g] for g in all_groups}
        for group in all_groups:
            base_vector = vectors["base_regression"][group]
            base_norm = float(base_vector.norm())
            for term, value in terms.items():
                vector = vectors[term][group]
                norm = float(vector.norm())
                raw = None
                weight_key = {"bbox":"loss_bbox", "giou":"loss_giou", "relative":"loss_rel_small", "classification":"loss_vfl", "quality":"loss_quality"}.get(term)
                if weight_key and criterion.weight_dict.get(weight_key, 0) != 0:
                    raw = float(value.detach()) / criterion.weight_dict[weight_key]
                weighted = value.detach().double().clone()
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(weighted)
                    weighted /= torch.distributed.get_world_size()
                if raw is not None:
                    raw = float(weighted)/criterion.weight_dict[weight_key]
                finite = bool(torch.isfinite(vector).all()) and math.isfinite(float(weighted))
                rows.append({"run":name,"batch":batch_index,"branch":branch,"group":group,"term":term,
                    "raw_loss":raw,"weighted_loss":float(weighted),"grad_norm":norm if finite else None,
                    "base_reg_grad_norm":base_norm,"ratio":norm/base_norm if base_norm>1e-12 and finite else None,
                    "base_near_zero":base_norm<=1e-12,
                    "cosine":float(torch.dot(vector,base_vector)/(norm*base_norm)) if norm>1e-12 and base_norm>1e-12 and finite else None,
                    "finite":finite,"global_gt":int(counts[0]),"small_640":int(counts[1]),
                    "small_input":int(counts[2]),"medium_input":int(counts[3]),"large_input":int(counts[4]),
                    "full_pre_clip_norm":pre_norm,"clip_coefficient":coefficient,
                    "group_post_clip_norm":norm*coefficient if finite and group!="prediction_boxes" else None,
                    "normalization":"all_GT_mean_per_rank_clamp1; parameter vectors rank-mean; prediction vectors concatenated/world",
                    "precision":precision,"evidence":"synthetic_fixture" if name=="SYNTHETIC" else "fixed_real_training_batch"})
    # Keep per-key losses as well, including every DN/aux/encoder branch.
    return rows, {k: float(v.detach()) for k,v in losses.items()}


def probe_trace(probe_spec, out, device, rank, world, expected_val):
    """Explicit historical-probe replay with image IDs; no new probe training."""
    from src.solver.query_stats import FinalQualityProbeStats, ClassConditionedQualityProbeStats
    from src.data import CocoEvaluator
    cfg = fresh_config(str(ROOT/probe_spec["config"]))
    if config_diff(cfg.yaml_cfg["val_dataloader"],expected_val):
        raise ValueError("probe validation config differs from main audit protocol")
    cfg.yaml_cfg["PResNet"]["pretrained"] = False
    model = cfg.model.to(device).eval()
    report = weight_load(model, probe_spec["checkpoint"], probe_spec.get("weights","model"))
    if not report["strict_load_passed"]:
        raise ValueError(f"probe checkpoint does not strictly match config: {report}")
    loader = cfg.val_dataloader
    # Replay the historical padding path and record the visited image ID at update.
    from torch.utils.data import DataLoader, DistributedSampler
    sampler = DistributedSampler(loader.dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                        collate_fn=loader.collate_fn, num_workers=0, drop_last=loader.drop_last)
    evaluator = CocoEvaluator(loader.dataset.coco,["bbox"])
    post = cfg.postprocessor.to(device).eval()
    probe = ClassConditionedQualityProbeStats(None) if model.decoder.class_conditioned_final_quality_probe else FinalQualityProbeStats(None)
    def sample_count():
        chunks = probe.scores if model.decoder.class_conditioned_final_quality_probe else probe.topn_chunks[100]["scores"]
        return sum(chunk.numel() for chunk in chunks)
    trace = []
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            outputs = model(samples)
            # One-image updates let us capture each contribution without altering algorithms.
            for i,t in enumerate(targets):
                before = sample_count()
                single = {k:v[i:i+1] for k,v in outputs.items() if k in ("pred_boxes","pred_logits","pred_quality_logits")}
                probe.update(single,[t],samples.shape[-2:])
                after = sample_count()
                trace.append({"image_id":int(t["image_id"]),"samples":after-before,"rank":rank})
            predictions = post(outputs,torch.stack([t["orig_size"] for t in targets]))
            evaluator.update({int(t["image_id"]):p for t,p in zip(targets,predictions)})
    dump(out / f"probe_trace_rank{rank}.json", {"loading":report,"trace":trace})
    evaluator.synchronize_between_processes()
    if rank==0:
        traces = []
        for r in range(world):
            traces.extend(json.loads((out/f"probe_trace_rank{r}.json").read_text(encoding="utf-8"))["trace"])
        visits = Counter(t["image_id"] for t in traces)
        first = {}
        for t in traces:
            first.setdefault(t["image_id"],t["samples"])
        dump(out/"probe_complete.json",{"raw_visits":len(traces),"unique_images":len(visits),
            "raw_sample_count":sum(t["samples"] for t in traces),"deduplicated_sample_count":sum(first.values()),
            "duplicated_image_ids":{str(k):v for k,v in visits.items() if v>1},
            "COCO_unique_images":len(evaluator.coco_eval["bbox"].params.imgIds),
            "drop_last":loader.drop_last,"world_size":world,"bad_image_skipping":False,
            "population":"class_top100_pairs" if model.decoder.class_conditioned_final_quality_probe else "class_top100_queries",
            "replay":"current code with explicit historical checkpoint; historical launch GPU count still needs logs"})


def run_eval(name, model, cfg, annotations, out, device, rank, world, limit=None):
    from torch.utils.data import DataLoader, Subset
    from src.data import CocoEvaluator
    loader = cfg.val_dataloader
    total = len(loader.dataset) if limit is None else min(limit,len(loader.dataset))
    indices = list(range(rank,total,world))
    loader = DataLoader(Subset(loader.dataset,indices),batch_size=loader.batch_size,
                        collate_fn=loader.collate_fn,num_workers=0,drop_last=False)
    capture = LayerCapture(model)
    evaluators, per_image, seen = {}, [], []
    post = cfg.postprocessor.to(device).eval()
    if post.final_score_method != "default" or post.final_quality_gamma != 0:
        raise ValueError("diagnostic query ID extraction requires the original scoring protocol")
    image_map = {x["id"]:x for x in annotations["images"]}
    ann_map = defaultdict(list)
    for ann in annotations["annotations"]:
        ann_map[ann["image_id"]].append(ann)
    export = out / f"queries_rank{rank}"
    export.mkdir(exist_ok=True)
    model.eval()
    try:
        with torch.no_grad():
            for batch_index,(samples,targets) in enumerate(loader):
                capture.clear()
                outputs = model(samples.to(device))
                stages = capture.collect(outputs)
                updates = {stage:{} for stage,_,_ in stages}
                for i,target in enumerate(targets):
                    image_id = int(target["image_id"])
                    if image_id in seen:
                        raise ValueError("unexpected duplicate in padding-free eval")
                    seen.append(image_id)
                    singles = [(s,b[i],l[i]) for s,b,l in stages]
                    records,trajectories,predictions = geometries(singles,image_map[image_id],ann_map[image_id],post,export/f"image_{image_id}.npz")
                    per_image.append({"image_id":image_id,"geometry":records,"trajectories":trajectories})
                    for stage,prediction in predictions.items():
                        updates[stage][image_id] = prediction
                    # Preserve encoder flattened token indices separately from regular query IDs.
                    np.save(export/f"image_{image_id}_encoder_token_ids.npy",outputs["enc_topk_indices"][i].cpu().numpy())
                for stage,predictions in updates.items():
                    if stage not in evaluators:
                        evaluators[stage] = CocoEvaluator(loader.dataset.dataset.coco,["bbox"])
                    evaluators[stage].update(predictions)
                if batch_index%20==0:
                    print(f"{name}: rank {rank} eval batch {batch_index}, visits={len(seen)}",flush=True)
    finally:
        capture.close()
    dump(out/f"geometry_rank{rank}.json",per_image)
    dump(out/f"image_visits_rank{rank}.json",seen)
    for evaluator in evaluators.values():
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
    if rank==0:
        items,visits = [],[]
        for r in range(world):
            items.extend(json.loads((out/f"geometry_rank{r}.json").read_text(encoding="utf-8")))
            visits.extend(json.loads((out/f"image_visits_rank{r}.json").read_text(encoding="utf-8")))
        if len(visits)!=total or len(set(visits))!=total:
            raise ValueError("eval visit counts differ from expected sample set")
        rows,refined = reduce_localization(name,items,annotations["categories"])
        metrics = {}
        for stage,evaluator in evaluators.items():
            rows.extend(coco_rows(name,stage,evaluator,annotations))
            ev = evaluator.coco_eval["bbox"]
            metrics[stage] = {"stats":np.asarray(ev.stats).tolist(),"maxDets":list(ev.params.maxDets),
                              "iouThrs":np.asarray(ev.params.iouThrs).tolist(),"areaRng":ev.params.areaRng,
                              "useCats":ev.params.useCats,"catIds":list(ev.params.catIds),"imgIds":list(ev.params.imgIds)}
            torch.save(ev.eval,out/f"{stage}_coco_eval.pth")
        write_csv(out/"localization.csv",rows,LOC_FIELDS)
        result = {"unique_images":total,"raw_visits":len(visits),"duplicate_visits":len(visits)-len(set(visits)),
                  "sample_set":"full_validation" if limit is None else "limited_smoke_NOT_full_AP",
                  "metrics":metrics,"refinement":refined,"no_DN_queries":True,"capture_final_exact":True}
        dump(out/"eval_complete.json",result)


def run_gradients(name,model,cfg,out,shared,device,rank,world,number,seed,epoch,precision):
    from torch.utils.data import DataLoader,DistributedSampler
    loader = cfg.train_dataloader
    if hasattr(loader.dataset,"set_epoch"):
        loader.dataset.set_epoch(epoch)
    if hasattr(loader.collate_fn,"set_epoch"):
        loader.collate_fn.set_epoch(epoch)
    sampler = DistributedSampler(loader.dataset,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=True)
    sampler.set_epoch(epoch)
    loader = DataLoader(loader.dataset,batch_size=loader.batch_size,sampler=sampler,
                        collate_fn=loader.collate_fn,num_workers=0,drop_last=True)
    if len(loader)<number:
        raise ValueError(f"requested {number} fixed batches but train loader only has {len(loader)}")
    criterion = cfg.criterion.to(device)
    # Baseline/QAQS audit their own criterion; the failed run also tests historical equivalence.
    historical = old_criterion(criterion).to(device)
    buffers = {n:b.clone() for n,b in model.named_buffers()}
    rows,loss_records,per_target,equivalence = [],[],[],None
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    shared.mkdir(exist_ok=True)
    batches = iter(loader)
    # Generate the entire fixed input cache before any model RNG consumption.
    for i in range(number):
        samples,targets = next(batches)
        cached = shared/f"batch_{i:03}_rank{rank}.pth"
        if not cached.exists():
            torch.save({"samples":samples,"targets":targets,"seed":seed,"audit_epoch":epoch,"rank":rank},cached)
    for i in range(number):
        cached = shared/f"batch_{i:03}_rank{rank}.pth"
        batch = torch.load(cached,map_location="cpu",weights_only=False) if "weights_only" in inspect.signature(torch.load).parameters else torch.load(cached,map_location="cpu")
        samples,targets = batch["samples"],batch["targets"]
        samples = samples.to(device)
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        model.train()
        with torch.no_grad():
            for n,b in model.named_buffers():
                b.copy_(buffers[n])
        torch.manual_seed(seed+rank+i)
        with torch.autocast(device_type=device.type,enabled=precision=="amp"):
            outputs = model(samples,targets=targets)
        if i==0:
            equivalence = zero_equivalence(model,outputs,targets,criterion,historical)
        with torch.autocast(device_type=device.type,enabled=False):
            batch_rows,raw_losses = gradient_rows(name,i,model,outputs,targets,criterion,device,samples.shape[-2:],precision,cfg.yaml_cfg.get("clip_max_norm",0))
            per_target.extend(target_losses(name,i,outputs,targets,criterion,samples.shape[-2:],rank,world))
        rows.extend(batch_rows)
        loss_records.append({"batch":i,"weighted_loss_dict":raw_losses,
                             "image_ids":[int(t["image_id"]) for t in targets],
                             "fixed_batch_sha256":sha(cached),"input_hw":list(samples.shape[-2:])})
        if rank==0:
            print(f"{name}: gradient batch {i+1}/{number}",flush=True)
    with torch.no_grad():
        for n,b in model.named_buffers():
            b.copy_(buffers[n])
    model.eval()
    dump(out/f"gradient_losses_rank{rank}.json",loss_records)
    dump(out/f"loss_per_target_rank{rank}.json",per_target)
    if world>1:
        torch.distributed.barrier()
    if rank==0:
        write_csv(out/"gradient.csv",rows,GRAD_FIELDS)
        groups = defaultdict(list)
        for row in rows:
            groups[f"{row['branch']}:{row['group']}:{row['term']}"].append(row)
        distributions = {k:{field:distribution([r[field] for r in rs]) for field in
                         ("raw_loss","weighted_loss","grad_norm","ratio","cosine","full_pre_clip_norm","clip_coefficient")}
                         for k,rs in groups.items()}
        all_targets = []
        for r in range(world):
            all_targets.extend(json.loads((out/f"loss_per_target_rank{r}.json").read_text(encoding="utf-8")))
        if all_targets:
            write_csv(out/"loss_per_target.csv",all_targets,list(all_targets[0]))
        strata = defaultdict(list)
        for row in all_targets:
            strata[f"{row['branch']}:{row['category_id']}:{row['training_scale']}"].append(row)
        target_summary = {k:{"matched_count":len(rs),**{field:{"sum":sum(r[field] for r in rs),
                         "distribution":distribution([r[field] for r in rs])} for field in
                         ("bbox_contribution","giou_contribution","relative_contribution")}} for k,rs in strata.items()}
        dump(out/"loss_gradient_distributions.json",{"gradient_strata":distributions,"class_scale_loss_contributions":target_summary,
             "scope":"matched branch-target contributions; DN repeated targets are counted per group; not additive gradient shares"})
        dump(out/"gradient_complete.json",{"batches":number,"zero_equivalence":equivalence,
                                          "optimizer_steps":0,"EMA_updates":0,"seed":seed,"audit_epoch":epoch,"precision":precision})


def self_test(out):
    runtime()
    from src.zoo.rtdetr.matcher import HungarianMatcher
    torch.set_num_threads(2)
    torch.manual_seed(0)
    criterion = RTDETRCriterionv2(HungarianMatcher({"cost_class":2,"cost_bbox":5,"cost_giou":2}),
                    {"loss_vfl":1,"loss_bbox":5,"loss_giou":2,"loss_rel_small":.5},["vfl","boxes"],num_classes=3,relative_small_loss=True)
    def prediction():
        return {"pred_boxes":torch.rand(2,10,4).requires_grad_(),"pred_logits":torch.rand(2,10,3).requires_grad_()}
    outputs = prediction()
    outputs["aux_outputs"] = [prediction(),prediction()]
    outputs["enc_aux_outputs"] = [prediction()]
    outputs["enc_meta"] = {"class_agnostic":False}
    targets = [{"labels":torch.tensor([0,1]),"boxes":torch.tensor([[.3,.3,8/640,8/640],[.6,.6,.2,.2]])} for _ in range(2)]
    outputs["dn_aux_outputs"] = [prediction() for _ in range(3)]
    outputs["dn_meta"] = {"dn_positive_idx":[torch.tensor([0,1]),torch.tensor([0,1])],"dn_num_group":1}
    old = old_criterion(criterion)
    # Compare gradients on every synthetic prediction tensor, across all branches.
    holder = torch.nn.ParameterList()
    for branch in [outputs]+outputs["aux_outputs"]+outputs["enc_aux_outputs"]+outputs["dn_aux_outputs"]:
        for key in ("pred_boxes","pred_logits"):
            holder.append(torch.nn.Parameter(branch[key].detach()))
            branch[key] = holder[-1]
    equivalent = zero_equivalence(holder,outputs,targets,criterion,old)
    active = criterion(outputs,targets)
    rel_keys = sorted(k for k in active if k.startswith("loss_rel_small"))
    assert rel_keys==["loss_rel_small","loss_rel_small_aux_0","loss_rel_small_aux_1"],rel_keys
    numeric = []
    for px in (1,4,8,16,31.99,32,32.01,64):
        t = [{"labels":torch.tensor([0]),"boxes":torch.tensor([[.5,.5,px/640,px/640]])}]
        b = t[0]["boxes"].clone();b[:,0]+=1/640;b.requires_grad_()
        loss = criterion.loss_relative_small_boxes({"pred_boxes":b[None]},t,[(torch.tensor([0]),torch.tensor([0]))],1)["loss_rel_small"]
        grad = torch.autograd.grad(loss,b)[0]
        assert torch.isfinite(loss) and torch.isfinite(grad).all()
        expected = .5*(1/max(px,4))**2 if px<32 else 0.
        assert math.isclose(float(loss),expected,rel_tol=1e-4,abs_tol=1e-8)
        if px>=32:
            assert loss.item()==0 and grad.abs().sum()==0
        numeric.append({"width_px":px,"one_pixel_dx_loss":float(loss),"dx_gradient":float(grad[0,0])})
    empty_t = [{"labels":torch.empty(0,dtype=torch.long),"boxes":torch.empty(0,4)}]
    empty_b = torch.empty(1,0,4,requires_grad=True)
    empty = criterion.loss_relative_small_boxes({"pred_boxes":empty_b},empty_t,[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))],1)["loss_rel_small"]
    assert empty.item()==0 and torch.isfinite(torch.autograd.grad(empty,empty_b)[0]).all()
    # Asymmetric box independently verifies x/w and y/h units and width/height terms.
    asym_t = [{"labels":torch.tensor([0]),"boxes":torch.tensor([[.5,.5,8/640,16/640]])}]
    asymmetric = []
    for coordinate in range(4):
        b = asym_t[0]["boxes"].clone();b[0,coordinate]+=1/640;b.requires_grad_()
        loss = criterion.loss_relative_small_boxes({"pred_boxes":b[None]},asym_t,[(torch.tensor([0]),torch.tensor([0]))],1)["loss_rel_small"]
        expected = .5*(1/(8 if coordinate in (0,2) else 16))**2
        assert math.isclose(float(loss),expected,rel_tol=1e-4)
        asymmetric.append({"coordinate":coordinate,"loss":float(loss),"expected":expected})
    # Maximum total IoU would select only one >=.75 edge; cardinality must find two.
    adversarial = np.array([[.99,.76],[.76,.74]])
    assert cardinality(adversarial,.75)==2
    from torch.utils.data import DistributedSampler
    visits = [list(DistributedSampler(range(3667),num_replicas=2,rank=r,shuffle=False,drop_last=False)) for r in range(2)]
    joined = visits[0]+visits[1]
    assert len(joined)==3668 and len(set(joined))==3667
    # Build all configs through repository factories, explicitly no pretrained download.
    construction = {}
    for name,path in CONFIGS.items():
        cfg = fresh_config(str(ROOT/path));cfg.yaml_cfg["PResNet"]["pretrained"]=False
        model = cfg.model
        construction[name] = {"parameters":sum(p.numel() for p in model.parameters()),"layers":model.decoder.num_layers,
                              "criterion":type(cfg.criterion).__name__,"postprocessor":type(cfg.postprocessor).__name__}
    cfg = fresh_config(str(ROOT/CONFIGS["qaqs"]))
    cfg.yaml_cfg["PResNet"]["pretrained"]=False
    cfg.yaml_cfg["eval_spatial_size"]=[64,64]
    cfg.yaml_cfg["RTDETRTransformerv2"]["num_queries"]=20
    model = cfg.model.eval()
    for head in model.decoder.dec_bbox_head:
        torch.nn.init.normal_(head.layers[-1].weight,std=.01)
        torch.nn.init.normal_(head.layers[-1].bias,std=.01)
    capture = LayerCapture(model)
    with torch.no_grad():
        outputs = model(torch.rand(2,3,64,64))
        stages = capture.collect(outputs)
    capture.close()
    assert len(stages)==4 and all(b.shape==(2,20,4) for _,b,_ in stages)
    torch.save({"model":model.state_dict(),"ema":{"module":model.state_dict()},"last_epoch":2},out/"synthetic_checkpoint.pth")
    for key in ("ema","model"):
        assert weight_load(model,out/"synthetic_checkpoint.pth",key)["strict_load_passed"]
    torch.save({"model":model.state_dict()},out/"synthetic_no_ema.pth")
    try:
        weight_load(model,out/"synthetic_no_ema.pth","ema")
    except ValueError:
        pass
    else:
        raise AssertionError("missing EMA silently fell back")
    # Exercise COCO, class/scale rows and gradient CSV paths with generated data.
    from PIL import Image
    fixture = out/"synthetic_fixture";fixture.mkdir()
    images,anns = [],[]
    for i in range(4):
        Image.fromarray(np.random.default_rng(i).integers(0,255,(64,64,3),dtype=np.uint8)).save(fixture/f"{i}.png")
        images.append({"id":i,"file_name":f"{i}.png","width":64,"height":64})
        anns.append({"id":i,"image_id":i,"category_id":i%3,"bbox":[12,14,2,3],"area":6,"iscrowd":0})
    fixture_data = {"images":images,"annotations":anns,"categories":[{"id":i,"name":f"synthetic_{i}"} for i in range(3)]}
    dump(fixture/"annotations.json",fixture_data)
    for loader_name in ("train_dataloader","val_dataloader"):
        cfg.yaml_cfg[loader_name]["total_batch_size"]=2
        cfg.yaml_cfg[loader_name]["num_workers"]=0
        cfg.yaml_cfg[loader_name]["dataset"]["img_folder"]=str(fixture)
        cfg.yaml_cfg[loader_name]["dataset"]["ann_file"]=str(fixture/"annotations.json")
        for op in cfg.yaml_cfg[loader_name]["dataset"]["transforms"]["ops"]:
            if op["type"]=="Resize":
                op["size"]=[64,64]
    cfg.yaml_cfg["RTDETRPostProcessor"]["num_top_queries"]=30
    cfg._postprocessor=None
    cfg._criterion=None
    cfg.yaml_cfg["RTDETRCriterionv2"]["relative_small_loss"]=True
    cfg.yaml_cfg["RTDETRCriterionv2"]["weight_dict"]["loss_rel_small"]=.5
    smoke = out/"synthetic_pipeline";smoke.mkdir()
    run_eval("SYNTHETIC",model,cfg,fixture_data,smoke,torch.device("cpu"),0,1)
    run_gradients("SYNTHETIC",model,cfg,smoke,out/"synthetic_fixed_batches",torch.device("cpu"),0,1,1,0,0,"fp32")
    dump(out/"synthetic_checks.json",{"status":"passed_synthetic_NOT_real_experiment", "historical_equivalence":equivalent,
         "relative_branch_keys":rel_keys,"numeric_cases":numeric,"empty_GT_finite":True,"maximum_cardinality_counterexample":True,
         "padding_example":{"visits":len(joined),"unique":len(set(joined))},"config_construction":construction,
         "asymmetric_coordinate_cases":asymmetric,"synthetic_pipeline_eval_and_gradients":True,
         "strict_model_EMA_loading_and_no_fallback":True,
         "eval_hook_final_exact":True,"stages":[s[0] for s in stages],"torch":torch.__version__})
    print("Synthetic checks passed; real checkpoints/dataset not evaluated",flush=True)


def run_audit(args,out,manifest):
    runtime()
    from src.data import CocoEvaluator
    import faster_coco_eval
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    world,rank = int(os.environ.get("WORLD_SIZE",1)),int(os.environ.get("RANK",0))
    local = int(os.environ.get("LOCAL_RANK",0))
    if world>1:
        torch.cuda.set_device(local)
        torch.distributed.init_process_group("nccl")
    device = torch.device("cpu" if args.preflight else f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if world>1 and device.type!="cuda":
        raise ValueError("two-rank server audit requires CUDA")
    seed = int(spec.get("seed",0))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    precision = spec.get("gradient_precision","fp32")
    if int(spec.get("gradient_batches",16))<1:
        raise ValueError("gradient_batches must be positive")
    if precision not in ("fp32","amp"):
        raise ValueError("gradient_precision must be fp32 or amp")
    weight_key = spec.get("weights","ema")
    gradient_weight_key = spec.get("gradient_weights","model")
    if weight_key not in ("ema","model"):
        raise ValueError("weights must explicitly be ema or model")
    if gradient_weight_key not in ("ema","model"):
        raise ValueError("gradient_weights must explicitly be ema or model")
    runs = spec["runs"]
    if {r["name"] for r in runs} != set(CONFIGS) or len(runs)!=3:
        raise ValueError("spec must contain exactly baseline, qaqs and relative_failed")
    for run in runs:
        if not run.get("checkpoint") or not Path(run["checkpoint"]).is_file():
            raise ValueError(f"explicit existing checkpoint required for {run['name']}")
    provenance = {"spec_sha256":sha(args.spec),"world_size":world,"gradient_batches":spec.get("gradient_batches",16),
                  "eval_limit":args.eval_limit,"source_sha256":sha(__file__),
                  "repository_sources":{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"src").rglob("*.py")},
                  "criterion_sha256":sha(ROOT/CRITERION_PATH),"checkpoints":{r["name"]:sha(r["checkpoint"]) for r in runs},
                  "probe_checkpoints":{p["name"]:sha(p["checkpoint"]) for p in spec.get("probes",[])}}
    configs = [read_yaml(ROOT/r["config"]) for r in runs]
    if any(config_diff(configs[0]["val_dataloader"],c["val_dataloader"]) for c in configs[1:]):
        raise ValueError("validation protocols differ; supply documented matching configs")
    if any(config_diff(configs[0]["train_dataloader"],c["train_dataloader"]) for c in configs[1:]):
        raise ValueError("training input protocols differ; fixed-batch comparison would not be fair")
    if any(c.get("remap_mscoco_category") for c in configs):
        raise ValueError("audit currently requires the direct OGSOD category protocol")
    val = configs[0]["val_dataloader"]["dataset"]
    train = configs[0]["train_dataloader"]["dataset"]
    val_info,annotations = annotation_audit(val["ann_file"],val["img_folder"],configs[0]["num_classes"])
    train_info,train_annotations = annotation_audit(train["ann_file"],train["img_folder"],configs[0]["num_classes"])
    if train_info["categories"]!=val_info["categories"]:
        raise ValueError("train and validation categories differ")
    train_names = {x["file_name"] for x in train_annotations["images"]}
    val_names = {x["file_name"] for x in annotations["images"]}
    overlap = sorted(train_names&val_names)
    if overlap:
        raise ValueError(f"train/val file-name overlap: {overlap[:10]}")
    provenance.update({"train_sha256":train_info["sha256"],"val_sha256":val_info["sha256"],
                       "resolved_configs":configs,"historical_criterion_sha256":hashlib.sha256(git("show",f"{OLD_REF}:{CRITERION_PATH}").encode()).hexdigest()})
    if (out/"audit_identity.json").exists():
        if json.loads((out/"audit_identity.json").read_text(encoding="utf-8"))!=provenance:
            raise ValueError("resume identity differs: spec/data/weights/source/config/world-size/limits")
    elif rank==0:
        dump(out/"audit_identity.json",provenance)
    if world>1:
        torch.distributed.barrier()
    manifest.update({"status":"real_audit_in_progress","missing":[],"dataset":{"train":train_info,"val":val_info,
        "file_name_overlap":overlap,"byte_hash_and_near_duplicate_check":"not_performed; file-name disjointness is not proof of source independence"},
        "audit_world_size":world,"audit_seed":seed,"weight_selection":weight_key,"gradient_weight_selection":gradient_weight_key,
        "runtime":{"torch":torch.__version__,"faster_coco_eval":faster_coco_eval.__version__,
                   "evaluator_module":inspect.getfile(CocoEvaluator),
                   "evaluator_library_source_sha256":sha(inspect.getfile(CocoEvaluator.__bases__[0]))}})
    if not args.preflight:
        estimate = int(spec.get("gradient_batches",16))*configs[0]["train_dataloader"]["total_batch_size"]*3*640*640*4
        cached_bytes = sum(p.stat().st_size for p in (out/"fixed_batches").glob("*.pth"))
        required_free = max(0,estimate-cached_bytes)+1024**3
        manifest["disk"] = {"fixed_input_estimated_bytes":estimate,"free_bytes":shutil.disk_usage(out).free,
                            "required_free_bytes":required_free}
        if manifest["disk"]["free_bytes"]<required_free:
            raise OSError("not enough free disk for fixed batches and diagnostic exports")
    for run in runs:
        name = run["name"]
        cfg = fresh_config(str(ROOT/run["config"]))
        cfg.yaml_cfg["PResNet"]["pretrained"]=False
        model = cfg.model.to(device)
        if world>1 and cfg.yaml_cfg.get("sync_bn",False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        report = weight_load(model,run["checkpoint"],weight_key)
        manifest["runs"][name].update(report)
        manifest["runs"][name].update({k:run.get(k) for k in ("run_git_commit","parent_checkpoint","launch_command")})
        manifest["runs"][name]["evidence"] = evidence_directory(run.get("run_dir",cfg.yaml_cfg["output_dir"]))
        if not report["strict_load_passed"]:
            if rank==0:
                dump(out/"eval_manifest.json",manifest)
            raise ValueError(f"strict state mismatch for {name}; report saved, evaluation refused")
        if any(run.get(k) is None for k in ("run_git_commit","parent_checkpoint","launch_command")):
            manifest["missing"].append(f"{name}: exact run commit/initialization/launch provenance incomplete; audit commit is not run commit")
        manifest["runs"][name]["gradient_sync_bn"] = world>1 and cfg.yaml_cfg.get("sync_bn",False)
        if args.preflight:
            gradient_report = weight_load(model,run["checkpoint"],gradient_weight_key)
            manifest["runs"][name]["gradient_weight_loading"] = gradient_report
            if not gradient_report["strict_load_passed"]:
                raise ValueError(f"gradient checkpoint mismatch: {name}")
            del model,cfg
            continue
        run_out = out/name
        run_out.mkdir(exist_ok=True)
        if not (run_out/"eval_complete.json").exists():
            run_eval(name,model,cfg,annotations,run_out,device,rank,world,args.eval_limit)
        if world>1:
            torch.distributed.barrier()
        if not (run_out/"gradient_complete.json").exists():
            gradient_report = weight_load(model,run["checkpoint"],gradient_weight_key)
            manifest["runs"][name]["gradient_weight_loading"] = gradient_report
            if not gradient_report["strict_load_passed"]:
                raise ValueError(f"gradient checkpoint mismatch: {name}")
            run_gradients(name,model,cfg,run_out,out/"fixed_batches",device,rank,world,
                          int(spec.get("gradient_batches",16)),seed,int(spec.get("audit_epoch",0)),precision)
        if world>1:
            torch.distributed.barrier()
        del model,cfg
        if device.type=="cuda":
            torch.cuda.empty_cache()
        if rank==0:
            dump(out/"eval_manifest.json",manifest)
    if rank==0:
        if not spec.get("probes"):
            manifest["missing"].append("366800/366700: historical probe checkpoints/launch GPU counts not provided; DDP padding hypothesis only")
    if args.preflight:
        if rank==0:
            manifest["status"]="preflight_weights_data_configs_checked_NO_evaluation"
            dump(out/"eval_manifest.json",manifest)
            summary(out,manifest)
            print("Preflight passed: strict checkpoint loads, annotation/image existence, category and config protocols",flush=True)
        return
    for probe in spec.get("probes",[]):
        if not probe.get("checkpoint"):
            raise ValueError("probe checkpoint must be explicit")
        probe_out = out/f"probe_{probe['name']}"
        probe_out.mkdir(exist_ok=True)
        if not (probe_out/"probe_complete.json").exists():
            probe_trace(probe,probe_out,device,rank,world,configs[0]["val_dataloader"])
        if world>1:
            torch.distributed.barrier()
    if rank==0:
        if spec.get("probes"):
            dump(out/"probe_count_audit.json",{"status":"current_code_checkpoint_replays_completed",
                "probes":{p["name"]:json.loads((out/f"probe_{p['name']}"/"probe_complete.json").read_text(encoding="utf-8")) for p in spec["probes"]},
                "historical_counts_conclusion":"compare replay IDs, val hashes and historical launch process count; current replay alone does not prove historical cause"})
        localization,gradients,refined = [],[],{}
        for run in runs:
            name = run["name"]
            with (out/name/"localization.csv").open(encoding="utf-8-sig",newline="") as stream:
                localization.extend(csv.DictReader(stream))
            with (out/name/"gradient.csv").open(encoding="utf-8-sig",newline="") as stream:
                gradients.extend(csv.DictReader(stream))
            refined[name] = json.loads((out/name/"eval_complete.json").read_text(encoding="utf-8"))
            manifest["runs"][name]["gradient_audit"] = json.loads((out/name/"gradient_complete.json").read_text(encoding="utf-8"))
        write_csv(out/"localization_by_class_scale.csv",localization,LOC_FIELDS)
        write_csv(out/"loss_gradient_audit.csv",gradients,GRAD_FIELDS)
        dump(out/"decoder_refinement_summary.json",{"status":"completed_real_audit","runs":refined})
        manifest["status"] = "completed_diagnostics_with_unresolved_provenance" if manifest["missing"] else "completed_real_audit"
        if args.eval_limit:
            manifest["status"] = "limited_real_smoke_NOT_full_evaluation"
        dump(out/"eval_manifest.json",manifest)
        summary(out,manifest)
    if world>1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--discover",action="store_true")
    mode.add_argument("--self-test",action="store_true")
    mode.add_argument("--spec",type=Path)
    directory = parser.add_mutually_exclusive_group(required=True)
    directory.add_argument("--output",type=Path)
    directory.add_argument("--resume-dir",type=Path)
    parser.add_argument("--eval-limit",type=int,help="smoke only; labels all metrics as incomplete")
    parser.add_argument("--preflight",action="store_true",help="with --spec: check paths, data, config and strict loads without evaluation")
    args = parser.parse_args()
    if args.preflight and (not args.spec or int(os.environ.get("WORLD_SIZE",1))!=1):
        parser.error("preflight requires --spec and ordinary Python, not torchrun")
    if args.eval_limit is not None and args.eval_limit<1:
        parser.error("eval-limit must be positive")
    rank = int(os.environ.get("RANK",0))
    out = args.output or args.resume_dir
    if args.resume_dir and not args.spec:
        parser.error("resume-dir is only valid with an explicit spec")
    if rank==0:
        if args.output:
            # A shell launcher may precreate only console.log and launcher.pid.
            if out.exists() and any(p.name not in ("console.log","launcher.pid") for p in out.iterdir()):
                raise FileExistsError(f"refusing nonempty output directory: {out}")
            out.mkdir(parents=True,exist_ok=True)
        elif not (out/"eval_manifest.json").exists():
            raise FileNotFoundError("resume directory has no audit manifest")
    if rank==0:
        manifest = static_audit(out) if not args.resume_dir else json.loads((out/"eval_manifest.json").read_text(encoding="utf-8"))
    else:
        manifest = {"runs":{name:{} for name in CONFIGS}}
    try:
        if args.self_test:
            self_test(out)
            manifest["local_verification"] = json.loads((out/"synthetic_checks.json").read_text(encoding="utf-8"))
            dump(out/"eval_manifest.json",manifest)
            summary(out,manifest)
        elif args.spec:
            run_audit(args,out,manifest)
        else:
            print(f"Evidence discovery written to {out}; exact checkpoints required",flush=True)
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=True)
        dump(out/f"failure_rank{rank}.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        if rank==0:
            manifest["status"]="incomplete_audit_failure"
            manifest.setdefault("missing",[]).append(str(exc))
            dump(out/"eval_manifest.json",manifest)
            summary(out,manifest)
        raise


if __name__=="__main__":
    main()
