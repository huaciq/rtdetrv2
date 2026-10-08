"""Prepare an external D-FINE R18 budget-matched config; never launch training."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import yaml

from localization_diagnosis import resolved_config, canonical_category
import baseline_error_analysis as base

UPSTREAM = "https://github.com/Peterande/D-FINE.git"
PIN = "956d1709314c2c6a4df6f34de232054578a7449f"
SHARED = ("num_classes", "remap_mscoco_category", "use_focal_loss", "eval_spatial_size", "PResNet",
          "optimizer", "lr_scheduler", "lr_warmup_scheduler", "use_amp", "scaler", "use_ema", "ema",
          "clip_max_norm", "sync_bn", "find_unused_parameters", "train_dataloader", "val_dataloader")
DECODER_COMMON = ("feat_channels", "feat_strides", "hidden_dim", "num_levels", "num_layers",
                  "num_queries", "num_denoising", "label_noise_ratio", "box_noise_scale", "eval_idx",
                  "num_points", "cross_attn_method", "query_select_method")


def build_config(pilot, native, overlay):
    cfg = copy.deepcopy(native)
    for key in SHARED:
        cfg[key] = copy.deepcopy(pilot[key])
    cfg["HybridEncoder"] = copy.deepcopy(pilot["HybridEncoder"])
    for key in DECODER_COMMON:
        if key in pilot["RTDETRTransformerv2"]:
            cfg["DFINETransformer"][key] = copy.deepcopy(pilot["RTDETRTransformerv2"][key])
    cfg["DFINECriterion"]["matcher"] = copy.deepcopy(pilot["RTDETRCriterionv2"]["matcher"])
    cfg["DFINEPostProcessor"]["num_top_queries"] = pilot["RTDETRPostProcessor"]["num_top_queries"]
    collate = cfg["train_dataloader"]["collate_fn"]
    if collate.pop("scales", None) is not None:
        raise ValueError("Unexpected baseline multiscale augmentation")
    collate.update(base_size=640, base_size_repeat=None, ema_restart_decay=.9999)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(copy.deepcopy(value))
        else:
            cfg[key] = copy.deepcopy(value)
    cfg.pop("__include__", None)
    if cfg["epochs"] != pilot["epoches"] or cfg["epochs"] != 20:
        raise ValueError("Exactly one fixed 20-epoch paired pilot is allowed")
    if cfg["train_dataloader"]["collate_fn"]["stop_epoch"] < cfg["epochs"]:
        raise ValueError("Native best-stg1 reload/EMA restart must remain outside pilot")
    if not cfg["PResNet"]["pretrained"] or cfg["PResNet"]["depth"] != 18:
        raise ValueError("Both pilots require the same ImageNet PResNet18 initialization")
    return cfg


def protocol(pilot, dfine):
    fields = {k: pilot[k] == dfine[k] for k in SHARED if k != "train_dataloader"}
    for key in ("dataset", "total_batch_size", "num_workers", "shuffle", "drop_last"):
        fields["train_"+key] = pilot["train_dataloader"].get(key) == dfine["train_dataloader"].get(key)
    fields["fixed_640_collate"] = (pilot["train_dataloader"]["collate_fn"].get("scales") is None
                                    and dfine["train_dataloader"]["collate_fn"]["base_size_repeat"] is None)
    fields["epochs"] = pilot["epoches"] == dfine["epochs"] == 20
    fields["matcher_cost_config"] = pilot["RTDETRCriterionv2"]["matcher"] == dfine["DFINECriterion"]["matcher"]
    for key in DECODER_COMMON:
        if key in pilot["RTDETRTransformerv2"]:
            fields["decoder_"+key] = pilot["RTDETRTransformerv2"][key] == dfine["DFINETransformer"][key]
    if not all(fields.values()):
        raise ValueError(f"Paired protocol mismatch: {fields}")
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-repo", required=True)
    parser.add_argument("--skip-data-check", action="store_true", help="Local source/config check only; NOT server readiness")
    args = parser.parse_args()
    repo = Path(args.dfine_repo).resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != PIN:
        raise ValueError(f"Pinned upstream required: {PIN}; got {actual}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip():
        raise ValueError("Upstream tracked source has modifications; inspect before preparing")
    pilot_path = base.ROOT/"configs/diagnosis/rtdetrv2_r18_ogsod_pilot20.yml"
    overlay_path = base.ROOT/"configs/diagnosis/dfine_r18_ogsod_pilot20_overlay.yml"
    pilot = resolved_config(pilot_path)
    native = resolved_config(repo/"configs/dfine/dfine_hgnetv2_s_coco.yml")
    overlay = resolved_config(overlay_path)
    cfg = build_config(pilot, native, overlay)
    checks = protocol(pilot, cfg)
    splits = {}
    if not args.skip_data_check:
        for name in ("train", "val"):
            data_cfg = cfg[name+"_dataloader"]["dataset"]
            gt = json.loads(Path(data_cfg["ann_file"]).read_text(encoding="utf-8"))
            base.preflight(gt, [], data_cfg["img_folder"])
            if set(c["id"] for c in gt["categories"]) != {0, 1, 2}:
                raise ValueError("Category IDs must match unremapped E0 labels 0,1,2")
            if set(canonical_category(c["name"]) for c in gt["categories"]) != {"Bridge", "Harbor", "Storage Tank"}:
                raise ValueError("Unexpected category semantics")
            splits[name] = {"sha256": base.sha(data_cfg["ann_file"]), "file_names": {i["file_name"] for i in gt["images"]}}
        if splits["train"]["file_names"] & splits["val"]["file_names"]:
            raise ValueError("Train/validation filenames overlap")
        for value in splits.values():
            value.pop("file_names")
    target = repo/"configs/ogsod/dfine_r18_ogsod_pilot20.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Existing pilot config is preserved: {target}")
    target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    base.dump(target.with_suffix(".protocol.json"), {
        "upstream": UPSTREAM, "upstream_commit": PIN, "prepared_config": str(target),
        "prepared_config_sha256": base.sha(target), "shared_protocol_checks": checks,
        "train_validation": splits, "server_data_checked": not args.skip_data_check,
        "source_pilot": str(pilot_path), "seed": 0, "epochs": 20, "gpu_count": 2,
        "initialization": "same ImageNet ResNet18_vd weights; no COCO/Objects365/E0 tuning",
        "differences": ["native D-FINE RepNCSPELAN4 encoder vs E0 CSPRepLayer",
                        "FDR, LQE, FGL and DDF/GO-LSD; native cross-layer union regression matching",
                        "native D-FINE pre-box prediction, supervision and checkpoint naming"],
        "comparison_scope": "budget/backbone/data matched detector comparison; not FDR-only causal ablation",
        "training_launched": False})
    print(f"Prepared, not trained: {target}")


if __name__ == "__main__":
    main()
