"""Paired detector checks/evaluation in an isolated source tree; never trains.

Without --checkpoint: synthetic construction/forward/loss/state checks only.
With --checkpoint: FP32 full-val export and common canonical COCO metrics.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import baseline_error_analysis as base
from localization_diagnosis import canonical_category
from prepare_dfine_pilot import PIN


def finite_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return True


def signature(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metrics(gt, predictions):
    import contextlib
    import io
    import numpy as np
    with contextlib.redirect_stdout(io.StringIO()):
        truth = base.PythonCOCO()
        truth.dataset = copy.deepcopy(gt)
        truth.createIndex()
        detected = truth.loadRes(copy.deepcopy(predictions))
        evaluator = base.PythonCOCOeval(truth, detected, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    t75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .75))[0])
    classes = {}
    by_id = {c["id"]: canonical_category(c["name"]) for c in gt["categories"]}
    for k, cat in enumerate(evaluator.params.catIds):
        p = evaluator.eval["precision"][t75, :, k, 0, -1]
        classes[by_id[cat]] = float(p[p >= 0].mean()) if (p >= 0).any() else None
    recalls = evaluator.eval["recall"][t75, :, 0, -1]
    return {**dict(zip(base.METRICS, map(float, evaluator.stats))),
            "AR75": float(recalls[recalls >= 0].mean()) if (recalls >= 0).any() else -1.,
            "class_AP75": classes, "units": "fractions", "maxDets": [1, 10, 100]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="RT-DETRv2 or pinned official D-FINE checkout")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--backbone-state", help="Local smoke: strictly load a shared backbone state")
    parser.add_argument("--save-backbone", help="Local smoke: save backbone state for equivalence check")
    parser.add_argument("--benchmark", action="store_true", help="Eval only: fixed batch=1 FP32, CUDA model+postprocessor latency")
    args = parser.parse_args()
    repo, out = Path(args.repo).resolve(), Path(args.output).resolve()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0 and any(p.name not in ("console.log", "launcher.pid") for p in out.iterdir()):
        raise FileExistsError(f"Preserve existing output: {out}")
    # baseline_error_analysis imports no src: choose the target registry here.
    sys.path.insert(0, str(repo))
    import torch
    from src.core import YAMLConfig
    from torch.utils.data import DataLoader
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if world > 1:
        torch.cuda.set_device(local)
        torch.distributed.init_process_group("nccl")
        torch.distributed.barrier()
    device = torch.device(f"cuda:{local}" if args.checkpoint and torch.cuda.is_available() else "cpu")
    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg["PResNet"]["pretrained"] = False  # strict checkpoint or synthetic check
    model = cfg.model.to(device)
    post = cfg.postprocessor.to(device).eval()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip()
    if cfg.yaml_cfg.get("epochs", cfg.yaml_cfg.get("epoches")) != 20:
        raise ValueError("Use the fixed 20-epoch pilot config, not a historical full-budget config")
    if cfg.yaml_cfg["model"] == "DFINE" and commit != PIN:
        raise ValueError("Use the pinned official D-FINE source")
    common = {"source_repo": str(repo), "git_commit": commit, "tracked_worktree_dirty": bool(dirty),
              "config": str(Path(args.config).resolve()), "config_sha256": base.sha(args.config),
              "resolved_config": cfg.yaml_cfg, "device": str(device), "seed": 0,
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "torch_version": torch.__version__, "gpu_count": world, "precision": "FP32"}
    hw = cfg.yaml_cfg["eval_spatial_size"]
    torch.manual_seed(42)  # identical probe image despite differing decoder initialization
    sample = torch.rand(1, 3, *hw, device=device)
    if not args.checkpoint:
        if world != 1 or args.benchmark:
            raise ValueError("Synthetic check uses one process; no benchmark claims")
        if args.backbone_state:
            model.backbone.load_state_dict(torch.load(args.backbone_state, map_location=device, weights_only=True), strict=True)
        model.eval()
        backbone_signature = signature(model.backbone)
        if args.save_backbone:
            torch.save(model.backbone.state_dict(), args.save_backbone)
        with torch.no_grad():
            features = model.backbone(sample)
            feature_sha = [hashlib.sha256(f.cpu().contiguous().numpy().tobytes()).hexdigest() for f in features]
            outputs = model(sample)
            assert outputs["pred_boxes"].shape == (1, 300, 4)
            assert outputs["pred_logits"].shape == (1, 300, 3)
            assert all(torch.isfinite(outputs[k]).all() for k in ("pred_boxes", "pred_logits"))
            results = post(outputs, torch.tensor([[hw[1], hw[0]]], device=device))
            assert results[0]["boxes"].shape == (300, 4)
            assert set(results[0]["labels"].tolist()) <= {0, 1, 2}
            # Exercise DN/aux native loss; no backward, optimizer or update.
            model.train()
            targets = [{"labels": torch.tensor([0, 2], device=device),
                        "boxes": torch.tensor([[.3, .3, .04, .02], [.6, .6, .1, .08]], device=device)}]
            train_outputs = model(sample, targets)
            assert finite_tensors(train_outputs), "Non-finite native/DN/aux predictions"
            losses = cfg.criterion.to(device)(train_outputs, targets, epoch=0, step=0, global_step=0)
            assert losses and all(torch.isfinite(v).all() for v in losses.values())
        # State round-trip, including detector heads, without optimizer updates.
        ema = cfg.ema
        torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, out/"synthetic_state.pth")
        stored = torch.load(out/"synthetic_state.pth", map_location=device, weights_only=True)
        model.load_state_dict(stored["model"], strict=True)
        ema.load_state_dict(stored["ema"], strict=True)
        (out/"synthetic_state.pth").unlink()
        base.dump(out/"validation.json", {**common, "mode": "synthetic_only_no_training",
            "backbone_state_sha256": backbone_signature, "backbone_feature_sha256": feature_sha,
            "feature_shapes": [list(f.shape) for f in features], "losses": {k: float(v) for k, v in losses.items()},
            "strict_state_roundtrip": True, "EMA_state_roundtrip": True,
            "prediction_shape": [1, 300, 4], "finite": True})
        print(f"Synthetic forward/loss/state checks passed: {out}", flush=True)
        return
    if dirty:
        raise ValueError("Evaluation requires clean tracked model source")
    log_path = Path(args.checkpoint).parent/"log.txt"
    epochs = {int(json.loads(line)["epoch"]) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if epochs != set(range(20)):
        raise ValueError("Checkpoint run must have exactly epochs 0..19 in its own log.txt")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    key = "ema" if cfg.yaml_cfg.get("use_ema", False) else "model"
    model.load_state_dict(state[key]["module"] if key == "ema" else state[key], strict=True)
    epoch = state.get("last_epoch")
    if not isinstance(epoch, int) or not 0 <= epoch < 20:
        raise ValueError("Checkpoint epoch outside this fixed pilot budget")
    del state
    model.eval()
    data = cfg.yaml_cfg["val_dataloader"]["dataset"]
    gt = json.loads(Path(data["ann_file"]).read_text(encoding="utf-8"))
    base.preflight(gt, [], data["img_folder"])
    original = cfg.val_dataloader
    loader = DataLoader(original.dataset, batch_size=original.batch_size,
                        sampler=list(range(rank, len(original.dataset), world)), collate_fn=original.collate_fn,
                        num_workers=original.num_workers, drop_last=False)
    with torch.inference_mode(), (out/f"predictions.rank{rank}.jsonl").open("x", encoding="utf-8") as stream:
        for batch, (samples, targets) in enumerate(loader):
            outputs = model(samples.to(device))
            sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = post(outputs, sizes)
            for target, result in zip(targets, results):
                iid = int(target["image_id"].item())
                boxes = result["boxes"].cpu().clone()
                boxes[:, 2:] -= boxes[:, :2]
                records = [{"image_id": iid, "category_id": int(label), "bbox": box, "score": float(score)}
                           for box, label, score in zip(boxes.tolist(), result["labels"].tolist(), result["scores"].tolist())]
                stream.write(json.dumps({"image_id": iid, "predictions": records}, allow_nan=False)+"\n")
            if batch % 20 == 0:
                print(f"rank={rank} validation {batch}/{len(loader)}", flush=True)
    latency = None
    if args.benchmark and rank == 0:
        if device.type != "cuda":
            raise ValueError("Inference overhead comparison requires CUDA on the same GPU")
        timing = []
        sizes = torch.tensor([[hw[1], hw[0]]], device=device)
        with torch.inference_mode():
            for i in range(120):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                post(model(sample), sizes)
                stop.record()
                stop.synchronize()
                if i >= 20:
                    timing.append(start.elapsed_time(stop))
        import numpy as np
        latency = {"GPU": torch.cuda.get_device_name(local), "batch_size": 1, "warmup": 20, "iterations": 100,
                   "model_and_postprocess_mean_ms": float(np.mean(timing)), "median_ms": float(np.median(timing)),
                   "P90_ms": float(np.quantile(timing, .9)), "includes_dataloader": False,
                   "condition": "Run with other GPU idle; same GPU/environment/input/FP32 for both"}
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank != 0:
        return
    merged = {}
    for r in range(world):
        for line in (out/f"predictions.rank{r}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["image_id"] in merged:
                raise ValueError("Repeated validation image")
            merged[row["image_id"]] = row["predictions"]
    if set(merged) != {i["id"] for i in gt["images"]}:
        raise ValueError("Incomplete validation coverage")
    predictions = [p for iid in sorted(merged) for p in merged[iid]]
    base.dump(out/"predictions.json", predictions)
    base.dump(out/"coco_metrics.json", metrics(gt, predictions))
    base.dump(out/"evaluation_manifest.json", {**common, "checkpoint": args.checkpoint,
        "checkpoint_sha256": base.sha(args.checkpoint), "checkpoint_epoch": epoch, "weights": key,
        "validation_sha256": base.sha(data["ann_file"]), "prediction_sha256": base.sha(out/"predictions.json"),
        "training_log_sha256": base.sha(log_path), "completed_epochs": sorted(epochs),
        "latency": latency, "training_time": "Read Training time from paired run console.log; no inference proxy"})
    base.dump(out/"COMPLETE.json", {"status": "complete"})
    print(f"Common full-validation evaluation complete: {out}", flush=True)


if __name__ == "__main__":
    main()
