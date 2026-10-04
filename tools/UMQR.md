# E3: Baseline + Uncertainty-guided Multi-keypoint Query Refinement

## What changed

Config: `configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_gbs64_gpu2_zxy.yml`, directly including the E0 GBS64 OGSOD baseline. No QAQS/SQ-QS or other quality branch is instantiated. E3 enforces baseline query selection. All PResNet18/HybridEncoder, matcher costs, 300 queries, three decoder layers, DN sampling and original VFL/bbox/GIoU losses, augmentation, optimizer, scheduler and checkpoint selection stay unchanged.

- `src/zoo/rtdetr/umqr.py`: GT/reference five-point construction, shared head, FP32 Laplace regression. One head is shared by all three layers, adding 35,732 parameters. `h -> Linear(256,64) -> ReLU -> Linear(64,20)` predicts 10 point-logit offsets and 10 log-scales. Point offsets refine the five-point template of the current detached reference box in sigmoid-logit space. Zero initialization starts at that template, with sigma=1.
- `src/zoo/rtdetr/rtdetrv2_decoder.py`: `h_refine = h + MLP([points, log_scales])`, with geometry MLP `20 -> 64 -> 256` and a small initial final projection. **Only the bbox head receives h_refine**; classification and subsequent decoder features receive the original h. The existing reference update/detach and iterative inverse-sigmoid residual rules are preserved. Guidance is active in both training and evaluation, including the existing DN sequence; no new DN supervision is added.
- `src/zoo/rtdetr/rtdetrv2_criterion.py`: use the existing final/auxiliary detection matching indices to supervise regular foreground queries. No new matching calls/costs. No background, encoder or DN keypoint regression. `loss_ukp`, `loss_ukp_aux_0`, `loss_ukp_aux_1` each use lambda=1.0. Ordinary detection auxiliary/DN/encoder losses retain their exact original weights.
- `src/solver/det_engine.py`: first-batch shape/GT range/uncertainty/loss/parameter-gradient/finite check, with a cross-rank pass flag, printed once per process run before gradients are cleared. No large diagnosis or extra backward pass.
- `src/solver/det_solver.py`: test-only `umqr_metrics.json` for the six requested evaluation metrics; original full COCO metrics and query diagnosis remain.
- `tools/test_umqr.py`: synthetic CPU verification, including full-model two-process DDP. This document is the server handoff.

GT points are `[center, left midpoint, right midpoint, top midpoint, bottom midpoint]`, shape `[N,5,2]`, automatically generated from the existing normalized cxcywh boxes after augmentation. Predictions and log-scales are `[B,300,5,2]` after DN removal; both auxiliary layers have the same shape. No keypoint annotations are needed.

Loss: clamp s to `[-5,3]`, compute `exp(-s)*abs(p-p_gt)+s` in FP32, **mean the ten coordinates per foreground object**, sum objects and divide by the same DDP-average GT count used by detection losses. The top-level clamp values are shared by model and criterion. Empty foreground sets return a differentiable zero. Negative Laplace loss values are valid (log-sigma can be negative); do not clamp the loss to zero. Point and uncertainty geometry remain differentiable into the bbox prediction path.

## Local verification

```powershell
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_umqr.py
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_sqqs.py
git diff --check
```

The Windows UTF-8 switch works around an existing non-UTF-8 `.pth` startup issue. Tests use generated 160x160 inputs with all 300 queries, unchanged three-layer architecture and ordinary DN; no pretrained download or real SAR data. Checks cover exact E0 shared initialization/RNG, initial predictions and detection losses, unchanged matcher call count, five-point order, foreground-only/empty/clamped/negative-loss numerics, CPU bfloat16 head autocast, three training steps through the actual engine, point-only and uncertainty-only interventions changing final boxes, detection-only gradients into both predicted point/scales, optimizer/model/EMA strict checkpoint recovery, full COCO/AR75/legacy diagnosis and two-process full-model CPU Gloo DDP with unequal foreground counts. Synthetic config construction isolates the repository YAML loader's existing mutable default, as the existing SAR audit does. Normal training CLI loads one config per process.

Local result: all five UMQR tests passed; all five existing SQ-QS regression tests passed; syntax compilation and `git diff --check` passed.

Real CUDA AMP, NCCL, memory and AP are not established by these CPU tests.

## Git transfer

Branch: `codex/ogsod-r18-umqr`. First inspect dirty/diverged server worktrees; stop and inspect instead of forcing checkout/reset.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/ogsod-r18-umqr
git pull --ff-only origin codex/ogsod-r18-umqr
git rev-parse HEAD
conda activate rtdetr_zxy
python tools/test_umqr.py
```

No new dependency or dataset migration. Train E3 from the same E0 ImageNet-pretrained PResNet18 initialization, without `--tuning` from E0/E1/E2. Strict E3 resume requires an E3 checkpoint with UMQR parameters; do not use `--resume` on an older detector checkpoint. Existing disabled-UMQR configs retain their model/criterion state keys and strict loading behavior.

## Server preflight and launch

Inherited data root: `/home/zxy/sar/datasets/OGSOD-1.0/sar`, annotations `RTDETR_COCO/train.json` and `val.json`. These are recorded E0 paths; verify actual files, class IDs and split integrity on the Linux server before training. Do not silently substitute another split.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
env CUDA_VISIBLE_DEVICES=0,1 python - <<'PY'
import json
from pathlib import Path
import torch
from src.core.yaml_utils import load_config

cfg = load_config('configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_gbs64_gpu2_zxy.yml', cfg={})
assert torch.cuda.device_count() == 2, 'Both GPUs must be visible'
paths, categories = {}, {}
for split in ('train', 'val'):
    dataset = cfg[f'{split}_dataloader']['dataset']
    root = Path(dataset['img_folder'])
    data = json.loads(Path(dataset['ann_file']).read_text())
    ids = [image['id'] for image in data['images']]
    id_set = set(ids)
    assert len(ids) == len(id_set), f'{split}: duplicate IDs'
    categories[split] = {c['id']: c['name'] for c in data['categories']}
    assert set(categories[split]) == set(range(cfg['num_classes'])), 'Check category IDs/remapping'
    paths[split] = {(root / image['file_name']).resolve() for image in data['images']}
    assert len(paths[split]) == len(ids), f'{split}: duplicate files'
    assert all(path.is_file() for path in paths[split]), f'{split}: missing images'
    assert all(a['image_id'] in id_set and a['category_id'] in categories[split]
               for a in data['annotations']), f'{split}: invalid annotations'
    print(split, len(ids), 'images', len(data['annotations']), 'annotations')
assert categories['train'] == categories['val'], 'Category definitions differ'
assert not paths['train'].intersection(paths['val']), 'Train/val overlap'
print('Data checks passed; independently verify split provenance.')
PY
test "$?" -eq 0 || exit 1
df -h /home/zxy/sar/experiments
nvidia-smi

OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr
test ! -e "$OUT" || { echo 'Output exists; inspect or resume the matching E3 run.'; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_gbs64_gpu2_zxy.yml \
  --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

80 epochs, global batch 64 (32/rank), val global batch 32, 640x640, LR 4e-4/backbone 4e-5, same original warmup/schedule/augmentation/AMP/EMA settings and seed 0. Only shared UMQR geometry and its keypoint supervision are added. Keep best checkpoint selection at original COCO AP@[.50:.95]; examine AP75/APs without changing method in response to AP50. Before the full run, a short two-GPU pilot in a separate directory is recommended.

## best.pth evaluation

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr
EVAL="$OUT/eval_best"
test -f "$OUT/best.pth" || exit 1
test ! -e "$EVAL" || { echo 'Evaluation output exists; choose a new directory.'; exit 1; }
mkdir -p "$EVAL"
git rev-parse HEAD > "$EVAL/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_gbs64_gpu2_zxy.yml \
  --resume "$OUT/best.pth" --test-only --seed 0 --output-dir "$EVAL" \
  > "$EVAL/console.log" 2>&1 &
EVAL_PID=$!
echo "$EVAL_PID" | tee "$EVAL/launcher.pid"
disown
```

Evaluation retains baseline class-score postprocessing and COCO protocol, including the existing EMA preference and FP32 evaluation. `umqr_metrics.json` reports AP, AP50, AP75, APs, AR100 and AR75, with full resolved config, seed, checkpoint source and GPU process count. `AR75` means COCO recall at IoU=.75, area=all, maxDets=100, averaged over valid categories; it is not encoder query recall. AP/AR values are on a 0–1 scale. `evaluation_metrics.json` retains all original metrics, including the same AR75; `eval.pth` retains COCO evaluation data. Compare against E0 using its actual recorded checkpoint/config/seed/global batch/split under that same protocol. No new query ranking or uncertainty score fusion is used in postprocessing.

## Monitoring and metrics

```bash
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr
tail -n 100 -f "$OUT/console.log"
grep 'UMQR_FIRST_BATCH' "$OUT/console.log"
watch -n 2 nvidia-smi
ps -fp "$(cat "$OUT/launcher.pid")"
tail -n 5 "$OUT/log.txt"
tail -n 1 "$OUT/log.txt" | python -m json.tool
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
python -m json.tool "$OUT/eval_best/umqr_metrics.json"
python -m json.tool "$OUT/eval_best/evaluation_metrics.json"
```

`UMQR_FIRST_BATCH` prints keypoint shape, GT range (null for empty GT), log-scale/sigma range, L_ukp and auxiliary values, parameter-gradient coverage and NaN/Inf flags, plus an all-rank pass result. Under AMP this check sees scaled gradients before clipping/optimizer clearing; it tests presence and finiteness, not a gradient norm. It adds no extra backward pass. A nonfinite/malformed first batch stops with a clear error. Geometry gradients can start at zero because original bbox final projections start at zero; synthetic tests verify they become active after optimization. The report repeats once on a new resumed process.

`log.txt` and TensorBoard retain `train_loss_ukp`, `train_loss_ukp_aux_0`, `train_loss_ukp_aux_1` together with all original detection losses. Original query diagnosis remains unchanged: full legacy diagnosis still runs only in single-process evaluation, and still skips in two-process evaluation.

## Outputs and resume

Output root: `/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr`.

Training outputs: `best.pth`, `last.pth`, periodic `checkpointNNNN.pth`, `console.log`, `log.txt`, `summary/`, `eval/latest.pth`. Final evaluation outputs: `eval_best/umqr_metrics.json`, `eval_best/evaluation_metrics.json`, `eval_best/eval.pth`, `eval_best/console.log`.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr
test -f "$OUT/last.pth" || exit 1
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_gbs64_gpu2_zxy.yml \
  --resume "$OUT/last.pth" --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/resume.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

## Caveats

Verify real server data/classes/splits, GPU memory, AMP/NCCL gradients and disk availability before expensive training. No real AP75/APs improvement is claimed locally. The original baseline DDP/DN path may have unused DN embedding parameters for a rank containing entirely empty GT; this implementation preserves that baseline policy rather than changing DN or DDP settings. No additional method variants or loss sweep are introduced.
