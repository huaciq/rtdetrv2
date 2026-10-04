# E4: Baseline + SAR Boundary Evidence Refinement

## What changed

Independent config:
`configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_gbs64_gpu2_zxy.yml`, directly
inheriting the E0 global-batch-64 OGSOD baseline. E4 instantiates no QAQS,
SQ-QS, UMQR or quality probe. It adds one shared SBER head (131,461 parameters),
not a head per decoder layer. PResNet18, HybridEncoder, query selection,
matcher/costs, 300 queries, three decoder layers, DN, augmentation, optimizer,
scheduler, original VFL/bbox/GIoU, EMA, AMP, initialization, best selection,
seed and 80-epoch schedule stay unchanged. No new supervision or sweep.

Files:

- `src/zoo/rtdetr/sber.py`: fixed boundary coordinates, deterministic
  size-dependent scale fusion, real bilinear feature sampling, contrast MLP,
  gated four-edge residual and legal box conversion.
- `src/zoo/rtdetr/rtdetrv2_decoder.py`: reconstruct actual encoder maps once;
  refine original bbox predictions and update the original iterative reference.
- `src/solver/det_engine.py`: first-batch summaries and all-rank checks.
- `src/solver/det_solver.py`: eight focused evaluation metrics in JSON.
- New E4 config, `tools/test_sber.py`, and this handoff.

### Feature sampling and multi-scale choice

Use the unchanged decoder `memory`, before the separate encoder proposal
`enc_output` projection. This is the concatenation of real HybridEncoder
P3/P4/P5 feature maps after baseline decoder input projection. Split by the
existing `spatial_shapes`, reshape to `[B,256,H,W]`, and convert to FP32 once.
These tensors remain connected to the encoder; there is no detach, new CNN,
FFT/wavelet or query-only substitute for appearance evidence.

For each layer's current detached reference `(cx,cy,w,h)`, edge midpoints are
left `(cx-w/2,cy)`, right `(cx+w/2,cy)`, top `(cx,cy-h/2)`, bottom
`(cx,cy+h/2)`. Sample inside/boundary/outside along each normal using fixed
rho=.1 times reference width/height. Shape `[B,Q,4,3,2]`, 12 points per query;
all normalized coordinates are clamped to [0,1]. Bilinear `grid_sample` uses
`align_corners=False`, `padding_mode=border`, consistent with map cell centers.
The DN sequence uses the same mechanism without changing DN construction.

Multi-scale choice has no learned attention: let
`extent=sqrt((w*W_P3)*(h*H_P3))`, `level=clamp(log2(extent/4),0,2)`.
Linearly interpolate between the two adjacent levels. At 640 input, normalized
square widths .05/.10/.20 select P3/P4/P5 respectively, matching the baseline
proposal scale convention. Smaller boxes use P3; larger boxes use P5. The four
cell rule is a fixed implementation choice, not another tuning sweep.

### Residual and gradient path

Concatenate four `F_inside-F_outside` vectors and four boundary feature
vectors to `[B,Q,2048]`. A shared `2048 -> 64 -> ReLU` trunk has two small
linear outputs: four offsets and one query gate. Gate logits are clamped to
[-10,10] before sigmoid. Offsets are tanh-bounded:

```text
E = concat(four inside-outside contrasts, four boundary features)
fractions = rho * sigmoid(clamp(gate_head(MLP(E)), -10, 10))
                * tanh(offset_head(MLP(E)))                 # dl,dr,dt,db
box_det = sigmoid(original_bbox_head(h) + logit(reference))
xyxy_final = xyxy(box_det) +
             [dl*w_det, dt*h_det, dr*w_det, db*h_det]
box_final = cxcywh(project_to_legal_image_box(xyxy_final))
```

Sampling uses the current reference; residual scale uses box_det width/height
so each side moves at most 10% of the original detector result. Before clipping,
width and height remain at least 80% of those of box_det. Clip image corners
and enforce minimum positive size 1e-6 to handle boxes touching the border.
Thus the original bbox prediction remains the base, with an additional box
residual. No evidence is added to decoder hidden features. Classification
receives the original h. Next-layer attention reads the refined reference
through the existing detach policy. Preserve the original non-detached
reference path of auxiliary training bbox predictions as well.

The final, decoder auxiliary and DN refined boxes receive the original
detection losses. Encoder proposals and their losses remain original.
No criterion/matcher change, no keypoint/uncertainty/edge loss or labels.

Offset output starts at zero and gate starts at .5. The offset head receives
nonzero detection gradients on the first batch; gate/trunk gradients can be
zero initially because the offset output is zero, and become nonzero after
updates. All six parameter tensors remain in the graph. Small floating-point
corner roundtrip differences or border projection can occur with enabled
zero-offset SBER. With `sber: False`, there is no sampling, projection or new
parameter; arbitrary baseline predictions and losses are exactly unchanged.

## Local verification

```powershell
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_sber.py
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_umqr.py
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_sqqs.py
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_umqr_direct.py UMQRDirectChecks.test_config_baseline_equivalence_and_training_gradient_path UMQRDirectChecks.test_geometry_legality_confidence_and_zero_alpha_gradients
git diff --check
```

The UTF-8 flag only works around this workstation's existing `.pth` encoding
issue. E4 uses synthetic 160 inputs, all 300 queries, three layers and baseline
DN; synthetic fixtures change only test image/batch/epoch limits and disable
pretrained downloads. Tests isolate repeated configs from the existing YAML
loader's mutable default. Normal CLI loads one config per process.

The five E4 tests cover analytical bilinear sampling and all scale weights;
exact map reconstruction; sampling-feature interventions; detection gradients
into real encoder maps; edge signs/bounds and tiny/border boxes; CPU bfloat16
autocast and extreme gate/offset values; config equality apart from E4 fields;
baseline parameter initialization/RNG; exact disabled outputs/losses and strict
baseline loading; unchanged matcher count/loss keys; actual engine training;
nonzero gradients of all SBER tensors after updates; model/optimizer/EMA
recovery; actual one-epoch solver fit and best.pth evaluation; eight COCO
metrics with independently indexed AR75; original query diagnosis; and
two-process full-model CPU Gloo DDP with unequal foreground counts.

Local result: five E4 tests, five existing UMQR tests, five SQ-QS regression
tests and the two selected E3' tests all passed. Syntax compilation and
`git diff --check` passed. One synthetic DDP first batch reported nonzero
contrast, coordinates [0,1], gate=.5, 6/6 finite parameter gradients and
legal refined boxes on all ranks. These are correctness checks, not SAR AP.

## Git transfer

Branch: `codex/ogsod-r18-sber`. Inspect dirty or diverged server worktrees
before switching; do not force checkout/reset. Complete the existing E3'
run before launching E4 on the same GPUs.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/ogsod-r18-sber
git pull --ff-only origin codex/ogsod-r18-sber
git rev-parse HEAD
conda activate rtdetr_zxy
python tools/test_sber.py
```

No new dependencies or dataset migration. Fresh training uses E0 pretrained
PResNet18 initialization, without tuning/resuming E1/E2/E3/E3' checkpoints.
Strict resume needs an E4 checkpoint with SBER keys. Disabled SBER preserves
all previous model state keys and behavior.

## Server launch

Data root and splits inherit E0:
`/home/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/{train,val}.json`.
Verify actual image files, category definitions, split integrity and both GPUs:

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
env CUDA_VISIBLE_DEVICES=0,1 python - <<'PY'
import json
from pathlib import Path
import torch
from src.core.yaml_utils import load_config
cfg = load_config('configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_gbs64_gpu2_zxy.yml', cfg={})
assert torch.cuda.device_count() == 2
paths, categories = {}, {}
for split in ('train', 'val'):
    dataset = cfg[f'{split}_dataloader']['dataset']
    root = Path(dataset['img_folder'])
    data = json.loads(Path(dataset['ann_file']).read_text())
    ids = [image['id'] for image in data['images']]
    id_set = set(ids)
    assert len(ids) == len(id_set), f'{split}: duplicate IDs'
    categories[split] = {c['id']: c['name'] for c in data['categories']}
    assert set(categories[split]) == set(range(cfg['num_classes']))
    paths[split] = {(root / image['file_name']).resolve() for image in data['images']}
    assert len(paths[split]) == len(ids), f'{split}: duplicate files'
    assert all(path.is_file() for path in paths[split]), f'{split}: missing images'
    assert all(a['image_id'] in id_set and a['category_id'] in categories[split]
               for a in data['annotations']), f'{split}: invalid annotations'
    print(split, len(ids), 'images', len(data['annotations']), 'annotations')
assert categories['train'] == categories['val']
assert not paths['train'].intersection(paths['val']), 'Train/val overlap'
print('Data checks passed; independently verify split provenance.')
PY
test "$?" -eq 0 || exit 1
df -h /home/zxy/sar/experiments
nvidia-smi
```

Keep global batch 64 (32/rank), validation batch 32, input 640, 80 epochs,
seed 0, original optimizer/scheduler/AMP/EMA. Checkpoint frequency stays at
baseline 1; earlier 80-epoch runs occupied about 25GB each. Reserve at least
30GB for E4 plus space needed by other ongoing jobs. Do not change checkpoint
or training settings to hide a disk-capacity issue.

```bash
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber
test ! -e "$OUT" || { echo 'Output exists; inspect or resume the matching E4 run.'; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_gbs64_gpu2_zxy.yml \
  --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

## Monitoring and metrics

```bash
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber
tail -n 100 -f "$OUT/console.log"
grep 'SBER_FIRST_BATCH' "$OUT/console.log"
watch -n 2 nvidia-smi
ps -fp "$(cat "$OUT/launcher.pid")"
df -h "$OUT"
tail -n 5 "$OUT/log.txt"
tail -n 1 "$OUT/log.txt" | python -m json.tool
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

`SBER_FIRST_BATCH` reports each layer's coordinate min/max, mean absolute
inside-outside contrast, fraction of different feature elements, size weights,
gate mean/min/max, bounded offset magnitude, actual cxcywh residual mean/max,
parameter gradient presence/magnitude, legal regular/aux/DN boxes, NaN/Inf
flags and an all-rank pass flag. This is one check after normal backward,
before clipping/clearing; under AMP it sees scaled gradients. No new backward
or large diagnosis. Contrast can legitimately be zero for constant maps or
clipped coincident points; the report exposes it rather than inventing a loss.
It repeats once on a resumed process, and debug tensors are small detached
summaries collected only until this check succeeds. Inference outputs retain
the existing schema and query diagnosis.

### best.pth evaluation

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber
EVAL="$OUT/eval_best"
test -f "$OUT/best.pth" || exit 1
test ! -e "$EVAL" || { echo 'Preserve existing evaluation output; choose a new directory.'; exit 1; }
mkdir -p "$EVAL"
git rev-parse HEAD > "$EVAL/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_gbs64_gpu2_zxy.yml \
  --resume "$OUT/best.pth" --test-only --seed 0 --output-dir "$EVAL" \
  > "$EVAL/console.log" 2>&1 &
EVAL_PID=$!
echo "$EVAL_PID" | tee "$EVAL/launcher.pid"
disown
```

```bash
python -m json.tool "$OUT/eval_best/sber_metrics.json"
python -m json.tool "$OUT/eval_best/evaluation_metrics.json"
```

Focus: AP, AP50, **AP75**, **APs**, APm, APl, AR100 and **AR75**. Files use
0-1 values; multiply by 100 for percentage reporting. AR75 is COCO recall at
IoU=.75, area=all, maxDets=100, averaged over valid categories, not encoder
query recall. Evaluation retains original FP32/class-score postprocessing,
COCO evaluator and EMA preference. The focused JSON records resolved config,
seed, checkpoint and process count. Full COCO/diagnosis outputs are preserved;
the existing full query diagnosis still skips multi-process evaluation.

Reported E0 comparison thresholds are AP75=51.7 and APs=46.3. User targets
are AP75>=52.5, APs>=47.0, AP>=54.2. These are experimental targets, not
locally measured results. Select best checkpoint by the original COCO AP;
compare against fixed E0 under the same split/seed/global batch/EMA protocol.

## Outputs and resume

Root: `/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber`.
`best.pth`, `last.pth`, `checkpointNNNN.pth`, `console.log`, `log.txt`,
`summary/`, `eval/latest.pth` keep their original roles.
Final evaluation: `eval_best/sber_metrics.json`, `evaluation_metrics.json`,
`eval.pth`, `console.log` under `eval_best`. Previous experiments are untouched.

Before resuming, ensure the previous launcher/workers have exited and storage
is available. Validate last.pth; disk-write failures can leave it incomplete:

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber
python - "$OUT/last.pth" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert {'model', 'optimizer', 'lr_scheduler', 'ema', 'scaler', 'last_epoch'} <= state.keys()
assert 'decoder.sber_head.offset_head.weight' in state['model']
print('Resume from last_epoch', state['last_epoch'])
PY
test "$?" -eq 0 || exit 1
STAMP=$(date +%Y%m%d_%H%M%S)
if test -f "$OUT/best.pth"; then
  mv -n "$OUT/best.pth" "$OUT/best_before_resume_${STAMP}.pth"
fi
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_gbs64_gpu2_zxy.yml \
  --resume "$OUT/last.pth" --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/resume_${STAMP}.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

Existing solver best-stat bookkeeping resets on resume, so retain historical
best checkpoints and compare them with the resumed segment's best by COCO AP
before final evaluation. This baseline bookkeeping is not modified by E4.

## Caveats

Local CPU tests do not establish SAR AP gains, CUDA AMP/NCCL, runtime or GPU
memory. SBER adds feature sampling and its detection gradient path; keep batch
and training settings fixed while verifying actual server memory. The server
is inaccessible by SSH from this workstation; transfer by Git. Preserve
original baseline DN/DDP behavior for entirely empty-GT ranks. No additional
method, backbone, attention, augmentation, matcher or loss sweep is included.
