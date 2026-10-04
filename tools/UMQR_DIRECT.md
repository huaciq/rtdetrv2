# E3': UMQR direct geometry box refinement

## What changed

This is the single final structural correction to E3. Config:
`configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_direct_gbs64_gpu2_zxy.yml`.
It inherits E3 and changes only the refinement mode and output directory.
There is no QAQS/SQ-QS. Backbone, encoder, query selection, matcher, 300 queries,
three decoder layers, DN, augmentation, optimizer/scheduler, initialization,
80 epochs, global batch 64, seed 0, EMA/AMP and all loss settings remain E3's.
The original E3 config and checkpoint keys retain their original behavior.

- `src/zoo/rtdetr/umqr.py`: keep the identical shared `256 -> 64 -> 20`
  point/log-scale MLP; remove the geometry embedding MLP in direct mode.
  Add one shared learnable scalar `alpha`, initialized to zero. E3' adds
  17,749 parameters to E0 (E3 added 35,732).
- `src/zoo/rtdetr/rtdetrv2_decoder.py`: the original bbox head receives
  unchanged decoder feature `h`. Add the gated geometry residual to its
  output before the original reference-logit addition/sigmoid, including
  the existing training reference-gradient path and DN sequence.
- `src/solver/det_engine.py`: extend the existing first-batch checks with
  legal geometry boxes, positive sizes, confidence range, zero initial
  alpha, alpha gradient presence and finite values/gradients on all ranks.
- `src/solver/det_solver.py`: record raw/EMA alpha each epoch and at the end
  of training; export seven requested best-checkpoint evaluation metrics.
- New independent config, `tools/test_umqr_direct.py`, and this handoff.

The point order stays `[center, left, right, top, bottom]`. The target generation
`box_keypoints`, foreground/auxiliary matching and FP32 Laplace `L_ukp` are
unchanged; lambda stays 1.0. No new loss, matcher cost, score variant or sweep.

For predicted points, construct width `x_right-x_left` and height
`y_bottom-y_top`, clamped to `[1e-5, 1-2e-5]`. Average the predicted center
and the center obtained from the opposing edge midpoints. Clamp this center
using the half-size so the resulting xyxy box also lies inside the image.
The direct refinement, computed in FP32, is:

```text
B_kp = keypoints_to_box(points)                 # normalized cxcywh
delta_kp = inverse_sigmoid(B_kp) - inverse_sigmoid(reference.detach())
C_geo = sigmoid(-mean(clamp(log_sigma, -5, 3))) # ten coordinates, per query
delta_final = original_bbox_head(h) + alpha * C_geo * delta_kp
box = sigmoid(delta_final + inverse_sigmoid(reference))
```

The reference addition/detach follows the original decoder semantics. Alpha
is unconstrained as requested; no additional gate parameter or regularizer.
At alpha=0 detection predictions are exactly baseline, even with arbitrary
nonzero bbox weights and point offsets. Points/scales independently train
through L_ukp. When initial point offsets are zero, B_kp approximates the
reference, so alpha's first gradient may be zero or numerically tiny; it is
attached to the graph and becomes active as the keypoint branch learns.
The runtime check requires gradient presence/finiteness, not an artificially
nonzero gradient at initialization.

## Local verification

```powershell
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_umqr_direct.py
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_umqr.py
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_sqqs.py
git diff --check
```

The UTF-8 flag works around this workstation's existing non-UTF-8 `.pth` file.
The direct tests use synthetic inputs, no real SAR dataset or pretrained
download: config equality except the two intended changes; baseline common
initialization/RNG; legal/crossed/extreme boxes; confidence/log-scale clamps;
alpha=0 equivalence and independent keypoint gradients; CPU bfloat16 head
numerics; actual-engine training; point/uncertainty interventions and
detection-only gradients; model/optimizer strict recovery; actual one-epoch
solver fit with final-alpha/last.pth agreement; best.pth EMA evaluation and
all seven COCO metrics; original query diagnosis; two-process CPU Gloo DDP.
Synthetic fixture image size/batch/epoch limits affect only tests.

Local result: all four direct tests, all five original E3 tests and all five
SQ-QS regression tests passed. `git diff --check` passed. The direct DDP
first batch reported alpha=0, confidence=0.5, 5/5 parameters with finite
gradients, finite alpha gradient and legal geometry boxes on all ranks.

## Git transfer

Branch: `codex/ogsod-r18-umqr-direct`. Stop and inspect any dirty or diverged
server worktree; do not force checkout or reset.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/ogsod-r18-umqr-direct
git pull --ff-only origin codex/ogsod-r18-umqr-direct
git rev-parse HEAD
conda activate rtdetr_zxy
python tools/test_umqr_direct.py
```

No new dependencies or migration. Train a fresh E3' from the same E3/E0
pretrained PResNet18 initialization; no tuning from the trained E3 checkpoint.
Resume requires a matching E3' checkpoint; an E3 hidden-fusion checkpoint has
different head keys and cannot strictly resume E3'. Existing E0/E1/E2/E3
configs still load their own checkpoints normally.

## Server launch

Inherited dataset root: `/home/zxy/sar/datasets/OGSOD-1.0/sar`, with
`RTDETR_COCO/train.json` and `val.json`. Verify these on the server:

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
env CUDA_VISIBLE_DEVICES=0,1 python - <<'PY'
import json
from pathlib import Path
import torch
from src.core.yaml_utils import load_config
cfg = load_config('configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_direct_gbs64_gpu2_zxy.yml', cfg={})
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

The training settings in the resolved direct config must match the completed
E3 run, including its recorded seed and pretrained initialization.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr_direct
test ! -e "$OUT" || { echo 'Output exists; inspect or resume this E3-prime run.'; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_direct_gbs64_gpu2_zxy.yml \
  --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

Global training batch stays 64 (32 per rank), validation global batch 32.
Best checkpoint selection stays COCO AP@[.50:.95], unchanged from E3.

## best.pth evaluation

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr_direct
EVAL="$OUT/eval_best"
test -f "$OUT/best.pth" || exit 1
test ! -e "$EVAL" || { echo 'Evaluation output exists; preserve it and choose a new directory.'; exit 1; }
mkdir -p "$EVAL"
git rev-parse HEAD > "$EVAL/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_direct_gbs64_gpu2_zxy.yml \
  --resume "$OUT/best.pth" --test-only --seed 0 --output-dir "$EVAL" \
  > "$EVAL/console.log" 2>&1 &
EVAL_PID=$!
echo "$EVAL_PID" | tee "$EVAL/launcher.pid"
disown
```

Focus: AP, AP50, **AP75**, **APs**, APm, AR100, **AR@0.75** and alpha.
Values are 0-1, multiply AP/AR by 100 for percentages. `AR75` is COCO recall
at IoU=.75, area=all, maxDets=100, averaged over valid categories, not query
recall. Evaluation uses the existing EMA preference and FP32 protocol.

- `eval_best/umqr_direct_metrics.json`: the seven metrics, evaluated
  checkpoint alpha, raw and EMA checkpoint alpha, resolved config and
  checkpoint provenance.
- `umqr_direct_alpha_final.json`: **training-ended** raw-model/EMA alpha
  corresponding to `last.pth`, which can differ from `best.pth` alpha.
- `log.txt`: each epoch's `umqr_alpha_model`, `umqr_alpha_ema`, original
  detection/keypoint losses and full COCO metrics.
- `eval_best/evaluation_metrics.json`, `eval_best/eval.pth`: original full
  evaluation metrics and serialized COCO evaluation remain available.

Report both final-training raw alpha and EMA alpha because inference uses
EMA. Do not substitute best-checkpoint alpha for the requested training-ended
alpha. Compare E0/E3/E3' under the same checkpoint/EMA/data protocol; do not
change the method based on AP50 or conduct another sweep.

## Monitoring and metrics

```bash
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr_direct
tail -n 100 -f "$OUT/console.log"
grep 'UMQR_FIRST_BATCH' "$OUT/console.log"
watch -n 2 nvidia-smi
ps -fp "$(cat "$OUT/launcher.pid")"
tail -n 5 "$OUT/log.txt"
tail -n 1 "$OUT/log.txt" | python -m json.tool
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
python -m json.tool "$OUT/umqr_direct_alpha_final.json"
python -m json.tool "$OUT/eval_best/umqr_direct_metrics.json"
```

The first-batch output includes all original E3 checks and `B_kp_legal`,
box/size/confidence ranges, alpha and alpha gradient. It runs after backward
and before clipping/clearing, with no extra backward pass. Under AMP, these
are scaled gradients. NaN/Inf, malformed boxes or missing gradients stop the
run. Original query diagnosis is unchanged, including its existing skip in
multi-process evaluation and availability in single-process evaluation.

## Outputs and resume

Output root: `/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr_direct`.
Original checkpoints (`best.pth`, `last.pth`, `checkpointNNNN.pth`),
`console.log`, `log.txt`, `summary/`, `eval/latest.pth` remain in this root.
It is distinct from every E0/E1/E2/E3 output directory.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_umqr_direct
test -f "$OUT/last.pth" || exit 1
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_direct_gbs64_gpu2_zxy.yml \
  --resume "$OUT/last.pth" --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/resume.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

## Caveats

Local synthetic checks do not establish real SAR AP improvements, CUDA AMP,
NCCL or server GPU memory. The server is inaccessible by SSH from this
workstation; transfer by Git and run there. Preserve the existing E3 run as
experimental evidence. The baseline's entirely empty-GT rank / DN-unused
parameter behavior is unchanged. E3' is the final structural experiment;
no additional modules, losses or parameter sweep are included.
