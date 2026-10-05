# E4': Scale-Aware SBER

## What changed

Config: `configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_scale_gbs64_gpu2_zxy.yml`.
It inherits E4 and changes only the independent output directory and
`RTDETRTransformerv2.sber_scale_aware: True`. The threshold is fixed at
0.0225, with no YAML threshold parameter or sweep interface.

Every decoder layer recomputes the mask from its current detached reference:

```text
use_sber = (reference.w * reference.h <= 0.0225)
masked_fractions = original_E4_boundary_fractions * use_sber
box_final = where(use_sber,
                  original_E4_boundary_refinement(box_det, masked_fractions),
                  box_det)
```

`box_det` is the original RT-DETR sigmoid-logit bbox refinement. Large queries
select it directly, bypassing the E4 corner projection/roundtrip as well as
the learned residual. Merely zeroing four edge offsets would still clip or
round the original box, so the final selection is necessary for exact bypass.
The same mask also applies to auxiliary bbox predictions with the original
non-detached training-reference path. The ordinary DN sequence uses this same
rule; its construction and losses remain unchanged.

Sampling coordinates, maps, deterministic multi-scale interpolation,
inside/outside contrast, MLP, gate, rho=.1, offset limit and head initialization
are unchanged. Sampling is still computed as in E4; bypass controls its box
effect. There are no new parameters/buffers/losses. The shared head still has
131,461 parameters and exactly the E4 state keys/initialization/RNG.
Computing the original branch before masking retains a differentiable graph
even for an all-large rank, with zero local SBER gradients instead of unused
parameters. DDP policy remains baseline's.

Backbone, encoder, query selection, matcher, 300 queries, three decoder layers,
optimizer/scheduler, augmentation, DN, VFL/bbox/GIoU, AMP/EMA, 80 epochs,
global batch 64, seed 0 and checkpoint frequency remain E4's. QAQS/SQ-QS/UMQR
are not instantiated. E4 configs retain their existing behavior.

Files:

- `src/zoo/rtdetr/sber.py`: fixed reference-area mask, optional final bypass,
  and small detached scale-debug summaries. The original SBERHead and all
  sampling/contrast functions are unchanged.
- `src/zoo/rtdetr/rtdetrv2_decoder.py`: mask/selection at both bbox refinement
  locations and pass regular query count for diagnostics excluding DN.
- `src/solver/det_engine.py`: verify exact large bypass and enabled fractions.
- `src/solver/det_solver.py`: independent nine-metric evaluation JSON.
- New config, `tools/test_sber_scale.py`, and this document.

## Local verification

```powershell
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_sber_scale.py
conda run -n pytorch --no-capture-output python -X utf8=0 tools/test_sber.py
git diff --check
```

All four E4' tests and five original E4 tests passed, along with production
syntax compilation and diff checks. E4' tests cover threshold equality and
just-above cases; reference/detector size disagreement; exact nonzero
small/medium E4 residuals; exact large box/zero residual and baseline box
gradients, including boxes extending outside the image; all-small full-model
E4 equality and all-large full-model E0 equality with nonzero bbox/SBER heads;
config/initialization/state-key equality; normal engine training and debug;
model/optimizer/EMA loading; original COCO/query diagnosis; all nine metrics;
and two-process full-model CPU DDP with one rank entirely bypassed.

Synthetic tests use 160 inputs, all 300 queries, three layers and original DN;
they disable pretrained downloads and use temporary fixtures. No experimental
training setting is modified by these test-only overrides. CPU bfloat16 head
numerics are tested, not server CUDA AMP/NCCL or real AP. The Windows UTF-8 flag
works around the workstation's existing non-UTF-8 `.pth` startup issue.

## Git transfer

Branch: `codex/ogsod-r18-sber-scale`. Inspect dirty/diverged server worktrees;
do not force checkout or reset.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/ogsod-r18-sber-scale
git pull --ff-only origin codex/ogsod-r18-sber-scale
git rev-parse HEAD
conda activate rtdetr_zxy
python tools/test_sber_scale.py
```

No dependencies or migrations. Fresh E4' training must use the same baseline
PResNet18 pretrained initialization as E4, without tuning or resuming trained
E4 weights. E4/E4' tensor state keys are identical, so checkpoint tensors alone
cannot identify whether bypass was enabled: use the matching config, output
directory and recorded Git revision. Resume only this E4' run.

## Server launch

Data paths/splits/classes are identical to completed E4, under
`/home/zxy/sar/datasets/OGSOD-1.0/sar`, annotations
`RTDETR_COCO/train.json` and `val.json`. The complete file/category/split
preflight in [SBER.md](SBER.md#server-launch) checks the same inherited data.
Verify both GPUs are available and disk has at least about 30GB for the new
80-epoch run, plus the needs of other jobs. Checkpoint frequency remains 1.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
CFG=configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_scale_gbs64_gpu2_zxy.yml
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber_scale
df -h /home/zxy/sar/experiments
nvidia-smi
test ! -e "$OUT" || { echo 'Output exists; inspect or resume the matching E4-prime run.'; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c "$CFG" --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

Global train batch 64 (32/rank), val batch 32, 640 input, 80 epochs and seed 0.
No threshold or other parameter sweep. Best selection stays original COCO AP.

## Monitoring and metrics

```bash
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber_scale
tail -n 100 -f "$OUT/console.log"
grep 'SBER_FIRST_BATCH' "$OUT/console.log"
watch -n 2 nvidia-smi
ps -fp "$(cat "$OUT/launcher.pid")"
df -h "$OUT"
tail -n 1 "$OUT/log.txt" | python -m json.tool
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

First-batch additions, for each layer:

- `small/medium/large_query_count` and `*_query_ratio` for rank0's regular
  queries, excluding DN; `*_sber_enabled_ratio` within each size group.
- Small proxy reference area <=.0025; medium >.0025 and <=.0225;
  large >.0225. This is reference-area diagnosis at 640, not GT COCO labels.
- Expected enabled rates: small=1, medium=1, large=0. An empty group reports
  `None` and count 0, avoiding NaN. Membership is recomputed at every layer.
- `large_boundary_residual_abs_max` and `large_offset_fraction_abs_max`
  must equal 0 exactly; `large_bbox_exact_baseline` must be True.
- `small_medium_fraction_exact_e4` and `enabled_box_corners_legal` must be
  True. Large bypass/finite checks include DN, even though ratios exclude it.
- All existing E4 sampling/contrast/gate/gradient/finite summaries remain.
  Cross-rank correctness uses the existing all-rank minimum flag.

Large baseline boxes retain baseline cxcywh semantics: finite normalized
coordinates and positive sizes, but corners can extend beyond the image.
Do not clip them in the debug path or bbox path. Enabled boxes retain E4's
corner legality. A violation stops the first-batch check clearly.

## Outputs and resume

Root: `/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber_scale`.
Original checkpoints/logs/TensorBoard/COCO data remain under this new root.
E4 outputs are preserved.

best.pth evaluation:

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
CFG=configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_scale_gbs64_gpu2_zxy.yml
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber_scale
EVAL="$OUT/eval_best"
test -f "$OUT/best.pth" || exit 1
test ! -e "$EVAL" || { echo 'Preserve existing evaluation output; choose a new directory.'; exit 1; }
mkdir -p "$EVAL"
git rev-parse HEAD > "$EVAL/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c "$CFG" --resume "$OUT/best.pth" --test-only --seed 0 --output-dir "$EVAL" \
  > "$EVAL/console.log" 2>&1 &
EVAL_PID=$!
echo "$EVAL_PID" | tee "$EVAL/launcher.pid"
disown
```

```bash
python -m json.tool "$OUT/eval_best/sber_scale_metrics.json"
python -m json.tool "$OUT/eval_best/evaluation_metrics.json"
```

Focus: AP, AP50, AP75, APs, APm, APl, AR100, ARs, AR75. Values are 0-1;
multiply by 100 for percent reporting. ARs is the original COCO small-object
recall; AR75 is recall at IoU=.75, area=all, maxDets=100, averaged over valid
categories. Original EMA preference, FP32 evaluation, class-score postprocess,
COCO evaluation and query diagnosis are preserved. E4 still exports its
eight-metric `sber_metrics.json`; E4' exports nine-metric
`eval_best/sber_scale_metrics.json` plus original `evaluation_metrics.json`
and `eval.pth` under eval_best. Training remains `log.txt`, `console.log`,
`summary/`, `best.pth`, `last.pth`, `checkpointNNNN.pth`, `eval/latest.pth`.

For resume, first ensure old workers exited, disk is available and last.pth
is complete. Preserve historic best because baseline best-stat bookkeeping
resets on resume:

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
CFG=configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_scale_gbs64_gpu2_zxy.yml
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sber_scale
python - "$OUT/last.pth" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert {'model', 'optimizer', 'lr_scheduler', 'ema', 'scaler', 'last_epoch'} <= state.keys()
assert 'decoder.sber_head.offset_head.weight' in state['model']
print('Resume last_epoch', state['last_epoch'])
PY
test "$?" -eq 0 || exit 1
STAMP=$(date +%Y%m%d_%H%M%S)
if test -f "$OUT/best.pth"; then
  mv -n "$OUT/best.pth" "$OUT/best_before_resume_${STAMP}.pth"
fi
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c "$CFG" --resume "$OUT/last.pth" --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/resume_${STAMP}.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

Compare historic best and resumed best by the same COCO AP before final
evaluation. This existing bookkeeping and checkpoint frequency are unchanged.

## Caveats

No real E4' AP is established locally. CUDA AMP/NCCL and server memory still
need verification. The server cannot be accessed by SSH here; transfer by Git.
Strict bypass means each layer's bbox branch at its current h/reference;
earlier enabled refinements can still affect later reference/features. A GT
large object is not guaranteed to have a large reference at every layer.
This is the requested current-reference rule, not a GT-based gate. Preserve
E4 evidence and compare all nine metrics under the same protocol. No further
structure/threshold sweep is introduced.
