# E2: SAR-aware Quality Query Selection (SQ-QS)

## What changed

- `src/zoo/rtdetr/sar_quality.py`: per-level 3x3 average-pool residual on encoder candidate `output_memory`. The quality input is `[f, f - AvgPool(f), sigmoid(proposal_logits).detach()]`; one linear head predicts localization-quality logits. Border pooling excludes padding. Feature levels never mix. No FFT, wavelet, extra attention or convolution.
- `src/zoo/rtdetr/rtdetrv2_decoder.py`: opt-in `sar_quality`, default false. E0/E1 keep their original path and state keys. E2 reuses `cls_score * Q_loc^beta`, quality sampling, max-IoU targets, BCE and weight unchanged. Geometry is detached so quality loss does not add gradients to the bbox head. Common detector initialization and subsequent RNG state match E1 at seed 0; Q starts at 0.5.
- `configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sqqs_gbs64_gpu2_zxy.yml`: inherits the E1 GBS64 QAQS config; only the quality head, independent output directory and test-only focused reporting change. Adds 260 parameters (517 vs E1's 257).
- `src/solver/query_selection_metrics.py`, `det_engine.py`, `det_solver.py`: optional test-only compact reporting on both GPUs, without modifying COCO evaluation, postprocessing, training metrics, checkpoint selection or the original query diagnosis.
- `tools/test_sqqs.py`: synthetic CPU checks. This guide contains the server handoff.

The focused report contains AP, AP50, AP75, APs (COCO values on a 0–1 scale), small Query Recall@IoU=0.75, and small best-query rank mean/median/P90. Recall uses **selected encoder proposals before decoder refinement**, class-agnostic max IoU, all 300 selected queries. Rank is one-based **actual selection-score order**, and is computed for every small GT including unrecalled/zero-IoU GTs; ties choose the first query. Small means annotation area `<32^2` in original image pixels, matching existing QueryStats. DistributedSampler padding is deduplicated by image ID. No small GT means null metrics, not zero. This rank measures ordering inside the selected set, not ranking of all encoder candidates.

## Local verification

```powershell
conda run -n pytorch python -X utf8=0 tools/test_sqqs.py
git diff --check
```

The Windows UTF-8 switch avoids the existing environment's non-UTF-8 `.pth` startup issue. The test uses CPU, generated inputs, mixed nonempty/empty GT, two ordinary training/backward/optimizer steps with DN, strict model/EMA/criterion/optimizer checkpoint roundtrips, rank formula/beta=0, per-level locality and two-process CPU Gloo DDP. It checks config equality outside the intended E2 changes and exact E1/E2 initial shared weights and predictions. It does not verify a real GPU experiment.

## Git transfer

Branch: `codex/ogsod-r18-sqqs`. Inspect server changes first. Stop if dirty or diverged; do not force checkout or reset.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/ogsod-r18-sqqs
git pull --ff-only origin codex/ogsod-r18-sqqs
git rev-parse HEAD
conda activate rtdetr_zxy
python -c 'import torch, faster_coco_eval, pycocotools; print(torch.__version__); print(torch.cuda.device_count()); assert torch.cuda.device_count() == 2'
python tools/test_sqqs.py
```

No new server dependency or checkpoint migration. Use the existing requirements/environment. E1 checkpoints have a different quality head: **do not resume E2 from an E1 checkpoint**. The formal E2 comparison starts fresh with the same PResNet-18 ImageNet pretrained initialization as E0/E1, with no `--tuning`. `--resume` below is exclusively for an existing E2 run.

## Server preflight and launch

The inherited dataset paths are `/home/zxy/sar/datasets/OGSOD-1.0/sar` and `RTDETR_COCO/{train,val}.json`, recorded in the existing E0/E1 config. Verify these actual paths on the server, the 3-category mapping, all referenced images and train/validation disjointness before expensive training. Do not silently replace them with another split. Check free disk space and GPU availability.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
test -d /home/zxy/sar/datasets/OGSOD-1.0/sar || exit 1
test -f /home/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/train.json || exit 1
test -f /home/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/val.json || exit 1
python - <<'PY'
import json
from pathlib import Path
from src.core.yaml_utils import load_config

cfg = load_config('configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sqqs_gbs64_gpu2_zxy.yml')
paths, categories = {}, {}
for split in ('train', 'val'):
    dataset = cfg[f'{split}_dataloader']['dataset']
    root = Path(dataset['img_folder'])
    data = json.loads(Path(dataset['ann_file']).read_text())
    ids = [image['id'] for image in data['images']]
    id_set = set(ids)
    assert len(ids) == len(id_set), f'{split}: duplicate image IDs'
    categories[split] = {c['id']: c['name'] for c in data['categories']}
    assert set(categories[split]) == set(range(cfg['num_classes'])), 'Check class IDs/remapping'
    paths[split] = {(root / image['file_name']).resolve() for image in data['images']}
    assert len(paths[split]) == len(ids), f'{split}: duplicate image files'
    assert all(path.is_file() for path in paths[split]), f'{split}: missing image file'
    assert all(a['image_id'] in id_set and a['category_id'] in categories[split]
               for a in data['annotations']), f'{split}: invalid annotation reference'
    print(split, len(ids), 'images', len(data['annotations']), 'annotations')
assert categories['train'] == categories['val'], 'Category definitions differ'
assert not paths['train'].intersection(paths['val']), 'Train/val image overlap'
print('Configured data/class/path checks passed; independently verify split provenance.')
PY
test "$?" -eq 0 || exit 1
df -h /home/zxy/sar/experiments
nvidia-smi
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sqqs
test ! -e "$OUT" || { echo 'Output already exists; inspect or resume the matching E2 run.'; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sqqs_gbs64_gpu2_zxy.yml \
  --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

80 epochs, global batch 64 (=32 per rank), LR 4e-4/backbone 4e-5, input 640, 300 queries, seed 0, existing AMP/EMA and augmentation. Keep these identical to the intended E0/E1 runs. A short two-GPU pilot in a separate output directory is recommended before the full run; synthetic tests do not establish SAR performance.

### Evaluate E2 best.pth

Evaluation uses the repository's existing EMA weights when present and the same FP32 evaluation path as E0/E1. Training AMP is not applied to evaluation by this code.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sqqs
EVAL="$OUT/eval_best"
test -f "$OUT/best.pth" || exit 1
test ! -e "$EVAL" || { echo 'Evaluation output exists; choose a new directory.'; exit 1; }
mkdir -p "$EVAL"
git rev-parse HEAD > "$EVAL/git_commit.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sqqs_gbs64_gpu2_zxy.yml \
  --resume "$OUT/best.pth" --test-only --seed 0 --output-dir "$EVAL" \
  > "$EVAL/console.log" 2>&1 &
EVAL_PID=$!
echo "$EVAL_PID" | tee "$EVAL/launcher.pid"
disown
```

### E0/E1 comparison under the same protocol

The checkpoint paths below are the **expected locations from the existing configs**. Confirm their historical config, seed, global batch, split and pretrained initialization from saved evidence before treating these as fair E0/E1 results. The focused reporting flag does not change their predictions or checkpoint state.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
for METHOD in baseline qaqs; do
  RUN=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_${METHOD}_gbs64_gpu2_zxy
  EVAL="$RUN/eval_best_sqqs_protocol"
  test -f "$RUN/best.pth" || { echo "Missing $RUN/best.pth"; break; }
  test ! -e "$EVAL" || { echo "Existing output $EVAL"; break; }
  mkdir -p "$EVAL"
  git rev-parse HEAD > "$EVAL/git_commit.txt"
  env CUDA_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_${METHOD}_gbs64_gpu2_zxy.yml \
    --resume "$RUN/best.pth" --test-only --seed 0 --output-dir "$EVAL" \
    -u query_selection_metrics=True > "$EVAL/console.log" 2>&1
done
```

Both older runs produce `eval_best_sqqs_protocol/query_selection_metrics.json`; E2 produces `eval_best/query_selection_metrics.json`. Compare the six focused metrics in these JSON files and retain the full `evaluation_metrics.json` and `eval.pth`. The JSON includes resolved config, seed, checkpoint path, weight source (EMA/model), query counts, small GT denominator and protocol. Each evaluation directory's `git_commit.txt` records the evaluation code; historical training commits must come from the original run evidence.

### One optional beta sweep

Default beta is 1.0, the E1 config value; no verified sweep result is stored in Git. If E1's verified best beta differs, update only E2 `quality_beta` **before training**, and retain that same value when resuming/evaluating. The only sweep interface is the existing CLI override:

```bash
-u RTDETRTransformerv2.quality_beta=0.5
```

If needed, evaluate the **same trained E2 best.pth** once at beta 0.5/1.0/2.0, each in a new directory such as `eval_beta_0p5`, using the best evaluation command above plus that override. Do not add scoring variants or repeat broad tuning. Record the beta with the metrics; a changed-beta evaluation is distinct from the primary fixed-beta E0/E1/E2 comparison.

## Monitoring and metrics

```bash
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sqqs
tail -n 100 -f "$OUT/console.log"
watch -n 2 nvidia-smi
ps -fp "$(cat "$OUT/launcher.pid")"
tail -n 5 "$OUT/log.txt"
tail -n 1 "$OUT/log.txt" | python -m json.tool
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
python -m json.tool "$OUT/eval_best/query_selection_metrics.json"
python -m json.tool "$OUT/eval_best/evaluation_metrics.json"
```

`log.txt` retains original per-epoch COCO metrics and `train_loss_quality`. Focused query metrics run only with `--test-only`, so there is no change to the training schedule, training diagnosis overhead or best.pth selection (AP@[.50:.95]). Existing single-process full diagnosis remains available in `query_diagnosis/`; it is still skipped during two-process evaluation. The new compact six-metric report works in both process modes.

## Outputs and resume

Training: `$OUT/{best.pth,last.pth,checkpointNNNN.pth,console.log,log.txt}`, `$OUT/summary/`, `$OUT/eval/latest.pth`. Best evaluation: `$OUT/eval_best/{query_selection_metrics.json,evaluation_metrics.json,eval.pth,console.log}`. Never reuse a directory for a different beta or checkpoint.

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_sqqs
test -f "$OUT/last.pth" || exit 1
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sqqs_gbs64_gpu2_zxy.yml \
  --resume "$OUT/last.pth" --use-amp --seed 0 --output-dir "$OUT" \
  > "$OUT/resume.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

## Caveats

Server paths and real SAR data/checkpoints cannot be verified from the Windows workspace. Check split/category/image integrity, memory use with real AMP/DDP and available checkpoint disk space on the Linux host. No real AP gain is claimed. This E2 tests adding residual plus geometry together; attributing gains solely to the residual would require a controlled geometry-only ablation, outside this implementation's scope.
