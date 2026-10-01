# OGSOD audit handoff

## 1. What changed

`tools/sar_audit.py` discovers experiment evidence and performs controlled evaluation, per-target loss and gradient diagnostics, and encoder/decoder query export. `tools/run_sar_audit.sh` launches only the audit, using two GPUs and no optimizer updates. Detector, criterion and historical experiment configs are unchanged.

The default config candidates are the GBS64 baseline, QAQS, and QAQS relative-loss configs. They are **not proof of which configs produced historical checkpoints**. Confirm or replace each `config` in the generated specification using the actual run evidence. The data paths recorded by these candidates are `/home/zxy/sar/datasets/OGSOD-1.0/sar`, with annotations in `RTDETR_COCO/train.json` and `val.json`.

## 2. Local verification

Use `conda activate pytorch` and:

```powershell
python tools/sar_audit.py --self-test --output outputs/ogsod_audit_selftest_new
```

The check covers historical `f102bb3^` criterion equivalence at lambda=0, all ordinary/encoder/DN synthetic prediction gradients, small-boundary/empty-selection/asymmetric-coordinate cases, an adversarial maximum-cardinality matching case, the 3667-image DistributedSampler example, construction of all three configs, exact eval hook equality with nonzero residuals, and the complete COCO/CSV/gradient output pipeline on generated images. These are synthetic checks, not real experiment metrics.

In this Windows workspace the existing `pytorch` environment fails during `.pth` decoding. Local verification uses the same interpreter with `-S`, an ignored bootstrap, and missing dependencies installed only under ignored `outputs/audit_runtime_deps`. The Conda environment itself is not edited. Command:

```powershell
conda run -n pytorch python -S outputs/audit_bootstrap.py --self-test --output outputs/ogsod_audit_selftest_new
```

## 3. Git transfer

The implementation is on `codex/ogsod-r18-baseline-gbs64`. Use the commit reported in the chat. On the Linux server:

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/ogsod-r18-baseline-gbs64
git pull --ff-only origin codex/ogsod-r18-baseline-gbs64
conda activate rtdetr_zxy
python -c 'import torch, torchvision, scipy, yaml, faster_coco_eval, pycocotools; print(torch.__version__, faster_coco_eval.__version__, torch.cuda.device_count())'
df -h /home/zxy/sar/experiments
```

Stop to inspect if `git status` is dirty or `pull --ff-only` fails. No new package is needed if the existing repository requirements are already installed. If the import check names a missing package, install that specific dependency from `requirements.txt`; do not upgrade the training environment as part of the audit. No model-weight downloads are made.

## 4. Server launch

First collect exact run-directory contents and hashes without selecting a checkpoint:

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
bash tools/run_sar_audit.sh discover /home/zxy/sar/experiments/ogsod_r18_audit_evidence_20261001_01
```

Inspect `eval_manifest.json`, then fill `run_spec.json`. Required: exact `checkpoint` for each of the three `runs`. Never infer best/last from a filename alone. Record `run_git_commit`, `launch_command`, and `parent_checkpoint` where evidence exists; use the string `none_from_scratch` when no parent was used. Leave unknown provenance null so it remains explicitly unresolved. Checkpoint SHA256 is computed by the tool. Choose one `weights` value for all three checkpoints: `ema` (the normal evaluator path) or `model`. A missing requested weight key or any missing/unexpected/shape-mismatched tensor blocks evaluation.

Evaluation uses `weights: "ema"`; gradient forwards use `gradient_weights: "model"` to inspect the saved training model, rather than its EMA smoothing. Each path is loaded strictly and reported separately. Both choices are uniform across the three runs and recorded. Change them only in a new audit specification/output.

To enter the three explicit paths interactively, without editing JSON by hand:

```bash
read -r -p 'Exact baseline checkpoint: ' BASELINE_CKPT
read -r -p 'Exact QAQS checkpoint: ' QAQS_CKPT
read -r -p 'Exact relative-loss failed checkpoint: ' RELATIVE_CKPT
python - /home/zxy/sar/experiments/ogsod_r18_audit_evidence_20261001_01/run_spec.json "$BASELINE_CKPT" "$QAQS_CKPT" "$RELATIVE_CKPT" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
s = json.loads(p.read_text())
paths = dict(zip(('baseline', 'qaqs', 'relative_failed'), sys.argv[2:]))
for run in s['runs']:
    run['checkpoint'] = paths[run['name']]
    if not pathlib.Path(run['checkpoint']).is_file():
        raise FileNotFoundError(run['checkpoint'])
p.write_text(json.dumps(s, indent=2) + '\n')
PY
```

`gradient_batches=16`, global training batch=64, audit seed=0, audit epoch=0. The tool saves and reuses the same augmented input tensors across models. Fixed batches alone require roughly 4.7 GiB at 640x640; allow additional space for query exports, evaluation states and per-image geometry. Default precision is FP32 for the controlled gradient comparison. Set `gradient_precision` to `amp` for a separate audit matching the original autocast-forward/FP32-criterion execution; use a new output directory for that audit. No loss scaling or optimizer step is needed for gradient measurement. The inherited SyncBN is used for two-rank gradient forwards, and model buffers are restored before each batch.

After the spec is filled:

```bash
bash tools/run_sar_audit.sh run \
  /home/zxy/sar/experiments/ogsod_r18_audit_evidence_20261001_01/run_spec.json \
  /home/zxy/sar/experiments/ogsod_r18_audit_20261001_01
```

This first performs strict checkpoint/data/config preflight, then starts `nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/sar_audit.py` in the background, saves the PID and calls `disown`. Existing output directories are refused. It evaluates all validation images using padding-free rank shards, while retaining the repository postprocessor and COCO evaluator. Gradients use the fixed training inputs and all original loss branches.

For the historical probe count question, provide exact probe config/checkpoint pairs in `probes`. All four known probe directories/configs are listed in `probe_evidence_candidates` in the manifest. For example, an entry has `name`, `config`, `checkpoint`, `weights: "model"`; these frozen probe configs disable EMA. The tool replays the current probe update code with the ordinary padded DistributedSampler and records per-image Top100 contributions plus COCO unique IDs. Scalar Top100 queries and class-conditioned Top100 query/class pairs are different statistical populations. Missing probe checkpoints leave the 366800/366700 historical explanation unresolved; the tool does not replace that evidence with a theoretical padding example.

## 5. Monitoring and metrics

```bash
OUT=/home/zxy/sar/experiments/ogsod_r18_audit_20261001_01
tail -n 100 -f "$OUT/console.log"
watch -n 2 nvidia-smi
ps -fp "$(cat "$OUT/launcher.pid")"
python -m json.tool "$OUT/eval_manifest.json"
python -m json.tool "$OUT/qaqs/eval_complete.json"
python -m json.tool "$OUT/relative_failed/loss_gradient_distributions.json"
```

`eval_complete.json` contains the original 12 COCO statistics (plus any additional statistics exposed by the installed evaluator), per-stage maxDets, thresholds and IDs. The main comparison uses final decoder AP/AP50/AP75; class-by-scale AP is in `localization_by_class_scale.csv`. Values are fractions, not percentages. AP for absent strata is null. Gradient distributions include mean/P50/P90/P99/max, class-by-scale scalar contributions, finite flags, and ratios/cosines against weighted L1+GIoU. A near-zero baseline yields a null ratio.

This audit creates no TensorBoard events. To inspect the original QAQS run:

```bash
tensorboard --logdir /home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_qaqs_gbs64_gpu2_zxy/summary --host 0.0.0.0 --port 6006
```

## 6. Outputs and resume

Main requested artifacts under `/home/zxy/sar/experiments/ogsod_r18_audit_20261001_01`:

- `audit_summary.md`, `eval_manifest.json`, `loss_gradient_audit.csv`;
- `localization_by_class_scale.csv`, `decoder_refinement_summary.json`;
- `probe_count_audit.json`, `fixed_batches/`, and each run's `queries_rank0/`, `queries_rank1/`;
- per-run `eval_complete.json`, `gradient_complete.json`, `loss_per_target.csv`, `loss_gradient_distributions.json`;
- per-stage serialized COCO evaluation data, per-rank image-visit records and loss dictionaries;
- `failure_rankN.json` on failure, `console.log`, `launcher.pid`.

No new model checkpoint, `last.pth` or `best.pth` is produced. Generated audit evidence stays outside Git.

Resume with the same spec, sources, checkpoint/data hashes, limits and two GPUs:

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
bash tools/run_sar_audit.sh resume \
  /home/zxy/sar/experiments/ogsod_r18_audit_evidence_20261001_01/run_spec.json \
  /home/zxy/sar/experiments/ogsod_r18_audit_20261001_01
```

Resume skips completed run stages and reruns incomplete stages from the start, reusing fixed input tensors. It creates a distinct resume log and refreshes launcher.pid. It is an audit-stage resume, not a training resume. A changed spec/code/weights/data/world size requires a new output directory.

## 7. Caveats and interpretation

Local synthetic checks do not verify real AP, server AMP/SyncBN collectives or the failed run's actual gradients. The server cannot be reached directly from Windows; transfer is via Git and the commands above must run on the server. Exact checkpoint paths and historical run provenance remain required.

The tool validates all annotation keys, class IDs, image existence and train/val file-name disjointness. File-name disjointness does not establish source independence: original-image byte hashes and near-duplicate/crop overlap still require further evidence. It records category/area/ignore/crowd counts and all original annotation hashes. Invalid boxes and area-vs-bbox differences are recorded, not silently relabeled. Current configs require direct category IDs 0,1,2 because remapping is off.

Geometry includes all valid, noncrowd, nonignored GT and all selected encoder/regular decoder queries. It reports zero-IoU targets. One-to-one coverage is maximum cardinality independently within each stratum; global coverage uses the entire GT set. COCO boundaries at 32^2 and 96^2 are inclusive and may appear in adjacent strata. Geometric boxes are clipped like the dataset loader; COCO scales use original annotation `area`. Encoder query IDs are selected-token IDs plus regular query order, not a correspondence across two models.

Fixed-query trajectories select the encoder best query once per GT. Layer-best trajectories reselect each stage. Query export uses eval hooks, excludes DN and asserts exact final prediction equality. All intermediate decoder APs use the same class-only postprocessing rule as the final stage. The tool's padding-free sample traversal deliberately audits unique images; optional probe replay separately reproduces the current padded update path.

Scalar per-target losses add under their branch reduction. Their norms are not reported as additive shares of the final gradient. The shared decoder parameter group includes decoder layers and query-position MLP; the regression group includes ordinary decoder bbox heads. Prediction-box gradients are measured on the full branch outputs in the actual computation graph, including graph dependencies. Parameter vectors are averaged across ranks before norms/cosines; prediction vectors are concatenated across disjoint samples and divided by world size. Post-clip parameter subgroup norms are calculated using the complete objective's parameter clip coefficient. Prediction gradients are not clipped and their post-clip field is null.

The lambda=0 comparison constructs the historical criterion from `f102bb3^` without registering it or changing repository source. It compares base losses and gradients on the same predictions. The manifest additionally records whether detector/data sources changed since that revision. If they changed, same-output criterion equivalence does not prove historical forward equivalence. The tool handles the repository YAML loader's mutable default by giving each config an independent dictionary.

Finite-loss/gradient failures stop the audit with a failure record; missing or partial metrics are never presented as completed real experiments. The static summary reports confirmed code behavior and unresolved evidence; it does not claim a measured scientific cause for AP degradation.
