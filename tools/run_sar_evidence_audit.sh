#!/usr/bin/env bash
set -euo pipefail
cd /home/zxy/sar/repos/rtdetrv2_pytorch

MODE=${1:-run}
OUT=${2:-/home/zxy/sar/experiments/ogsod_r18_baseline_evidence_v1_20261008}
CONFIG=configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml
CHECKPOINT=/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_baseline_gbs64_gpu2_zxy/best.pth
PROTOCOL=configs/analysis/sar_evidence_v1.yml
if [[ "$MODE" != run && "$MODE" != smoke && "$MODE" != resume ]]; then
  echo "Usage: bash tools/run_sar_evidence_audit.sh run|smoke|resume [output_dir]" >&2
  exit 2
fi
test -f "$CHECKPOINT"
env CUDA_VISIBLE_DEVICES=0,1 python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Both GPUs must be visible"'
ARGS=(--config "$CONFIG" --checkpoint "$CHECKPOINT" --weights ema --protocol "$PROTOCOL")
if [[ "$MODE" == smoke ]]; then
  ARGS+=(--eval-limit 32)
fi
if [[ "$MODE" == resume ]]; then
  test -f "$OUT/identity.json"
  if [[ -f "$OUT/launcher.pid" ]] && kill -0 "$(cat "$OUT/launcher.pid")" 2>/dev/null; then
    echo "Recorded launcher PID is still alive; inspect it before resuming." >&2
    exit 2
  fi
  # Recover the original smoke limit; resume identity will reject other differences.
  LIMIT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_limit"] or "")' "$OUT/identity.json")
  if [[ -n "$LIMIT" ]]; then ARGS+=(--eval-limit "$LIMIT"); fi
  DIR_ARGS=(--resume-dir "$OUT")
  LOG="$OUT/resume_$(date +%Y%m%d_%H%M%S).log"
else
  if [[ -e "$OUT" || -e "$OUT.preflight" ]]; then
    echo "Output or preflight already exists; use a new directory or resume." >&2
    exit 2
  fi
  # Single CPU process: annotations/images/hashes/config/strict EMA before using GPUs.
  python tools/sar_evidence_audit.py "${ARGS[@]}" --preflight --output "$OUT.preflight"
  mkdir -p "$OUT"
  DIR_ARGS=(--output "$OUT")
  LOG="$OUT/console.log"
fi
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/sar_evidence_audit.py \
  "${ARGS[@]}" "${DIR_ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$PID" | tee "$OUT/launcher.pid"
disown "$PID"
echo "Two-GPU read-only evaluation started. Log: $LOG"
echo "No new detector checkpoint is produced; results: $OUT/report.md and $OUT/diagnostics.json"
