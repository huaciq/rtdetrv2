#!/usr/bin/env bash
# Evidence audit only. Run inside the activated rtdetr_zxy environment.
set -euo pipefail
cd /home/zxy/sar/repos/rtdetrv2_pytorch

mode=${1:?Usage: run_sar_audit.sh discover OUT | run SPEC OUT | resume SPEC OUT}
if [[ "$mode" == discover ]]; then
    exec python tools/sar_audit.py --discover --output "${2:?new evidence directory required}"
fi

spec=${2:?explicit run_spec.json required}
out=${3:?audit output directory required}
if [[ "$mode" == run ]]; then
    if [[ -e "$out" || -e "$out.preflight" ]]; then
        echo "Output or preflight directory already exists; use new paths or resume: $out" >&2
        exit 1
    fi
    # This checks actual checkpoint keys, paths and annotations before GPU diagnostics.
    python tools/sar_audit.py --spec "$spec" --preflight --output "$out.preflight"
    mkdir -p "$out"
    directory_option=--output
    log="$out/console.log"
elif [[ "$mode" == resume ]]; then
    if [[ ! -f "$out/audit_identity.json" ]]; then
        echo "Cannot resume an audit without audit_identity.json: $out" >&2
        exit 1
    fi
    directory_option=--resume-dir
    log="$out/resume_$(date +%Y%m%d_%H%M%S)_$$.log"
else
    echo "Unknown mode: $mode" >&2
    exit 1
fi

nohup env CUDA_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
    tools/sar_audit.py --spec "$spec" "$directory_option" "$out" \
    > "$log" 2>&1 &
audit_pid=$!
printf '%s\n' "$audit_pid" > "$out/launcher.pid"
disown "$audit_pid"
printf 'Audit PID: %s\nOutput: %s\nLog: %s\n' "$audit_pid" "$out" "$log"
