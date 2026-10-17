#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ $# -ne 2 ]]; then
  echo 'Usage: GPU=0 HAZE4K_ROOT=/path/to/Haze4K bash scripts/haze4k/eval_compare.sh FULL_RUN WO_RUN' >&2
  exit 2
fi
full_run="$1"
wo_run="$2"
resolution="${RESOLUTION:-256}"
root_args=()
if [[ -n "${HAZE4K_ROOT:-}" ]]; then root_args=(--data-root "$HAZE4K_ROOT"); fi
python -m src.haze4k_transfer.eval --run-dir "$full_run" --checkpoint best_psnr --gpu "${GPU:-0}" --resolution "$resolution" "${root_args[@]}"
python -m src.haze4k_transfer.eval --run-dir "$wo_run" --checkpoint best_psnr --gpu "${GPU:-0}" --resolution "$resolution" "${root_args[@]}"
python -m src.haze4k_transfer.compare --full-run "$full_run" --wo-run "$wo_run" --resolution "$resolution"
