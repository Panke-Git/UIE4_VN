#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:-dry-run}"
stamp="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)"
output="${2:-$root/experiments/eval_sift_reference_v2/${stage}_${stamp}}"
python_bin="${SIFT_V2_PYTHON:-/home/paxton/miniconda3/envs/duo_yolo/bin/python}"
args=(--config "$root/configs/sift/reference_v2.home_wsl.json" --output-dir "$output")
case "$stage" in
  dry-run) args+=(--dry-run) ;;
  smoke) args+=(--only-ids 1930 220 895 1757 1544 2753 3035 2010 2519 1812) ;;
  regression) args+=(--regression-only --K unlimited) ;;
  full) ;;
  *) echo "Usage: bash scripts/run_sift_reference_v2_wsl.sh {dry-run|smoke|regression|full} [NEW_OUTPUT_DIR]" >&2; exit 2 ;;
esac
exec "$python_bin" "$root/tools/sift_reference_consistency_v2.py" "${args[@]}"
