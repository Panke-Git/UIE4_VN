#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
  echo "Usage: $0 P1_RUN_DIR --uie-python /path/to/pax/python --gpu 0 --start-training"
  echo "Create the P1 plan first. This command enhances train/val, trains once, and evaluates E->R/E->E."
  exit 0
fi
run_dir="$1"
shift
uie_python=""
gpu="0"
confirmed="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --uie-python) uie_python="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --start-training) confirmed="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$confirmed" != "true" || -z "$uie_python" ]]; then
  echo "Need explicit --start-training and original V16 --uie-python; nothing started" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
detector_python="${PYTHON:-python}"
finish() {
  result=$?
  "$detector_python" -m src.downstream_duo.diagnostics.p1 report --run-dir "$run_dir" || true
  exit "$result"
}
trap finish EXIT
"$detector_python" -m src.downstream_duo.diagnostics.p1 check --run-dir "$run_dir"
"$uie_python" -m src.downstream_duo.diagnostics.p1 enhance --run-dir "$run_dir" --device "cuda:$gpu"
"$detector_python" -m src.downstream_duo.diagnostics.p1 train --run-dir "$run_dir" --gpu "$gpu" --start-training
"$detector_python" -m src.downstream_duo.diagnostics.p1 evaluate --run-dir "$run_dir" --gpu "$gpu"
