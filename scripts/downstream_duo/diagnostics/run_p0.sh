#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "${PYTHON:-python}" -m src.downstream_duo.diagnostics.run_p0 --help
fi
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 PREPARED_DUO RAW_BEST_PT ENHANCED_TILED OUTPUT_RUN [--uie-run-dir DIR] [--uie-python PYTHON] [--gpu 0] [--data-root DUO] [--evaluation-root DIR] [--sample-count 32]" >&2
  exit 2
fi
prepared="$1"
weights="$2"
enhanced="$3"
output="$4"
shift 4
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -m src.downstream_duo.diagnostics.run_p0 \
  --prepared "$prepared" --weights "$weights" --enhanced "$enhanced" --output "$output" "$@"
