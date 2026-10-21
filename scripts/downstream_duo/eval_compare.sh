#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 PREPARED BEST_PT ENHANCED [OUTPUT_PARENT] [DEVICE]" >&2
  exit 2
fi
prepared="$1"
weights="$2"
enhanced="$3"
output="${4:-experiments/downstream_duo/evaluation}"
device="${5:-0}"
py="${PYTHON:-python}"
"$py" -m src.downstream_duo.evaluate --prepared "$prepared" --weights "$weights" --condition raw --output "$output/raw" --device "$device"
"$py" -m src.downstream_duo.evaluate --prepared "$prepared" --weights "$weights" --condition full --enhanced "$enhanced" --output "$output/full" --device "$device"
"$py" -m src.downstream_duo.compare --raw "$output/raw/metrics_raw.json" --full "$output/full/metrics_full.json" --output "$output/comparison"
"$py" -m src.downstream_duo.visualize --raw "$output/raw/metrics_raw.json" --full "$output/full/metrics_full.json" --output "$output/figures" --conf 0.25 --rows 3 --gt
