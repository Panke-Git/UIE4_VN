#!/usr/bin/env bash
# User-run WSL helper. This script neither selects a dataset path nor changes batch size.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ "$#" -ne 2 ]; then
  echo "Usage: bash $0 /absolute/LSUI_ROOT GPU_INDEX" >&2
  exit 2
fi
PYTHON="${PYTHON:-python}"
data_root="$1"
gpu="$2"
for variant in direct_residual without_anchor fixed_anchor; do
  "$PYTHON" -m src.uicf_parameterization_ablations.train \
    --config "configs/ablations/uicf_parameterization/${variant}.yaml" \
    --variant "$variant" --seed 3520 --gpu "$gpu" --data-root "$data_root" \
    --output-dir "experiments/uicf_parameterization/${variant}"
done
