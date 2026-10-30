#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ "$#" -ne 5 ]; then
  echo "Usage: bash $0 /absolute/LSUI_ROOT GPU_INDEX DIRECT_RUN WITHOUT_RUN FIXED_RUN" >&2
  exit 2
fi
PYTHON="${PYTHON:-python}"
data_root="$1"
gpu="$2"
"$PYTHON" -m src.uicf_parameterization_ablations.test --variant direct_residual --run-dir "$3" --checkpoint best_psnr --gpu "$gpu" --data-root "$data_root"
"$PYTHON" -m src.uicf_parameterization_ablations.test --variant without_anchor --run-dir "$4" --checkpoint best_psnr --gpu "$gpu" --data-root "$data_root"
"$PYTHON" -m src.uicf_parameterization_ablations.test --variant fixed_anchor --run-dir "$5" --checkpoint best_psnr --gpu "$gpu" --data-root "$data_root"
