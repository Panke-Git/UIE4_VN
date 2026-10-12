#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GPU=0
SEED=3520
DATA_ROOT=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
for VARIANT in no_qgrd shared_query single_stage no_local_refine; do
  CMD=("${PYTHON_BIN}" -m "src.qgrd_ablations.${VARIANT}.train"
       --config "configs/ablations/qgrd/lsui/${VARIANT}.yaml" --seed "${SEED}" --gpu "${GPU}")
  if [[ -n "${DATA_ROOT}" ]]; then CMD+=(--data-root "${DATA_ROOT}"); fi
  printf '%q ' "${CMD[@]}"; printf '\n'
  if [[ "${DRY_RUN}" == 0 ]]; then "${CMD[@]}"; fi
done
