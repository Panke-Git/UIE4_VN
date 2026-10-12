#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <A1_RUN> <A2_RUN> <A3_RUN> <A4_RUN> [--gpu 0] [--data-root PATH] [--dry-run]" >&2
  exit 2
fi
RUNS=("$1" "$2" "$3" "$4")
shift 4
VARIANTS=(no_qgrd shared_query single_stage no_local_refine)
GPU=0
DATA_ROOT=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
for INDEX in 0 1 2 3; do
  RUN_DIR="${RUNS[${INDEX}]}"
  if [[ "${DRY_RUN}" == 0 ]]; then
    "${PYTHON_BIN}" -c 'import sys,pathlib,yaml; c=yaml.safe_load((pathlib.Path(sys.argv[1])/"config_resolved.yaml").read_text()); assert c["data"]["dataset"] == sys.argv[2], "Wrong dataset for test script"' "${RUN_DIR}" "LSUI19"
  fi
  CMD=("${PYTHON_BIN}" -m "src.qgrd_ablations.${VARIANTS[${INDEX}]}.test"
       --run-dir "${RUN_DIR}" --checkpoint best_psnr --gpu "${GPU}")
  if [[ -n "${DATA_ROOT}" ]]; then CMD+=(--data-root "${DATA_ROOT}"); fi
  printf '%q ' "${CMD[@]}"; printf '\n'
  if [[ "${DRY_RUN}" == 0 ]]; then "${CMD[@]}"; fi
done
