#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pytest tests/qgrd_ablations/ -q
"${PYTHON_BIN}" -m pytest tests/test_color_query_unet.py tests/test_uicf.py tests/test_training_integration.py tests/test_v19_sdqr.py -q
