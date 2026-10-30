#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
mkdir -p analysis/uicf_parameterization/local_tests
"$PYTHON" -m pytest tests/uicf_parameterization_ablations -q \
  --basetemp=analysis/uicf_parameterization/local_tests/tmp \
  --junitxml=analysis/uicf_parameterization/local_tests/new_tests.xml
"$PYTHON" -m pytest tests/test_uicf.py tests/test_uicf_unet.py \
  tests/test_color_query_unet.py tests/test_e00.py tests/test_experiment_status.py -q \
  --basetemp=analysis/uicf_parameterization/local_tests/regression_tmp \
  --junitxml=analysis/uicf_parameterization/local_tests/regressions.xml
for script in scripts/uicf_parameterization_ablations/*.sh; do bash -n "$script"; done
