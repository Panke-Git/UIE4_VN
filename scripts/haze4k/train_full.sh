#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m src.haze4k_transfer.train --config configs/haze4k/full_v16.yaml "$@"
