#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-python}" -B -m src.downstream_usod.export "$@"
