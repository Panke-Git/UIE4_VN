#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python="${PYTHON:-python}"
if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  exec "$python" -B -m src.downstream_usod.cli pipeline --phase p0 "$@"
fi
"$python" -B -m src.downstream_usod.cli smoke --phase p0 "$@"
exec "$python" -B -m src.downstream_usod.cli pipeline --phase p0 "$@"
