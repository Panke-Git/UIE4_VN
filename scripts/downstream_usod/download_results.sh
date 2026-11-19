#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || "$1" == "--help" ]]; then
  printf '%s\n' 'Usage: download_results.sh /absolute/remote/report.tar.gz LOCAL_REPORT.tar.gz'
  [[ $# -eq 1 && "$1" == "--help" ]] && exit 0
  exit 2
fi
[[ "$1" == /* ]] || { printf '%s\n' 'Use the absolute remote archive path.' >&2; exit 2; }
[[ ! -e "$2" && ! -L "$2" ]] || { printf 'Local output already exists: %s\n' "$2" >&2; exit 2; }
temporary=$(mktemp "$(dirname "$2")/.usod-download.XXXXXXXX")
trap 'unlink "$temporary"' EXIT
scp -o BatchMode=yes -- "school:$1" "$temporary"
tar -tzf "$temporary" >/dev/null
ln "$temporary" "$2"
printf 'Downloaded %s\n' "$2"
