#!/usr/bin/env bash
# Main 200-epoch protocol. Run smoke first; this script never changes batch/architecture on OOM.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HAZE4K_ROOT="${HAZE4K_ROOT:-/root/autodl-tmp/pro/publicdata/Haze4K}"
export GPU="${GPU:-0}"
realpath "$HAZE4K_ROOT"
ls -ld "$HAZE4K_ROOT"/{train,test}/{haze,gt}
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required for main experiment"; print(torch.__version__, torch.cuda.get_device_name(int(__import__("os").environ["GPU"])))'
python -m src.haze4k_transfer.prepare --data-root "$HAZE4K_ROOT" --seed 3520
full_log="$(mktemp /tmp/haze4k_full.XXXXXX)"
wo_log="$(mktemp /tmp/haze4k_wo.XXXXXX)"
# Fair hardware adjustment can be supplied identically: BATCH_SIZE=8 ACCUMULATION_STEPS=2.
shared=(--gpu "$GPU" --data-root "$HAZE4K_ROOT" --batch-size "${BATCH_SIZE:-16}" --accumulation-steps "${ACCUMULATION_STEPS:-1}")
if [[ -n "${AMP_INIT_SCALE:-}" ]]; then shared+=(--amp-init-scale "$AMP_INIT_SCALE"); fi
# Actual selected GPU, unchanged 256-pixel patch and model; two batches per variant.
# Validation still uses the entire fixed validation split. Smoke checkpoints are never tested for paper use.
bash scripts/haze4k/train_full.sh "${shared[@]}" --smoke-epochs 1 --smoke-steps 2
bash scripts/haze4k/train_wo_uicf.sh "${shared[@]}" --smoke-epochs 1 --smoke-steps 2
bash scripts/haze4k/train_full.sh "${shared[@]}" 2>&1 | tee "$full_log"
bash scripts/haze4k/train_wo_uicf.sh "${shared[@]}" 2>&1 | tee "$wo_log"
FULL_RUN="$(sed -n 's/^RUN_DIR=//p' "$full_log" | tail -n 1)"
WO_RUN="$(sed -n 's/^RUN_DIR=//p' "$wo_log" | tail -n 1)"
[[ -d "$FULL_RUN" && -d "$WO_RUN" ]]
printf 'Actual FULL_RUN=%s\nActual WO_RUN=%s\n' "$FULL_RUN" "$WO_RUN"
bash scripts/haze4k/eval_compare.sh "$FULL_RUN" "$WO_RUN"
