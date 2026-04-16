#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/scripts/cloudSeg/configs/full_finetune_vitg16_fpn.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/cloudseg/full_finetune_vitg16_fpn}"
WANDB_FLAG=""
RESUME_FLAG=""
AUTO_RESUME_FLAG=""

cd "$REPO_ROOT"

if [[ "${USE_XFORMERS:-0}" == "1" ]]; then
  unset XFORMERS_DISABLED
else
  export XFORMERS_DISABLED=1
fi

if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_FLAG="--wandb"
fi

if [[ -n "${RESUME_PATH:-}" ]]; then
  RESUME_FLAG="--resume-path ${RESUME_PATH}"
fi

if [[ "${AUTO_RESUME:-1}" == "0" ]]; then
  AUTO_RESUME_FLAG="--no-auto-resume"
fi

/opt/conda/envs/qwen3/bin/python scripts/cloudSeg/full_finetune_train.py \
  --output-dir "$OUTPUT_DIR" \
  --config "$CONFIG" \
  $RESUME_FLAG \
  $AUTO_RESUME_FLAG \
  $WANDB_FLAG
