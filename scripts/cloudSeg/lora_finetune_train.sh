#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/data1/dinov2-seg"
CONFIG="${CONFIG:-/mnt/data1/dinov2-seg/scripts/cloudSeg/configs/lora_vitg16.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data1/dinov2-seg/outputs/cloudseg/lora_vitg16}"
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

python scripts/cloudSeg/lora_finetune_train.py \
  --output-dir "$OUTPUT_DIR" \
  --config "$CONFIG" \
  $RESUME_FLAG \
  $AUTO_RESUME_FLAG \
  $WANDB_FLAG
