#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Physical GPUs used by the project's vision/text DataParallel wrappers.
GPU_ID="${GPU_ID:-0,1,2,5}"
# Both .../DATA and .../DATA/Imagenet are accepted by train.py.
DATA_ROOT="${DATA_ROOT:-/home/duanzhenya/ood/gallo/DATA/Imagenet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/NR_RAPL/imagenet_16shot}"
DATASET_NAME="${DATASET_NAME:-imagenet}"
NUM_SHOTS="${NUM_SHOTS:-16}"
CLIP_NAME="${CLIP_NAME:-clip_vit_b16}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
# GPU 0 (the first ID in GPU_ID) gathers DataParallel outputs and therefore
# needs more headroom than the remaining cards. These conservative defaults
# keep four 24-GB cards stable; increase only after a successful epoch.
BATCH_SIZE="${BATCH_SIZE:-32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
PROMPTS_BATCH_SIZE="${PROMPTS_BATCH_SIZE:-1}"
CHECKPOINTING_SEGMENTS="${CHECKPOINTING_SEGMENTS:-4}"
PARALLEL_TEXT_ENCODER="${PARALLEL_TEXT_ENCODER:-true}"
PARALLEL_VISION_ENCODER="${PARALLEL_VISION_ENCODER:-true}"
# max_split_size_mb is supported by the PyTorch 1.13 environment supplied by
# this project. On newer PyTorch, users may override this with
# expandable_segments:True if desired.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m RAPL.train \
  --method_name RAPL --data_dir "${DATA_ROOT}" --save_dir "${OUTPUT_DIR}" \
  --dataset_name "${DATASET_NAME}" --num_shots "${NUM_SHOTS}" --clip_name "${CLIP_NAME}" \
  --batch_size "${BATCH_SIZE}" --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --prompts_batch_size "${PROMPTS_BATCH_SIZE}" --checkpointing_segments "${CHECKPOINTING_SEGMENTS}" \
  --parallel_text_encoder "${PARALLEL_TEXT_ENCODER}" \
  --parallel_vision_encoder "${PARALLEL_VISION_ENCODER}" \
  --n_global_prompts 4 --n_local_prompts 4 --topk 10 20 30 40 \
  --use_local_features true --use_global_loss true --use_local_loss true \
  --max_epoch "${STAGE1_EPOCHS}" --eval_freq 1 --checkpoint_name stage1_RAPL_best.pth
