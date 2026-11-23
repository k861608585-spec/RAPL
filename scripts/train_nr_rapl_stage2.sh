#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the dataset root before running this script}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/NR_RAPL/imagenet_16shot}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${OUTPUT_DIR}/stage1_RAPL_best.pth}"
DATASET_NAME="${DATASET_NAME:-imagenet}"
NUM_SHOTS="${NUM_SHOTS:-16}"
CLIP_NAME="${CLIP_NAME:-clip_vit_b16}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then echo "Missing Stage 1 checkpoint: ${STAGE1_CHECKPOINT}" >&2; exit 1; fi
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m RAPL.train \
  --method_name NR_RAPL --nr_train_stage regional --checkpoint_path "${STAGE1_CHECKPOINT}" \
  --data_dir "${DATA_ROOT}" --save_dir "${OUTPUT_DIR}" --dataset_name "${DATASET_NAME}" \
  --num_shots "${NUM_SHOTS}" --clip_name "${CLIP_NAME}" --n_global_prompts 4 --n_local_prompts 4 --topk 10 20 30 40 \
  --use_local_features true --use_global_loss true --use_local_loss true --max_epoch "${STAGE2_EPOCHS}" --eval_freq 1 \
  --nr_crop_enabled true --nr_use_guided_crop true --nr_neg_prompt_enabled true --nr_use_neg_prompt true \
  --nr_use_neg_loss true --nr_use_div_loss true --nr_ood_score_type NR_GLMCM --nr_use_nr_glmcm true \
  --checkpoint_name stage2_NR_RAPL_best.pth
