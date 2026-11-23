#!/usr/bin/env bash
set -euo pipefail

# Full NR_RAPL regional training. It starts from a Stage-1 RAPL checkpoint
# and enables guided crops, positive/negative regional objectives, diversity,
# and the NR-GLMCM evaluation score. No data or weights are downloaded.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the dataset root before running this script}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/NR_RAPL/full_imagenet_16shot}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${OUTPUT_DIR}/stage1_RAPL_best.pth}"
DATASET_NAME="${DATASET_NAME:-imagenet}"
NUM_SHOTS="${NUM_SHOTS:-16}"
CLIP_NAME="${CLIP_NAME:-clip_vit_b16}"
SEED="${SEED:-1}"
FULL_EPOCHS="${FULL_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-256}"
LR="${LR:-0.002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GLOBAL_PROMPTS="${GLOBAL_PROMPTS:-4}"
LOCAL_PROMPTS="${LOCAL_PROMPTS:-4}"
LOCAL_SCALES="${LOCAL_SCALES:-10 20 30 40}"
CROP_CANDIDATES="${CROP_CANDIDATES:-8}"
CROP_NEGATIVES="${CROP_NEGATIVES:-1}"
CROP_MIN_SCALE="${CROP_MIN_SCALE:-0.08}"
CROP_MAX_SCALE="${CROP_MAX_SCALE:-1.0}"
NEG_PROMPTS="${NEG_PROMPTS:-100}"
LAMBDA_POS="${LAMBDA_POS:-1.0}"
LAMBDA_NEG="${LAMBDA_NEG:-1.0}"
LAMBDA_DIV="${LAMBDA_DIV:-0.05}"
NEG_WARMUP_EPOCHS="${NEG_WARMUP_EPOCHS:-5}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-NR_RAPL_full_best.pth}"

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Missing Stage-1 RAPL checkpoint: ${INIT_CHECKPOINT}" >&2
  echo "Set INIT_CHECKPOINT, or run scripts/train_NR_RAPL_stage1.sh first." >&2
  exit 1
fi
read -r -a SCALES <<< "${LOCAL_SCALES}"
if [[ "${#SCALES[@]}" -ne "${LOCAL_PROMPTS}" ]]; then
  echo "LOCAL_PROMPTS (${LOCAL_PROMPTS}) must equal the number of LOCAL_SCALES (${#SCALES[@]})." >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m RAPL.train \
  --method_name NR_RAPL --nr_train_stage regional --checkpoint_path "${INIT_CHECKPOINT}" \
  --data_dir "${DATA_ROOT}" --save_dir "${OUTPUT_DIR}" --dataset_name "${DATASET_NAME}" \
  --num_shots "${NUM_SHOTS}" --clip_name "${CLIP_NAME}" --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --optimizer sgd --lr_init "${LR}" --weight_decay "${WEIGHT_DECAY}" \
  --n_global_prompts "${GLOBAL_PROMPTS}" --n_local_prompts "${LOCAL_PROMPTS}" --topk "${SCALES[@]}" \
  --use_local_features true --use_global_loss true --use_local_loss true \
  --max_epoch "${FULL_EPOCHS}" --eval_freq 1 --eval_ood true --use_fp16 true \
  --nr_crop_enabled true --nr_crop_num_candidates "${CROP_CANDIDATES}" \
  --nr_crop_num_positive 1 --nr_crop_num_negative "${CROP_NEGATIVES}" \
  --nr_crop_min_scale "${CROP_MIN_SCALE}" --nr_crop_max_scale "${CROP_MAX_SCALE}" \
  --nr_use_guided_crop true --nr_use_margin_selection true \
  --nr_neg_prompt_enabled true --nr_neg_prompt_num "${NEG_PROMPTS}" --nr_use_neg_prompt true \
  --nr_use_neg_loss true --nr_use_div_loss true \
  --nr_lambda_pos "${LAMBDA_POS}" --nr_lambda_neg "${LAMBDA_NEG}" --nr_lambda_div "${LAMBDA_DIV}" \
  --nr_neg_warmup_epochs "${NEG_WARMUP_EPOCHS}" --nr_keep_original_local_loss true \
  --nr_ood_score_type NR_GLMCM --nr_use_nr_glmcm true \
  --checkpoint_name "${CHECKPOINT_NAME}"
