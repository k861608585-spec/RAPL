#!/usr/bin/env bash
set -euo pipefail

# One invocation reports both unchanged RAPL ID top-1 and NR-GLMCM OOD
# metrics. The checkpoint/model hyperparameters must match training.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the dataset root before running this script}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/NR_RAPL/full_imagenet_16shot}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/NR_RAPL_full_best.pth}"
DATASET_NAME="${DATASET_NAME:-imagenet}"
NUM_SHOTS="${NUM_SHOTS:-16}"
CLIP_NAME="${CLIP_NAME:-clip_vit_b16}"
SEED="${SEED:-1}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-256}"
GLOBAL_PROMPTS="${GLOBAL_PROMPTS:-4}"
LOCAL_PROMPTS="${LOCAL_PROMPTS:-4}"
LOCAL_SCALES="${LOCAL_SCALES:-10 20 30 40}"
NEG_PROMPTS="${NEG_PROMPTS:-100}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing full NR_RAPL checkpoint: ${CHECKPOINT}" >&2
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
  --method_name NR_RAPL --eval_only true --eval_ood true --checkpoint_path "${CHECKPOINT}" \
  --data_dir "${DATA_ROOT}" --save_dir "${OUTPUT_DIR}" --dataset_name "${DATASET_NAME}" \
  --num_shots "${NUM_SHOTS}" --clip_name "${CLIP_NAME}" --seed "${SEED}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --n_global_prompts "${GLOBAL_PROMPTS}" --n_local_prompts "${LOCAL_PROMPTS}" --topk "${SCALES[@]}" \
  --use_local_features true --use_fp16 true \
  --nr_neg_prompt_enabled true --nr_neg_prompt_num "${NEG_PROMPTS}" --nr_use_neg_prompt true \
  --nr_ood_score_type NR_GLMCM --nr_use_nr_glmcm true
