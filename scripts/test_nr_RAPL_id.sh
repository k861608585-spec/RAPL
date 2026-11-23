#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the dataset root before running this script}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/NR_RAPL/imagenet_16shot}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/stage3_NR_RAPL_best.pth}"
if [[ ! -f "${CHECKPOINT}" ]]; then echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; fi
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m RAPL.train \
  --method_name NR_RAPL --eval_only true --checkpoint_path "${CHECKPOINT}" \
  --data_dir "${DATA_ROOT}" --save_dir "${OUTPUT_DIR}" --dataset_name "${DATASET_NAME:-imagenet}" \
  --num_shots "${NUM_SHOTS:-16}" --clip_name "${CLIP_NAME:-clip_vit_b16}" --n_global_prompts 4 --n_local_prompts 4 --topk 10 20 30 40 \
  --use_local_features true --nr_neg_prompt_enabled true --nr_use_neg_prompt true
