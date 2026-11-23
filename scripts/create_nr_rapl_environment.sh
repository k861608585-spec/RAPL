#!/usr/bin/env bash
set -euo pipefail

# Create the environment; this does not activate it and does not download any
# dataset or CLIP checkpoint. Internet access is required for package install.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/environment/NR_RAPL-cu117.yml}"
ENV_NAME="${ENV_NAME:-NR-RAPL}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  echo "Install Miniconda/Mambaforge or set CONDA_BIN to its absolute path." >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

"${CONDA_BIN}" env create --name "${ENV_NAME}" --file "${ENV_FILE}"
echo "Environment created. Verify it with:"
echo "  ${CONDA_BIN} run -n ${ENV_NAME} python ${REPO_ROOT}/scripts/verify_NR_RAPL_environment.py"
