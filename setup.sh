#!/usr/bin/env bash
set -euo pipefail

# Fast dataset setup via Hugging Face CLI downloads (avoids slow git-lfs clones)
# Requires: `pip install huggingface_hub` to provide `huggingface-cli` command.

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli not found. Install with: pip install huggingface_hub" >&2
  exit 1
fi

# Base directory where datasets will be downloaded
# 1st arg overrides; otherwise use RFM_DATASET_PATH or ./rfm_dataset
BASE_DIR=${1:-${RFM_DATASET_PATH:-./rfm_dataset}}

mkdir -p "${BASE_DIR}"

download_dataset() {
  local repo_id="$1"  # e.g., abraranwar/libero_rfm
  local name
  name="${repo_id##*/}"  # take last path segment as folder name
  local target_dir="${BASE_DIR}/${name}"

  echo "Downloading ${repo_id} -> ${target_dir}"
  # --local-dir-use-symlinks False ensures actual files are materialized
  huggingface-cli download "${repo_id}" \
    --repo-type dataset \
    --local-dir "${target_dir}" \
    --local-dir-use-symlinks True
}

# download_dataset abraranwar/agibotworld_rfm
# download_dataset abraranwar/libero_rfm
# download_dataset abraranwar/egodex_rfm
download_dataset ykorkmaz/libero_failure_rfm

echo "Done. Set RFM_DATASET_PATH=${BASE_DIR} for training/eval."