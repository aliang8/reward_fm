#!/usr/bin/env bash
set -euo pipefail

# Fast dataset setup via either Hugging Face CLI downloads (default)
# or git-lfs clones from the Hub.
# - HF CLI path requires: `pip install huggingface_hub` (provides `hf` CLI)
# - Git path requires: `git` and `git-lfs`

# Parse args
# Usage examples:
#   ./setup.sh                                # HF CLI (default), base dir from RFM_DATASET_PATH or ./rfm_dataset
#   ./setup.sh --git                          # git clone method
#   ./setup.sh --method git --dir /data/rfm   # explicit method and base dir
#   ./setup.sh /data/rfm                      # positional base dir

METHOD=${RFM_DOWNLOAD_METHOD:-hf}  # hf | git
BASE_DIR_DEFAULT=${RFM_DATASET_PATH:-./rfm_dataset}
BASE_DIR="$BASE_DIR_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --git)
      METHOD="git"
      shift
      ;;
    --hf)
      METHOD="hf"
      shift
      ;;
    --method=*)
      METHOD="${1#*=}"
      shift
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --dir|--base-dir|-d)
      BASE_DIR="$2"
      shift 2
      ;;
    *)
      BASE_DIR="$1"
      shift
      ;;
  esac
done

case "$METHOD" in
  hf)
    if ! command -v hf >/dev/null 2>&1; then
      echo "Error: 'hf' CLI not found. Install with: uv pip install huggingface_hub (or ensure your venv is activated)" >&2
      exit 1
    fi
    ;;
  git)
    if ! command -v git >/dev/null 2>&1; then
      echo "Error: git not found. Please install git." >&2
      exit 1
    fi
    if ! git lfs version >/dev/null 2>&1; then
      echo "Warning: git-lfs not found. You may end up with pointer files. Install git-lfs for full downloads." >&2
    fi
    ;;
  *)
    echo "Error: Unknown METHOD='${METHOD}'. Use 'hf' or 'git'." >&2
    exit 1
    ;;
esac

mkdir -p "${BASE_DIR}"

download_dataset() {
  local repo_id="$1"  # e.g., abraranwar/libero_rfm
  local name
  name="${repo_id##*/}"  # take last path segment as folder name
  local target_dir="${BASE_DIR}/${name}"

  echo "Downloading ${repo_id} -> ${target_dir} via ${METHOD}"
  if [[ "$METHOD" == "hf" ]]; then
    hf download "${repo_id}" \
      --repo-type dataset \
      --local-dir "${target_dir}"
  else
    local url="https://huggingface.co/datasets/${repo_id}.git"
    if [[ -d "${target_dir}/.git" ]]; then
      echo "Updating existing clone at ${target_dir}"
      git -C "${target_dir}" remote set-url origin "${url}" || true
      git -C "${target_dir}" fetch --all --tags
      git -C "${target_dir}" pull --ff-only
    else
      git clone "${url}" "${target_dir}"
    fi
    if git lfs version >/dev/null 2>&1; then
      git -C "${target_dir}" lfs install --local >/dev/null 2>&1 || true
      git -C "${target_dir}" lfs pull || true
    fi
  fi
}

download_dataset abraranwar/agibotworld_rfm
download_dataset abraranwar/libero_rfm
download_dataset abraranwar/egodex_rfm
download_dataset ykorkmaz/libero_failure_rfm

echo "Done. Set RFM_DATASET_PATH=${BASE_DIR} for training/eval."