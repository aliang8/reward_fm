#!/usr/bin/env bash
set -euo pipefail
umask 0002

# Ensure HOME is writable (important when running as non-root)
export HOME=${HOME:-/workspace}
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

# 1) Create a venv that INHERITS base site-packages (Torch + FA2 stay from the image)
# VENV="/workspace/.venv"
# if [[ ! -d "$VENV" ]]; then
#   echo "[entrypoint] Creating venv (system-site-packages) at $VENV"
#   python -m venv --system-site-packages "$VENV"
# fi
# # shellcheck disable=SC1090
# source "$VENV/bin/activate"

# Create a constraints file that pins the currently-installed torch & flash-attn
python - <<'PY' >/tmp/constraints.txt
import importlib, sys
def ver(pkg):
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", None)
        if v: print(f"{pkg}=={v}")
    except Exception:
        pass
for p in ("torch","flash-attn"):
    ver(p)
PY
echo "[entrypoint] Using constraints:"; cat /tmp/constraints.txt || true


# 2) Ensure build backend bits for editable installs (no build isolation)
pip install -q --upgrade pip

# 3) Install your repo (editable) WITHOUT touching Torch from the base image
if [[ -f /workspace/pyproject.toml ]]; then
  echo "[entrypoint] Installing project (editable): pip install --no-build-isolation -e ."
  # --no-build-isolation avoids a temp build env that might pull a different torch
  pip install -e . -c /tmp/constraints.txt
else
  echo "[entrypoint] No pyproject.toml in /workspace; skipping install."
fi

# 4) Sanity print (optional)
python - <<'PY'
import sys, importlib.util as u, torch
print("[runtime] PY:", sys.executable)
print("[runtime] torch:", torch.__version__, "cu:", torch.version.cuda, "file:", torch.__file__)
for m in ("flash_attn","flash_attn_2_cuda"):
    spec = u.find_spec(m)
    print(f"[runtime] {m}:", bool(spec), getattr(spec, "origin", None))
PY

echo "[entrypoint] RFM_DATASET_PATH=$RFM_DATASET_PATH"
echo "[entrypoint] RFM_PROCESSED_DATASETS_PATH=$RFM_PROCESSED_DATASETS_PATH"

exec "$@"
