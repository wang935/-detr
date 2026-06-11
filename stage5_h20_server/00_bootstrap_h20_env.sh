#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-daq310}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PYTORCH_FALLBACK_INDEX_URL="${PYTORCH_FALLBACK_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-}"

echo "[info] project root: $PROJECT_ROOT"
echo "[info] conda env: $ENV_NAME"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader || true
else
  echo "[warn] nvidia-smi not found yet. The NVIDIA driver must be visible before training."
fi

ensure_miniconda() {
  if [[ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base)"
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
      MINICONDA_DIR="$base"
      return 0
    fi
  fi

  echo "[info] installing Miniconda to $MINICONDA_DIR"
  local installer="/tmp/miniconda-h20.sh"
  local url="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$installer" "$url"
  else
    echo "[error] need curl or wget to install Miniconda" >&2
    exit 2
  fi
  bash "$installer" -b -p "$MINICONDA_DIR"
}

ensure_miniconda
# shellcheck disable=SC1090
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

if ! conda run -n "$ENV_NAME" python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >/dev/null 2>&1; then
  echo "[info] creating conda env $ENV_NAME with Python 3.10"
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
  conda create -n "$ENV_NAME" python=3.10 pip -y
fi

PIP_ARGS=(--prefer-binary --timeout 120 --retries 5)
if [[ -n "$PYPI_INDEX_URL" ]]; then
  PIP_ARGS+=(-i "$PYPI_INDEX_URL")
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade pip "${PIP_ARGS[@]}"

torch_cuda_ok() {
  conda run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

if ! torch_cuda_ok; then
  echo "[info] installing CUDA PyTorch from $PYTORCH_INDEX_URL"
  conda run --no-capture-output -n "$ENV_NAME" python -m pip uninstall -y torch torchvision torchaudio || true
  if ! conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"; then
    echo "[warn] cu124 install failed; trying $PYTORCH_FALLBACK_INDEX_URL"
    conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url "$PYTORCH_FALLBACK_INDEX_URL"
  fi
fi

echo "[info] installing Stage 5 Python packages"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${PIP_ARGS[@]}" \
  numpy pyyaml matplotlib opencv-python-headless pillow tqdm requests scipy psutil py-cpuinfo pandas seaborn
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${PIP_ARGS[@]}" --no-deps polars ultralytics-thop ultralytics

echo "[info] verifying CUDA on all visible GPUs"
conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import sys
import torch
import ultralytics
import numpy

print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("ultralytics", ultralytics.__version__)
print("numpy", numpy.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
count = torch.cuda.device_count()
print("cuda_device_count", count)
assert count >= 4, f"expected at least 4 GPUs, got {count}"
for idx in range(count):
    name = torch.cuda.get_device_name(idx)
    x = torch.randn((512, 512), device=f"cuda:{idx}")
    y = (x @ x).sum()
    torch.cuda.synchronize(idx)
    print(f"gpu{idx}", name, "smoke_sum", round(float(y.detach().cpu()), 4))
PY

echo "[ok] H20 Stage 5 environment is ready: $ENV_NAME"
