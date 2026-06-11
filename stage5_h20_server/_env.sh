#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export ENV_NAME="${ENV_NAME:-daq310}"
export LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/h20}"

find_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH:-}" ]]; then
    printf '%s\n' "$CONDA_SH"
    return 0
  fi
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/usr/local/conda/etc/profile.d/conda.sh"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

activate_daq() {
  local conda_sh
  if [[ "${SKIP_CONDA:-0}" == "1" ]]; then
    return 0
  fi
  conda_sh="$(find_conda_sh)" || {
    if command -v python >/dev/null 2>&1 && python -c "import torch, ultralytics" >/dev/null 2>&1; then
      echo "[info] conda not found; using system Python with torch/ultralytics"
      return 0
    fi
    echo "[error] conda.sh not found. Run stage5_h20_server/00_bootstrap_h20_env.sh first, or use the H20 Docker image." >&2
    return 2
  }
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$ENV_NAME"
}

stage5_log_path() {
  local name="$1"
  mkdir -p "$LOG_DIR"
  printf '%s/%s_%s.log\n' "$LOG_DIR" "$name" "$(date +%Y%m%d-%H%M%S)"
}
