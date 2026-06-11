#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${GEMINI_DATA_OUT:-/gemini/output}"
WORK_ROOT="${GEMINI_WORK_ROOT:-$OUTPUT_ROOT/detr_Q3_work}"

bash "$SCRIPT_DIR/gemini_prepare_workdir.sh"
cd "$WORK_ROOT"

# Gemini check is intentionally scoped to the Stage 5 v1 formal run. The
# broader SSH check probes the PV-v2 yolo26n/rtdetr assets.
# shellcheck disable=SC1091
source stage5_h20_server/_env.sh
activate_daq

echo "[info] GPU inventory"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader

echo "[info] compiling Stage 5 v1 scripts"
python -m py_compile \
  scripts/stage5_formal_runner.py \
  scripts/stage5_aggregate_repeats.py \
  scripts/stage5_external_negatives.py

echo "[info] smoke-checking Stage 5 v1 on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/stage5_formal_runner.py --mode smoke --family yolo --seed 11 --device 0
CUDA_VISIBLE_DEVICES=0 python scripts/stage5_formal_runner.py --mode smoke --family rtdetr --seed 11 --device 0

echo "[ok] Gemini Stage 5 v1 assets and GPU smoke checks passed."
