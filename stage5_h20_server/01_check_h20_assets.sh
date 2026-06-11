#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"

python stage5_h20_server/02_rewrite_paths_for_server.py --root "$PROJECT_ROOT"

echo "[info] GPU inventory"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader

echo "[info] compiling Stage 5 scripts"
python -m py_compile \
  scripts/stage5_formal_runner.py \
  scripts/stage5_aggregate_repeats.py \
  scripts/stage5_external_negatives.py \
  scripts/stage5_pv_v2_runner.py \
  scripts/stage5_pv_v2_aggregate.py \
  scripts/stage5_pv_v2_prepare_dfs.py \
  scripts/stage5_pv_v2_map_metrics.py

echo "[info] smoke-checking Stage 5 v1 on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/stage5_formal_runner.py --mode smoke --family yolo --seed 11 --device 0
CUDA_VISIBLE_DEVICES=0 python scripts/stage5_formal_runner.py --mode smoke --family rtdetr --seed 11 --device 0

echo "[info] smoke-checking Stage5-PV v2 D-Fire on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/stage5_pv_v2_runner.py --mode smoke --dataset dfire --family yolo26n --seed 11 --arm all4 --device 0
CUDA_VISIBLE_DEVICES=0 python scripts/stage5_pv_v2_runner.py --mode smoke --dataset dfire --family rtdetr --seed 11 --arm hardneg_sched --device 0

echo "[ok] H20 assets and GPU smoke checks passed."
