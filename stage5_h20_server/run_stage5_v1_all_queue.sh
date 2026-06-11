#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"
python stage5_h20_server/02_rewrite_paths_for_server.py --root "$PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

launch_worker() {
  local worker="$1"
  local gpu="$2"
  local session="stage5v1_${worker}"
  local cmd="bash '$PROJECT_ROOT/stage5_h20_server/_stage5_v1_worker.sh' '$worker' '$gpu'"
  if command -v screen >/dev/null 2>&1; then
    screen -dmS "$session" bash -lc "$cmd"
    echo "[launch] screen=$session gpu=$gpu worker=$worker"
  else
    local log="$LOG_DIR/${session}_launcher_$(date +%Y%m%d-%H%M%S).log"
    nohup bash -lc "$cmd" > "$log" 2>&1 &
    echo "[launch] pid=$! gpu=$gpu worker=$worker log=$log"
  fi
}

launch_worker gpu0_rtdetr_seed11 0
launch_worker gpu1_rtdetr_seed22 1
launch_worker gpu2_rtdetr_seed33 2
launch_worker gpu3_yolo_all_seeds 3

if command -v screen >/dev/null 2>&1; then
  screen -ls || true
fi

echo "[ok] Stage 5 v1 H20 formal queue launched."
