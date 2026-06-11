#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"
python stage5_h20_server/02_rewrite_paths_for_server.py --root "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

SESSION="stage5pv2_yolo26n_seed22_sched"
CMD="cd '$PROJECT_ROOT' && source '$PROJECT_ROOT/stage5_h20_server/_env.sh' && activate_daq && python scripts/stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 22 --mode all --arm hardneg_sched --epochs 300 --batch 16 --workers 4 --device '${GPU_ID:-0}' --val false"

if command -v screen >/dev/null 2>&1; then
  screen -dmS "$SESSION" bash -lc "$CMD"
  echo "[launch] screen=$SESSION gpu=${GPU_ID:-0}"
  screen -ls || true
else
  LOG="$LOG_DIR/${SESSION}_$(date +%Y%m%d-%H%M%S).log"
  nohup bash -lc "$CMD" > "$LOG" 2>&1 &
  echo "[launch] pid=$! gpu=${GPU_ID:-0} log=$LOG"
fi

echo "[ok] Stage5-PV v2 yolo26n seed22 hardneg_sched launched."
