#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
python stage5_h20_server/02_rewrite_paths_for_server.py --root "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"
PIDS=()
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

run_mincheck() {
  local name="$1"
  local gpu="$2"
  shift 2
  local log="$LOG_DIR/${name}_$(date +%Y%m%d-%H%M%S).log"
  echo "[launch] $name gpu=$gpu log=$log"
  (
    set -euo pipefail
    cd "$PROJECT_ROOT"
    "$@"
  ) > >(tee "$log") 2>&1 &
  PIDS+=("$!")
}

wait_all() {
  local failed=0
  local pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  return "$failed"
}

run_mincheck pv2_yolo26n_baseline 0 python scripts/stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 901 --mode all --arm baseline --epochs 1 --batch 4 --workers 2 --device 0 --limit 128 --run-root runs/detect/runs_stage5_pv_v2_mincheck --out-root formal_results/stage5_pv_v2_mincheck --overwrite
run_mincheck pv2_yolo26n_sched 1 python scripts/stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 902 --mode all --arm hardneg_sched --epochs 1 --batch 4 --workers 2 --device 1 --limit 128 --run-root runs/detect/runs_stage5_pv_v2_mincheck --out-root formal_results/stage5_pv_v2_mincheck --overwrite
run_mincheck pv2_rtdetr_baseline 2 python scripts/stage5_pv_v2_runner.py --dataset dfire --family rtdetr --seed 903 --mode all --arm baseline --epochs 1 --batch 2 --workers 2 --device 2 --limit 128 --run-root runs/detect/runs_stage5_pv_v2_mincheck --out-root formal_results/stage5_pv_v2_mincheck --overwrite
run_mincheck pv2_rtdetr_sched 3 python scripts/stage5_pv_v2_runner.py --dataset dfire --family rtdetr --seed 904 --mode all --arm hardneg_sched --epochs 1 --batch 2 --workers 2 --device 3 --limit 128 --run-root runs/detect/runs_stage5_pv_v2_mincheck --out-root formal_results/stage5_pv_v2_mincheck --overwrite
wait_all

echo "[ok] Stage5-PV v2 D-Fire H20 minchecks completed."
