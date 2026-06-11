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

run_mincheck yolo_seed11_min 0 python scripts/stage5_formal_runner.py --family yolo --seed 11 --mode all --arm all3 --epochs 1 --batch 16 --workers 2 --device 0 --val false --limit 256 --run-root runs/detect/runs_stage5_mincheck --out-root formal_results/stage5_mincheck --overwrite
run_mincheck yolo_seed22_min 1 python scripts/stage5_formal_runner.py --family yolo --seed 22 --mode all --arm all3 --epochs 1 --batch 16 --workers 2 --device 1 --val false --limit 256 --run-root runs/detect/runs_stage5_mincheck --out-root formal_results/stage5_mincheck --overwrite
run_mincheck yolo_seed33_min 2 python scripts/stage5_formal_runner.py --family yolo --seed 33 --mode all --arm all3 --epochs 1 --batch 16 --workers 2 --device 2 --val false --limit 256 --run-root runs/detect/runs_stage5_mincheck --out-root formal_results/stage5_mincheck --overwrite
run_mincheck rtdetr_seed11_min 3 python scripts/stage5_formal_runner.py --family rtdetr --seed 11 --mode all --arm all3 --epochs 1 --batch 4 --workers 2 --device 3 --val false --limit 256 --run-root runs/detect/runs_stage5_mincheck --out-root formal_results/stage5_mincheck --overwrite
wait_all

run_mincheck rtdetr_seed22_min 0 python scripts/stage5_formal_runner.py --family rtdetr --seed 22 --mode all --arm all3 --epochs 1 --batch 4 --workers 2 --device 0 --val false --limit 256 --run-root runs/detect/runs_stage5_mincheck --out-root formal_results/stage5_mincheck --overwrite
run_mincheck rtdetr_seed33_min 1 python scripts/stage5_formal_runner.py --family rtdetr --seed 33 --mode all --arm all3 --epochs 1 --batch 4 --workers 2 --device 1 --val false --limit 256 --run-root runs/detect/runs_stage5_mincheck --out-root formal_results/stage5_mincheck --overwrite
wait_all

python scripts/stage5_aggregate_repeats.py --result-root formal_results/stage5_mincheck --allow-empty --allow-partial --require-eqstep --min-seeds 1 --required-seeds 11,22,33
echo "[ok] Stage 5 v1 H20 minchecks completed."
