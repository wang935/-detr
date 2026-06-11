#!/usr/bin/env bash
set -euo pipefail

WORKER="${1:?worker name required}"
GPU="${2:?gpu id required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${WORKER}_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo "[start] worker=$WORKER gpu=$GPU root=$PROJECT_ROOT"
if [[ "${STAGE5_SKIP_PATH_REWRITE:-0}" != "1" ]]; then
  python stage5_h20_server/02_rewrite_paths_for_server.py --root "$PROJECT_ROOT"
fi

run_formal() {
  local family="$1"
  local seed="$2"
  local batch="$3"
  echo "[run] family=$family seed=$seed batch=$batch physical_gpu=$GPU"
  python scripts/stage5_formal_runner.py \
    --family "$family" \
    --seed "$seed" \
    --mode all \
    --arm all3 \
    --epochs 300 \
    --batch "$batch" \
    --workers 4 \
    --device "$GPU" \
    --val false
}

case "$WORKER" in
  gpu0_rtdetr_seed11)
    run_formal rtdetr 11 54
    ;;
  gpu1_rtdetr_seed22)
    run_formal rtdetr 22 54
    ;;
  gpu2_rtdetr_seed33)
    run_formal rtdetr 33 54
    ;;
  gpu3_yolo_all_seeds)
    run_formal yolo 11 16
    run_formal yolo 22 16
    run_formal yolo 33 16
    ;;
  *)
    echo "[error] unknown worker: $WORKER" >&2
    exit 2
    ;;
esac

echo "[ok] worker completed: $WORKER"
