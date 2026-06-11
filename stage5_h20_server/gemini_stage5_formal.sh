#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${GEMINI_DATA_OUT:-/gemini/output}"
WORK_ROOT="${GEMINI_WORK_ROOT:-$OUTPUT_ROOT/detr_Q3_work}"

bash "$SCRIPT_DIR/gemini_prepare_workdir.sh"
cd "$WORK_ROOT"
export STAGE5_SKIP_PATH_REWRITE=1

# Offline training jobs should keep the main process alive until all GPU
# workers finish. The generic queue script detaches into screen/nohup, which is
# useful on SSH servers but can make managed platforms mark the job as done.
declare -a pids=()

cleanup() {
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM

launch_worker() {
  local worker="$1"
  local gpu="$2"
  echo "[launch] worker=$worker gpu=$gpu"
  bash stage5_h20_server/_stage5_v1_worker.sh "$worker" "$gpu" &
  pids+=("$!")
}

launch_worker gpu0_rtdetr_seed11 0
launch_worker gpu1_rtdetr_seed22 1
launch_worker gpu2_rtdetr_seed33 2
launch_worker gpu3_yolo_all_seeds 3

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "[error] One or more Stage 5 formal workers failed." >&2
  exit "$status"
fi

echo "[ok] Stage 5 v1 Gemini formal training completed."
