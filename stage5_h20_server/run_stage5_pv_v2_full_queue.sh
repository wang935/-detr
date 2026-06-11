#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
python stage5_h20_server/02_rewrite_paths_for_server.py --root "$PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
mkdir -p "$LOG_DIR"

DATASETS=(${STAGE5_PV2_DATASETS:-dfire dfs})
FAMILIES=(${STAGE5_PV2_FAMILIES:-yolo26n rtdetr})
SEEDS=(${STAGE5_PV2_SEEDS:-11 22 33})
GPU_COUNT="${GPU_COUNT:-4}"
YOLO_BATCH="${YOLO_BATCH:-16}"
RTDETR_BATCH="${RTDETR_BATCH:-54}"
YOLO_WORKERS="${YOLO_WORKERS:-4}"
RTDETR_WORKERS="${RTDETR_WORKERS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

ensure_dfire_link() {
  if [[ -d "data/D-Fire/train/images" && -d "data/D-Fire/test/images" ]]; then
    return
  fi
  if [[ -e "data/D-Fire" && ! -L "data/D-Fire" ]]; then
    echo "[error] data/D-Fire exists but is not a usable D-Fire dataset or symlink" >&2
    return 2
  fi
  local candidates=(
    "${DFIRE_DATA_ROOT:-}"
    "${GEMINI_DATA_IN1:-/gemini/data-1}/D-Fire"
    "${GEMINI_DATA_IN1:-/gemini/data-1}/data/D-Fire"
    "${GEMINI_DATA_IN2:-/gemini/data-2}/D-Fire"
    "${GEMINI_DATA_IN2:-/gemini/data-2}/data/D-Fire"
    "${GEMINI_DATA_IN3:-/gemini/data-3}/D-Fire"
    "${GEMINI_DATA_IN3:-/gemini/data-3}/data/D-Fire"
  )
  local root
  for root in "${candidates[@]}"; do
    [[ -z "$root" ]] && continue
    if [[ -d "$root/train/images" && -d "$root/test/images" ]]; then
      mkdir -p data
      ln -sfn "$root" data/D-Fire
      echo "[prepare] linked data/D-Fire -> $root"
      return
    fi
  done
  echo "[error] missing D-Fire mount; expected data/D-Fire or /gemini/data-*/D-Fire" >&2
  return 2
}

check_dataset_ready() {
  local dataset="$1"
  if [[ "$dataset" == "dfire" ]]; then
    ensure_dfire_link
  fi
  if [[ "$dataset" == "dfs" ]]; then
    local required=(
      "data/stage5_pv_v2/dfs/dfs_posonly.yaml"
      "data/stage5_pv_v2/dfs/dfs_full.yaml"
      "data/stage5_pv_v2/dfs/dfs_hardneg_sched.yaml"
      "data/stage5_pv_v2/dfs/eval_labels.csv"
      "data/stage5_pv_v2/dfs/eval_calib_labels.csv"
      "data/stage5_pv_v2/dfs/eval_test_labels.csv"
    )
    local path
    for path in "${required[@]}"; do
      if [[ ! -f "$path" ]]; then
        echo "[error] missing DFS prepared file: $PROJECT_ROOT/$path" >&2
        echo "[hint] run: python scripts/stage5_pv_v2_prepare_dfs.py --source external_data/DFS-FIRE-SMOKE-Dataset" >&2
        return 2
      fi
    done
  fi
}

prepare_derived_dataset() {
  local dataset="$1"
  echo "[prepare] derived Stage5-PV v2 lists dataset=$dataset"
  python scripts/stage5_pv_v2_runner.py \
    --mode smoke \
    --dataset "$dataset" \
    --family yolo26n \
    --seed 11 \
    --arm all4 \
    --device cpu
}

for dataset in "${DATASETS[@]}"; do
  check_dataset_ready "$dataset"
done

for dataset in "${DATASETS[@]}"; do
  prepare_derived_dataset "$dataset"
done

declare -a pids=()
declare -a labels=()

wait_wave() {
  local status=0
  local idx
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[error] job failed: ${labels[$idx]}" >&2
      status=1
    fi
  done
  pids=()
  labels=()
  return "$status"
}

launch_job() {
  local dataset="$1"
  local family="$2"
  local seed="$3"
  local gpu="$4"
  local batch="$YOLO_BATCH"
  local workers="$YOLO_WORKERS"
  if [[ "$family" == "rtdetr" ]]; then
    batch="$RTDETR_BATCH"
    workers="$RTDETR_WORKERS"
  fi
  local name="pv2_${dataset}_${family}_seed${seed}"
  local log="$LOG_DIR/${name}_$(date +%Y%m%d-%H%M%S).log"
  echo "[launch] name=$name gpu=$gpu batch=$batch workers=$workers log=$log"
  (
    set -euo pipefail
    cd "$PROJECT_ROOT"
    python scripts/stage5_pv_v2_runner.py \
      --dataset "$dataset" \
      --family "$family" \
      --seed "$seed" \
      --mode all \
      --arm all4 \
      --epochs 300 \
      --batch "$batch" \
      --workers "$workers" \
      --device "$gpu" \
      --val false
  ) > >(tee "$log") 2>&1 &
  pids+=("$!")
  labels+=("$name")
}

gpu=0
for dataset in "${DATASETS[@]}"; do
  for family in "${FAMILIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      launch_job "$dataset" "$family" "$seed" "$gpu"
      gpu=$((gpu + 1))
      if (( gpu >= GPU_COUNT )); then
        wait_wave
        gpu=0
      fi
    done
  done
done

if (( ${#pids[@]} > 0 )); then
  wait_wave
fi

echo "[ok] Stage5-PV v2 full H20 queue completed."
