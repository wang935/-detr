#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

python stage5_h20_server/02_rewrite_paths_for_server.py --root "$PROJECT_ROOT"

SEEDS="${SEEDS:-22,33}"
GPU_IDS="${GPU_IDS:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-54}"
WORKERS="${WORKERS:-4}"
CONF="${CONF:-0.001}"
RUN_ROOT="${RUN_ROOT:-runs/detect/runs_stage5_formal}"
OUT_ROOT="${OUT_ROOT:-formal_results/stage5}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-external_data/BoWFireDataset/dataset/img}"
EXTERNAL_SOURCE="${EXTERNAL_SOURCE:-BoWFire non-fire fire-like negatives}"
EXTERNAL_TOKENS="${EXTERNAL_TOKENS:-neutral,normal,negative,none,other,no_fire,no-fire,nonfire,non_fire,non-fire,nofire,nor,not_fire}"
EXTERNAL_OUT_ROOT="${EXTERNAL_OUT_ROOT:-formal_results/stage5_external/bowfire}"
TARGETS="${TARGETS:-0.80,0.85,0.90,0.95}"
MIN_EXTERNAL_IMAGES="${MIN_EXTERNAL_IMAGES:-100}"
EXTERNAL_CONF="${EXTERNAL_CONF:-$CONF}"
EXTERNAL_IMG_SZ="${EXTERNAL_IMG_SZ:-640}"
EXTERNAL_MAX_DET="${EXTERNAL_MAX_DET:-100}"
OVERWRITE="${OVERWRITE:-0}"
PARALLEL="${PARALLEL:-0}"
bg_pids=()

safe_rm_dir() {
  local path="$1"
  local parent="$2"
  if [[ -z "$path" || -z "$parent" ]]; then
    echo "[error] unsafe path guard failed: path=$path parent=$parent"
    return 1
  fi
  case "$path" in
    "/"|""|"."|".." )
      echo "[error] unsafe rm target: $path"
      return 1
      ;;
  esac
  if [[ "$path" != "$parent"/* ]]; then
    echo "[error] refuse rm outside parent: $path (parent=$parent)"
    return 1
  fi
  rm -rf "$path"
}

cleanup_background_jobs() {
  for pid in "${bg_pids[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}

run_seed() {
  local seed="$1"
  local gpu="$2"
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "[error] invalid seed=$seed"
    return 2
  fi
  local run_dir="${RUN_ROOT}/rtdetr_seed${seed}"
  local out_dir="${OUT_ROOT}/rtdetr_seed${seed}"
  local log="$LOG_DIR/rt_detr_seed${seed}_$(date +%Y%m%d-%H%M%S).log"

  if [[ "${OVERWRITE}" == "1" ]]; then
    safe_rm_dir "$run_dir" "$RUN_ROOT"
    safe_rm_dir "$out_dir" "$OUT_ROOT"
  else
    if [[ -d "$run_dir" || -d "$out_dir" ]]; then
      echo "[error] output exists for seed=$seed. Set OVERWRITE=1 to clear and rerun."
      echo "  run_dir=$run_dir"
      echo "  out_dir=$out_dir"
      return 2
    fi
  fi

  echo "[run] seed=$seed gpu=$gpu log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/stage5_formal_runner.py \
    --family rtdetr \
    --seed "$seed" \
    --mode all \
    --arm all3 \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device 0 \
    --val false \
    --conf "$CONF" \
    --run-root "$RUN_ROOT" \
    --out-root "$OUT_ROOT" \
    >> "$log" 2>&1
}

trap 'cleanup_background_jobs' EXIT INT TERM

if [[ -z "${SEEDS// }" ]]; then
  echo "[error] SEEDS is empty."
  exit 2
fi
if [[ -z "${GPU_IDS// }" ]]; then
  echo "[error] GPU_IDS is empty."
  exit 2
fi

split_csv_list() {
  local text="$1"
  local -n out="$2"
  local -a raw
  raw=()
  IFS=',' read -r -a raw <<< "${text// /}"
  for item in "${raw[@]}"; do
    [[ -n "$item" ]] && out+=("$item")
  done
}

SEED_LIST=()
GPU_LIST=()
split_csv_list "$SEEDS" SEED_LIST
split_csv_list "$GPU_IDS" GPU_LIST

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "[error] no valid seeds parsed from SEEDS=$SEEDS"
  exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "[error] no valid GPU ids parsed from GPU_IDS=$GPU_IDS"
  exit 2
fi

echo "[info] seeds=${SEEDS} gpus=${GPU_IDS} epochs=$EPOCHS batch=$BATCH workers=$WORKERS conf=$CONF"

if [[ ! -d "$EXTERNAL_ROOT" ]]; then
  echo "[error] external root missing: $EXTERNAL_ROOT"
  exit 2
fi

if [[ "$PARALLEL" == "1" && ${#SEED_LIST[@]} -gt 1 && ${#GPU_LIST[@]} -ge ${#SEED_LIST[@]} ]]; then
  pids=()
  for i in "${!SEED_LIST[@]}"; do
    seed="${SEED_LIST[$i]}"
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
      echo "[skip] invalid seed=$seed"
      continue
    fi
    gpu="${GPU_LIST[$i]}"
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      echo "[skip] invalid gpu=$gpu"
      continue
    fi
    run_seed "$seed" "$gpu" &
    child_pid="$!"
    pids+=("$child_pid")
    bg_pids+=("$child_pid")
    echo "[launch] seed=$seed gpu=$gpu pid=$child_pid (log: $LOG_DIR/rt_detr_seed${seed}_*.log)"
  done
  if (( ${#bg_pids[@]} == 0 )); then
    echo "[error] no valid seed/gpu pairs were launched."
    exit 2
  fi

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
else
  if [[ "$PARALLEL" == "1" ]]; then
    echo "[warn] PARALLEL=1 but GPU count < seed count, fallback to sequential scheduling."
  fi
  for i in "${!SEED_LIST[@]}"; do
    seed="${SEED_LIST[$i]}"
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
      echo "[skip] invalid seed=$seed"
      continue
    fi
    gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
    run_seed "$seed" "$gpu"
  done
fi

echo "[ok] rtdetr all3 formal runs done for seeds: ${SEEDS}"

python scripts/stage5_external_negatives.py \
  --result-root "$OUT_ROOT" \
  --external-root "$EXTERNAL_ROOT" \
  --external-source "$EXTERNAL_SOURCE" \
  --neutral-tokens "$EXTERNAL_TOKENS" \
  --out-dir "$EXTERNAL_OUT_ROOT" \
  --targets "$TARGETS" \
  --min-external-images "$MIN_EXTERNAL_IMAGES" \
  --imgsz "$EXTERNAL_IMG_SZ" \
  --conf "$EXTERNAL_CONF" \
  --max-det "$EXTERNAL_MAX_DET"

echo "[ok] Stage 5 RT-DETR-L seed batch completed."
