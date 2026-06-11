#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SESSION="${SESSION:-stage5pv2_rtdetr_dfire_12}"
QUEUE_ID="${QUEUE_ID:-rtdetr_dfire_12_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/detect/runs_stage5_pv_v2}"
OUT_ROOT="${OUT_ROOT:-formal_results/stage5_pv_v2}"
TRAIN_TMP_OUT_ROOT="${TRAIN_TMP_OUT_ROOT:-formal_results/stage5_pv_v2_trainmeta_tmp}"
EPOCHS="${EPOCHS:-300}"
RTDETR_BATCH="${RTDETR_BATCH:-54}"
RTDETR_WORKERS="${RTDETR_WORKERS:-8}"
EXPORT_DEVICE="${EXPORT_DEVICE:-0}"
STAGE5_PV2_SEEDS="${STAGE5_PV2_SEEDS:-11 22 33}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
PREPARE_FORCE="${PREPARE_FORCE:-0}"
AUTO_FILL_GPU="${AUTO_FILL_GPU:-1}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"

if [[ "${1:-}" != "--driver" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[error] tmux not found on this host" >&2
    exit 2
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[error] tmux session already exists: $SESSION" >&2
    echo "[hint] attach: tmux attach -t $SESSION" >&2
    exit 2
  fi
  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$SESSION" \
    "cd '$PROJECT_ROOT' && env PATH=/opt/conda/bin:\$PATH SKIP_CONDA='${SKIP_CONDA:-1}' SESSION='$SESSION' QUEUE_ID='$QUEUE_ID' RUN_ROOT='$RUN_ROOT' OUT_ROOT='$OUT_ROOT' TRAIN_TMP_OUT_ROOT='$TRAIN_TMP_OUT_ROOT' EPOCHS='$EPOCHS' RTDETR_BATCH='$RTDETR_BATCH' RTDETR_WORKERS='$RTDETR_WORKERS' EXPORT_DEVICE='$EXPORT_DEVICE' STAGE5_PV2_SEEDS='$STAGE5_PV2_SEEDS' GPU_IDS='$GPU_IDS' PREPARE_FORCE='$PREPARE_FORCE' AUTO_FILL_GPU='$AUTO_FILL_GPU' ALLOW_RESUME='$ALLOW_RESUME' bash '$SCRIPT_DIR/$(basename "$0")' --driver"
  echo "[launch] tmux=$SESSION queue_id=$QUEUE_ID"
  echo "[attach] tmux attach -t $SESSION"
  echo "[logs] $LOG_DIR/${QUEUE_ID}_driver.log"
  tmux ls || true
  exit 0
fi

export PATH="/opt/conda/bin:$PATH"
export SKIP_CONDA="${SKIP_CONDA:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

activate_daq
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/${QUEUE_ID}_driver.log"
PID_FILE="$LOG_DIR/${QUEUE_ID}.pids"
: > "$PID_FILE"
exec > >(tee -a "$DRIVER_LOG") 2>&1

IFS=' ' read -r -a SEEDS <<< "$STAGE5_PV2_SEEDS"
IFS=' ' read -r -a GPUS <<< "$GPU_IDS"
ARMS=(baseline baseline_eqstep hardneg hardneg_sched)
ARM_COUNT="${#ARMS[@]}"

if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "[error] expected exactly 4 GPU ids, got: $GPU_IDS" >&2
  exit 2
fi

if [[ "${#ARMS[@]}" -ne 4 ]]; then
  echo "[error] expected exactly 4 arms, got: ${ARMS[*]}" >&2
  exit 2
fi

echo "[start] queue_id=$QUEUE_ID session=$SESSION root=$PROJECT_ROOT"
echo "[config] dataset=dfire family=rtdetr seeds=${SEEDS[*]} arms=${ARMS[*]} epochs=$EPOCHS batch=$RTDETR_BATCH workers=$RTDETR_WORKERS"
echo "[config] run_root=$RUN_ROOT out_root=$OUT_ROOT train_tmp_out_root=$TRAIN_TMP_OUT_ROOT"
echo "[config] mode=AUTO_FILL_GPU=$AUTO_FILL_GPU ALLOW_RESUME=$ALLOW_RESUME"
echo "[config] OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
python stage5_h20_server/02_rewrite_paths_for_server.py --root "$PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

declare -A SEED_SUCCESS_COUNT
declare -A SEED_EXPORT_DONE
declare -a TASK_SEEDS=()
declare -a TASK_ARMS=()
declare -a PIDS=()
declare -a LABELS=()
declare -a RUN_PIDS=()
declare -A RUN_SEED
declare -A RUN_ARM
declare -A RUN_GPU
declare -A RUN_LABEL
declare -A RUN_LOG

TASK_LAUNCH_PID=""
TASK_LAUNCH_LABEL=""
TASK_LAUNCH_LOG=""
DONE_PID=""
DONE_RC=0
DONE_GPU=""

is_task_completed() {
  local seed="$1"
  local arm="$2"
  local train_dir="$RUN_ROOT/dfire_rtdetr_seed${seed}/${arm}"
  local log_pattern="$LOG_DIR/*_seed${seed}_${arm}_gpu*.log"
  local -a metas=()
  local meta
  local has_meta=0

  if [[ ! -f "$train_dir/weights/last.pt" ]] || [[ ! -s "$train_dir/weights/last.pt" ]]; then
    return 1
  fi
  if [[ ! -f "$train_dir/results.csv" ]]; then
    return 1
  fi
  if ! compgen -G "$log_pattern" >/dev/null; then
    return 1
  fi

  shopt -s nullglob
  metas=(
    "$TRAIN_TMP_OUT_ROOT/$QUEUE_ID"/*/dfire_rtdetr_seed"${seed}"/run_meta.json
    "$TRAIN_TMP_OUT_ROOT"/*/dfire_rtdetr_seed"${seed}"/run_meta.json
  )
  shopt -u nullglob
  for meta in "${metas[@]}"; do
    if [[ ! -f "$meta" ]]; then
      continue
    fi
    if grep -q "\"seed\": ${seed}" "$meta" \
      && grep -q "\"family\": \"rtdetr\"" "$meta" \
      && grep -q "\"${arm}\"" "$meta"; then
      has_meta=1
      break
    fi
  done

  return $((1 - has_meta))
}

seed_output_complete() {
  local seed="$1"
  local out_dir="$OUT_ROOT/dfire_rtdetr_seed${seed}"
  local arm
  if [[ ! -d "$out_dir" ]]; then
    return 1
  fi
  if [[ ! -f "$out_dir/gonogo.json" ]]; then
    return 1
  fi
  for arm in "${ARMS[@]}"; do
    if [[ ! -f "$out_dir/${arm}.csv" ]]; then
      return 1
    fi
  done
  return 0
}

check_clean_outputs() {
  local seed arm train_dir out_dir
  local done_count=0
  for seed in "${SEEDS[@]}"; do
    SEED_SUCCESS_COUNT["$seed"]=0
    SEED_EXPORT_DONE["$seed"]=0
    out_dir="$OUT_ROOT/dfire_rtdetr_seed${seed}"

    if [[ "$ALLOW_RESUME" == "0" ]]; then
      if [[ -e "$out_dir" ]]; then
        echo "[error] output dir already exists: $out_dir" >&2
        echo "[hint] move/remove only this RT-DETR D-Fire seed dir before relaunching if it is an interrupted leftover" >&2
        exit 2
      fi
      for arm in "${ARMS[@]}"; do
        train_dir="$RUN_ROOT/dfire_rtdetr_seed${seed}/${arm}"
        if [[ -e "$train_dir" ]]; then
          echo "[error] train dir already exists: $train_dir" >&2
          echo "[hint] move/remove only this arm dir before relaunching if it is an interrupted leftover" >&2
          exit 2
        fi
      done
      continue
    fi

    done_count=0
    for arm in "${ARMS[@]}"; do
      if is_task_completed "$seed" "$arm"; then
        done_count=$((done_count + 1))
        continue
      fi
      train_dir="$RUN_ROOT/dfire_rtdetr_seed${seed}/${arm}"
      if [[ -e "$train_dir" ]]; then
        echo "[error] partial/unfinished train dir exists with resume enabled: $train_dir" >&2
        echo "[hint] remove this arm dir before relaunching this seed in ALLOW_RESUME mode" >&2
        exit 2
      fi
    done

    if (( done_count < ARM_COUNT )) && [[ -e "$out_dir" ]]; then
      echo "[error] output dir already exists but seed is incomplete: $out_dir" >&2
      echo "[hint] remove this seed dir or use a clean seed dir before relaunching" >&2
      exit 2
    fi

    SEED_SUCCESS_COUNT["$seed"]=$done_count
  done
}

check_clean_outputs

prepare_files=(
  data/stage5_pv_v2/dfire/dfire_posonly_equalstep.yaml
  data/stage5_pv_v2/dfire/dfire_hardneg_sched.yaml
  data/stage5_pv_v2/dfire/train_posonly_equalstep.txt
  data/stage5_pv_v2/dfire/train_hardneg_sched.txt
)
prepare_ready=1
for path in "${prepare_files[@]}"; do
  if [[ ! -s "$path" ]]; then
    prepare_ready=0
  fi
done

if [[ "$PREPARE_FORCE" != "1" && "$prepare_ready" == "1" ]]; then
  echo "[prepare] derived D-Fire lists already exist; skipping smoke"
else
  echo "[prepare] Stage5-PV v2 D-Fire derived lists"
  python scripts/stage5_pv_v2_runner.py \
    --mode smoke \
    --dataset dfire \
    --family yolo26n \
    --seed "${SEEDS[0]}" \
    --arm all4 \
    --device cpu
fi

run_export_eval() {
  local seed="$1"
  local log="$LOG_DIR/${QUEUE_ID}_seed${seed}_export_eval.log"
  echo "[export/eval] seed=$seed log=$log"
  {
    env PATH="/opt/conda/bin:$PATH" SKIP_CONDA=1 OMP_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$MKL_NUM_THREADS" NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
      python scripts/stage5_pv_v2_runner.py \
        --mode export \
        --dataset dfire \
        --family rtdetr \
        --seed "$seed" \
        --arm all4 \
        --batch "$RTDETR_BATCH" \
        --workers "$RTDETR_WORKERS" \
        --device "$EXPORT_DEVICE" \
        --run-root "$RUN_ROOT" \
        --out-root "$OUT_ROOT"
    python scripts/stage5_pv_v2_runner.py \
      --mode eval \
      --dataset dfire \
      --family rtdetr \
      --seed "$seed" \
      --arm all4 \
      --run-root "$RUN_ROOT" \
      --out-root "$OUT_ROOT"
  } > "$log" 2>&1
  tail -n 20 "$log"
}

start_train_process() {
  local seed="$1"
  local arm="$2"
  local gpu="$3"
  local label="seed${seed}_${arm}_gpu${gpu}"
  local log="$LOG_DIR/${QUEUE_ID}_${label}.log"
  local tmp_out="$TRAIN_TMP_OUT_ROOT/${QUEUE_ID}/${label}"
  echo "[launch] label=$label gpu=$gpu log=$log"
  env PATH="/opt/conda/bin:$PATH" SKIP_CONDA=1 OMP_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$MKL_NUM_THREADS" NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
    python scripts/stage5_pv_v2_runner.py \
      --mode train \
      --dataset dfire \
      --family rtdetr \
      --seed "$seed" \
      --arm "$arm" \
      --epochs "$EPOCHS" \
      --batch "$RTDETR_BATCH" \
      --workers "$RTDETR_WORKERS" \
      --device "$gpu" \
      --val false \
      --run-root "$RUN_ROOT" \
      --out-root "$tmp_out" \
      > "$log" 2>&1 &
  TASK_LAUNCH_PID="$!"
  TASK_LAUNCH_LABEL="$label"
  TASK_LAUNCH_LOG="$log"
}

launch_train_wave() {
  local seed="$1"
  local arm="$2"
  local gpu="$3"
  start_train_process "$seed" "$arm" "$gpu"
  PIDS+=("$TASK_LAUNCH_PID")
  LABELS+=("$TASK_LAUNCH_LABEL")
  echo "$TASK_LAUNCH_PID $TASK_LAUNCH_LABEL $TASK_LAUNCH_LOG" >> "$PID_FILE"
}

launch_train_fill() {
  local seed="$1"
  local arm="$2"
  local gpu="$3"
  local pid
  start_train_process "$seed" "$arm" "$gpu"
  pid="$TASK_LAUNCH_PID"
  RUN_PIDS+=("$pid")
  RUN_SEED["$pid"]="$seed"
  RUN_ARM["$pid"]="$arm"
  RUN_GPU["$pid"]="$gpu"
  RUN_LABEL["$pid"]="$TASK_LAUNCH_LABEL"
  RUN_LOG["$pid"]="$TASK_LAUNCH_LOG"
  echo "$pid $TASK_LAUNCH_LABEL $TASK_LAUNCH_LOG" >> "$PID_FILE"
}

wait_wave() {
  local status=0
  local idx
  for idx in "${!PIDS[@]}"; do
    local rc=0
    if wait "${PIDS[$idx]}"; then
      rc=0
    else
      rc=$?
      status=1
    fi
    if (( rc == 0 )); then
      echo "[done] label=${LABELS[$idx]} rc=$rc"
    else
      echo "[error] label=${LABELS[$idx]} rc=$rc log=$LOG_DIR/${QUEUE_ID}_${LABELS[$idx]}.log" >&2
    fi
  done
  PIDS=()
  LABELS=()
  return "$status"
}

run_export_if_ready() {
  local seed="$1"
  if [[ "${SEED_EXPORT_DONE[$seed]}" == "1" ]]; then
    return 0
  fi
  if (( SEED_SUCCESS_COUNT["$seed"] < ARM_COUNT )); then
    return 0
  fi
  if seed_output_complete "$seed"; then
    echo "[skip] seed=$seed reason=existing_export_complete"
    SEED_EXPORT_DONE["$seed"]=1
    return 0
  fi
  if ! run_export_eval "$seed"; then
    echo "[error] seed=$seed export/eval failed" >&2
    return 1
  fi
  SEED_EXPORT_DONE["$seed"]=1
  return 0
}

remove_running() {
  local pid="$1"
  local p
  local -a remain=()
  for p in "${RUN_PIDS[@]}"; do
    [[ "$p" != "$pid" ]] && remain+=("$p")
  done
  RUN_PIDS=("${remain[@]}")
  unset "RUN_SEED[$pid]"
  unset "RUN_ARM[$pid]"
  unset "RUN_GPU[$pid]"
  unset "RUN_LABEL[$pid]"
  unset "RUN_LOG[$pid]"
}

wait_one_done() {
  local pid
  local rc
  while true; do
    for pid in "${RUN_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      if wait "$pid"; then
        rc=0
      else
        rc=$?
      fi
      DONE_PID="$pid"
      DONE_RC="$rc"
      return 0
    done
    sleep 3
  done
}

build_task_queue() {
  local seed arm label
  TASK_SEEDS=()
  TASK_ARMS=()
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      if is_task_completed "$seed" "$arm"; then
        label="seed${seed}_${arm}"
        echo "[skip] label=$label reason=existing_done"
        continue
      fi
      TASK_SEEDS+=("$seed")
      TASK_ARMS+=("$arm")
    done
  done
}

declare -i FAIL_CODE=0

run_seed_wave_static() {
  local seed idx arm
  local wave_status=0
  for seed in "${SEEDS[@]}"; do
    echo "[wave] seed=$seed training ${ARM_COUNT} arms"
    PIDS=()
    LABELS=()
    for idx in "${!ARMS[@]}"; do
      arm="${ARMS[$idx]}"
      if is_task_completed "$seed" "$arm"; then
        echo "[skip] label=seed${seed}_${arm}_gpu${GPUS[$idx]} reason=existing_done"
        continue
      fi
      launch_train_wave "$seed" "$arm" "${GPUS[$idx]}"
    done
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    if (( ${#PIDS[@]} > 0 )); then
      if ! wait_wave; then
        wave_status=1
      fi
    else
      echo "[skip] seed=$seed reason=all_arms_completed"
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    if ! run_export_if_ready "$seed"; then
      wave_status=1
    fi
  done
  return "$wave_status"
}

run_seed_wave_fill() {
  local task_idx=0
  local total_tasks="${#TASK_SEEDS[@]}"
  local status=0
  local idx
  local seed arm gpu done_pid done_rc done_seed done_label done_log
  local -i i=0
  if (( total_tasks == 0 )); then
    for seed in "${SEEDS[@]}"; do
      if ! run_export_if_ready "$seed"; then
        status=1
      fi
    done
    return "$status"
  fi

  for i in "${!GPUS[@]}"; do
    if (( task_idx >= total_tasks )); then
      break
    fi
    launch_train_fill "${TASK_SEEDS[$task_idx]}" "${TASK_ARMS[$task_idx]}" "${GPUS[$i]}"
    task_idx=$((task_idx + 1))
  done

  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
  while (( ${#RUN_PIDS[@]} > 0 )); do
    wait_one_done
    done_pid="$DONE_PID"
    done_rc="$DONE_RC"
    done_seed="${RUN_SEED[$done_pid]:-}"
    done_label="${RUN_LABEL[$done_pid]:-}"
    done_arm="${RUN_ARM[$done_pid]:-}"
    done_gpu="${RUN_GPU[$done_pid]:-}"
    done_log="${RUN_LOG[$done_pid]:-}"
    remove_running "$done_pid"

    if (( done_rc == 0 )); then
      echo "[done] label=$done_label rc=$done_rc"
      SEED_SUCCESS_COUNT["$done_seed"]=$((SEED_SUCCESS_COUNT["$done_seed"] + 1))
    else
      echo "[error] label=$done_label rc=$done_rc log=$done_log" >&2
      status=1
    fi

    if (( task_idx < total_tasks )); then
      launch_train_fill "${TASK_SEEDS[$task_idx]}" "${TASK_ARMS[$task_idx]}" "$done_gpu"
      task_idx=$((task_idx + 1))
    fi

    if ! run_export_if_ready "$done_seed"; then
      status=1
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
  done

  for seed in "${SEEDS[@]}"; do
    if ! run_export_if_ready "$seed"; then
      status=1
    fi
  done
  return "$status"
}

build_task_queue

if [[ "$AUTO_FILL_GPU" == "0" ]]; then
  run_seed_wave_static
  FAIL_CODE=$?
else
  run_seed_wave_fill
  FAIL_CODE=$?
fi

if (( FAIL_CODE != 0 )); then
  echo "[error] queue finished with failures."
  exit "$FAIL_CODE"
fi

echo "[ok] RT-DETR D-Fire 12-task tmux queue completed."
echo "[ok] run_root=$PROJECT_ROOT/$RUN_ROOT"
echo "[ok] out_root=$PROJECT_ROOT/$OUT_ROOT"
