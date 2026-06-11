#!/usr/bin/env bash
set -euo pipefail

SESSION=stage5pv2_rtdetr_dfire_12
QUEUE=rtdetr_dfire_12_20260606-205531
LOG_DIR=/gemini/output/detr_Q3_work/logs/h20

for i in $(seq 1 40); do
  ts=$(date "+%F %T")
  echo "===== $ts ====="

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[tmux] $SESSION exists"
    tmux list-sessions -t "$SESSION" -F "#S #W created=#{session_created} attached=#{session_attached}"
  else
    echo "[tmux] $SESSION missing"
  fi

  echo "[gpu]"
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader --unit=MiB || true

  pidfile="$LOG_DIR/${QUEUE}.pids"
  if [ -f "$pidfile" ]; then
    echo "[pids] $pidfile"
    running=0
    while read -r pid label log; do
      [ -z "$pid" ] && continue
      if kill -0 "$pid" 2>/dev/null; then
        state=RUN
        running=$((running+1))
      else
        state=DONE
      fi
      printf 'pid=%s state=%s label=%s\n' "$pid" "$state" "$label"
    done < "$pidfile"
    echo "[pids] running=$running"
  else
    echo "[pids] no pid file"
  fi

  echo "[log_tail] driver"
  if [ -f "$LOG_DIR/${QUEUE}_driver.log" ]; then
    sed -r 's/\x1B\[[0-9;]*[a-zA-Z]//g' "$LOG_DIR/${QUEUE}_driver.log" | tail -n 12
  else
    echo "(no driver log)"
  fi

  echo "[latest task logs]"
  ls -1t "$LOG_DIR"/${QUEUE}_seed*_*.log 2>/dev/null | head -n 4 | while read -r f; do
    echo "-- $f"
    sed -r 's/\x1B\[[0-9;]*[a-zA-Z]//g' "$f" | tail -n 3
  done

  echo
  sleep 30
done