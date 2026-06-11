#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/gemini/code}"
export RESULT_ROOT="${RESULT_ROOT:-/gemini/output/detr_Q3_results}"
export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
export TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
export SSH_PORT="${SSH_PORT:-22}"
export LOG_ROOT="${LOG_ROOT:-/var/log/h20-services}"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT" /run/sshd /root/.ssh /tmp/matplotlib /tmp/ultralytics
chmod 700 /root/.ssh

if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
  touch /root/.ssh/authorized_keys
  grep -qxF "$SSH_PUBLIC_KEY" /root/.ssh/authorized_keys || echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

if [[ -n "${ROOT_PASSWORD:-}" ]]; then
  echo "root:${ROOT_PASSWORD}" | chpasswd
fi

if [[ "${ENABLE_SSH:-1}" == "1" ]]; then
  if [[ "$SSH_PORT" != "22" ]]; then
    sed -i "s/^#\?Port .*/Port ${SSH_PORT}/" /etc/ssh/sshd_config
  fi
  /usr/sbin/sshd
  echo "[ok] sshd listening on port ${SSH_PORT}"
fi

if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
  JUPYTER_TOKEN="$(python - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"
  export JUPYTER_TOKEN
fi

if [[ "${ENABLE_JUPYTER:-1}" == "1" ]]; then
  nohup jupyter lab \
    --ip=0.0.0.0 \
    --port="$JUPYTER_PORT" \
    --no-browser \
    --allow-root \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --ServerApp.allow_origin="*" \
    > "$LOG_ROOT/jupyterlab.log" 2>&1 &
  echo "[ok] JupyterLab listening on port ${JUPYTER_PORT}"
  echo "[info] JUPYTER_TOKEN=${JUPYTER_TOKEN}"
fi

if [[ "${ENABLE_TENSORBOARD:-1}" == "1" ]]; then
  nohup tensorboard \
    --logdir "$RESULT_ROOT/runs" \
    --host 0.0.0.0 \
    --port="$TENSORBOARD_PORT" \
    > "$LOG_ROOT/tensorboard.log" 2>&1 &
  echo "[ok] TensorBoard listening on port ${TENSORBOARD_PORT}, logdir=${RESULT_ROOT}/runs"
fi

echo "[info] logs: $LOG_ROOT"
echo "[info] PROJECT_ROOT=$PROJECT_ROOT"
echo "[info] RESULT_ROOT=$RESULT_ROOT"

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

tail -f /dev/null
