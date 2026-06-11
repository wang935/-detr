#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_TAG="${LOCAL_TAG:-detr-q3-h20:cu124-lab}"
REMOTE_TAG="${REMOTE_TAG:-hpc.chzu.edu.cn:32402/tcvmpvo5lnx3/detr_q3:cu124-lab}"

docker build -t "$LOCAL_TAG" "$SCRIPT_DIR"
docker tag "$LOCAL_TAG" "$REMOTE_TAG"
docker push "$REMOTE_TAG"
echo "[ok] pushed $REMOTE_TAG"
