#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/stage5_external_negatives.py \
  --result-root formal_results/stage5 \
  --external-root external_data/BoWFireDataset/dataset/img \
  --external-source "BoWFire non-fire fire-like negatives" \
  --neutral-tokens "not_fire" \
  --out-dir formal_results/stage5_external/bowfire \
  --min-external-images 100
echo "[ok] Stage 5 v1 BoWFire external negative stress test completed."
