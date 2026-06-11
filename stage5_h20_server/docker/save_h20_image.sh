#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-detr-q3-h20:cu124}"
OUT="${OUT:-$PROJECT_ROOT/transfer_packages/detr-q3-h20-cu124.tar}"

mkdir -p "$(dirname "$OUT")"
docker save "$IMAGE_TAG" -o "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
ls -lh "$OUT" "$OUT.sha256"
echo "[ok] saved Docker image: $OUT"
