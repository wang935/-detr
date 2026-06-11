#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${GEMINI_CODE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUTPUT_ROOT="${GEMINI_DATA_OUT:-/gemini/output}"
WORK_ROOT="${GEMINI_WORK_ROOT:-$OUTPUT_ROOT/detr_Q3_work}"
RESULT_ROOT="${GEMINI_STAGE5_RESULT_ROOT:-$OUTPUT_ROOT/detr_Q3_results}"
PRETRAIN_ROOT="${GEMINI_PRETRAIN:-/gemini/pretrain}"

find_dfire_root() {
  local roots=(
    "${DFIRE_DATA_ROOT:-}"
    "${GEMINI_DATA_IN1:-/gemini/data-1}/D-Fire"
    "${GEMINI_DATA_IN1:-/gemini/data-1}/data/D-Fire"
    "${GEMINI_DATA_IN1:-/gemini/data-1}"
    "${GEMINI_DATA_IN2:-/gemini/data-2}/D-Fire"
    "${GEMINI_DATA_IN2:-/gemini/data-2}/data/D-Fire"
    "${GEMINI_DATA_IN3:-/gemini/data-3}/D-Fire"
    "${GEMINI_DATA_IN3:-/gemini/data-3}/data/D-Fire"
  )
  local root
  for root in "${roots[@]}"; do
    [[ -z "$root" ]] && continue
    if [[ -d "$root/train/images" && -d "$root/train/labels" ]]; then
      printf '%s\n' "$root"
      return 0
    fi
  done
  return 1
}

find_weight() {
  local name="$1"
  local roots=(
    "$PRETRAIN_ROOT"
    "$PRETRAIN_ROOT/detr_Q3"
    "$PRETRAIN_ROOT/weights"
    "$CODE_ROOT"
  )
  local root
  for root in "${roots[@]}"; do
    if [[ -f "$root/$name" ]]; then
      printf '%s\n' "$root/$name"
      return 0
    fi
  done
  return 1
}

DFIRE_ROOT="$(find_dfire_root)" || {
  echo "[error] Cannot find D-Fire. Expected one of:" >&2
  echo "  \$GEMINI_DATA_IN1/D-Fire/train/images" >&2
  echo "  \$GEMINI_DATA_IN1/data/D-Fire/train/images" >&2
  echo "  or set DFIRE_DATA_ROOT=/path/to/D-Fire" >&2
  exit 2
}
YOLO_WEIGHT="$(find_weight yolo26n.pt)" || {
  echo "[error] Cannot find yolo26n.pt under $PRETRAIN_ROOT" >&2
  exit 2
}
RTDETR_WEIGHT="$(find_weight rtdetr-l.pt)" || {
  echo "[error] Cannot find rtdetr-l.pt under $PRETRAIN_ROOT" >&2
  exit 2
}

echo "[info] CODE_ROOT=$CODE_ROOT"
echo "[info] WORK_ROOT=$WORK_ROOT"
echo "[info] RESULT_ROOT=$RESULT_ROOT"
echo "[info] DFIRE_ROOT=$DFIRE_ROOT"
echo "[info] YOLO_WEIGHT=$YOLO_WEIGHT"
echo "[info] RTDETR_WEIGHT=$RTDETR_WEIGHT"

mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"
cp -a "$CODE_ROOT"/. "$WORK_ROOT"/

rm -rf "$WORK_ROOT/data/D-Fire" "$WORK_ROOT/yolo26n.pt" "$WORK_ROOT/rtdetr-l.pt"
mkdir -p "$WORK_ROOT/data"
ln -s "$DFIRE_ROOT" "$WORK_ROOT/data/D-Fire"
ln -s "$YOLO_WEIGHT" "$WORK_ROOT/yolo26n.pt"
ln -s "$RTDETR_WEIGHT" "$WORK_ROOT/rtdetr-l.pt"

mkdir -p "$RESULT_ROOT/runs" "$RESULT_ROOT/formal_results" "$RESULT_ROOT/logs"
rm -rf "$WORK_ROOT/runs" "$WORK_ROOT/formal_results" "$WORK_ROOT/logs"
ln -s "$RESULT_ROOT/runs" "$WORK_ROOT/runs"
ln -s "$RESULT_ROOT/formal_results" "$WORK_ROOT/formal_results"
ln -s "$RESULT_ROOT/logs" "$WORK_ROOT/logs"

python "$WORK_ROOT/stage5_h20_server/02_rewrite_paths_for_server.py" --root "$WORK_ROOT"

echo "[ok] Gemini workdir prepared."
echo "[next] cd $WORK_ROOT"
