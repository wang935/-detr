#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${GEMINI_DATA_OUT:-/gemini/output}"
WORK_ROOT="${GEMINI_WORK_ROOT:-$OUTPUT_ROOT/detr_Q3_work}"

cd "$WORK_ROOT"
bash stage5_h20_server/run_stage5_v1_aggregate.sh
