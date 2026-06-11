#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${GEMINI_DATA_OUT:-/gemini/output}"
WORK_ROOT="${GEMINI_WORK_ROOT:-$OUTPUT_ROOT/detr_Q3_work}"

bash "$SCRIPT_DIR/gemini_prepare_workdir.sh"
cd "$WORK_ROOT"
bash stage5_h20_server/run_stage5_v1_mincheck_4gpu.sh
