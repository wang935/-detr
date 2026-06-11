#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
python scripts/stage5_pv_v2_aggregate.py --require-eqstep --require-sched --min-seeds 3 --required-seeds 11,22,33 --required-datasets dfire,dfs --required-families yolo26n,rtdetr --strict
echo "[ok] Stage5-PV v2 strict aggregation passed."
