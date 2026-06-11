#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

activate_daq
cd "$PROJECT_ROOT"
python scripts/stage5_aggregate_repeats.py --require-eqstep --min-seeds 3 --required-seeds 11,22,33 --strict
echo "[ok] Stage 5 v1 strict aggregation passed."
