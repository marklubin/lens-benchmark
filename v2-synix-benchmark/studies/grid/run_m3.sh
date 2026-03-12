#!/usr/bin/env bash
# Run M=3 replicates for a scope. Banks are built on r01 (cached) and reused for r02, r03.
# r01 uses cache (for bank builds). r02+ disable cache (for inference variance).
# Usage: bash studies/grid/run_m3.sh <scope_id> <scope_dir> [start_replicate] [extra_args...]
set -euo pipefail

SCOPE_ID="$1"
SCOPE_DIR="$2"
START_REP="${3:-1}"
shift 3 2>/dev/null || shift $# 2>/dev/null
EXTRA_ARGS="$*"

cd "$(dirname "$0")/../.."

for rep in 1 2 3; do
    if [ "$rep" -lt "$START_REP" ]; then
        echo "Skipping r0${rep} (start_replicate=${START_REP})"
        continue
    fi

    CACHE_FLAG=""
    if [ "$rep" -gt 1 ]; then
        CACHE_FLAG="--no-cache"
    fi

    echo "=== ${SCOPE_ID} replicate r0${rep} ${CACHE_FLAG} ${EXTRA_ARGS} ==="
    uv run python studies/grid/run_scope.py \
        --scope-id "$SCOPE_ID" \
        --scope-dir "$SCOPE_DIR" \
        --temperature 0.3 \
        --replicate-id "r0${rep}" \
        $CACHE_FLAG $EXTRA_ARGS
done

echo "=== ${SCOPE_ID} ALL 3 REPLICATES DONE ==="
