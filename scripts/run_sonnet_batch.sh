#!/usr/bin/env bash
# Run 9 Sonnet evaluation runs in parallel
# 3 adapters × 3 scopes (S07, S08, S11)

set -euo pipefail

# SSL cert for proxy
export SSL_CERT_FILE="/home/mark/.local/share/synix-proxy/certs/ca.pem"

export LENS_EMBED_BASE_URL="https://synix--lens-embed-serve.modal.run/v1"
export LENS_EMBED_API_KEY="not-needed"
export LENS_EMBED_MODEL="gte-modernbert-base"

# graphrag-light entity extraction uses OPENAI_API_KEY/OPENAI_BASE_URL fallback
# (NOT LENS_LLM_* — those would conflict with resolve_env() overriding the Anthropic key)
CEREBRAS_KEY="$(grep CEREBRAS_API_KEY .env | cut -d= -f2)"

CONFIGS_DIR="configs/sonnet_runs"
PIDS=()

for config in "$CONFIGS_DIR"/*.json; do
    name=$(basename "$config" .json)
    echo "[$(date +%H:%M:%S)] Starting $name"
    if [[ "$name" == graphrag_light_* ]]; then
        # graphrag adapter reads GRAPHRAG_LLM_* for entity extraction (avoids LENS_LLM_* / resolve_env() conflict)
        GRAPHRAG_LLM_API_KEY="$CEREBRAS_KEY" \
        GRAPHRAG_LLM_API_BASE="https://api.cerebras.ai/v1" \
        GRAPHRAG_LLM_MODEL="qwen-3-235b-a22b-instruct-2507" \
        uv run lens run --config "$config" -v > "output/sonnet_${name}.log" 2>&1 &
    else
        uv run lens run --config "$config" -v > "output/sonnet_${name}.log" 2>&1 &
    fi
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} runs in parallel. PIDs: ${PIDS[*]}"
echo "Logs: output/sonnet_*.log"
echo ""

# Wait for all
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    config_name=$(ls "$CONFIGS_DIR"/*.json | sed -n "$((i+1))p" | xargs basename .json)
    if wait "$pid"; then
        echo "[$(date +%H:%M:%S)] ✓ $config_name completed"
    else
        echo "[$(date +%H:%M:%S)] ✗ $config_name FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Done. $((${#PIDS[@]} - FAILED))/${#PIDS[@]} succeeded."
