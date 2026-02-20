#!/usr/bin/env bash
# Launch vLLM on a RunPod A100-80GB pod with auto-shutdown watchdog.
#
# Usage (on the RunPod pod):
#   bash runpod_vllm.sh
#
# Environment variables:
#   IDLE_TIMEOUT  - seconds before auto-shutdown (default: 3600 = 1 hour)
#   VLLM_API_KEY  - API key for the vLLM server (default: lens-benchmark)
#   PORT          - server port (default: 8000)
set -euo pipefail

MODEL="hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
PORT="${PORT:-8000}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-3600}"
VLLM_API_KEY="${VLLM_API_KEY:-lens-benchmark}"

echo "=== RunPod vLLM Launcher ==="
echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "Idle timeout: ${IDLE_TIMEOUT}s"
echo "Pod ID:       ${RUNPOD_POD_ID:-unknown}"

# Install vLLM if not already available
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install -q vllm
fi

# Auto-shutdown watchdog (background)
# Stops pod after $IDLE_TIMEOUT seconds of no requests
if [ -n "${RUNPOD_POD_ID:-}" ]; then
    (
        echo "Watchdog: will stop pod after ${IDLE_TIMEOUT}s idle"
        sleep "${IDLE_TIMEOUT}"
        echo "Idle timeout reached (${IDLE_TIMEOUT}s) — stopping pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "$RUNPOD_POD_ID" || echo "WARNING: runpodctl stop failed"
    ) &
    WATCHDOG_PID=$!
    echo "Watchdog PID: $WATCHDOG_PID"
else
    echo "WARNING: RUNPOD_POD_ID not set — auto-shutdown disabled"
fi

# Launch vLLM OpenAI-compatible server
echo "Starting vLLM server on port $PORT..."
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization awq \
    --dtype auto \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --port "$PORT" \
    --api-key "$VLLM_API_KEY"
