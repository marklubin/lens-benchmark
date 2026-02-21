#!/usr/bin/env bash
# Launch Ollama embedding server on a lightweight GPU pod (lens-embed).
#
# Serves nomic-embed-text for all LENS worker pods.
# Throughput: ~500 req/s on GPU — handles 6 workers easily.
#
# Prefer the Docker image at ~/runpod-infra/images/ollama-embed-ssh/ which
# pre-loads the model and includes SSH access. This script is a fallback
# for running on a bare ollama/ollama container.
#
# Usage (on the RunPod embed pod):
#   bash embed_server.sh
set -euo pipefail

EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export OLLAMA_HOST

echo "=== LENS Embedding Server ==="
echo "Model: $EMBED_MODEL"
echo "Host:  $OLLAMA_HOST"

# Start Ollama in the background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        break
    fi
    sleep 1
done

# Pull the embedding model (no-op if using the pre-built image)
echo "Pulling $EMBED_MODEL..."
ollama pull "$EMBED_MODEL"
echo "Embedding server ready on port 11434"

# Keep running
wait $OLLAMA_PID
