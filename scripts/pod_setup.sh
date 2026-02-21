#!/usr/bin/env bash
# Bootstrap a LENS worker pod on RunPod.
#
# This script is the pod entrypoint — called by pod_manager.py via dockerStartCmd.
# It launches vLLM, starts adapter containers, clones the repo, and runs pod_worker.py.
#
# Docker image: vllm/vllm-openai:v0.8.5.post1
#
# Required environment variables:
#   LENS_BRANCH      - Git branch to commit results to
#   LENS_GROUP       - Worker group name (fast, letta, mem0, cognee, graphiti, hindsight)
#   LENS_JOBS        - JSON list of jobs [{adapter, scope, budget}, ...]
#   LENS_REPO_URL    - Git repo URL (SSH or HTTPS)
#
# Optional environment variables:
#   EMBED_URL        - Embedding server URL (default: http://lens-embed:11434/v1)
#   VLLM_MODEL       - Model to serve (default: Qwen/Qwen3-32B-AWQ)
#   VLLM_API_KEY     - vLLM API key (default: lens-benchmark)
#   LENS_CONTAINERS  - Space-separated container names to start (e.g., "falkordb letta")
set -euo pipefail

echo "=== LENS Pod Setup ==="
echo "Group:      ${LENS_GROUP:-unknown}"
echo "Branch:     ${LENS_BRANCH:-unknown}"
echo "Pod ID:     ${RUNPOD_POD_ID:-unknown}"

# ---------------------------------------------------------------------------
# 1. Configure HuggingFace cache on network volume
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/runpod-volume/hub}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"

# ---------------------------------------------------------------------------
# 2. Launch vLLM in background
# ---------------------------------------------------------------------------
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-32B-AWQ}"
VLLM_API_KEY="${VLLM_API_KEY:-lens-benchmark}"

echo "Starting vLLM with model $VLLM_MODEL ..."
python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --port 8000 --host 0.0.0.0 \
    --max-model-len 14000 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 16 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --trust-remote-code \
    --disable-log-requests \
    --api-key "$VLLM_API_KEY" &
VLLM_PID=$!

# ---------------------------------------------------------------------------
# 3. Start adapter containers (if specified)
# ---------------------------------------------------------------------------
CONTAINERS="${LENS_CONTAINERS:-}"
for container in $CONTAINERS; do
    echo "Starting container: $container"
    case "$container" in
        falkordb)
            # FalkorDB for Graphiti
            if command -v podman &>/dev/null; then
                podman run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest
            else
                echo "WARNING: podman not available, skipping $container"
            fi
            ;;
        letta)
            # Letta server
            if command -v podman &>/dev/null; then
                podman run -d --name letta -p 8283:8283 \
                    -e OPENAI_API_KEY=dummy \
                    -e OPENAI_API_BASE=http://host.containers.internal:8000/v1 \
                    letta/letta:latest
            else
                echo "WARNING: podman not available, skipping $container"
            fi
            ;;
        qdrant)
            # Qdrant for mem0
            if command -v podman &>/dev/null; then
                podman run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
            else
                echo "WARNING: podman not available, skipping $container"
            fi
            ;;
        hindsight)
            # Hindsight server
            if command -v podman &>/dev/null; then
                podman run -d --name hindsight -p 8090:8090 \
                    -e OPENAI_API_KEY=dummy \
                    -e OPENAI_API_BASE=http://host.containers.internal:8000/v1 \
                    hindsight/hindsight:latest
            else
                echo "WARNING: podman not available, skipping $container"
            fi
            ;;
        *)
            echo "WARNING: Unknown container: $container"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 4. Wait for vLLM health
# ---------------------------------------------------------------------------
echo "Waiting for vLLM to be ready..."
VLLM_TIMEOUT=300
VLLM_START=$SECONDS
while true; do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "vLLM ready after $((SECONDS - VLLM_START))s"
        break
    fi

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi

    if (( SECONDS - VLLM_START > VLLM_TIMEOUT )); then
        echo "ERROR: vLLM not ready after ${VLLM_TIMEOUT}s"
        exit 1
    fi

    sleep 5
done

# ---------------------------------------------------------------------------
# 5. Verify embedding pod
# ---------------------------------------------------------------------------
EMBED_URL="${EMBED_URL:-}"
if [ -n "$EMBED_URL" ]; then
    echo "Checking embedding server at $EMBED_URL ..."
    # Strip /v1 suffix for model list check
    EMBED_BASE="${EMBED_URL%/v1}"
    if curl -sf "${EMBED_BASE}/v1/models" >/dev/null 2>&1; then
        echo "Embedding server OK"
    else
        echo "WARNING: Embedding server not reachable at $EMBED_URL"
    fi
fi

# ---------------------------------------------------------------------------
# 6. Repo setup and install dependencies
# ---------------------------------------------------------------------------
LENS_BRANCH="${LENS_BRANCH:-main}"
WORKDIR="/workspace/lens-benchmark"

# The repo should already be cloned by the docker start command.
# If running manually, clone it now.
if [ -d "$WORKDIR/.git" ]; then
    cd "$WORKDIR"
    echo "Repo already cloned at $WORKDIR"
else
    LENS_REPO_URL="${LENS_REPO_URL:-}"
    if [ -n "$LENS_REPO_URL" ]; then
        echo "Cloning repo..."
        git clone --branch "$LENS_BRANCH" --depth 1 "$LENS_REPO_URL" "$WORKDIR"
        cd "$WORKDIR"
    else
        echo "ERROR: No repo at $WORKDIR and LENS_REPO_URL not set"
        exit 1
    fi
fi

# Install uv if not available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing dependencies..."
uv sync --all-extras 2>&1 | tail -5

# Configure git for commits
git config user.email "lens-sweep@runpod.io"
git config user.name "LENS Sweep Worker"

# Unshallow and set up remote tracking so we can push
git fetch --unshallow origin "$LENS_BRANCH" 2>/dev/null || true
git branch --set-upstream-to="origin/$LENS_BRANCH" "$LENS_BRANCH" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 7. Set environment and launch pod_worker.py
# ---------------------------------------------------------------------------
export VLLM_URL="http://localhost:8000/v1"
export EMBED_URL="${EMBED_URL:-http://localhost:11434/v1}"

echo "=== Launching pod_worker.py ==="
python3 scripts/pod_worker.py
WORKER_EXIT=$?

# ---------------------------------------------------------------------------
# 8. Auto-shutdown
# ---------------------------------------------------------------------------
echo "Worker finished with exit code $WORKER_EXIT"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "Stopping pod ${RUNPOD_POD_ID}..."
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "WARNING: runpodctl stop failed"
fi

exit $WORKER_EXIT
