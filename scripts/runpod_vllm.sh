#!/usr/bin/env bash
# Launch vLLM on a RunPod A100-80GB pod.
#
# Usage (on the RunPod pod):
#   bash runpod_vllm.sh
#
# Environment variables:
#   VLLM_MODEL    - model to serve (default: Qwen/Qwen3-32B-AWQ)
#   VLLM_API_KEY  - API key for the vLLM server (default: lens-benchmark)
#   PORT          - server port (default: 8000)
#   HF_HOME       - HuggingFace cache dir (default: /runpod-volume/hub)
#
# Docker image: vllm/vllm-openai:v0.8.5.post1
# Flags aligned with ~/runpod-infra/profiles/qwen3-32b-awq.env
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-lens-benchmark}"

# Use network volume for model cache if available
export HF_HOME="${HF_HOME:-/runpod-volume/hub}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"

echo "=== RunPod vLLM Launcher (V1 engine) ==="
echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "HF_HOME:      $HF_HOME"
echo "Pod ID:       ${RUNPOD_POD_ID:-unknown}"

# V1 engine (vLLM 0.8.5+): prefix caching + chunked prefill ON by default.
# DO NOT pass --enable-prefix-caching or --enable-chunked-prefill (forces V0 fallback).
# AWQ quantization is auto-detected from model config — no --quantization flag needed.
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 14000 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 16 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --trust-remote-code \
    --disable-log-requests \
    --api-key "$VLLM_API_KEY"
