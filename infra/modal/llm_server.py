"""Modal vLLM server for LENS benchmark — Qwen3.5-35B-A3B on H100.

Serves an OpenAI-compatible /v1/chat/completions endpoint with tool calling
support via the qwen3_coder parser.

Deploy:
    cd infra/modal && modal deploy llm_server.py

Test:
    curl https://synix--lens-llm-llm-serve.modal.run/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "Qwen/Qwen3.5-35B-A3B",
           "messages": [{"role": "user", "content": "Say hello"}],
           "max_tokens": 64}'

Debugging:
    modal app logs lens-llm
"""
import os
import subprocess

import modal

app = modal.App("lens-llm")

# Volumes for caching model weights and vLLM compilation artifacts
hf_cache = modal.Volume.from_name("lens-model-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B-FP8"
# Serve as the non-FP8 name so configs don't need updating
SERVED_MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
VLLM_PORT = 8000
MINUTES = 60

MIN_CONTAINERS = int(os.environ.get("LENS_MIN_CONTAINERS", "0"))

# Qwen3.5 requires vLLM nightly (qwen3_5_moe arch not in 0.13.0)
CHAT_TEMPLATE = "/root/qwen3_permissive.jinja"

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm",
        "huggingface-hub==0.36.0",
        extra_index_url="https://wheels.vllm.ai/nightly",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file("qwen3_permissive.jinja", CHAT_TEMPLATE)
)


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
    min_containers=MIN_CONTAINERS,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def llm_serve():
    """Start vLLM OpenAI-compatible server via subprocess."""
    cmd = [
        "vllm", "serve", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--served-model-name", SERVED_MODEL_NAME,
        "--max-model-len", "262144",
        "--gpu-memory-utilization", "0.95",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--chat-template", CHAT_TEMPLATE,
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--trust-remote-code",
        "--disable-log-requests",
        "--uvicorn-log-level", "info",
    ]
    subprocess.Popen(cmd)
