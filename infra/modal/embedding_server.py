"""Modal embedding server for LENS benchmark — OpenAI-compatible.

Serves embedding endpoint using Alibaba-NLP/gte-modernbert-base (768 dims).
Responds at /, /embeddings, and /v1/embeddings for compatibility.

Deploy:
    cd infra/modal && modal deploy embedding_server.py

Test:
    curl -X POST https://synix--lens-embed-serve.modal.run/embeddings \
      -H "Content-Type: application/json" \
      -d '{"model": "Alibaba-NLP/gte-modernbert-base", "input": ["hello world"]}'

Debugging:
    modal app logs lens-embed
"""
from __future__ import annotations

import subprocess
import textwrap

import modal

app = modal.App("lens-embed")
model_cache = modal.Volume.from_name("lens-model-cache", create_if_missing=True)
MODEL_CACHE_PATH = "/root/.cache/huggingface"

MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
PORT = 8080

embed_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers>=3.0",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        "uvicorn",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


SERVER_CODE = '''
import logging
from fastapi import FastAPI, Request

log = logging.getLogger("embed-server")
app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "MODEL_ID_PLACEHOLDER",
        cache_folder="CACHE_PLACEHOLDER",
        trust_remote_code=True,
    )
    log.info("Loaded model (dim=%d)", model.get_sentence_embedding_dimension())

async def do_embed(request: Request):
    body = await request.json()
    inp = body.get("input", [])
    if isinstance(inp, str):
        inp = [inp]
    embeddings = model.encode(inp, normalize_embeddings=True)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": emb.tolist()}
            for i, emb in enumerate(embeddings)
        ],
        "model": body.get("model", "MODEL_ID_PLACEHOLDER"),
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in inp),
            "total_tokens": sum(len(t.split()) for t in inp),
        },
    }

@app.post("/")
async def embed_root(request: Request):
    return await do_embed(request)

@app.post("/embeddings")
async def embed_path(request: Request):
    return await do_embed(request)

@app.post("/v1/embeddings")
async def embed_v1(request: Request):
    return await do_embed(request)
'''


@app.function(
    image=embed_image,
    gpu="T4",
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=120,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=30)
@modal.web_server(port=PORT, startup_timeout=120)
def serve():
    """Start FastAPI embedding server with /embeddings endpoint."""
    code = SERVER_CODE.replace("MODEL_ID_PLACEHOLDER", MODEL_ID).replace(
        "CACHE_PLACEHOLDER", MODEL_CACHE_PATH
    )
    with open("/tmp/embed_app.py", "w") as f:
        f.write(code)
    subprocess.Popen([
        "uvicorn", "embed_app:app",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--app-dir", "/tmp",
    ])
