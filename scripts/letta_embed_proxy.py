"""LLM + embedding translation proxy for Letta.

Routes requests from Letta's single BYOK provider to the right backends:
- /v1/embeddings → Ollama (local, nomic-embed-text)
- /v1/chat/completions → RunPod vLLM (remote, Qwen3-32B)
- /v1/models → combined model list

Usage:
    VLLM_URL=https://xxx-8000.proxy.runpod.net/v1 python scripts/letta_embed_proxy.py

Port: 7878 (configure Letta's together-oai provider base_url to http://localhost:7878)
"""
import http.server
import json
import logging
import os
import urllib.request
import urllib.error

# Embedding config — local Ollama
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/v1")
TARGET_EMBED_MODEL = os.environ.get("TARGET_EMBED_MODEL", "nomic-embed-text")

# LLM config — RunPod vLLM
VLLM_URL = os.environ.get("VLLM_URL", "")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-32B")

# Fallback to Together AI for embeddings if TOGETHER_API_KEY is set
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"

PORT = int(os.environ.get("PROXY_PORT", "7878"))

log = logging.getLogger("letta-proxy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _forward(url: str, payload: bytes, headers: dict, timeout: int = 60) -> tuple[int, bytes]:
    """Forward a request and return (status_code, body)."""
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read()
        except Exception:
            pass
        return e.code, body


class LettaProxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)

        if self.path.rstrip("/").endswith("/embeddings"):
            self._handle_embeddings(raw_body)
        elif self.path.rstrip("/").endswith("/chat/completions"):
            self._handle_chat(raw_body)
        else:
            # Forward unknown POST to vLLM as-is
            if VLLM_URL:
                self._proxy_to_vllm(self.path, raw_body)
            else:
                self.send_error(404, f"Unknown path: {self.path}")

    def _handle_embeddings(self, raw_body: bytes):
        """Route embeddings to local Ollama."""
        body = json.loads(raw_body)
        body["model"] = TARGET_EMBED_MODEL

        payload = json.dumps(body).encode()
        target_url = f"{OLLAMA_EMBED_URL}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "lens-letta-proxy/2.0",
        }

        # If Ollama fails, fall back to Together AI
        status, resp_body = _forward(target_url, payload, headers, timeout=30)
        if status >= 500 and TOGETHER_API_KEY:
            log.warning("Ollama embed failed (%d), falling back to Together AI", status)
            body["model"] = TOGETHER_EMBED_MODEL
            payload = json.dumps(body).encode()
            headers["Authorization"] = f"Bearer {TOGETHER_API_KEY}"
            status, resp_body = _forward(
                f"{TOGETHER_BASE_URL}/embeddings", payload, headers, timeout=30
            )

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def _handle_chat(self, raw_body: bytes):
        """Route chat completions to RunPod vLLM."""
        if not VLLM_URL:
            self.send_error(503, "VLLM_URL not configured")
            return

        body = json.loads(raw_body)
        # Rewrite model to the one running on vLLM
        body["model"] = VLLM_MODEL
        payload = json.dumps(body).encode()

        target_url = f"{VLLM_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "lens-letta-proxy/2.0",
        }

        status, resp_body = _forward(target_url, payload, headers, timeout=120)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def _proxy_to_vllm(self, path: str, raw_body: bytes):
        """Forward arbitrary request to vLLM."""
        target_url = f"{VLLM_URL}{path}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "lens-letta-proxy/2.0",
        }
        status, resp_body = _forward(target_url, raw_body, headers, timeout=60)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            # Combined model list: embed models + vLLM model
            models = {
                "object": "list",
                "data": [
                    {"id": "text-embedding-3-small", "object": "model"},
                    {"id": "text-embedding-ada-002", "object": "model"},
                    {"id": VLLM_MODEL, "object": "model"},
                ],
            }
            data = json.dumps(models).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)


if __name__ == "__main__":
    log.info("Letta proxy on port %d", PORT)
    log.info("  Embeddings → %s (model: %s)", OLLAMA_EMBED_URL, TARGET_EMBED_MODEL)
    log.info("  Chat       → %s (model: %s)", VLLM_URL or "NOT SET", VLLM_MODEL)
    server = http.server.HTTPServer(("0.0.0.0", PORT), LettaProxy)
    server.serve_forever()
