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
import socketserver
import urllib.request
import urllib.error

# Embedding config — external endpoint (Modal, Together, Ollama, etc.)
EMBED_API_BASE = os.environ.get("LENS_EMBED_BASE_URL", os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/v1"))
EMBED_API_KEY = os.environ.get("LENS_EMBED_API_KEY", "")
EMBED_MODEL = os.environ.get("LENS_EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base")
# If True, the embed endpoint uses custom format ({"texts": [...]} → {"embeddings": [...]})
# and needs translation to/from OpenAI format
EMBED_CUSTOM_FORMAT = os.environ.get("LENS_EMBED_CUSTOM_FORMAT", "").lower() in ("1", "true", "yes")

# LLM config — Cerebras (primary) or vLLM (fallback)
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_API_BASE = "https://api.cerebras.ai/v1"
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
VLLM_URL = os.environ.get("VLLM_URL", "")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-32B")

# Legacy Together AI fallback (deprecated — use CEREBRAS or VLLM)
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

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
        """Route embeddings to configured endpoint (Modal, Together, Ollama)."""
        body = json.loads(raw_body)

        if EMBED_CUSTOM_FORMAT:
            # Translate OpenAI format → custom format (Modal endpoint)
            inp = body.get("input", [])
            if isinstance(inp, str):
                inp = [inp]
            custom_payload = json.dumps({"texts": inp}).encode()
            target_url = EMBED_API_BASE  # Direct endpoint, not /embeddings
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "lens-letta-proxy/2.0",
            }
            if EMBED_API_KEY:
                headers["Authorization"] = f"Bearer {EMBED_API_KEY}"
            try:
                status, resp_body = _forward(target_url, custom_payload, headers, timeout=30)
            except Exception as exc:
                log.warning("Embed endpoint unreachable (%s): %s", target_url, exc)
                status, resp_body = 503, b'{"error": "embed endpoint unreachable"}'
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp_body)
                return

            if status == 200:
                # Translate custom response → OpenAI format
                try:
                    custom_resp = json.loads(resp_body)
                    embeddings = custom_resp.get("embeddings", [])
                    openai_resp = {
                        "object": "list",
                        "data": [
                            {"object": "embedding", "index": i, "embedding": emb}
                            for i, emb in enumerate(embeddings)
                        ],
                        "model": body.get("model", EMBED_MODEL),
                        "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    }
                    resp_body = json.dumps(openai_resp).encode()
                except Exception as exc:
                    log.warning("Failed to translate embed response: %s", exc)
        else:
            # Standard OpenAI-compatible endpoint
            body["model"] = EMBED_MODEL
            payload = json.dumps(body).encode()
            target_url = f"{EMBED_API_BASE}/embeddings"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "lens-letta-proxy/2.0",
            }
            if EMBED_API_KEY:
                headers["Authorization"] = f"Bearer {EMBED_API_KEY}"
            try:
                status, resp_body = _forward(target_url, payload, headers, timeout=30)
            except Exception as exc:
                log.warning("Embed endpoint unreachable (%s): %s", target_url, exc)
                status, resp_body = 503, b'{"error": "embed endpoint unreachable"}'

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def _handle_chat(self, raw_body: bytes):
        """Route chat completions to Cerebras, vLLM, or Together AI (fallback)."""
        body = json.loads(raw_body)

        if CEREBRAS_API_KEY:
            body["model"] = CEREBRAS_MODEL
            target_url = f"{CEREBRAS_API_BASE}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "lens-letta-proxy/2.0",
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            }
        elif VLLM_URL:
            body["model"] = VLLM_MODEL
            target_url = f"{VLLM_URL}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "lens-letta-proxy/2.0",
            }
        elif TOGETHER_API_KEY:
            target_url = f"{TOGETHER_BASE_URL}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "lens-letta-proxy/2.0",
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
            }
            log.info("Routing chat to Together AI (fallback)")
        else:
            self.send_error(503, "No chat backend configured (set CEREBRAS_API_KEY, VLLM_URL, or TOGETHER_API_KEY)")
            return

        payload = json.dumps(body).encode()
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
            # Combined model list: embed models + LLM model
            llm_model = CEREBRAS_MODEL if CEREBRAS_API_KEY else VLLM_MODEL
            models = {
                "object": "list",
                "data": [
                    {"id": "text-embedding-3-small", "object": "model", "type": "embedding"},
                    {"id": "text-embedding-ada-002", "object": "model", "type": "embedding"},
                    {"id": llm_model, "object": "model", "type": "chat", "context_length": 131072},
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
    log.info("  Embeddings → %s (model: %s)", EMBED_API_BASE, EMBED_MODEL)
    if CEREBRAS_API_KEY:
        log.info("  Chat       → Cerebras (model: %s)", CEREBRAS_MODEL)
    elif VLLM_URL:
        log.info("  Chat       → %s (model: %s)", VLLM_URL, VLLM_MODEL)
    elif TOGETHER_API_KEY:
        log.info("  Chat       → Together AI (deprecated fallback)")
    else:
        log.warning("  Chat       → NO BACKEND (set CEREBRAS_API_KEY, VLLM_URL, or TOGETHER_API_KEY)")
    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", PORT), LettaProxy)
    server.serve_forever()
