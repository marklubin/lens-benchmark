"""LLM + embedding translation proxy for Letta.

Routes requests from Letta's single BYOK provider to Modal endpoints:
- /v1/embeddings → Modal embedding server (gte-modernbert-base)
- /v1/chat/completions → Modal LLM server (Llama 3.3 70B)
- /v1/models → combined model list

Usage:
    MODAL_LLM_URL=https://... MODAL_EMBED_URL=https://... python scripts/letta_embed_proxy.py

Port: 7878 (configure Letta's provider base_url to http://localhost:7878)
"""
import http.server
import json
import logging
import os
import socketserver
import urllib.request
import urllib.error

# Modal endpoints
LLM_API_BASE = os.environ.get("MODAL_LLM_URL") or os.environ.get("LENS_LLM_API_BASE", "")
LLM_MODEL = os.environ.get("LENS_LLM_MODEL", "casperhansen/Meta-Llama-3.3-70B-Instruct-AWQ-INT4")
LLM_API_KEY = os.environ.get("MODAL_API_KEY") or os.environ.get("LENS_LLM_API_KEY", "")

EMBED_API_BASE = os.environ.get("MODAL_EMBED_URL") or os.environ.get("LENS_EMBED_BASE_URL", "")
EMBED_MODEL = os.environ.get("LENS_EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base")
EMBED_API_KEY = os.environ.get("MODAL_API_KEY") or os.environ.get("LENS_EMBED_API_KEY", "")

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
            self.send_error(404, f"Unknown path: {self.path}")

    def _handle_embeddings(self, raw_body: bytes):
        """Route embeddings to Modal embedding endpoint (OpenAI-compatible)."""
        body = json.loads(raw_body)
        body["model"] = EMBED_MODEL
        payload = json.dumps(body).encode()
        target_url = EMBED_API_BASE.rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "lens-letta-proxy/3.0",
        }
        if EMBED_API_KEY:
            headers["Authorization"] = f"Bearer {EMBED_API_KEY}"
        try:
            status, resp_body = _forward(target_url, payload, headers, timeout=30)
        except Exception as exc:
            log.warning("Embed endpoint unreachable (%s): %s", target_url, exc)
            status = 503
            resp_body = json.dumps({"error": "embed endpoint unreachable"}).encode()

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def _handle_chat(self, raw_body: bytes):
        """Route chat completions to Modal LLM endpoint."""
        if not LLM_API_BASE:
            self.send_error(503, "No LLM backend configured (set MODAL_LLM_URL)")
            return

        body = json.loads(raw_body)
        body["model"] = LLM_MODEL
        target_url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "lens-letta-proxy/3.0",
        }
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"

        payload = json.dumps(body).encode()
        try:
            status, resp_body = _forward(target_url, payload, headers, timeout=120)
        except Exception as exc:
            log.warning("LLM endpoint unreachable (%s): %s", target_url, exc)
            status = 503
            resp_body = json.dumps({"error": "LLM endpoint unreachable"}).encode()

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            models = {
                "object": "list",
                "data": [
                    {"id": "text-embedding-3-small", "object": "model", "type": "embedding"},
                    {"id": "text-embedding-ada-002", "object": "model", "type": "embedding"},
                    {"id": LLM_MODEL, "object": "model", "type": "chat", "context_length": 131072},
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
    if LLM_API_BASE:
        log.info("  Chat       → %s (model: %s)", LLM_API_BASE, LLM_MODEL)
    else:
        log.warning("  Chat       → NO BACKEND (set MODAL_LLM_URL)")

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", PORT), LettaProxy)
    server.serve_forever()
