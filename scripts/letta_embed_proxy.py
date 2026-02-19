"""Minimal embedding translation proxy for Letta.

Letta registers OpenAI embedding model IDs (text-embedding-3-small, etc.)
for its BYOK openai-style providers. Together AI doesn't accept these names.
This proxy intercepts embedding requests and rewrites the model name to
the actual Together AI model, then forwards to Together AI.

Usage:
    python scripts/letta_embed_proxy.py

Port: 7878 (configure Letta's together-oai provider base_url to http://localhost:7878)
"""
import http.server
import json
import os
import urllib.request
import urllib.error

TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TARGET_EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"
PORT = 7878


class EmbedProxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def do_POST(self):
        if not self.path.endswith("/embeddings"):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        # Rewrite model name
        original_model = body.get("model", "")
        body["model"] = TARGET_EMBED_MODEL

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{TOGETHER_BASE_URL}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "lens-letta-proxy/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            err = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err)

    def do_GET(self):
        # Letta calls /v1/models to list models from the provider
        if self.path in ("/v1/models", "/v1/models/"):
            # Return a minimal model list that includes our embedding model
            models = {
                "object": "list",
                "data": [
                    {"id": "text-embedding-3-small", "object": "model"},
                    {"id": "text-embedding-ada-002", "object": "model"},
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
    server = http.server.HTTPServer(("0.0.0.0", PORT), EmbedProxy)
    print(f"Letta embedding proxy running on port {PORT}")
    print(f"Translating all models → {TARGET_EMBED_MODEL} @ Together AI")
    server.serve_forever()
