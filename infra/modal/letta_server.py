"""Modal Letta server for LENS benchmark.

Runs a Letta server with an embedded proxy that routes LLM and embedding
requests to the Modal vLLM and embedding servers.

Architecture:
    [benchmark runner] → HTTPS → [Modal Letta server :8283]
                                        ↕ localhost
                                   [proxy :7878]
                                    ↙         ↘
                         [Modal vLLM]    [Modal embed]

Deploy:
    cd infra/modal && modal deploy letta_server.py

Test:
    curl https://synix--lens-letta-serve.modal.run/v1/health

Usage in benchmark:
    LETTA_BASE_URL=https://synix--lens-letta-serve.modal.run \
    LETTA_LLM_MODEL=openai/gpt-4o \
    LETTA_EMBED_MODEL=openai/text-embedding-3-small \
    uv run lens run --config configs/static_letta_scope10d.json -v

The proxy rewrites model names to the actual Modal model IDs, so the
Letta-side model names (openai/gpt-4o) are just placeholders.
"""
import subprocess
import textwrap

import modal

app = modal.App("lens-letta")

LETTA_PORT = 8283
PROXY_PORT = 7878
MINUTES = 60

# Modal backend URLs
MODAL_LLM_URL = "https://synix--lens-llm-llm-serve.modal.run/v1"
MODAL_EMBED_URL = "https://synix--lens-embed-serve.modal.run"
LLM_MODEL = "Qwen/Qwen3.5-35B-A3B"
EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"

letta_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("postgresql", "postgresql-client", "libpq-dev")
    .run_commands(
        # Install pgvector extension
        "apt-get install -y postgresql-16-pgvector || "
        "apt-get install -y postgresql-server-dev-all && "
        "cd /tmp && git clone https://github.com/pgvector/pgvector.git && "
        "cd pgvector && make && make install || true"
    )
    .pip_install("letta", "httpx", "psycopg2-binary")
)

# Inline proxy server code (routes to Modal LLM + embed)
PROXY_CODE = textwrap.dedent(f'''\
import http.server
import json
import logging
import socketserver
import urllib.request
import urllib.error

LLM_API_BASE = "{MODAL_LLM_URL}"
LLM_MODEL = "{LLM_MODEL}"
EMBED_API_BASE = "{MODAL_EMBED_URL}"
EMBED_MODEL = "{EMBED_MODEL}"
PORT = {PROXY_PORT}

log = logging.getLogger("letta-proxy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _forward(url, payload, headers, timeout=120):
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
            self.send_error(404, f"Unknown path: {{self.path}}")

    def _handle_embeddings(self, raw_body):
        body = json.loads(raw_body)
        body["model"] = EMBED_MODEL
        payload = json.dumps(body).encode()
        target_url = EMBED_API_BASE.rstrip("/")
        headers = {{"Content-Type": "application/json"}}
        try:
            status, resp_body = _forward(target_url, payload, headers, timeout=30)
        except Exception as exc:
            log.warning("Embed unreachable: %s", exc)
            status, resp_body = 503, json.dumps({{"error": str(exc)}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def _handle_chat(self, raw_body):
        body = json.loads(raw_body)
        body["model"] = LLM_MODEL
        target_url = f"{{LLM_API_BASE.rstrip('/')}}/chat/completions"
        headers = {{"Content-Type": "application/json"}}
        payload = json.dumps(body).encode()
        try:
            status, resp_body = _forward(target_url, payload, headers, timeout=120)
        except Exception as exc:
            log.warning("LLM unreachable: %s", exc)
            status, resp_body = 503, json.dumps({{"error": str(exc)}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            models = {{
                "object": "list",
                "data": [
                    {{"id": "text-embedding-3-small", "object": "model"}},
                    {{"id": "text-embedding-ada-002", "object": "model"}},
                    {{"id": LLM_MODEL, "object": "model"}},
                    {{"id": "gpt-4o", "object": "model"}},
                ],
            }}
            data = json.dumps(models).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)


if __name__ == "__main__":
    log.info("Letta proxy on port %d", PORT)
    log.info("  LLM   -> %s (%s)", LLM_API_BASE, LLM_MODEL)
    log.info("  Embed -> %s (%s)", EMBED_API_BASE, EMBED_MODEL)

    class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    Threaded(("0.0.0.0", PORT), LettaProxy).serve_forever()
''')

# Startup script: postgres → proxy → letta → provider config
STARTUP_SCRIPT = textwrap.dedent(f'''\
#!/usr/bin/env python3
"""Start PostgreSQL, proxy, and Letta server, then configure providers."""
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("startup")

LETTA_PORT = {LETTA_PORT}
PROXY_PORT = {PROXY_PORT}
PG_USER = "letta"
PG_DB = "letta"

def wait_for_port(port, timeout=120, label="service"):
    """Wait for a TCP port to accept connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket()
            s.settimeout(2)
            s.connect(("127.0.0.1", port))
            s.close()
            log.info("%s ready on port %d", label, port)
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    log.error("%s failed to start on port %d", label, port)
    return False

def setup_postgres():
    """Initialize and start PostgreSQL with pgvector."""
    pg_data = "/var/lib/postgresql/data"
    os.makedirs(pg_data, exist_ok=True)
    os.system("chown -R postgres:postgres /var/lib/postgresql")

    if not os.path.exists(os.path.join(pg_data, "PG_VERSION")):
        log.info("Initializing PostgreSQL...")
        subprocess.run(
            ["su", "-", "postgres", "-c", f"initdb -D {{pg_data}}"],
            check=True,
        )
        # Configure pg_hba for local trust auth
        with open(os.path.join(pg_data, "pg_hba.conf"), "a") as f:
            f.write("\\nlocal all all trust\\nhost all all 127.0.0.1/32 trust\\n")

    log.info("Starting PostgreSQL...")
    subprocess.run(
        ["su", "-", "postgres", "-c", f"pg_ctl -D {{pg_data}} -l /tmp/pg.log start"],
        check=True,
    )
    wait_for_port(5432, label="PostgreSQL")

    # Create user and database
    subprocess.run(
        ["su", "-", "postgres", "-c",
         f"psql -c \\"CREATE USER {{PG_USER}} WITH SUPERUSER PASSWORD 'letta';\\" 2>/dev/null || true"],
        shell=False,
    )
    subprocess.run(
        ["su", "-", "postgres", "-c",
         f"psql -c \\"CREATE DATABASE {{PG_DB}} OWNER {{PG_USER}};\\" 2>/dev/null || true"],
        shell=False,
    )
    # Enable pgvector
    subprocess.run(
        ["su", "-", "postgres", "-c",
         f"psql -d {{PG_DB}} -c \\"CREATE EXTENSION IF NOT EXISTS vector;\\" 2>/dev/null || true"],
        shell=False,
    )
    log.info("PostgreSQL ready with pgvector")

def start_proxy():
    log.info("Starting proxy on port %d...", PROXY_PORT)
    subprocess.Popen([sys.executable, "/root/proxy.py"])
    wait_for_port(PROXY_PORT, label="Proxy")

def start_letta():
    log.info("Starting Letta server on port %d...", LETTA_PORT)
    env = {{
        **os.environ,
        "LETTA_PG_URI": f"postgresql://{{PG_USER}}:letta@localhost:5432/{{PG_DB}}",
        "OPENAI_API_BASE": f"http://localhost:{{PROXY_PORT}}/v1",
        "OPENAI_API_KEY": "dummy",
    }}
    subprocess.Popen(
        ["letta", "server", "--host", "0.0.0.0", "--port", str(LETTA_PORT)],
        env=env,
    )
    wait_for_port(LETTA_PORT, timeout=180, label="Letta")

def configure_providers():
    """Register LLM and embedding providers via Letta REST API."""
    base = f"http://localhost:{{LETTA_PORT}}"

    # Check health
    try:
        req = urllib.request.urlopen(f"{{base}}/v1/health", timeout=5)
        log.info("Letta health: %s", req.read().decode()[:200])
    except Exception as e:
        log.warning("Health check failed: %s", e)

    log.info("Provider configuration complete (using OPENAI_API_BASE env var)")

if __name__ == "__main__":
    setup_postgres()
    start_proxy()
    start_letta()
    configure_providers()
    log.info("All services running. Letta at :%d, Proxy at :%d", LETTA_PORT, PROXY_PORT)
    # Keep alive
    import signal
    signal.pause()
''')


@app.function(
    image=letta_image,
    cpu=2.0,
    memory=4096,
    timeout=30 * MINUTES,
    scaledown_window=10 * MINUTES,
    min_containers=0,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=LETTA_PORT, startup_timeout=5 * MINUTES)
def serve():
    """Start PostgreSQL + proxy + Letta server."""
    # Write proxy and startup scripts
    with open("/root/proxy.py", "w") as f:
        f.write(PROXY_CODE)
    with open("/root/startup.py", "w") as f:
        f.write(STARTUP_SCRIPT)

    subprocess.Popen(["python3", "/root/startup.py"])
