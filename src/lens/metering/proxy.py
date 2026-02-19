"""Lightweight HTTP proxy that forwards OpenAI-compatible API calls and meters token usage."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler

from lens.metering import MeteringUsage

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Single API call record."""
    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class UsageStore:
    """Thread-safe accumulator for API call records."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float) -> None:
        rec = UsageRecord(
            timestamp=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        with self._lock:
            self._records.append(rec)

    def get_usage(self) -> MeteringUsage:
        with self._lock:
            records = list(self._records)
        if not records:
            return MeteringUsage()
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_tokens = total_prompt + total_completion
        avg_latency = sum(r.latency_ms for r in records) / len(records)
        by_model: dict[str, int] = {}
        for r in records:
            by_model[r.model] = by_model.get(r.model, 0) + r.prompt_tokens + r.completion_tokens
        return MeteringUsage(
            total_calls=len(records),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_tokens,
            avg_latency_ms=avg_latency,
            by_model=by_model,
        )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


def _make_handler_class(upstream_url: str, store: UsageStore) -> type:
    """Create a request handler class bound to the given upstream and store."""

    class MeteringHandler(BaseHTTPRequestHandler):
        """Forwards /v1/ requests to upstream, meters usage from responses."""

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            logger.debug(format, *args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/metering/usage":
                self._serve_usage()
            elif self.path.startswith("/v1/"):
                self._forward("GET")
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/metering/reset":
                store.reset()
                self._send_json(200, {"status": "ok"})
            elif self.path.startswith("/v1/"):
                self._forward("POST")
            else:
                self._send_json(404, {"error": "not found"})

        def _serve_usage(self) -> None:
            usage = store.get_usage()
            self._send_json(200, usage.to_dict())

        def _send_json(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _forward(self, method: str) -> None:
            # Read incoming body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

            # Build upstream request
            target_url = upstream_url.rstrip("/") + self.path
            req = urllib.request.Request(target_url, data=body, method=method)

            # Copy relevant headers
            for header in ("Authorization", "Content-Type", "Accept"):
                value = self.headers.get(header)
                if value:
                    req.add_header(header, value)

            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp_body = resp.read()
                    elapsed_ms = (time.monotonic() - start) * 1000
                    status = resp.status
                    resp_headers = resp.headers
            except urllib.error.HTTPError as e:
                resp_body = e.read()
                elapsed_ms = (time.monotonic() - start) * 1000
                status = e.code
                resp_headers = e.headers

            # Extract usage if present
            self._extract_usage(resp_body, elapsed_ms)

            # Forward response to caller
            self.send_response(status)
            content_type = resp_headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def _extract_usage(self, resp_body: bytes, latency_ms: float) -> None:
            try:
                data = json.loads(resp_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            usage = data.get("usage")
            if not isinstance(usage, dict):
                return
            model = data.get("model", "unknown")
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            store.record(model, prompt_tokens, completion_tokens, latency_ms)

    return MeteringHandler


def create_proxy_server(upstream_url: str, store: UsageStore, port: int = 0) -> HTTPServer:
    """Create an HTTPServer bound to the given port (0 = auto-assign)."""
    handler_cls = _make_handler_class(upstream_url, store)
    server = HTTPServer(("127.0.0.1", port), handler_cls)
    return server
