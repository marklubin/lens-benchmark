"""Lifecycle manager for the metering proxy."""
from __future__ import annotations

import threading

from lens.metering import MeteringUsage
from lens.metering.proxy import UsageStore, create_proxy_server


class MeteringManager:
    """Start/stop the metering proxy and query usage."""

    def __init__(self) -> None:
        self._server = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None
        self._store: UsageStore | None = None

    def start(self, upstream_url: str = "https://api.openai.com") -> str:
        """Start proxy in background thread, return local URL like 'http://localhost:PORT'."""
        if self._server is not None:
            raise RuntimeError("Metering proxy already running")
        self._store = UsageStore()
        self._server = create_proxy_server(upstream_url, self._store)
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="metering-proxy",
        )
        self._thread.start()
        return f"http://localhost:{self._port}"

    def stop(self) -> None:
        """Shutdown proxy server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None
            self._port = None

    def get_usage(self) -> MeteringUsage:
        """Query proxy for accumulated stats (via direct store reference)."""
        if self._store is None:
            return MeteringUsage()
        return self._store.get_usage()

    def reset(self) -> None:
        """Reset counters."""
        if self._store is not None:
            self._store.reset()

    @property
    def is_running(self) -> bool:
        return self._server is not None
