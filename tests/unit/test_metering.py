"""Tests for the LLM metering proxy."""
from __future__ import annotations

import json
import urllib.request

import pytest

from lens.metering import MeteringUsage
from lens.metering.proxy import UsageStore
from lens.metering.manager import MeteringManager


class TestUsageStore:
    def test_empty_store(self):
        store = UsageStore()
        usage = store.get_usage()
        assert usage.total_calls == 0
        assert usage.total_tokens == 0

    def test_record_and_get(self):
        store = UsageStore()
        store.record("gpt-4o", prompt_tokens=100, completion_tokens=50, latency_ms=200.0)
        usage = store.get_usage()
        assert usage.total_calls == 1
        assert usage.total_prompt_tokens == 100
        assert usage.total_completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.by_model == {"gpt-4o": 150}

    def test_multiple_records(self):
        store = UsageStore()
        store.record("gpt-4o", 100, 50, 200.0)
        store.record("gpt-4o-mini", 50, 25, 100.0)
        usage = store.get_usage()
        assert usage.total_calls == 2
        assert usage.total_tokens == 225
        assert usage.by_model == {"gpt-4o": 150, "gpt-4o-mini": 75}
        assert usage.avg_latency_ms == 150.0

    def test_reset(self):
        store = UsageStore()
        store.record("gpt-4o", 100, 50, 200.0)
        store.reset()
        usage = store.get_usage()
        assert usage.total_calls == 0

    def test_to_dict(self):
        usage = MeteringUsage(
            total_calls=1,
            total_prompt_tokens=10,
            total_completion_tokens=5,
            total_tokens=15,
            avg_latency_ms=100.0,
            by_model={"gpt-4o": 15},
        )
        d = usage.to_dict()
        assert d["total_calls"] == 1
        assert d["by_model"] == {"gpt-4o": 15}


class TestMeteringManager:
    def test_start_and_stop(self):
        mgr = MeteringManager()
        url = mgr.start(upstream_url="https://api.openai.com")
        assert url.startswith("http://localhost:")
        assert mgr.is_running
        mgr.stop()
        assert not mgr.is_running

    def test_usage_query(self):
        mgr = MeteringManager()
        mgr.start()
        try:
            usage = mgr.get_usage()
            assert usage.total_calls == 0
            mgr.reset()
            usage = mgr.get_usage()
            assert usage.total_calls == 0
        finally:
            mgr.stop()

    def test_metering_endpoint_reachable(self):
        """GET /metering/usage returns valid JSON."""
        mgr = MeteringManager()
        url = mgr.start()
        try:
            req = urllib.request.Request(f"{url}/metering/usage")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                assert "total_calls" in data
        finally:
            mgr.stop()

    def test_reset_endpoint(self):
        """POST /metering/reset clears stats."""
        mgr = MeteringManager()
        url = mgr.start()
        try:
            req = urllib.request.Request(f"{url}/metering/reset", method="POST", data=b"")
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 200
        finally:
            mgr.stop()
