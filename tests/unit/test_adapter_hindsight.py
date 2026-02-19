"""Tests for HindsightAdapter batch ingest behavior.

These tests verify that ingest() buffers episodes and prepare() flushes them
via aretain_batch() — no real Hindsight server required.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_client():
    """Mock Hindsight client with all necessary methods."""
    client = MagicMock()
    client.create_bank = MagicMock(return_value=None)
    client.retain = MagicMock(return_value=None)
    client.aretain_batch = AsyncMock(return_value=MagicMock(success=True, items_count=2))
    return client


@pytest.fixture
def adapter(mock_client):
    """HindsightAdapter with mocked client, already reset."""
    with patch.dict("sys.modules", {"hindsight_client": MagicMock(Hindsight=MagicMock(return_value=mock_client))}):
        from lens.adapters.hindsight import HindsightAdapter  # noqa: PLC0415
        a = HindsightAdapter()
        a._client = mock_client
        a._bank_id = "test-scope-abcd1234"
        return a


class TestHindsightBatchIngest:
    def test_ingest_buffers_episodes(self, adapter, mock_client):
        """ingest() should append to _pending_episodes and NOT call client.retain()."""
        adapter.ingest("ep_001", "scope01", "2025-01-01T00:00:00Z", "Log entry one")
        adapter.ingest("ep_002", "scope01", "2025-01-02T00:00:00Z", "Log entry two")

        assert len(adapter._pending_episodes) == 2
        assert adapter._pending_episodes[0]["document_id"] == "ep_001"
        assert adapter._pending_episodes[1]["document_id"] == "ep_002"
        assert "[ep_001]" in adapter._pending_episodes[0]["content"]
        # retain() must NOT have been called — we buffer, not retain immediately
        mock_client.retain.assert_not_called()

    def test_prepare_calls_aretain_batch(self, adapter, mock_client):
        """prepare() should call aretain_batch() with buffered items and clear the buffer."""
        adapter.ingest("ep_001", "scope01", "2025-01-01T00:00:00Z", "Log one")
        adapter.ingest("ep_002", "scope01", "2025-01-02T00:00:00Z", "Log two")

        assert len(adapter._pending_episodes) == 2

        adapter.prepare("scope01", 5)

        # aretain_batch must have been called once with both items
        mock_client.aretain_batch.assert_awaited_once()
        call_kwargs = mock_client.aretain_batch.call_args
        assert call_kwargs.kwargs["bank_id"] == "test-scope-abcd1234"
        items = call_kwargs.kwargs["items"]
        assert len(items) == 2
        assert items[0]["document_id"] == "ep_001"
        assert items[1]["document_id"] == "ep_002"

        # Buffer must be cleared after flush
        assert adapter._pending_episodes == []

    def test_prepare_noop_when_empty(self, adapter, mock_client):
        """prepare() on an empty buffer should do nothing and not raise."""
        adapter.prepare("scope01", 5)
        mock_client.aretain_batch.assert_not_called()
        assert adapter._pending_episodes == []

    def test_reset_clears_buffer(self, adapter, mock_client):
        """reset() should clear _pending_episodes along with other state."""
        adapter.ingest("ep_001", "scope01", "2025-01-01T00:00:00Z", "Log one")
        assert len(adapter._pending_episodes) == 1

        # adapter._client is already set to mock_client; _get_client() returns it directly
        adapter.reset("new-scope")

        assert adapter._pending_episodes == []
