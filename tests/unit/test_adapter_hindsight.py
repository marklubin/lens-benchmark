"""Tests for HindsightAdapter batch ingest behavior.

These tests verify that ingest() buffers episodes and prepare() flushes them
via individual retain() calls — no real Hindsight server required.

Note: retain_batch() was tried but causes HTTP 413 from Together AI's embedding API
when batching multiple episodes. Individual retain() calls are the safe approach.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_client():
    """Mock Hindsight client with all necessary methods."""
    client = MagicMock()
    client.create_bank = MagicMock(return_value=None)
    client.retain = MagicMock(return_value=None)
    client.retain_batch = MagicMock(return_value=MagicMock(success=True, items_count=2))
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

    def test_prepare_calls_retain_per_episode(self, adapter, mock_client):
        """prepare() should call retain() once per buffered episode and clear the buffer.

        retain_batch() was tried but caused HTTP 413 from Together AI's embedding API.
        Individual retain() calls are the safe approach.
        """
        adapter.ingest("ep_001", "scope01", "2025-01-01T00:00:00Z", "Log one")
        adapter.ingest("ep_002", "scope01", "2025-01-02T00:00:00Z", "Log two")

        assert len(adapter._pending_episodes) == 2

        adapter.prepare("scope01", 5)

        # retain() must have been called once per episode (not retain_batch)
        assert mock_client.retain.call_count == 2
        mock_client.retain_batch.assert_not_called()

        # Verify both episodes were retained with correct document_id
        call_args_list = mock_client.retain.call_args_list
        doc_ids = [c.kwargs["document_id"] for c in call_args_list]
        assert "ep_001" in doc_ids
        assert "ep_002" in doc_ids

        # Both must use the correct bank_id
        assert all(c.kwargs["bank_id"] == "test-scope-abcd1234" for c in call_args_list)

        # Buffer must be cleared after flush
        assert adapter._pending_episodes == []

    def test_prepare_noop_when_empty(self, adapter, mock_client):
        """prepare() on an empty buffer should do nothing and not raise."""
        adapter.prepare("scope01", 5)
        mock_client.retain.assert_not_called()
        mock_client.retain_batch.assert_not_called()
        assert adapter._pending_episodes == []

    def test_reset_clears_buffer(self, adapter, mock_client):
        """reset() should clear _pending_episodes along with other state."""
        adapter.ingest("ep_001", "scope01", "2025-01-01T00:00:00Z", "Log one")
        assert len(adapter._pending_episodes) == 1

        # adapter._client is already set to mock_client; _get_client() returns it directly
        adapter.reset("new-scope")

        assert adapter._pending_episodes == []
