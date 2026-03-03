"""Tests for the LettaEntity memory adapter.

Mocks all Letta client and httpx calls — no live server needed.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from lens.adapters.base import CapabilityManifest, Document
from lens.adapters.registry import get_adapter
from lens.core.errors import AdapterError


# ---------------------------------------------------------------------------
# Mock Letta client
# ---------------------------------------------------------------------------


class _MockAgent:
    def __init__(self, name: str):
        self.name = name
        self.id = str(uuid.uuid4())


class _MockAgentsNamespace:
    def __init__(self):
        self._agents: dict[str, _MockAgent] = {}
        self.messages = _MockMessagesNamespace()

    def create(self, name: str, model: str, embedding: str,
               enable_sleeptime: bool = False, memory_blocks: list | None = None):
        a = _MockAgent(name)
        self._agents[a.id] = a
        return a

    def delete(self, agent_id: str):
        self._agents.pop(agent_id, None)

    def list(self):
        return list(self._agents.values())


class _MockMessagesNamespace:
    def create(self, agent_id: str, input: str, max_steps: int = 10):
        return _MockLettaResponse()


class _MockLettaResponse:
    def __init__(self):
        self.messages = []


class MockLettaClient:
    def __init__(self):
        self.agents = _MockAgentsNamespace()


# ---------------------------------------------------------------------------
# Mock httpx for block / tool operations
# ---------------------------------------------------------------------------


def _mock_httpx_success():
    """Return a mock httpx module where all HTTP methods return 200 OK."""
    mock = MagicMock()

    def _make_response(json_data=None, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.text = "{}"
        return resp

    # POST (create block) returns a block with an id
    mock.post.side_effect = lambda *a, **kw: _make_response(
        {"id": str(uuid.uuid4()), "label": "test", "value": "test"}
    )
    # PATCH (attach block, update system prompt) returns 200
    mock.patch.side_effect = lambda *a, **kw: _make_response({})
    # GET (list tools, get blocks) returns empty list
    mock.get.side_effect = lambda *a, **kw: _make_response([])
    # DELETE returns 200
    mock.delete.side_effect = lambda *a, **kw: _make_response({})

    return mock


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_adapter():
    """Create a LettaEntityAdapter with mocked internals."""
    from lens.adapters.letta_entity import LettaEntityAdapter

    adapter = LettaEntityAdapter()
    adapter._client = MockLettaClient()
    return adapter


def _reset_adapter(adapter=None, scope_id="scope_test"):
    """Create and reset an adapter with all httpx calls mocked."""
    if adapter is None:
        adapter = _make_adapter()
    mock_httpx = _mock_httpx_success()
    with patch("lens.adapters.letta_entity.httpx", mock_httpx):
        adapter.reset(scope_id)
    return adapter


def _ingested_adapter(n: int = 3) -> "LettaEntityAdapter":
    """Adapter with reset + n ingested episodes."""
    adapter = _reset_adapter()
    mock_httpx = _mock_httpx_success()
    # GET for _get_agent_blocks during _sync_entity_blocks
    mock_httpx.get.side_effect = lambda *a, **kw: MagicMock(
        status_code=200,
        json=MagicMock(return_value=[
            {"label": "persona", "id": adapter._persona_block_id or "persona-id"},
        ]),
    )
    with patch("lens.adapters.letta_entity.httpx", mock_httpx):
        for i in range(1, n + 1):
            adapter.ingest(
                episode_id=f"ep_{i:03d}",
                scope_id="scope_test",
                timestamp=f"2024-01-0{i}T00:00:00",
                text=f"System report {i}: metric_a={i * 10}, metric_b={i * 5}.",
            )
    return adapter


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestLettaEntityRegistration:
    def test_registered_by_name(self):
        cls = get_adapter("letta-entity")
        assert cls.__name__ == "LettaEntityAdapter"

    def test_distinct_from_letta_v4(self):
        assert get_adapter("letta-entity") is not get_adapter("letta-v4")


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestLettaEntityReset:
    def test_reset_creates_two_agents(self):
        adapter = _reset_adapter()
        assert adapter._ingest_agent_id is not None
        assert adapter._qa_agent_id is not None

    def test_reset_agents_are_distinct(self):
        adapter = _reset_adapter()
        assert adapter._ingest_agent_id != adapter._qa_agent_id

    def test_reset_clears_entity_blocks(self):
        adapter = _make_adapter()
        adapter._entity_blocks = {"old_entity": "block-id-123"}
        _reset_adapter(adapter)
        assert adapter._entity_blocks == {}

    def test_reset_clears_text_cache(self):
        adapter = _ingested_adapter(n=2)
        assert len(adapter._text_cache) == 2
        _reset_adapter(adapter, scope_id="scope_new")
        assert adapter._text_cache == {}

    def test_reset_creates_persona_block(self):
        adapter = _reset_adapter()
        assert adapter._persona_block_id is not None


# ---------------------------------------------------------------------------
# Tests: ingest()
# ---------------------------------------------------------------------------


class TestLettaEntityIngest:
    def test_ingest_caches_text(self):
        adapter = _ingested_adapter(n=2)
        assert "ep_001" in adapter._text_cache
        assert "ep_002" in adapter._text_cache

    def test_ingest_requires_reset(self):
        adapter = _make_adapter()
        with pytest.raises(AdapterError):
            adapter.ingest("ep_001", "scope_x", "2024-01-01", "text")

    def test_ingest_sends_message_to_ingest_agent(self):
        adapter = _reset_adapter()
        mock_httpx = _mock_httpx_success()
        mock_httpx.get.side_effect = lambda *a, **kw: MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"label": "persona", "id": adapter._persona_block_id or "p-id"},
            ]),
        )
        # Spy on _send_message
        original_send = adapter._send_message
        calls = []

        def spy_send(agent_id, message, max_steps=10):
            calls.append((agent_id, message))
            return ""

        adapter._send_message = spy_send
        with patch("lens.adapters.letta_entity.httpx", mock_httpx):
            adapter.ingest("ep_001", "scope_test", "2024-01-01T00:00:00", "Hello data")

        assert len(calls) == 1
        assert calls[0][0] == adapter._ingest_agent_id
        assert "[ep_001]" in calls[0][1]


# ---------------------------------------------------------------------------
# Tests: _sync_entity_blocks
# ---------------------------------------------------------------------------


class TestLettaEntityBlockSync:
    def test_sync_attaches_new_blocks_to_qa(self):
        adapter = _reset_adapter()
        new_block_id = str(uuid.uuid4())
        mock_httpx = _mock_httpx_success()
        # Simulate ingest agent has persona + a new entity block
        mock_httpx.get.side_effect = lambda *a, **kw: MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"label": "persona", "id": adapter._persona_block_id or "p-id"},
                {"label": "alice_jones", "id": new_block_id},
            ]),
        )
        with patch("lens.adapters.letta_entity.httpx", mock_httpx):
            adapter._sync_entity_blocks()

        assert "alice_jones" in adapter._entity_blocks
        assert adapter._entity_blocks["alice_jones"] == new_block_id

    def test_sync_detaches_deleted_blocks(self):
        adapter = _reset_adapter()
        old_block_id = str(uuid.uuid4())
        adapter._entity_blocks["bob_smith"] = old_block_id

        mock_httpx = _mock_httpx_success()
        # Ingest agent only has persona (bob_smith was deleted)
        mock_httpx.get.side_effect = lambda *a, **kw: MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"label": "persona", "id": adapter._persona_block_id or "p-id"},
            ]),
        )
        with patch("lens.adapters.letta_entity.httpx", mock_httpx):
            adapter._sync_entity_blocks()

        assert "bob_smith" not in adapter._entity_blocks

    def test_sync_ignores_persona_block(self):
        adapter = _reset_adapter()
        mock_httpx = _mock_httpx_success()
        mock_httpx.get.side_effect = lambda *a, **kw: MagicMock(
            status_code=200,
            json=MagicMock(return_value=[
                {"label": "persona", "id": "persona-block-id"},
            ]),
        )
        with patch("lens.adapters.letta_entity.httpx", mock_httpx):
            adapter._sync_entity_blocks()

        assert "persona" not in adapter._entity_blocks


# ---------------------------------------------------------------------------
# Tests: answer_question
# ---------------------------------------------------------------------------


class TestLettaEntityAnswer:
    def test_answer_extracts_refs(self):
        adapter = _reset_adapter()
        adapter._scope_id = "scope_test"

        # Mock _send_message to return text with refs
        adapter._send_message = lambda aid, msg, max_steps=15: (
            "Based on [ep_001] and ep_003, there is a pattern."
        )
        answer = adapter.answer_question("What happened?", question_id="q1")
        assert "scope_test_ep_001" in answer.refs_cited
        assert "scope_test_ep_003" in answer.refs_cited

    def test_answer_returns_agent_answer(self):
        from lens.core.models import AgentAnswer

        adapter = _reset_adapter()
        adapter._send_message = lambda aid, msg, max_steps=15: "Some answer text."
        answer = adapter.answer_question("Question?", question_id="q2")
        assert isinstance(answer, AgentAnswer)
        assert answer.question_id == "q2"
        assert answer.answer_text == "Some answer text."

    def test_answer_not_initialized(self):
        adapter = _make_adapter()
        answer = adapter.answer_question("Question?", question_id="q3")
        assert "not initialized" in answer.answer_text.lower()


# ---------------------------------------------------------------------------
# Tests: retrieve
# ---------------------------------------------------------------------------


class TestLettaEntityRetrieve:
    def test_retrieve_from_cache(self):
        adapter = _ingested_adapter(n=2)
        doc = adapter.retrieve("ep_001")
        assert isinstance(doc, Document)
        assert "metric_a=10" in doc.text

    def test_retrieve_missing_returns_none(self):
        adapter = _ingested_adapter(n=1)
        assert adapter.retrieve("ep_999") is None


# ---------------------------------------------------------------------------
# Tests: search and capabilities
# ---------------------------------------------------------------------------


class TestLettaEntitySearch:
    def test_search_returns_empty(self):
        adapter = _reset_adapter()
        assert adapter.search("anything") == []


class TestLettaEntityCapabilities:
    def test_capabilities_has_letta_native(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "letta-native" in caps.search_modes
