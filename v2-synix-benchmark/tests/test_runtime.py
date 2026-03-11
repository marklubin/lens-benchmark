"""Tests for runtime, policy, agent, and runner (T013)."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bench.agent import AgentHarness, _extract_inline_refs
from bench.policy import POLICY_REGISTRY, create_policy, make_null_policy
from bench.runtime import BenchmarkRuntime
from bench.schemas import (
    BankManifest,
    BankStatus,
    EventType,
    FusionConfig,
    PolicyManifest,
    RetrievalCaps,
    RunStatus,
    StudyManifest,
)
from bench.state import EventWriter, StateStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def store(tmp_dir):
    s = StateStore(tmp_dir / "state.db")
    yield s
    s.close()


@pytest.fixture
def event_writer(store):
    return EventWriter(store, "test-study")


@dataclass
class FakeSearchResult:
    content: str
    label: str
    score: float
    layer: str
    provenance: list[str]
    metadata: dict | None = None
    mode: str = "hybrid"
    layer_level: int = 0


@dataclass
class FakeArtifact:
    content: str
    label: str
    artifact_type: str = "core_memory"


def _make_release(*, search_results=None, artifacts=None):
    """Create a mock Synix Release."""
    release = MagicMock()

    if search_results is None:
        search_results = [
            FakeSearchResult(
                content="Episode 1 content about metrics.",
                label="test_ep_001",
                score=0.95,
                layer="episodes",
                provenance=["test_ep_001"],
            ),
            FakeSearchResult(
                content="Chunk from episode 2.",
                label="chunks-test_ep_002-abc",
                score=0.80,
                layer="chunks",
                provenance=["test_ep_002"],
            ),
        ]
    release.search.return_value = search_results

    artifact_map = {}
    if artifacts:
        for art in artifacts:
            artifact_map[art.label] = art
    release.artifact.side_effect = lambda label: artifact_map.get(label) or _raise_not_found(label)

    return release


def _raise_not_found(label):
    raise KeyError(f"Artifact {label!r} not found")


def _base_policy():
    return create_policy("policy_base", "v1")


def _core_policy():
    return create_policy("policy_core", "v1")


def _summary_policy():
    return create_policy("policy_summary", "v1")


def _null_policy():
    return create_policy("null", "v1")


# ---------------------------------------------------------------------------
# Policy Registry Tests
# ---------------------------------------------------------------------------


class TestPolicyRegistry:
    def test_all_policies_registered(self):
        assert set(POLICY_REGISTRY.keys()) == {"null", "policy_base", "policy_core", "policy_summary"}

    def test_null_policy_no_surfaces(self):
        p = _null_policy()
        assert p.query_surfaces == []
        assert p.visible_artifact_families == []
        assert p.retrieval_caps.max_results == 0

    def test_base_policy_has_search(self):
        p = _base_policy()
        assert "keyword" in p.query_surfaces
        assert "semantic" in p.query_surfaces
        assert "episodes" in p.visible_artifact_families
        assert "chunks" in p.visible_artifact_families
        assert "core_memory" not in p.visible_artifact_families

    def test_core_policy_includes_core_memory(self):
        p = _core_policy()
        assert "core_memory" in p.visible_artifact_families
        assert "summary" not in p.visible_artifact_families

    def test_summary_policy_includes_summary(self):
        p = _summary_policy()
        assert "summary" in p.visible_artifact_families
        assert "core_memory" not in p.visible_artifact_families

    def test_create_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown policy_id"):
            create_policy("nonexistent", "v1")

    def test_policy_ids_unique(self):
        policies = [create_policy(pid, "v1") for pid in POLICY_REGISTRY]
        ids = [p.policy_manifest_id for p in policies]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Runtime Tests
# ---------------------------------------------------------------------------


class TestRuntimeSearch:
    def test_null_policy_returns_empty(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _null_policy())
        results = runtime.search("test query")
        assert results == []
        release.search.assert_not_called()

    def test_base_policy_returns_results(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _base_policy())
        results = runtime.search("test query")
        assert len(results) == 2
        assert results[0]["label"] == "test_ep_001"
        assert results[0]["score"] == 0.95

    def test_search_passes_mode_hybrid(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _base_policy())
        runtime.search("test")
        release.search.assert_called_once_with("test", mode="hybrid", limit=10)

    def test_search_respects_limit(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _base_policy())
        runtime.search("test", limit=5)
        release.search.assert_called_once_with("test", mode="hybrid", limit=5)


class TestRuntimeContext:
    def test_null_policy_no_context(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _null_policy())
        assert runtime.get_context() is None

    def test_base_policy_no_context(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _base_policy())
        assert runtime.get_context() is None

    def test_core_policy_returns_core_memory(self):
        core_art = FakeArtifact(content="Key observations from episodes.", label="core-memory")
        release = _make_release(artifacts=[core_art])
        runtime = BenchmarkRuntime(release, _core_policy())
        ctx = runtime.get_context()
        assert ctx == "Key observations from episodes."

    def test_summary_policy_returns_summary(self):
        summary_art = FakeArtifact(content="Overview of all episodes.", label="summary", artifact_type="summary")
        release = _make_release(artifacts=[summary_art])
        runtime = BenchmarkRuntime(release, _summary_policy())
        ctx = runtime.get_context()
        assert ctx == "Overview of all episodes."

    def test_missing_artifact_returns_none(self):
        release = _make_release(artifacts=[])
        runtime = BenchmarkRuntime(release, _core_policy())
        assert runtime.get_context() is None


class TestRuntimeTools:
    def test_null_policy_no_tools(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _null_policy())
        assert runtime.get_tools() == []

    def test_base_policy_has_memory_search(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _base_policy())
        tools = runtime.get_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "memory_search"

    def test_dispatch_unknown_tool_raises(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _base_policy())
        with pytest.raises(ValueError, match="Unknown tool"):
            runtime.dispatch_tool("unknown_tool", {})

    def test_dispatch_blocked_by_null_policy(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _null_policy())
        with pytest.raises(ValueError, match="not allowed"):
            runtime.dispatch_tool("memory_search", {"query": "test"})

    def test_dispatch_memory_search_returns_json(self):
        release = _make_release()
        runtime = BenchmarkRuntime(release, _base_policy())
        result = runtime.dispatch_tool("memory_search", {"query": "test"})
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["label"] == "test_ep_001"


# ---------------------------------------------------------------------------
# Citation Extraction Tests
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    def test_extract_ep_refs(self):
        text = "Based on [scope_01_ep_001] and [scope_01_ep_003], the metrics show..."
        refs = _extract_inline_refs(text)
        assert refs == ["scope_01_ep_001", "scope_01_ep_003"]

    def test_extract_no_refs(self):
        text = "No citations here."
        assert _extract_inline_refs(text) == []

    def test_deduplicates(self):
        text = "[scope_01_ep_001] confirms [scope_01_ep_001] again."
        refs = _extract_inline_refs(text)
        assert refs == ["scope_01_ep_001"]


# ---------------------------------------------------------------------------
# Agent Harness Tests
# ---------------------------------------------------------------------------


class TestAgentHarness:
    def _mock_broker_response(self, *, content="Answer text.", tool_calls=None, finish_reason="stop"):
        """Create a mock chat completion response."""
        choice = MagicMock()
        choice.finish_reason = finish_reason
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        response.usage = usage
        return response

    def test_no_tools_returns_direct_answer(self):
        broker = MagicMock()
        broker.chat_completion.return_value = self._mock_broker_response(
            content="The answer is 42."
        )

        runtime = BenchmarkRuntime(_make_release(), _null_policy())
        harness = AgentHarness(broker, runtime)
        result = harness.answer("What is the answer?", question_id="q01")

        assert result.answer_text == "The answer is 42."
        assert result.question_id == "q01"
        assert result.tool_calls_made == 0

    def test_tool_call_flow(self):
        """Agent makes a tool call, gets results, then answers."""
        broker = MagicMock()

        # First response: tool call
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "memory_search"
        tc.function.arguments = json.dumps({"query": "metrics"})
        first_response = self._mock_broker_response(
            content=None,
            tool_calls=[tc],
            finish_reason="tool_calls",
        )

        # Second response: final answer
        second_response = self._mock_broker_response(
            content="Based on the search, metrics increased. [test_scope_ep_001]"
        )

        broker.chat_completion.side_effect = [first_response, second_response]

        runtime = BenchmarkRuntime(_make_release(), _base_policy())
        harness = AgentHarness(broker, runtime, max_turns=5)
        result = harness.answer("What happened?", question_id="q01")

        assert result.tool_calls_made == 1
        assert result.answer_text.startswith("Based on the search")
        assert "test_scope_ep_001" in result.cited_refs

    def test_max_turns_enforced(self):
        """Agent should stop after max_turns even if still calling tools."""
        broker = MagicMock()

        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "memory_search"
        tc.function.arguments = json.dumps({"query": "test"})

        # Always return tool call — never produce final answer
        response = self._mock_broker_response(content=None, tool_calls=[tc], finish_reason="tool_calls")
        broker.chat_completion.return_value = response

        runtime = BenchmarkRuntime(_make_release(), _base_policy())
        harness = AgentHarness(broker, runtime, max_turns=3)
        result = harness.answer("What?")

        # Should have stopped after 3 turns
        assert broker.chat_completion.call_count == 3

    def test_tokens_tracked(self):
        broker = MagicMock()
        broker.chat_completion.return_value = self._mock_broker_response(content="Answer.")
        runtime = BenchmarkRuntime(_make_release(), _null_policy())
        harness = AgentHarness(broker, runtime)
        result = harness.answer("Question?")

        assert result.total_prompt_tokens == 100
        assert result.total_completion_tokens == 50


# ---------------------------------------------------------------------------
# Runner Tests
# ---------------------------------------------------------------------------


class TestRunner:
    def _make_study(self) -> StudyManifest:
        from datetime import datetime, timezone
        return StudyManifest(
            study_id="test-study-001",
            scope_ids=["test_scope_01"],
            policy_ids=["null"],
            agent_model="test-model",
            judge_model="test-judge",
            embedding_model="test-embed",
            prompt_set_version="v1",
            scoring_version="v1",
            code_sha="abc123",
            artifact_family_set_version="v1",
            created_at=datetime.now(timezone.utc),
        )

    def _make_scope_dir(self, base: Path) -> Path:
        """Create a minimal test scope directory."""
        import yaml

        scope_dir = base / "test_scope"
        ep_dir = scope_dir / "generated" / "episodes"
        ep_dir.mkdir(parents=True)

        spec = {"scope_id": "test_scope_01", "domain": "test", "description": "Test"}
        with open(scope_dir / "spec.yaml", "w") as f:
            yaml.dump(spec, f)

        for i in range(1, 4):
            (ep_dir / f"signal_{i:03d}.txt").write_text(f"Signal {i}.")
        for i in range(1, 4):
            (ep_dir / f"distractor_alpha_{i:03d}.txt").write_text(f"Distractor {i}.")

        questions = [
            {
                "question_id": "q01",
                "scope_id": "test_scope_01",
                "checkpoint_after": 3,
                "question_type": "longitudinal",
                "prompt": "What happened?",
                "ground_truth": {
                    "canonical_answer": "Metrics.",
                    "key_facts": ["fact1"],
                    "required_evidence_refs": [],
                },
            },
        ]
        with open(scope_dir / "generated" / "questions.json", "w") as f:
            json.dump(questions, f)

        return scope_dir

    def test_run_cell_creates_run_manifest(self, tmp_dir, store):
        from bench.dataset import load_scope
        from bench.runner import StudyRunner

        study = self._make_study()
        scope_dir = self._make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        broker = MagicMock()
        broker._llm_base_url = "http://localhost/v1"
        broker._llm_api_key = "test"
        broker._embed_base_url = "http://localhost"

        # Mock the broker to return a simple answer
        choice = MagicMock()
        choice.finish_reason = "stop"
        msg = MagicMock()
        msg.content = "The answer."
        msg.tool_calls = None
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        response.usage = usage
        broker.chat_completion.return_value = response

        # Mock Synix for bank building
        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=3, cached=0, skipped=0, total_time=1.0,
            snapshot_oid="snap-1", manifest_oid="m-1",
        )
        mock_project.source.return_value = MagicMock()
        mock_release = _make_release()
        mock_project.release.return_value = mock_release

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project
            mock_synix.open_project.return_value = mock_project

            runner = StudyRunner(
                study, store, broker, tmp_dir / "work",
                scope_dirs={"test_scope_01": scope_dir},
                families=["chunks"],
            )
            runs = runner.run_study()

        assert len(runs) == 1
        assert runs[0].status == RunStatus.completed
        assert runs[0].policy_id == "null"

        # Answer should be saved
        answer = store.get_answer(runs[0].run_id, "q01")
        assert answer is not None
        assert answer["answer_text"] == "The answer."

    def test_resume_skips_completed_questions(self, tmp_dir, store):
        from bench.dataset import load_scope
        from bench.runner import StudyRunner

        study = self._make_study()
        scope_dir = self._make_scope_dir(tmp_dir)

        # Pre-save an answer for q01
        run_id = f"run-test_scope_01-null-{study.study_id[:8]}"
        store.save_answer(run_id, "q01", "cp03", {
            "question_id": "q01",
            "answer_text": "Previous answer.",
            "cited_refs": [],
            "tool_calls_made": 0,
        })

        broker = MagicMock()
        broker._llm_base_url = "http://localhost/v1"
        broker._llm_api_key = "test"
        broker._embed_base_url = "http://localhost"

        # If agent is called, it should NOT be — that's what we're testing
        broker.chat_completion.side_effect = RuntimeError("Should not be called")

        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=3, cached=0, skipped=0, total_time=1.0,
            snapshot_oid="snap-1", manifest_oid="m-1",
        )
        mock_project.source.return_value = MagicMock()
        mock_project.release.return_value = _make_release()

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project
            mock_synix.open_project.return_value = mock_project

            runner = StudyRunner(
                study, store, broker, tmp_dir / "work",
                scope_dirs={"test_scope_01": scope_dir},
                families=["chunks"],
            )
            runs = runner.run_study()

        assert len(runs) == 1
        assert runs[0].status == RunStatus.completed
        # Agent should NOT have been called
        broker.chat_completion.assert_not_called()

    def test_events_emitted_for_run(self, tmp_dir, store):
        from bench.runner import StudyRunner

        study = self._make_study()
        scope_dir = self._make_scope_dir(tmp_dir)

        broker = MagicMock()
        broker._llm_base_url = "http://localhost/v1"
        broker._llm_api_key = "test"
        broker._embed_base_url = "http://localhost"

        choice = MagicMock()
        choice.finish_reason = "stop"
        msg = MagicMock()
        msg.content = "Answer."
        msg.tool_calls = None
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        response.usage = usage
        broker.chat_completion.return_value = response

        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=3, cached=0, skipped=0, total_time=1.0,
            snapshot_oid="snap-1", manifest_oid="m-1",
        )
        mock_project.source.return_value = MagicMock()
        mock_project.release.return_value = _make_release()

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project
            mock_synix.open_project.return_value = mock_project

            runner = StudyRunner(
                study, store, broker, tmp_dir / "work",
                scope_dirs={"test_scope_01": scope_dir},
                families=["chunks"],
            )
            runner.run_study()

        events = store.get_events(study_id="test-study-001")
        event_types = [e.event_type for e in events]

        # Should have run lifecycle events
        assert EventType.run_started in event_types
        assert EventType.run_completed in event_types
        assert EventType.checkpoint_started in event_types
        assert EventType.checkpoint_completed in event_types
        assert EventType.question_started in event_types
        assert EventType.question_completed in event_types

        # Plus bank build events
        assert EventType.bank_build_started in event_types
        assert EventType.bank_build_completed in event_types

    def test_bank_reused_across_policies(self, tmp_dir, store):
        """Banks should be built once and reused across policy runs."""
        from datetime import datetime, timezone
        from bench.runner import StudyRunner

        study = StudyManifest(
            study_id="test-study-002",
            scope_ids=["test_scope_01"],
            policy_ids=["null", "policy_base"],
            agent_model="test-model",
            judge_model="test-judge",
            embedding_model="test-embed",
            prompt_set_version="v1",
            scoring_version="v1",
            code_sha="abc123",
            artifact_family_set_version="v1",
            created_at=datetime.now(timezone.utc),
        )
        scope_dir = self._make_scope_dir(tmp_dir)

        broker = MagicMock()
        broker._llm_base_url = "http://localhost/v1"
        broker._llm_api_key = "test"
        broker._embed_base_url = "http://localhost"

        choice = MagicMock()
        choice.finish_reason = "stop"
        msg = MagicMock()
        msg.content = "Answer."
        msg.tool_calls = None
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        response.usage = usage
        broker.chat_completion.return_value = response

        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=3, cached=0, skipped=0, total_time=1.0,
            snapshot_oid="snap-1", manifest_oid="m-1",
        )
        mock_project.source.return_value = MagicMock()
        mock_project.release.return_value = _make_release()

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project
            mock_synix.open_project.return_value = mock_project

            runner = StudyRunner(
                study, store, broker, tmp_dir / "work",
                scope_dirs={"test_scope_01": scope_dir},
                families=["chunks"],
            )
            runs = runner.run_study()

        assert len(runs) == 2

        # Bank should only be built once (first policy), second should hit resume
        build_events = [
            e for e in store.get_events(study_id="test-study-002")
            if e.event_type == EventType.bank_build_started
        ]
        # Only 1 bank build (1 checkpoint) — second policy reuses it
        assert len(build_events) == 1
