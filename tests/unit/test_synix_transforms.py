"""Tests for lens.datagen.synix.transforms — synix transform classes.

These tests mock synix's Artifact and LLM client to test transform logic
without requiring synix to be installed in the test environment.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add synix directory to path
SYNIX_DIR = str(Path(__file__).resolve().parents[2] / "src" / "lens" / "datagen" / "synix")
sys.path.insert(0, SYNIX_DIR)

import spec_utils  # noqa: E402


# ---------------------------------------------------------------------------
# Mock synix types (so we don't need synix installed for testing)
# ---------------------------------------------------------------------------

@dataclass
class MockArtifact:
    label: str = ""
    artifact_type: str = ""
    content: str = ""
    artifact_id: str = ""
    input_ids: list = field(default_factory=list)
    prompt_id: str | None = None
    model_config: dict | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.artifact_id:
            import hashlib
            self.artifact_id = f"sha256:{hashlib.sha256(self.content.encode()).hexdigest()}"


@dataclass
class MockLLMResponse:
    content: str = ""
    model: str = "test-model"
    input_tokens: int = 100
    output_tokens: int = 200
    total_tokens: int = 300


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

MINIMAL_SPEC_RAW = {
    "scope_id": "test_scope",
    "domain": "test_domain",
    "episodes": {
        "count": 10,
        "timeline": {"start": "2024-01-01", "interval": "1d"},
        "format": "Daily log",
        "target_words": 50,
    },
    "scenario": {"setting": "Test scenario", "voice": "Terse style"},
    "arc": [
        {"id": "baseline", "episodes": "1-5", "description": "Normal", "signal_density": "none"},
        {"id": "signal", "episodes": "6-10", "description": "Degradation", "signal_density": "low"},
    ],
    "noise": {"description": "Normal noise", "examples": []},
    "key_facts": [
        {"id": "kf1", "fact": "latency increasing", "first_appears": "signal:1", "reinforced_in": ["signal:3"]},
    ],
    "questions": [
        {
            "id": "q01",
            "checkpoint_after": 8,
            "type": "longitudinal",
            "prompt": "What patterns?",
            "ground_truth": {
                "canonical_answer": "Latency increasing",
                "key_facts": ["kf1"],
                "evidence": ["signal:1"],
            },
        },
    ],
    "distractors": {
        "count": 5,
        "target_words": 50,
        "themes": [
            {"id": "dns", "scenario": "DNS migration.", "excluded_terms": ["latency"]},
        ],
        "seed": 99,
        "max_similarity": 0.3,
    },
}


def _make_spec_artifact(raw=None):
    """Create a mock spec artifact."""
    spec = spec_utils.parse_spec(raw or MINIMAL_SPEC_RAW)
    content = json.dumps(spec, indent=2)
    return MockArtifact(
        label="spec",
        artifact_type="scope_spec",
        content=content,
        metadata={"scope_id": spec["scope_id"]},
    )


def _make_signal_episode(label: str, text: str, phase: str = "signal", metadata: dict | None = None):
    """Create a mock signal episode artifact."""
    return MockArtifact(
        label=label,
        artifact_type="signal_episode",
        content=text,
        metadata={
            "episode_id": label,
            "scope_id": "test_scope",
            "timestamp": "2024-01-01T10:00:00",
            "phase": phase,
            "signal_density": "low",
            "episode_type": "signal",
            **(metadata or {}),
        },
    )


def _make_signal_outline(episodes: list[dict]):
    """Create a mock signal_outline artifact."""
    content = json.dumps({"episodes": episodes}, indent=2)
    return MockArtifact(
        label="signal_outline",
        artifact_type="signal_outline",
        content=content,
        metadata={"episode_count": len(episodes)},
    )


def _make_distractor_outline(theme: str, episodes: list[dict], theme_index: int = 0):
    """Create a mock distractor_outline artifact."""
    content = json.dumps({"episodes": episodes}, indent=2)
    return MockArtifact(
        label=f"distractor_outline_{theme}",
        artifact_type="distractor_outline",
        content=content,
        metadata={"theme": theme, "theme_index": theme_index, "episode_count": len(episodes)},
    )


# ---------------------------------------------------------------------------
# Patch synix imports so transforms module can load without synix installed
# ---------------------------------------------------------------------------

# Create mock synix modules — v0.11 API: Source and Transform at top level
mock_synix = MagicMock()


class _MockSource:
    def __init__(self, name=None, *, dir=None, config=None):
        self.name = name
        self.dir = dir
        self.config = config or {}

    def load(self, config):
        return []


class _MockTransform:
    prompt_name = None

    def __init__(self, name=None, *, depends_on=None, config=None, context_budget=None):
        self.name = name
        self.depends_on = depends_on or []
        self.config = config or {}

    def execute(self, inputs, config):
        return []

    def split(self, inputs, config):
        return [(inputs, {})]


mock_synix.Source = _MockSource
mock_synix.Transform = _MockTransform

mock_synix_llm = MagicMock()
mock_synix_llm._get_llm_client = MagicMock()
mock_synix_llm._logged_complete = MagicMock()

mock_synix_models = MagicMock()
mock_synix_models.Artifact = MockArtifact

# Install mocks
sys.modules["synix"] = mock_synix
sys.modules["synix.build"] = MagicMock()
sys.modules["synix.build.transforms"] = MagicMock()
sys.modules["synix.build.llm_transforms"] = mock_synix_llm
sys.modules["synix.core"] = MagicMock()
sys.modules["synix.core.models"] = mock_synix_models

# Now import the transforms module (it will use our mocks)
if "transforms" in sys.modules:
    del sys.modules["transforms"]
import transforms as transforms_mod  # noqa: E402


# ---------------------------------------------------------------------------
# LoadSpec (Source)
# ---------------------------------------------------------------------------


class TestLoadSpec:
    def test_loads_spec_from_yaml(self, tmp_path):
        import yaml
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(MINIMAL_SPEC_RAW))

        t = transforms_mod.LoadSpec()
        result = t.load({"source_dir": str(tmp_path)})

        assert len(result) == 1
        art = result[0]
        assert art.artifact_type == "scope_spec"
        assert art.label == "spec"

        spec_data = json.loads(art.content)
        assert spec_data["scope_id"] == "test_scope"

    def test_spec_hash_in_input_ids(self, tmp_path):
        import yaml
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(MINIMAL_SPEC_RAW))

        t = transforms_mod.LoadSpec()
        result = t.load({"source_dir": str(tmp_path)})
        assert result[0].input_ids[0].startswith("sha256:")

    def test_invalid_spec_raises(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("not: valid\n")  # Missing scope_id

        t = transforms_mod.LoadSpec()
        with pytest.raises(ValueError, match="scope_id"):
            t.load({"source_dir": str(tmp_path)})


# ---------------------------------------------------------------------------
# PlanOutline
# ---------------------------------------------------------------------------


class TestPlanOutline:
    def test_split_signal_plus_distractor_groups(self):
        spec_art = _make_spec_artifact()
        t = transforms_mod.PlanOutline()
        groups = t.split([spec_art], {})
        # 1 signal group + 1 distractor theme group = 2
        assert len(groups) == 2
        assert groups[0][1]["_plan_type"] == "signal"
        assert groups[1][1]["_plan_type"] == "distractor"
        assert groups[1][1]["_theme_idx"] == 0

    def test_split_no_distractors(self):
        raw = {**MINIMAL_SPEC_RAW, "distractors": None}
        spec_art = _make_spec_artifact(raw)
        t = transforms_mod.PlanOutline()
        groups = t.split([spec_art], {})
        assert len(groups) == 1
        assert groups[0][1]["_plan_type"] == "signal"

    def test_plan_signal_returns_outline(self):
        spec_art = _make_spec_artifact()

        # Mock LLM returns structured episode data sheets
        outline_data = {"episodes": [
            {"index": i + 1, "date": f"2024-01-{i + 1:02d}",
             "metrics": {"checkout": {"p99": 200 + i * 10}},
             "events": ["CDN cache 95%"], "on_call": "Routine."}
            for i in range(10)
        ]}
        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.reset_mock(side_effect=True)
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(
            content=json.dumps(outline_data)
        )

        t = transforms_mod.PlanOutline()
        result = t.execute([spec_art], {"_plan_type": "signal", "llm_config": {}})

        assert len(result) == 1
        assert result[0].artifact_type == "signal_outline"
        assert result[0].label == "signal_outline"
        parsed = json.loads(result[0].content)
        assert len(parsed["episodes"]) == 10

    def test_plan_distractor_returns_outline(self):
        spec_art = _make_spec_artifact()

        outline_data = {"episodes": [
            {"index": 1, "theme": "dns", "date": "2024-01-15",
             "metrics": {"zones_migrated": 120},
             "events": ["BIND sync"], "notes": "Normal."}
        ]}
        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.reset_mock(side_effect=True)
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(
            content=json.dumps(outline_data)
        )

        t = transforms_mod.PlanOutline()
        result = t.execute([spec_art], {
            "_plan_type": "distractor",
            "_theme_idx": 0,
            "_theme_count": 1,
            "llm_config": {},
        })

        assert len(result) == 1
        assert result[0].artifact_type == "distractor_outline"
        assert result[0].metadata["theme"] == "dns"

    def test_plan_signal_handles_parse_failure(self):
        spec_art = _make_spec_artifact()

        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.reset_mock(side_effect=True)
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(content="not json")

        t = transforms_mod.PlanOutline()
        result = t.execute([spec_art], {"_plan_type": "signal", "llm_config": {}})

        assert len(result) == 1
        parsed = json.loads(result[0].content)
        assert parsed["episodes"] == []


# ---------------------------------------------------------------------------
# RenderSignalEpisodes
# ---------------------------------------------------------------------------


class TestRenderSignalEpisodes:
    def test_split_returns_one_group_per_brief(self):
        spec_art = _make_spec_artifact()
        outline = _make_signal_outline([
            {"index": 1, "date": "2024-01-01", "metrics": {}},
            {"index": 2, "date": "2024-01-02", "metrics": {}},
            {"index": 3, "date": "2024-01-03", "metrics": {}},
        ])

        t = transforms_mod.RenderSignalEpisodes()
        groups = t.split([spec_art, outline], {})
        assert len(groups) == 3
        assert groups[0][1]["_brief_idx"] == 0
        assert groups[2][1]["_brief_idx"] == 2

    def test_renders_episode_from_brief(self):
        spec_art = _make_spec_artifact()
        outline = _make_signal_outline([
            {"index": 1, "date": "2024-01-01",
             "metrics": {"checkout": {"requests": 50000, "p99": 200}},
             "events": ["CDN cache 95%"], "on_call": "Routine."},
        ])

        rendered_text = (
            "## 2024-01-01 Daily API Gateway Summary\n"
            "- /checkout: 50K | p99: 200ms | err: 0.1%\n"
            "- CDN cache hit rate 95%\n"
            "- On-call: Routine."
        )
        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.reset_mock(side_effect=True)
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(content=rendered_text)

        t = transforms_mod.RenderSignalEpisodes()
        result = t.execute([spec_art, outline], {"_brief_idx": 0, "llm_config": {}})

        assert len(result) == 1
        assert result[0].artifact_type == "signal_episode"
        assert result[0].label == "test_scope_ep_001"
        assert "checkout" in result[0].content

    def test_renderer_prompt_has_no_key_facts(self):
        """Critical test: renderer must NOT see key facts or arc descriptions."""
        spec_art = _make_spec_artifact()
        outline = _make_signal_outline([
            {"index": 6, "date": "2024-01-06",
             "metrics": {"geo_lookup": {"p99": 600}},
             "events": [], "on_call": "Routine."},
        ])

        captured_messages = []
        def capture_complete(client, config, *, messages, artifact_desc):
            captured_messages.extend(messages)
            return MockLLMResponse(content="## Log entry\n- metrics here")

        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.reset_mock(side_effect=True)
        mock_synix_llm._logged_complete.side_effect = capture_complete

        t = transforms_mod.RenderSignalEpisodes()
        t.execute([spec_art, outline], {"_brief_idx": 0, "llm_config": {}})

        # Combine all message content
        all_content = " ".join(m["content"] for m in captured_messages)

        # Must NOT contain key fact text
        assert "latency increasing" not in all_content.lower()
        # Must NOT contain arc descriptions
        assert "Degradation" not in all_content
        # Must NOT contain signal density labels
        assert "signal_density" not in all_content
        # Must NOT contain question text
        assert "What patterns" not in all_content

    def test_assigns_correct_phase_metadata(self):
        spec_art = _make_spec_artifact()
        outline = _make_signal_outline([
            {"index": 1, "date": "2024-01-01", "metrics": {}},
        ])

        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.reset_mock(side_effect=True)
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(content="log text")

        t = transforms_mod.RenderSignalEpisodes()
        result = t.execute([spec_art, outline], {"_brief_idx": 0, "llm_config": {}})

        assert result[0].metadata["phase"] == "baseline"
        assert result[0].metadata["signal_density"] == "none"


# ---------------------------------------------------------------------------
# RenderDistractorEpisodes
# ---------------------------------------------------------------------------


class TestRenderDistractorEpisodes:
    def test_split_returns_one_group_per_brief(self):
        spec_art = _make_spec_artifact()
        outline = _make_distractor_outline("dns", [
            {"index": 1, "theme": "dns", "metrics": {}},
            {"index": 2, "theme": "dns", "metrics": {}},
        ])

        t = transforms_mod.RenderDistractorEpisodes()
        groups = t.split([spec_art, outline], {})
        assert len(groups) == 2
        assert groups[0][1]["_theme"] == "dns"

    def test_split_no_outlines_returns_no_distractors(self):
        spec_art = _make_spec_artifact()
        t = transforms_mod.RenderDistractorEpisodes()
        groups = t.split([spec_art], {})
        assert len(groups) == 1
        assert groups[0][1].get("_no_distractors") is True

    def test_renders_distractor_episode(self):
        spec_art = _make_spec_artifact()
        outline = _make_distractor_outline("dns", [
            {"index": 1, "theme": "dns", "date": "2024-01-15",
             "metrics": {"zones_migrated": 120},
             "events": ["BIND sync"], "notes": "Normal."},
        ])

        rendered_text = "## DNS Migration Log\n- Zones migrated: 120\n- BIND sync complete"
        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.reset_mock(side_effect=True)
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(content=rendered_text)

        t = transforms_mod.RenderDistractorEpisodes()
        result = t.execute([spec_art, outline], {
            "_outline_label": f"distractor_outline_dns",
            "_brief_idx": 0,
            "_theme": "dns",
            "_theme_idx": 0,
            "llm_config": {},
        })

        assert len(result) == 1
        assert result[0].artifact_type == "distractor_episode"
        assert result[0].label == "test_scope_dx_dns_001"
        assert result[0].metadata["theme"] == "dns"

    def test_no_distractors_returns_empty(self):
        spec_art = _make_spec_artifact()
        t = transforms_mod.RenderDistractorEpisodes()
        result = t.execute([spec_art], {"_no_distractors": True})
        assert result == []

    def test_renderer_prompt_has_no_key_facts(self):
        """Critical test: distractor renderer must NOT see key facts."""
        spec_art = _make_spec_artifact()
        outline = _make_distractor_outline("dns", [
            {"index": 1, "theme": "dns", "metrics": {"zones": 100}},
        ])

        captured_messages = []
        def capture_complete(client, config, *, messages, artifact_desc):
            captured_messages.extend(messages)
            return MockLLMResponse(content="## DNS Log\n- zones: 100")

        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.reset_mock(side_effect=True)
        mock_synix_llm._logged_complete.side_effect = capture_complete

        t = transforms_mod.RenderDistractorEpisodes()
        t.execute([spec_art, outline], {
            "_outline_label": "distractor_outline_dns",
            "_brief_idx": 0,
            "_theme": "dns",
            "_theme_idx": 0,
            "llm_config": {},
        })

        all_content = " ".join(m["content"] for m in captured_messages)
        assert "latency increasing" not in all_content.lower()
        assert "Degradation" not in all_content


# ---------------------------------------------------------------------------
# ResolveQuestions
# ---------------------------------------------------------------------------


class TestResolveQuestions:
    def test_resolves_questions(self):
        spec_art = _make_spec_artifact()
        signal_ep = _make_signal_episode("test_scope_ep_006", "Latency is increasing in the system")

        t = transforms_mod.ResolveQuestions()
        result = t.execute([spec_art, signal_ep], {})

        assert len(result) == 1
        q_art = result[0]
        assert q_art.artifact_type == "question"
        assert q_art.label == "q_q01"

        q_data = json.loads(q_art.content)
        assert q_data["question_id"] == "q01"
        assert q_data["question_type"] == "longitudinal"
        assert q_data["checkpoint_after"] == 8

    def test_evidence_refs_resolved(self):
        spec_art = _make_spec_artifact()
        signal_ep = _make_signal_episode("test_scope_ep_006", "text")

        t = transforms_mod.ResolveQuestions()
        result = t.execute([spec_art, signal_ep], {})

        q_data = json.loads(result[0].content)
        assert "test_scope_ep_006" in q_data["ground_truth"]["required_evidence_refs"]

    def test_key_facts_mapped_to_text(self):
        spec_art = _make_spec_artifact()
        t = transforms_mod.ResolveQuestions()
        result = t.execute([spec_art], {})

        q_data = json.loads(result[0].content)
        assert "latency increasing" in q_data["ground_truth"]["key_facts"]


# ---------------------------------------------------------------------------
# AuditKeyFacts
# ---------------------------------------------------------------------------


class TestAuditKeyFacts:
    def test_coverage_found(self):
        spec_art = _make_spec_artifact()
        # Episode 6 (signal:1) should contain "latency increasing"
        ep6 = _make_signal_episode("test_scope_ep_006", "The latency is increasing across the board")
        # Episode 8 (signal:3) should also contain it
        ep8 = _make_signal_episode("test_scope_ep_008", "latency increasing confirmed in all regions")

        t = transforms_mod.AuditKeyFacts()
        result = t.execute([spec_art, ep6, ep8], {})

        assert len(result) == 1
        assert result[0].artifact_type == "audit"
        assert result[0].label == "key-fact-audit"

        audit = json.loads(result[0].content)
        assert audit["hit_rate"] == 1.0
        assert "kf1" in audit["key_fact_coverage"]

    def test_coverage_missing(self):
        spec_art = _make_spec_artifact()
        # Episode text doesn't mention the key fact
        ep6 = _make_signal_episode("test_scope_ep_006", "Everything is normal today")
        ep8 = _make_signal_episode("test_scope_ep_008", "Still all clear")

        t = transforms_mod.AuditKeyFacts()
        result = t.execute([spec_art, ep6, ep8], {})

        audit = json.loads(result[0].content)
        assert audit["hit_rate"] == 0.0

    def test_partial_coverage(self):
        spec_art = _make_spec_artifact()
        # Only ep6 has the fact, ep8 doesn't
        ep6 = _make_signal_episode("test_scope_ep_006", "latency is increasing in all services")
        ep8 = _make_signal_episode("test_scope_ep_008", "Everything looks fine today")

        t = transforms_mod.AuditKeyFacts()
        result = t.execute([spec_art, ep6, ep8], {})

        audit = json.loads(result[0].content)
        assert 0 < audit["hit_rate"] < 1.0


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_find_artifact(self):
        art = MockArtifact(label="spec", artifact_type="scope_spec", content="{}")
        found = transforms_mod._find_artifact([art], "scope_spec")
        assert found is art

    def test_find_artifact_missing(self):
        with pytest.raises(ValueError, match="No artifact of type"):
            transforms_mod._find_artifact([], "scope_spec")

    def test_parse_json_response(self):
        result = transforms_mod._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_markdown_fences(self):
        result = transforms_mod._parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_invalid(self):
        with pytest.raises(json.JSONDecodeError):
            transforms_mod._parse_json_response("not json")
