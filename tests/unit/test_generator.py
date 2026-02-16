"""Unit tests for datagen generator internals.

These test the pure functions without any LLM calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lens.core.errors import DatasetError
from lens.datagen.generator import (
    _assign_distractor_ids_and_timestamps,
    _compute_distractor_similarity,
    _compute_key_fact_coverage,
    _parse_phase_response,
    _resolve_questions,
    _validate_phase_output,
)
from lens.datagen.spec import (
    DistractorConfig,
    DistractorTheme,
    EpisodeConfig,
    GenerationConfig,
    KeyFact,
    NoiseConfig,
    PhaseArc,
    QuestionGroundTruth,
    QuestionSpec,
    ScenarioConfig,
    ScopeSpec,
    TimelineConfig,
    make_episode_id,
)


@pytest.fixture
def spec() -> ScopeSpec:
    return ScopeSpec(
        scope_id="gen_test_01",
        domain="testing",
        generation=GenerationConfig(temperature=0.7, seed=42),
        episodes=EpisodeConfig(
            count=6,
            timeline=TimelineConfig(start="2024-01-01"),
            target_words=50,
        ),
        scenario=ScenarioConfig(setting="Test", voice="Brief"),
        arc=[
            PhaseArc(id="baseline", episodes="1-3", description="Normal", signal_density="none"),
            PhaseArc(id="signal", episodes="4-6", description="Signal", signal_density="high"),
        ],
        noise=NoiseConfig(),
        key_facts=[
            KeyFact(
                id="fact_a",
                fact="server latency increasing",
                first_appears="signal:1",
                reinforced_in=["signal:3"],
            ),
            KeyFact(
                id="fact_b",
                fact="disk usage critical",
                first_appears="signal:2",
            ),
        ],
        questions=[
            QuestionSpec(
                id="q01",
                checkpoint_after=6,
                type="longitudinal",
                prompt="What patterns emerged?",
                ground_truth=QuestionGroundTruth(
                    canonical_answer="Server latency increased and disk usage hit critical.",
                    key_facts=["fact_a", "fact_b"],
                    evidence=["signal:1", "signal:2"],
                ),
            ),
            QuestionSpec(
                id="q02",
                checkpoint_after=3,
                type="null_hypothesis",
                prompt="What happened on day 2?",
                ground_truth=QuestionGroundTruth(
                    canonical_answer="Normal operations.",
                    key_facts=[],
                    evidence=["baseline:2"],
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# _parse_phase_response
# ---------------------------------------------------------------------------


class TestParsePhaseResponse:
    def test_valid_json(self):
        content = json.dumps({
            "episodes": [
                {"index": 1, "text": "Episode one.", "meta": {}},
                {"index": 2, "text": "Episode two.", "meta": {}},
            ],
            "phase_summary": "Two normal episodes."
        })
        result = _parse_phase_response(content)
        assert len(result["episodes"]) == 2
        assert result["phase_summary"] == "Two normal episodes."

    def test_invalid_json_raises(self):
        with pytest.raises(DatasetError, match="invalid JSON"):
            _parse_phase_response("not json at all")

    def test_missing_episodes_key_raises(self):
        with pytest.raises(DatasetError, match="missing 'episodes'"):
            _parse_phase_response(json.dumps({"phase_summary": "oops"}))

    def test_episodes_not_list_raises(self):
        with pytest.raises(DatasetError, match="must be a list"):
            _parse_phase_response(json.dumps({"episodes": "not a list"}))

    def test_minimal_valid(self):
        result = _parse_phase_response(json.dumps({"episodes": []}))
        assert result["episodes"] == []


# ---------------------------------------------------------------------------
# _validate_phase_output
# ---------------------------------------------------------------------------


class TestValidatePhaseOutput:
    def test_correct_count_no_warnings(self, spec):
        phase = spec.arc[0]  # baseline: 1-3, expects 3 episodes
        parsed = {
            "episodes": [
                {"text": "Normal day one with some filler content padding." + " word" * 20},
                {"text": "Normal day two with some filler content padding." + " word" * 20},
                {"text": "Normal day three with some filler content." + " word" * 20},
            ]
        }
        warnings = _validate_phase_output(parsed, phase, spec)
        assert warnings == []

    def test_wrong_count_warns(self, spec):
        phase = spec.arc[0]  # expects 3
        parsed = {
            "episodes": [
                {"text": "Only one episode here." + " word" * 20},
            ]
        }
        warnings = _validate_phase_output(parsed, phase, spec)
        assert any("expected 3" in w for w in warnings)

    def test_empty_text_warns(self, spec):
        phase = spec.arc[0]
        parsed = {
            "episodes": [
                {"text": ""},
                {"text": "OK text here." + " word" * 20},
                {"text": "Another fine episode." + " word" * 20},
            ]
        }
        warnings = _validate_phase_output(parsed, phase, spec)
        assert any("empty text" in w for w in warnings)

    def test_short_text_warns(self, spec):
        phase = spec.arc[0]  # target_words=50
        parsed = {
            "episodes": [
                {"text": "tiny"},
                {"text": "also tiny"},
                {"text": "very short too"},
            ]
        }
        warnings = _validate_phase_output(parsed, phase, spec)
        # All 3 should warn about word count (< 30% of 50 = 15 words)
        word_warnings = [w for w in warnings if "words" in w]
        assert len(word_warnings) == 3

    def test_key_fact_missing_warns(self, spec):
        phase = spec.arc[1]  # signal phase, expects fact_a in ep 1, fact_b in ep 2
        parsed = {
            "episodes": [
                {"text": "Completely unrelated content about nothing." + " word" * 20},
                {"text": "More irrelevant stuff that misses everything." + " word" * 20},
                {"text": "Server latency increasing noticeably today." + " word" * 20},
            ]
        }
        warnings = _validate_phase_output(parsed, phase, spec)
        # fact_a should warn for episode 1, fact_b should warn for episode 2
        fact_warnings = [w for w in warnings if "key fact" in w]
        assert len(fact_warnings) >= 1

    def test_key_fact_present_no_warning(self, spec):
        phase = spec.arc[1]  # signal phase
        parsed = {
            "episodes": [
                {"text": "The server latency is increasing beyond normal." + " word" * 20},
                {"text": "Disk usage has become critical on the cluster." + " word" * 20},
                {"text": "Server latency still increasing, confirmed pattern." + " word" * 20},
            ]
        }
        warnings = _validate_phase_output(parsed, phase, spec)
        fact_warnings = [w for w in warnings if "key fact" in w]
        assert fact_warnings == []


# ---------------------------------------------------------------------------
# _resolve_questions
# ---------------------------------------------------------------------------


class TestResolveQuestions:
    def test_resolves_all_questions(self, spec):
        questions = _resolve_questions(spec)
        assert len(questions) == 2

    def test_question_structure(self, spec):
        questions = _resolve_questions(spec)
        q1 = questions[0]
        assert q1["question_id"] == "q01"
        assert q1["scope_id"] == "gen_test_01"
        assert q1["checkpoint_after"] == 6
        assert q1["question_type"] == "longitudinal"
        assert q1["prompt"] == "What patterns emerged?"

    def test_resolves_evidence_refs(self, spec):
        questions = _resolve_questions(spec)
        q1 = questions[0]
        refs = q1["ground_truth"]["required_evidence_refs"]
        # signal:1 = global 4 -> gen_test_01_ep_004
        # signal:2 = global 5 -> gen_test_01_ep_005
        assert "gen_test_01_ep_004" in refs
        assert "gen_test_01_ep_005" in refs

    def test_resolves_key_facts_to_text(self, spec):
        questions = _resolve_questions(spec)
        q1 = questions[0]
        facts = q1["ground_truth"]["key_facts"]
        assert "server latency increasing" in facts
        assert "disk usage critical" in facts

    def test_empty_key_facts(self, spec):
        questions = _resolve_questions(spec)
        q2 = questions[1]
        assert q2["ground_truth"]["key_facts"] == []

    def test_question_matches_dataset_schema(self, spec):
        """Resolved questions should match the dataset question schema."""
        from lens.datasets.schema import QUESTION_REQUIRED_KEYS, GROUND_TRUTH_REQUIRED_KEYS

        questions = _resolve_questions(spec)
        for q in questions:
            for key in QUESTION_REQUIRED_KEYS:
                assert key in q, f"Missing key: {key}"
            for key in GROUND_TRUTH_REQUIRED_KEYS:
                assert key in q["ground_truth"], f"Missing ground_truth key: {key}"


# ---------------------------------------------------------------------------
# _compute_key_fact_coverage
# ---------------------------------------------------------------------------


class TestComputeKeyFactCoverage:
    def test_full_coverage(self, spec):
        episodes = [
            {"episode_id": "gen_test_01_ep_001", "text": "Normal day."},
            {"episode_id": "gen_test_01_ep_002", "text": "Another day."},
            {"episode_id": "gen_test_01_ep_003", "text": "Still normal."},
            {"episode_id": "gen_test_01_ep_004", "text": "The server latency is increasing."},
            {"episode_id": "gen_test_01_ep_005", "text": "Disk usage has become critical."},
            {"episode_id": "gen_test_01_ep_006", "text": "Server latency still increasing."},
        ]
        manifest: dict = {"key_fact_coverage": {}, "validation": {"key_fact_hit_rate": 0.0}}
        _compute_key_fact_coverage(spec, episodes, manifest)

        assert manifest["validation"]["key_fact_hit_rate"] > 0.0
        assert "fact_a" in manifest["key_fact_coverage"]
        assert "fact_b" in manifest["key_fact_coverage"]

    def test_no_coverage(self, spec):
        episodes = [
            {"episode_id": "gen_test_01_ep_001", "text": "Nothing."},
            {"episode_id": "gen_test_01_ep_002", "text": "Nothing."},
            {"episode_id": "gen_test_01_ep_003", "text": "Nothing."},
            {"episode_id": "gen_test_01_ep_004", "text": "Nothing relevant here."},
            {"episode_id": "gen_test_01_ep_005", "text": "Nothing relevant either."},
            {"episode_id": "gen_test_01_ep_006", "text": "Nothing."},
        ]
        manifest: dict = {"key_fact_coverage": {}, "validation": {"key_fact_hit_rate": 0.0}}
        _compute_key_fact_coverage(spec, episodes, manifest)

        assert manifest["validation"]["key_fact_hit_rate"] == 0.0

    def test_partial_coverage(self, spec):
        episodes = [
            {"episode_id": "gen_test_01_ep_001", "text": "Nothing."},
            {"episode_id": "gen_test_01_ep_002", "text": "Nothing."},
            {"episode_id": "gen_test_01_ep_003", "text": "Nothing."},
            {"episode_id": "gen_test_01_ep_004", "text": "The server latency is increasing."},
            {"episode_id": "gen_test_01_ep_005", "text": "All good here."},  # fact_b missing
            {"episode_id": "gen_test_01_ep_006", "text": "Nothing."},  # fact_a reinforcement missing
        ]
        manifest: dict = {"key_fact_coverage": {}, "validation": {"key_fact_hit_rate": 0.0}}
        _compute_key_fact_coverage(spec, episodes, manifest)

        rate = manifest["validation"]["key_fact_hit_rate"]
        assert 0.0 < rate < 1.0

    def test_coverage_records_target_episodes(self, spec):
        episodes = [
            {"episode_id": f"gen_test_01_ep_{i:03d}", "text": "text"}
            for i in range(1, 7)
        ]
        manifest: dict = {"key_fact_coverage": {}, "validation": {"key_fact_hit_rate": 0.0}}
        _compute_key_fact_coverage(spec, episodes, manifest)

        cov_a = manifest["key_fact_coverage"]["fact_a"]
        # fact_a: first_appears=signal:1 (ep 4), reinforced_in=signal:3 (ep 6)
        assert "gen_test_01_ep_004" in cov_a["target_episodes"]
        assert "gen_test_01_ep_006" in cov_a["target_episodes"]

        cov_b = manifest["key_fact_coverage"]["fact_b"]
        # fact_b: first_appears=signal:2 (ep 5)
        assert "gen_test_01_ep_005" in cov_b["target_episodes"]


# ---------------------------------------------------------------------------
# Overlapping phase dedup
# ---------------------------------------------------------------------------


class TestOverlappingPhaseDedup:
    """Test that overlapping arc phases deduplicate episodes correctly."""

    def test_overlapping_phases_dedup_by_episode_id(self):
        """Simulate the episode_map dedup logic the generator uses.

        When phases overlap (e.g., early_signal: 3-5, red_herring: 5-7),
        the later phase should win for shared episodes.
        """
        scope_id = "overlap_test"

        # Simulate what generate_scope does internally with episode_map
        episode_map: dict[str, dict] = {}

        # Phase 1 produces eps 1-5
        for i in range(1, 6):
            eid = make_episode_id(scope_id, i)
            episode_map[eid] = {
                "episode_id": eid,
                "text": f"phase_a content for ep {i}",
            }

        # Phase 2 overlaps: produces eps 5-7 (ep 5 shared)
        for i in range(5, 8):
            eid = make_episode_id(scope_id, i)
            episode_map[eid] = {
                "episode_id": eid,
                "text": f"phase_b content for ep {i}",
            }

        all_episodes = [episode_map[eid] for eid in sorted(episode_map)]

        # 7 unique episodes, not 8
        assert len(all_episodes) == 7

        # Episode 5 should have phase_b content (later phase wins)
        ep5 = [e for e in all_episodes if e["episode_id"] == f"{scope_id}_ep_005"][0]
        assert "phase_b" in ep5["text"]

        # No duplicates
        ids = [e["episode_id"] for e in all_episodes]
        assert len(ids) == len(set(ids))

    def test_three_way_overlap(self):
        """Three phases overlap on the same episode — last phase wins."""
        scope_id = "tri_overlap"
        episode_map: dict[str, dict] = {}

        # Phase A: 1-3
        for i in range(1, 4):
            eid = make_episode_id(scope_id, i)
            episode_map[eid] = {"episode_id": eid, "text": f"A_{i}"}

        # Phase B: 3-5 (overlaps ep 3)
        for i in range(3, 6):
            eid = make_episode_id(scope_id, i)
            episode_map[eid] = {"episode_id": eid, "text": f"B_{i}"}

        # Phase C: 5-7 (overlaps ep 5)
        for i in range(5, 8):
            eid = make_episode_id(scope_id, i)
            episode_map[eid] = {"episode_id": eid, "text": f"C_{i}"}

        all_episodes = [episode_map[eid] for eid in sorted(episode_map)]

        assert len(all_episodes) == 7
        ep3 = [e for e in all_episodes if "ep_003" in e["episode_id"]][0]
        assert ep3["text"] == "B_3"
        ep5 = [e for e in all_episodes if "ep_005" in e["episode_id"]][0]
        assert ep5["text"] == "C_5"


# ---------------------------------------------------------------------------
# Compiled suite validation
# ---------------------------------------------------------------------------


class TestCompiledSuiteValidation:
    """Verify the compiled v0.1 suite is loadable and valid."""

    @pytest.fixture
    def suite_path(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent / "datasets" / "suites" / "v0.1.json"

    def test_suite_exists(self, suite_path: Path):
        assert suite_path.exists(), f"Compiled suite not found at {suite_path}"

    def test_suite_schema_valid(self, suite_path: Path):
        if not suite_path.exists():
            pytest.skip("Suite not compiled yet")
        from lens.datasets.schema import validate_dataset

        data = json.loads(suite_path.read_text())
        errors = validate_dataset(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_suite_loadable_by_runner(self, suite_path: Path):
        """Suite must be loadable by the standard dataset loader."""
        if not suite_path.exists():
            pytest.skip("Suite not compiled yet")
        from lens.datasets.loader import load_dataset, load_episodes, load_questions

        data = load_dataset(str(suite_path))
        episodes_by_scope = load_episodes(data)
        questions = load_questions(data)

        assert "cascading_failure_01" in episodes_by_scope
        assert len(episodes_by_scope["cascading_failure_01"]) == 30
        assert len(questions) == 4
        assert all(e.scope_id == "cascading_failure_01" for e in episodes_by_scope["cascading_failure_01"])

    def test_suite_no_duplicate_ids(self, suite_path: Path):
        if not suite_path.exists():
            pytest.skip("Suite not compiled yet")

        data = json.loads(suite_path.read_text())
        ep_ids = [ep["episode_id"] for s in data["scopes"] for ep in s["episodes"]]
        assert len(ep_ids) == len(set(ep_ids)), "Duplicate episode IDs found"

        q_ids = [q["question_id"] for q in data["questions"]]
        assert len(q_ids) == len(set(q_ids)), "Duplicate question IDs found"

    def test_suite_checkpoints_are_reachable(self, suite_path: Path):
        """All question checkpoint_after values should be <= total episodes in scope."""
        if not suite_path.exists():
            pytest.skip("Suite not compiled yet")

        data = json.loads(suite_path.read_text())
        scope_ep_count = {
            s["scope_id"]: len(s["episodes"]) for s in data["scopes"]
        }
        for q in data["questions"]:
            sid = q["scope_id"]
            cp = q["checkpoint_after"]
            assert cp <= scope_ep_count[sid], (
                f"Question {q['question_id']} has checkpoint_after={cp} "
                f"but scope {sid} only has {scope_ep_count[sid]} episodes"
            )

    def test_suite_question_types(self, suite_path: Path):
        if not suite_path.exists():
            pytest.skip("Suite not compiled yet")

        data = json.loads(suite_path.read_text())
        types = {q["question_type"] for q in data["questions"]}
        assert "longitudinal" in types
        assert "null_hypothesis" in types
        assert "action_recommendation" in types


# ---------------------------------------------------------------------------
# Distractor similarity scoring
# ---------------------------------------------------------------------------


class TestComputeDistractorSimilarity:
    def test_no_overlap(self):
        text = "DNS zone transfers completed successfully today."
        facts = ["connection pool exhaustion", "geo-lookup latency increasing"]
        sim = _compute_distractor_similarity(text, facts)
        assert sim == 0.0

    def test_full_overlap(self):
        text = "The geo-lookup API latency is increasing rapidly today."
        facts = ["geo-lookup API latency increasing"]
        sim = _compute_distractor_similarity(text, facts)
        assert sim == 1.0

    def test_partial_overlap(self):
        text = "Service latency was observed during the migration."
        facts = ["API latency increasing"]  # 3 words, "latency" overlaps = 1/3
        sim = _compute_distractor_similarity(text, facts)
        assert 0.0 < sim < 1.0

    def test_empty_facts(self):
        sim = _compute_distractor_similarity("any text", [])
        assert sim == 0.0

    def test_max_across_facts(self):
        text = "Connection pool metrics are normal."
        facts = ["connection pool exhaustion", "geo-lookup latency"]
        sim = _compute_distractor_similarity(text, facts)
        # "connection" and "pool" overlap with first fact (2/3),
        # no overlap with second fact
        assert sim > 0.5


# ---------------------------------------------------------------------------
# Distractor ID and timestamp assignment
# ---------------------------------------------------------------------------


class TestAssignDistractorIdsAndTimestamps:
    def test_assigns_dx_ids(self):
        spec = ScopeSpec(
            scope_id="test_01",
            episodes=EpisodeConfig(
                count=10,
                timeline=TimelineConfig(start="2024-01-01", interval="1d"),
            ),
        )
        episodes = [{"text": f"Ep {i}", "meta": {}} for i in range(3)]
        _assign_distractor_ids_and_timestamps(spec, episodes)

        assert episodes[0]["episode_id"] == "test_01_dx_001"
        assert episodes[1]["episode_id"] == "test_01_dx_002"
        assert episodes[2]["episode_id"] == "test_01_dx_003"

    def test_assigns_scope_id(self):
        spec = ScopeSpec(
            scope_id="my_scope",
            episodes=EpisodeConfig(
                count=5,
                timeline=TimelineConfig(start="2024-01-01"),
            ),
        )
        episodes = [{"text": "test", "meta": {}}]
        _assign_distractor_ids_and_timestamps(spec, episodes)
        assert episodes[0]["scope_id"] == "my_scope"

    def test_timestamps_span_signal_timeline(self):
        spec = ScopeSpec(
            scope_id="test_01",
            episodes=EpisodeConfig(
                count=10,
                timeline=TimelineConfig(start="2024-01-01", interval="1d"),
            ),
        )
        episodes = [{"text": f"Ep {i}", "meta": {}} for i in range(5)]
        _assign_distractor_ids_and_timestamps(spec, episodes)

        # First distractor should start at signal start
        assert episodes[0]["timestamp"].startswith("2024-01-01")
        # Last distractor should end at signal end (day 9 = Jan 10)
        assert episodes[4]["timestamp"].startswith("2024-01-10")
        # Timestamps should be in order
        timestamps = [ep["timestamp"] for ep in episodes]
        assert timestamps == sorted(timestamps)

    def test_empty_list_noop(self):
        spec = ScopeSpec(
            scope_id="test_01",
            episodes=EpisodeConfig(
                count=5,
                timeline=TimelineConfig(start="2024-01-01"),
            ),
        )
        episodes: list[dict] = []
        _assign_distractor_ids_and_timestamps(spec, episodes)
        assert episodes == []

    def test_distractor_episode_type_preserved(self):
        spec = ScopeSpec(
            scope_id="test_01",
            episodes=EpisodeConfig(
                count=5,
                timeline=TimelineConfig(start="2024-01-01"),
            ),
        )
        episodes = [
            {"text": "test", "meta": {"episode_type": "distractor", "theme": "dns"}},
        ]
        _assign_distractor_ids_and_timestamps(spec, episodes)
        assert episodes[0]["meta"]["episode_type"] == "distractor"
        assert episodes[0]["meta"]["theme"] == "dns"


# ---------------------------------------------------------------------------
# Signal episode meta tagging
# ---------------------------------------------------------------------------


class TestSignalEpisodeMetaTagging:
    """Verify that the episode_map building logic tags signal episodes."""

    def test_signal_episodes_get_tagged(self):
        """Simulate the episode building logic from generate_scope."""
        scope_id = "tag_test"
        phase_episodes = [
            {"text": "Normal day one." + " word" * 20, "meta": {}},
            {"text": "Normal day two." + " word" * 20, "meta": {"existing": "data"}},
        ]

        episode_map: dict[str, dict] = {}
        start = 1
        for i, ep in enumerate(phase_episodes):
            global_idx = start + i
            eid = make_episode_id(scope_id, global_idx)
            meta = ep.get("meta", {})
            meta["episode_type"] = "signal"
            episode_map[eid] = {
                "episode_id": eid,
                "scope_id": scope_id,
                "text": ep.get("text", ""),
                "meta": meta,
            }

        for ep in episode_map.values():
            assert ep["meta"]["episode_type"] == "signal"
        # Existing meta preserved
        assert episode_map[f"{scope_id}_ep_002"]["meta"]["existing"] == "data"
