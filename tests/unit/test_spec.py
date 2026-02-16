from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

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
    get_checkpoints,
    get_key_facts_for_phase,
    get_phase_for_episode,
    load_spec,
    make_episode_id,
    make_episode_timestamp,
    resolve_phase_ref,
    validate_spec,
    validate_spec_or_raise,
)
from lens.core.errors import DatasetError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_spec() -> ScopeSpec:
    """A minimal valid spec for testing."""
    return ScopeSpec(
        scope_id="test_scope_01",
        domain="testing",
        description="Test scope",
        generation=GenerationConfig(temperature=0.7, seed=42),
        episodes=EpisodeConfig(
            count=10,
            timeline=TimelineConfig(start="2024-01-15", interval="1d"),
            format="Daily log entry",
            target_words=100,
        ),
        scenario=ScenarioConfig(setting="Test setting", voice="Test voice"),
        arc=[
            PhaseArc(id="baseline", episodes="1-5", description="Normal", signal_density="none"),
            PhaseArc(id="signal", episodes="6-10", description="Signal", signal_density="high"),
        ],
        noise=NoiseConfig(description="Noise", examples=["example noise"]),
        key_facts=[
            KeyFact(
                id="fact_a",
                fact="something happened",
                first_appears="signal:1",
                reinforced_in=["signal:3"],
            ),
        ],
        questions=[
            QuestionSpec(
                id="q01",
                checkpoint_after=10,
                type="longitudinal",
                prompt="What happened?",
                ground_truth=QuestionGroundTruth(
                    canonical_answer="Something happened.",
                    key_facts=["fact_a"],
                    evidence=["signal:1"],
                ),
            ),
        ],
    )


@pytest.fixture
def sample_spec_yaml(tmp_path: Path) -> Path:
    """Write a minimal spec YAML and return its path."""
    content = textwrap.dedent("""\
        scope_id: yaml_test_01
        domain: testing
        description: "YAML test scope"

        generation:
          temperature: 0.5
          seed: 99

        episodes:
          count: 6
          timeline:
            start: "2024-03-01"
            interval: "7d"
          format: "Weekly report"
          target_words: 200

        scenario:
          setting: "A test scenario"
          voice: "Professional"

        arc:
          - id: phase_a
            episodes: "1-3"
            description: "Phase A"
            signal_density: none
          - id: phase_b
            episodes: "4-6"
            description: "Phase B"
            signal_density: high

        noise:
          description: "Background noise"
          examples:
            - "routine event"

        key_facts:
          - id: kf1
            fact: "important thing"
            first_appears: "phase_b:1"
            reinforced_in:
              - "phase_b:3"

        questions:
          - id: q1
            checkpoint_after: 6
            type: longitudinal
            prompt: "What happened?"
            ground_truth:
              canonical_answer: "An important thing."
              key_facts: [kf1]
              evidence: ["phase_b:1"]
    """)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(content)
    return spec_file


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadSpec:
    def test_load_from_yaml(self, sample_spec_yaml: Path) -> None:
        spec = load_spec(sample_spec_yaml)
        assert spec.scope_id == "yaml_test_01"
        assert spec.domain == "testing"
        assert spec.generation.temperature == 0.5
        assert spec.generation.seed == 99
        assert spec.episodes.count == 6
        assert spec.episodes.timeline.start == "2024-03-01"
        assert spec.episodes.timeline.interval == "7d"
        assert len(spec.arc) == 2
        assert spec.arc[0].id == "phase_a"
        assert len(spec.key_facts) == 1
        assert spec.key_facts[0].id == "kf1"
        assert len(spec.questions) == 1
        assert spec.questions[0].type == "longitudinal"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="not found"):
            load_spec(tmp_path / "nonexistent.yaml")

    def test_load_missing_scope_id(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text("domain: testing\n")
        with pytest.raises(DatasetError, match="scope_id"):
            load_spec(tmp_path / "bad.yaml")

    def test_load_non_mapping(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text("- item1\n- item2\n")
        with pytest.raises(DatasetError, match="YAML mapping"):
            load_spec(tmp_path / "bad.yaml")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateSpec:
    def test_valid_spec(self, minimal_spec: ScopeSpec) -> None:
        errors = validate_spec(minimal_spec)
        assert errors == []

    def test_empty_scope_id(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.scope_id = ""
        errors = validate_spec(minimal_spec)
        assert any("scope_id" in e for e in errors)

    def test_zero_episodes(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.episodes.count = 0
        errors = validate_spec(minimal_spec)
        assert any("count" in e for e in errors)

    def test_invalid_timeline_interval(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.episodes.timeline.interval = "bad"
        errors = validate_spec(minimal_spec)
        assert any("interval" in e.lower() for e in errors)

    def test_duplicate_phase_id(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.arc.append(
            PhaseArc(id="baseline", episodes="1-5", description="Dupe")
        )
        errors = validate_spec(minimal_spec)
        assert any("Duplicate phase" in e for e in errors)

    def test_arc_exceeds_episode_count(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.arc[1] = PhaseArc(id="signal", episodes="6-20", description="Too big")
        errors = validate_spec(minimal_spec)
        assert any("exceeds episode count" in e for e in errors)

    def test_uncovered_episodes(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.episodes.count = 15
        errors = validate_spec(minimal_spec)
        assert any("not covered" in e for e in errors)

    def test_invalid_signal_density(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.arc[0].signal_density = "extreme"
        errors = validate_spec(minimal_spec)
        assert any("signal_density" in e for e in errors)

    def test_unknown_key_fact_in_question(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.questions[0].ground_truth.key_facts = ["nonexistent"]
        errors = validate_spec(minimal_spec)
        assert any("nonexistent" in e for e in errors)

    def test_invalid_question_type(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.questions[0].type = "invalid_type"
        errors = validate_spec(minimal_spec)
        assert any("invalid type" in e for e in errors)

    def test_checkpoint_out_of_range(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.questions[0].checkpoint_after = 99
        errors = validate_spec(minimal_spec)
        assert any("out of range" in e for e in errors)

    def test_invalid_phase_ref_format(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.key_facts[0].first_appears = "no_colon"
        errors = validate_spec(minimal_spec)
        assert any("invalid phase ref" in e for e in errors)

    def test_unknown_phase_in_ref(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.key_facts[0].first_appears = "unknown_phase:1"
        errors = validate_spec(minimal_spec)
        assert any("unknown phase" in e for e in errors)

    def test_validate_or_raise(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.scope_id = ""
        with pytest.raises(DatasetError, match="validation failed"):
            validate_spec_or_raise(minimal_spec)


# ---------------------------------------------------------------------------
# Phase-relative ref resolution
# ---------------------------------------------------------------------------


class TestResolvePhaseRef:
    def test_basic_resolve(self, minimal_spec: ScopeSpec) -> None:
        # signal phase is episodes 6-10, so signal:1 = global 6
        eid = resolve_phase_ref("signal:1", minimal_spec)
        assert eid == "test_scope_01_ep_006"

    def test_resolve_later_index(self, minimal_spec: ScopeSpec) -> None:
        eid = resolve_phase_ref("signal:3", minimal_spec)
        assert eid == "test_scope_01_ep_008"

    def test_resolve_baseline(self, minimal_spec: ScopeSpec) -> None:
        eid = resolve_phase_ref("baseline:1", minimal_spec)
        assert eid == "test_scope_01_ep_001"

    def test_resolve_unknown_phase(self, minimal_spec: ScopeSpec) -> None:
        with pytest.raises(DatasetError, match="Unknown phase"):
            resolve_phase_ref("nonexistent:1", minimal_spec)

    def test_resolve_invalid_format(self, minimal_spec: ScopeSpec) -> None:
        with pytest.raises(DatasetError, match="Invalid phase ref"):
            resolve_phase_ref("no_colon", minimal_spec)


# ---------------------------------------------------------------------------
# Episode ID generation
# ---------------------------------------------------------------------------


class TestMakeEpisodeId:
    def test_basic(self) -> None:
        assert make_episode_id("scope_01", 1) == "scope_01_ep_001"

    def test_padding(self) -> None:
        assert make_episode_id("scope_01", 42) == "scope_01_ep_042"

    def test_three_digits(self) -> None:
        assert make_episode_id("scope_01", 100) == "scope_01_ep_100"


# ---------------------------------------------------------------------------
# Episode timestamps
# ---------------------------------------------------------------------------


class TestMakeEpisodeTimestamp:
    def test_first_episode(self, minimal_spec: ScopeSpec) -> None:
        ts = make_episode_timestamp(minimal_spec, 1)
        assert ts == "2024-01-15T10:00:00"

    def test_second_episode(self, minimal_spec: ScopeSpec) -> None:
        ts = make_episode_timestamp(minimal_spec, 2)
        assert ts == "2024-01-16T10:00:00"

    def test_weekly_interval(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.episodes.timeline.interval = "7d"
        ts = make_episode_timestamp(minimal_spec, 3)
        assert ts == "2024-01-29T10:00:00"


# ---------------------------------------------------------------------------
# Phase lookup
# ---------------------------------------------------------------------------


class TestGetPhaseForEpisode:
    def test_baseline_episode(self, minimal_spec: ScopeSpec) -> None:
        phase = get_phase_for_episode(minimal_spec, 3)
        assert phase is not None
        assert phase.id == "baseline"

    def test_signal_episode(self, minimal_spec: ScopeSpec) -> None:
        phase = get_phase_for_episode(minimal_spec, 8)
        assert phase is not None
        assert phase.id == "signal"

    def test_no_phase(self, minimal_spec: ScopeSpec) -> None:
        # Episode 11 is beyond all phases (count is 10)
        phase = get_phase_for_episode(minimal_spec, 11)
        assert phase is None


# ---------------------------------------------------------------------------
# Key facts per phase
# ---------------------------------------------------------------------------


class TestGetKeyFactsForPhase:
    def test_signal_phase_has_fact(self, minimal_spec: ScopeSpec) -> None:
        signal_phase = minimal_spec.arc[1]
        facts = get_key_facts_for_phase(minimal_spec, signal_phase)
        assert len(facts) == 1
        local_idx, kf = facts[0]
        assert kf.id == "fact_a"
        assert local_idx == 1  # first_appears: signal:1

    def test_baseline_phase_no_facts(self, minimal_spec: ScopeSpec) -> None:
        baseline_phase = minimal_spec.arc[0]
        facts = get_key_facts_for_phase(minimal_spec, baseline_phase)
        assert len(facts) == 0


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class TestGetCheckpoints:
    def test_single_checkpoint(self, minimal_spec: ScopeSpec) -> None:
        cps = get_checkpoints(minimal_spec)
        assert cps == [10]

    def test_multiple_checkpoints(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.questions.append(
            QuestionSpec(
                id="q02", checkpoint_after=5, type="null_hypothesis",
                prompt="?", ground_truth=QuestionGroundTruth(canonical_answer=""),
            )
        )
        cps = get_checkpoints(minimal_spec)
        assert cps == [5, 10]


# ---------------------------------------------------------------------------
# PhaseArc
# ---------------------------------------------------------------------------


class TestPhaseArc:
    def test_episode_range(self) -> None:
        phase = PhaseArc(id="test", episodes="3-7")
        assert phase.episode_range() == (3, 7)

    def test_invalid_range(self) -> None:
        phase = PhaseArc(id="test", episodes="bad")
        with pytest.raises(DatasetError, match="Invalid episode range"):
            phase.episode_range()


# ---------------------------------------------------------------------------
# TimelineConfig
# ---------------------------------------------------------------------------


class TestTimelineConfig:
    def test_start_date(self) -> None:
        tl = TimelineConfig(start="2024-06-15")
        assert tl.start_date().isoformat() == "2024-06-15"

    def test_interval_days(self) -> None:
        tl = TimelineConfig(start="2024-01-01", interval="7d")
        assert tl.interval_days() == 7

    def test_invalid_interval(self) -> None:
        tl = TimelineConfig(start="2024-01-01", interval="weekly")
        with pytest.raises(DatasetError, match="Invalid interval"):
            tl.interval_days()


# ---------------------------------------------------------------------------
# DistractorConfig parsing & validation
# ---------------------------------------------------------------------------


class TestDistractorConfig:
    def test_parse_spec_with_distractors(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            scope_id: dist_test_01
            episodes:
              count: 10
              timeline:
                start: "2024-01-01"
              target_words: 500
            arc:
              - id: baseline
                episodes: "1-10"
                description: "Normal"
                signal_density: none
            distractors:
              count: 30
              target_words: 500
              seed: 99
              max_similarity: 0.25
              themes:
                - id: dns_migration
                  scenario: "DNS team ops"
                  excluded_terms:
                    - geo-lookup
                    - retry
                - id: storage
                  scenario: "Storage team ops"
        """)
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(content)
        spec = load_spec(spec_file)

        assert spec.distractors is not None
        assert spec.distractors.count == 30
        assert spec.distractors.target_words == 500
        assert spec.distractors.seed == 99
        assert spec.distractors.max_similarity == 0.25
        assert len(spec.distractors.themes) == 2
        assert spec.distractors.themes[0].id == "dns_migration"
        assert spec.distractors.themes[0].excluded_terms == ["geo-lookup", "retry"]
        assert spec.distractors.themes[1].id == "storage"

    def test_spec_without_distractors_is_none(self, minimal_spec: ScopeSpec) -> None:
        assert minimal_spec.distractors is None

    def test_backward_compat_no_distractors(self, sample_spec_yaml: Path) -> None:
        spec = load_spec(sample_spec_yaml)
        assert spec.distractors is None
        errors = validate_spec(spec)
        assert errors == []

    def test_validate_negative_count(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.distractors = DistractorConfig(
            count=-1,
            themes=[DistractorTheme(id="t1", scenario="test")],
        )
        errors = validate_spec(minimal_spec)
        assert any("count" in e for e in errors)

    def test_validate_invalid_max_similarity(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.distractors = DistractorConfig(
            count=10,
            max_similarity=1.5,
            themes=[DistractorTheme(id="t1", scenario="test")],
        )
        errors = validate_spec(minimal_spec)
        assert any("max_similarity" in e for e in errors)

    def test_validate_count_without_themes(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.distractors = DistractorConfig(count=10, themes=[])
        errors = validate_spec(minimal_spec)
        assert any("themes required" in e for e in errors)

    def test_validate_duplicate_theme_ids(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.distractors = DistractorConfig(
            count=10,
            themes=[
                DistractorTheme(id="dup", scenario="test 1"),
                DistractorTheme(id="dup", scenario="test 2"),
            ],
        )
        errors = validate_spec(minimal_spec)
        assert any("Duplicate distractor theme" in e for e in errors)

    def test_validate_empty_theme_scenario(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.distractors = DistractorConfig(
            count=10,
            themes=[DistractorTheme(id="t1", scenario="")],
        )
        errors = validate_spec(minimal_spec)
        assert any("scenario is required" in e for e in errors)

    def test_validate_valid_distractors(self, minimal_spec: ScopeSpec) -> None:
        minimal_spec.distractors = DistractorConfig(
            count=10,
            themes=[DistractorTheme(id="t1", scenario="A valid scenario")],
        )
        errors = validate_spec(minimal_spec)
        assert errors == []
