from __future__ import annotations

import pytest

from lens.datagen.prompt import (
    SYSTEM_PROMPT,
    build_contamination_prompt,
    build_naive_baseline_prompt,
    build_phase_prompt,
)
from lens.datagen.spec import (
    EpisodeConfig,
    GenerationConfig,
    KeyFact,
    NoiseConfig,
    PhaseArc,
    ScenarioConfig,
    ScopeSpec,
    TimelineConfig,
)


@pytest.fixture
def spec_with_facts() -> ScopeSpec:
    return ScopeSpec(
        scope_id="test_01",
        generation=GenerationConfig(temperature=0.7, seed=42),
        episodes=EpisodeConfig(
            count=10,
            timeline=TimelineConfig(start="2024-01-01"),
            format="Daily log",
            target_words=150,
        ),
        scenario=ScenarioConfig(
            setting="A test scenario with services.",
            voice="Terse log style.",
        ),
        arc=[
            PhaseArc(id="baseline", episodes="1-5", description="Normal ops", signal_density="none"),
            PhaseArc(id="signal", episodes="6-10", description="Signal phase", signal_density="high"),
        ],
        noise=NoiseConfig(
            description="Background noise items",
            examples=["routine deploy", "disk alert"],
        ),
        key_facts=[
            KeyFact(
                id="latency_issue",
                fact="API latency increasing",
                first_appears="signal:1",
                reinforced_in=["signal:3"],
            ),
        ],
    )


class TestBuildPhasePrompt:
    def test_includes_scenario(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "A test scenario" in prompt

    def test_includes_voice(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "Terse log style" in prompt

    def test_includes_format(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "Daily log" in prompt

    def test_includes_target_words(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "150 words" in prompt

    def test_includes_phase_description(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "Normal ops" in prompt

    def test_includes_noise(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "Background noise" in prompt
        assert "routine deploy" in prompt

    def test_no_signal_requirements_for_baseline(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "Signal Requirements" not in prompt

    def test_signal_requirements_for_signal_phase(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[1], [])
        assert "Signal Requirements" in prompt
        assert "API latency increasing" in prompt

    def test_includes_prior_summaries(self, spec_with_facts: ScopeSpec) -> None:
        summaries = ["Phase 1 was normal operations."]
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[1], summaries)
        assert "Prior Context" in prompt
        assert "Phase 1 was normal" in prompt

    def test_no_prior_context_for_first_phase(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "Prior Context" not in prompt

    def test_output_section_has_json_format(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "Return JSON" in prompt
        assert '"episodes"' in prompt
        assert '"phase_summary"' in prompt

    def test_correct_episode_count(self, spec_with_facts: ScopeSpec) -> None:
        prompt = build_phase_prompt(spec_with_facts, spec_with_facts.arc[0], [])
        assert "exactly 5 episodes" in prompt


class TestBuildContaminationPrompt:
    def test_includes_episode_text(self) -> None:
        prompt = build_contamination_prompt("Episode text here.", "What happened?")
        assert "Episode text here" in prompt

    def test_includes_question(self) -> None:
        prompt = build_contamination_prompt("Episode text.", "What pattern emerged?")
        assert "What pattern emerged?" in prompt

    def test_instructs_single_record(self) -> None:
        prompt = build_contamination_prompt("text", "question")
        assert "single record" in prompt


class TestBuildNaiveBaselinePrompt:
    def test_includes_all_episodes(self) -> None:
        texts = ["Episode 1 text", "Episode 2 text", "Episode 3 text"]
        prompt = build_naive_baseline_prompt(texts, "What happened?")
        assert "Episode 1 text" in prompt
        assert "Episode 2 text" in prompt
        assert "Episode 3 text" in prompt

    def test_includes_question(self) -> None:
        prompt = build_naive_baseline_prompt(["text"], "The big question?")
        assert "The big question?" in prompt

    def test_includes_all_records_label(self) -> None:
        prompt = build_naive_baseline_prompt(["text"], "q")
        assert "all records" in prompt


class TestSystemPrompt:
    def test_system_prompt_mentions_json(self) -> None:
        assert "JSON" in SYSTEM_PROMPT
