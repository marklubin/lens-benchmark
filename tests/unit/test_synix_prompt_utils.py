"""Tests for lens.datagen.synix.prompt_utils — standalone prompt building."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add synix directory to path for standalone imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "lens" / "datagen" / "synix"))

import prompt_utils  # noqa: E402
import spec_utils  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SPEC_RAW = {
    "scope_id": "test_scope",
    "domain": "test_domain",
    "episodes": {
        "count": 10,
        "timeline": {"start": "2024-01-01", "interval": "1d"},
        "format": "Daily log",
        "target_words": 500,
    },
    "scenario": {
        "setting": "Microservices platform with gateway logging.",
        "voice": "Terse ops style. Bullet points.",
    },
    "arc": [
        {"id": "baseline", "episodes": "1-5", "description": "Normal ops", "signal_density": "none"},
        {"id": "signal", "episodes": "6-10", "description": "Degradation begins", "signal_density": "low"},
    ],
    "noise": {"description": "Normal traffic", "examples": ["CDN cache 94%"]},
    "key_facts": [
        {"id": "kf_lat", "fact": "API latency increasing", "first_appears": "signal:1", "reinforced_in": []},
    ],
    "questions": [
        {
            "id": "q01",
            "checkpoint_after": 8,
            "type": "longitudinal",
            "prompt": "What patterns do you see?",
            "ground_truth": {
                "canonical_answer": "API latency is increasing",
                "key_facts": ["kf_lat"],
                "evidence": ["signal:1"],
            },
        },
    ],
    "distractors": {
        "count": 10,
        "target_words": 500,
        "themes": [
            {"id": "dns", "scenario": "DNS migration ops.", "excluded_terms": ["latency", "API"]},
        ],
        "seed": 99,
        "max_similarity": 0.3,
    },
}


@pytest.fixture
def spec():
    return spec_utils.parse_spec(MINIMAL_SPEC_RAW)


# ---------------------------------------------------------------------------
# build_plan_signal_prompt
# ---------------------------------------------------------------------------


class TestBuildPlanSignalPrompt:
    def test_contains_scenario(self, spec):
        prompt = prompt_utils.build_plan_signal_prompt(spec)
        assert "Microservices platform" in prompt

    def test_contains_arc_phases(self, spec):
        prompt = prompt_utils.build_plan_signal_prompt(spec)
        assert "baseline" in prompt
        assert "signal" in prompt
        assert "episodes 1-5" in prompt
        assert "episodes 6-10" in prompt

    def test_contains_key_facts(self, spec):
        prompt = prompt_utils.build_plan_signal_prompt(spec)
        assert "API latency increasing" in prompt

    def test_contains_episode_count(self, spec):
        prompt = prompt_utils.build_plan_signal_prompt(spec)
        assert "exactly 10 episode data sheets" in prompt

    def test_forbids_commentary_words(self, spec):
        prompt = prompt_utils.build_plan_signal_prompt(spec)
        assert "LANGUAGE RULES" in prompt
        assert "FORBIDDEN" in prompt
        assert "increasing" in prompt

    def test_contains_noise_guidance(self, spec):
        prompt = prompt_utils.build_plan_signal_prompt(spec)
        assert "Normal traffic" in prompt

    def test_output_format_is_json(self, spec):
        prompt = prompt_utils.build_plan_signal_prompt(spec)
        assert '"episodes"' in prompt
        assert '"endpoint_metrics"' in prompt

    def test_signal_density_labels_present(self, spec):
        prompt = prompt_utils.build_plan_signal_prompt(spec)
        assert "signal_density: none" in prompt
        assert "signal_density: low" in prompt


# ---------------------------------------------------------------------------
# build_plan_distractor_prompt
# ---------------------------------------------------------------------------


class TestBuildPlanDistractorPrompt:
    def test_contains_theme(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_plan_distractor_prompt(spec, theme, 10)
        assert "dns" in prompt
        assert "DNS migration ops" in prompt

    def test_contains_key_facts_to_avoid(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_plan_distractor_prompt(spec, theme, 10)
        assert "API latency increasing" in prompt
        assert "AVOID" in prompt

    def test_contains_excluded_terms(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_plan_distractor_prompt(spec, theme, 10)
        assert '"latency"' in prompt
        assert '"API"' in prompt

    def test_contains_count(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_plan_distractor_prompt(spec, theme, 15)
        assert "exactly 15 episode data sheets" in prompt

    def test_does_not_contain_main_scenario(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_plan_distractor_prompt(spec, theme, 10)
        assert "Microservices platform" not in prompt


# ---------------------------------------------------------------------------
# build_render_prompt
# ---------------------------------------------------------------------------


class TestBuildRenderPrompt:
    def test_contains_voice(self, spec):
        brief = {"index": 1, "metrics": {"checkout": {"p99": 200}}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "Terse ops style" in prompt

    def test_contains_format(self, spec):
        brief = {"index": 1, "metrics": {}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "Daily log" in prompt

    def test_contains_data_sheet(self, spec):
        brief = {"index": 5, "metrics": {"checkout": {"p99": 200, "requests": 50000}}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert '"p99": 200' in prompt
        assert '"requests": 50000' in prompt

    def test_contains_strict_rules(self, spec):
        brief = {"index": 1, "metrics": {}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "Bullet points and metrics ONLY" in prompt
        assert "Do NOT add interpretation" in prompt

    def test_forbids_trend_words(self, spec):
        brief = {"index": 1, "metrics": {}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "'increasing'" in prompt
        assert "'degrading'" in prompt

    def test_does_not_contain_key_facts(self, spec):
        """Critical: renderer prompt must NOT contain key fact text."""
        brief = {"index": 6, "metrics": {"geo_lookup": {"p99": 600}}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "API latency increasing" not in prompt

    def test_does_not_contain_arc_descriptions(self, spec):
        """Critical: renderer prompt must NOT contain arc phase descriptions."""
        brief = {"index": 6, "metrics": {}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "Degradation begins" not in prompt
        assert "Normal ops" not in prompt

    def test_does_not_contain_questions(self, spec):
        """Critical: renderer prompt must NOT contain question text."""
        brief = {"index": 6, "metrics": {}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "What patterns" not in prompt

    def test_does_not_contain_signal_density(self, spec):
        """Critical: renderer prompt must NOT contain signal density labels."""
        brief = {"index": 6, "metrics": {}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "signal_density" not in prompt

    def test_target_words(self, spec):
        brief = {"index": 1, "metrics": {}}
        prompt = prompt_utils.build_render_prompt(spec, brief)
        assert "500 words" in prompt

    def test_distractor_target_words(self, spec):
        brief = {"index": 1, "metrics": {}}
        prompt = prompt_utils.build_render_prompt(spec, brief, episode_type="distractor")
        assert "500 words" in prompt


# ---------------------------------------------------------------------------
# build_phase_prompt (legacy — kept for backward compat)
# ---------------------------------------------------------------------------


class TestBuildPhasePrompt:
    def test_contains_scenario(self, spec):
        phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, phase, [])
        assert "Microservices platform" in prompt

    def test_contains_voice(self, spec):
        phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, phase, [])
        assert "Terse ops style" in prompt

    def test_contains_format(self, spec):
        phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, phase, [])
        assert "Daily log" in prompt

    def test_contains_word_count_requirement(self, spec):
        phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, phase, [])
        assert "MINIMUM 500 words" in prompt

    def test_contains_phase_info(self, spec):
        phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, phase, [])
        assert "baseline" in prompt
        assert "episodes 1-5" in prompt

    def test_contains_episode_count(self, spec):
        phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, phase, [])
        assert "exactly 5 episodes" in prompt

    def test_signal_requirements_present(self, spec):
        signal_phase = spec["arc"][1]
        prompt = prompt_utils.build_phase_prompt(spec, signal_phase, [])
        assert "API latency increasing" in prompt
        assert "Signal Requirements" in prompt

    def test_no_signal_requirements_in_baseline(self, spec):
        baseline_phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, baseline_phase, [])
        assert "Signal Requirements" not in prompt

    def test_prior_summaries_included(self, spec):
        phase = spec["arc"][1]
        prompt = prompt_utils.build_phase_prompt(spec, phase, ["Phase 1 was normal."])
        assert "Prior Context" in prompt
        assert "Phase 1 was normal." in prompt

    def test_noise_included(self, spec):
        phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, phase, [])
        assert "Normal traffic" in prompt
        assert "CDN cache 94%" in prompt

    def test_json_output_format(self, spec):
        phase = spec["arc"][0]
        prompt = prompt_utils.build_phase_prompt(spec, phase, [])
        assert '"episodes"' in prompt
        assert '"phase_summary"' in prompt


# ---------------------------------------------------------------------------
# build_distractor_prompt (legacy — kept for backward compat)
# ---------------------------------------------------------------------------


class TestBuildDistractorPrompt:
    def test_uses_theme_scenario(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_distractor_prompt(spec, theme, 10, [])
        assert "DNS migration ops" in prompt
        # Should NOT use the main scenario
        assert "Microservices platform" not in prompt

    def test_contains_excluded_terms(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_distractor_prompt(spec, theme, 10, [])
        assert "NEUTRALITY CONSTRAINT" in prompt
        assert '"latency"' in prompt
        assert '"API"' in prompt

    def test_contains_key_facts_for_neutrality(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_distractor_prompt(spec, theme, 10, [])
        # Key fact text should appear so LLM knows what to be neutral about
        assert "API latency increasing" in prompt
        assert "Ground truth facts" in prompt

    def test_contains_purpose_framing(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_distractor_prompt(spec, theme, 10, [])
        assert "DISTRACTOR episodes" in prompt
        assert "entirely neutral" in prompt
        assert "materially sway" in prompt

    def test_contains_count(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_distractor_prompt(spec, theme, 15, [])
        assert "exactly 15 episodes" in prompt

    def test_uses_distractor_target_words(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_distractor_prompt(spec, theme, 10, [])
        assert "MINIMUM 500 words" in prompt

    def test_prior_summaries(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_distractor_prompt(spec, theme, 10, ["Batch 1 done."])
        assert "Prior Batches" in prompt
        assert "Batch 1 done." in prompt

    def test_same_voice(self, spec):
        theme = spec["distractors"]["themes"][0]
        prompt = prompt_utils.build_distractor_prompt(spec, theme, 10, [])
        assert "Terse ops style" in prompt


# ---------------------------------------------------------------------------
# build_expand_prompt
# ---------------------------------------------------------------------------


class TestBuildExpandPrompt:
    def test_contains_target_words(self):
        prompt = prompt_utils.build_expand_prompt("Short text here.", 500)
        assert "500 words" in prompt
        assert "Short text here." in prompt


# ---------------------------------------------------------------------------
# build_contamination_prompt
# ---------------------------------------------------------------------------


class TestBuildContaminationPrompt:
    def test_contains_episode_and_question(self):
        prompt = prompt_utils.build_contamination_prompt("Log entry text.", "What happened?")
        assert "Log entry text." in prompt
        assert "What happened?" in prompt
        assert "single record" in prompt
        assert "ONLY" in prompt

    def test_structure(self):
        prompt = prompt_utils.build_contamination_prompt("ep", "q")
        assert "## Record" in prompt
        assert "## Question" in prompt
        assert "## Answer" in prompt


# ---------------------------------------------------------------------------
# build_naive_baseline_prompt
# ---------------------------------------------------------------------------


class TestBuildNaiveBaselinePrompt:
    def test_contains_all_records(self):
        texts = ["Record A text.", "Record B text.", "Record C text."]
        prompt = prompt_utils.build_naive_baseline_prompt(texts, "Question?")
        assert "Record A text." in prompt
        assert "Record B text." in prompt
        assert "Record C text." in prompt
        assert "Question?" in prompt

    def test_record_numbering(self):
        prompt = prompt_utils.build_naive_baseline_prompt(["a", "b"], "q")
        assert "Record 1:" in prompt
        assert "Record 2:" in prompt

    def test_structure(self):
        prompt = prompt_utils.build_naive_baseline_prompt(["text"], "q")
        assert "## Records" in prompt
        assert "## Question" in prompt
        assert "## Answer" in prompt
