from __future__ import annotations

import pytest

from lens.datagen.verify_report import generate_verification_report
from lens.datagen.spec import (
    EpisodeConfig,
    KeyFact,
    NoiseConfig,
    PhaseArc,
    ScopeSpec,
    TimelineConfig,
)


def _make_spec() -> ScopeSpec:
    return ScopeSpec(
        scope_id="test_scope_01",
        domain="system_logs",
        arc=[
            PhaseArc(id="baseline", episodes="1-5", description="Normal ops", signal_density="none"),
            PhaseArc(id="signal", episodes="6-10", description="Early signal", signal_density="low"),
            PhaseArc(id="escalation", episodes="11-15", description="Escalation", signal_density="high"),
        ],
        episodes=EpisodeConfig(
            count=15,
            timeline=TimelineConfig(start="2024-01-01"),
        ),
        key_facts=[
            KeyFact(id="kf_latency", fact="latency increasing", first_appears="signal:1"),
            KeyFact(id="kf_pool", fact="pool exhaustion", first_appears="escalation:1"),
        ],
    )


def _make_manifest() -> dict:
    return {
        "scope_id": "test_scope_01",
        "generated_at": "2025-01-01T00:00:00",
        "key_fact_coverage": {
            "kf_latency": {
                "target_episodes": ["ep_006", "ep_008"],
                "found_in": ["ep_006"],
            },
            "kf_pool": {
                "target_episodes": ["ep_011"],
                "found_in": ["ep_011"],
            },
        },
        "validation": {
            "contamination_check": "pass",
            "naive_baseline": {"longitudinal": 0.65},
        },
    }


def _make_distractor_purity() -> dict:
    return {
        "summary": "pass",
        "threshold": 0.3,
        "total_distractors": 3,
        "flagged_count": 0,
        "avg_similarity": 0.12,
        "max_similarity": 0.2,
        "episodes": [
            {
                "episode_id": "test_dx_001",
                "theme": "dns_migration",
                "max_similarity": 0.1,
                "flagged": False,
            },
            {
                "episode_id": "test_dx_002",
                "theme": "storage_capacity",
                "max_similarity": 0.15,
                "flagged": False,
            },
            {
                "episode_id": "test_dx_003",
                "theme": "auth_audit",
                "max_similarity": 0.2,
                "flagged": False,
            },
        ],
    }


def _make_results() -> dict:
    return {
        "_meta": {
            "scope_id": "test_scope_01",
            "domain": "system_logs",
            "episode_count": 15,
            "question_count": 3,
            "verified_at": "2025-01-01T12:00:00+00:00",
            "model": "gpt-4o-mini",
        },
        "contamination": {
            "summary": "pass",
            "questions": [
                {
                    "question_id": "q01_longitudinal",
                    "max_single_episode_coverage": 0.25,
                    "worst_episode": "ep_006",
                    "contaminated": False,
                    "episode_scores": [
                        {"episode_id": "ep_006", "coverage": 0.25, "answer": "Some partial answer"},
                        {"episode_id": "ep_007", "coverage": 0.0, "answer": "No info"},
                    ],
                },
                {
                    "question_id": "q02_longitudinal",
                    "max_single_episode_coverage": 0.6,
                    "worst_episode": "ep_011",
                    "contaminated": True,
                    "episode_scores": [],
                },
            ],
        },
        "naive_baseline": {
            "summary": {"longitudinal": 0.75, "null_hypothesis": 1.0},
            "questions": [
                {
                    "question_id": "q01_longitudinal",
                    "question_type": "longitudinal",
                    "fact_coverage": 0.75,
                    "answer": "The latency was increasing and the pool was exhausted.",
                    "per_fact_matches": [
                        {"fact": "latency increasing", "matched": True, "overlap_ratio": 1.0},
                        {"fact": "pool exhaustion", "matched": False, "overlap_ratio": 0.333},
                    ],
                },
                {
                    "question_id": "q02_null",
                    "question_type": "null_hypothesis",
                    "fact_coverage": 1.0,
                    "answer": "Nothing notable happened.",
                    "per_fact_matches": [],
                },
            ],
        },
    }


class TestVerifyReport:
    def test_generates_valid_html(self) -> None:
        html = generate_verification_report(_make_results(), _make_spec(), _make_manifest())
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "</body>" in html

    def test_contains_scope_metadata(self) -> None:
        html = generate_verification_report(_make_results(), _make_spec(), _make_manifest())
        assert "test_scope_01" in html
        assert "system_logs" in html
        assert "gpt-4o-mini" in html

    def test_contains_contamination_results(self) -> None:
        html = generate_verification_report(_make_results(), _make_spec(), _make_manifest())
        assert "q01_longitudinal" in html
        assert "q02_longitudinal" in html
        # Pass badge for clean question
        assert "PASS" in html
        # Fail badge for contaminated question
        assert "FAIL" in html

    def test_contains_naive_baseline_results(self) -> None:
        html = generate_verification_report(_make_results(), _make_spec(), _make_manifest())
        assert "75.0%" in html
        assert "100.0%" in html
        assert "The latency was increasing" in html

    def test_contains_key_fact_coverage(self) -> None:
        html = generate_verification_report(_make_results(), _make_spec(), _make_manifest())
        assert "kf_latency" in html
        assert "kf_pool" in html
        assert "ep_008" in html  # missing episode shown

    def test_contains_episode_timeline(self) -> None:
        html = generate_verification_report(_make_results(), _make_spec(), _make_manifest())
        assert "baseline" in html
        assert "signal" in html
        assert "escalation" in html
        assert "density-none" in html
        assert "density-low" in html
        assert "density-high" in html

    def test_score_colors(self) -> None:
        html = generate_verification_report(_make_results(), _make_spec(), _make_manifest())
        # Green for high scores (>=70%)
        assert "#22c55e" in html
        # Yellow for medium (40-70%)
        assert "#eab308" in html
        # Red for low (<40%) — contamination inverse color for 25% single-ep
        assert "#ef4444" in html

    def test_html_escaping(self) -> None:
        results = _make_results()
        results["naive_baseline"]["questions"][0]["answer"] = "<script>alert('xss')</script>"
        html = generate_verification_report(results, _make_spec(), _make_manifest())
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_missing_contamination_section(self) -> None:
        results = _make_results()
        del results["contamination"]
        html = generate_verification_report(results, _make_spec(), _make_manifest())
        assert "Not run" in html
        assert "<!DOCTYPE html>" in html.strip()

    def test_missing_naive_baseline_section(self) -> None:
        results = _make_results()
        del results["naive_baseline"]
        html = generate_verification_report(results, _make_spec(), _make_manifest())
        assert "Not run" in html
        assert "<!DOCTYPE html>" in html.strip()

    def test_contains_distractor_purity_section(self) -> None:
        results = _make_results()
        results["distractor_purity"] = _make_distractor_purity()
        html = generate_verification_report(results, _make_spec(), _make_manifest())
        assert "Distractor Purity" in html
        assert "test_dx_001" in html
        assert "dns_migration" in html
        assert "storage_capacity" in html
        assert "auth_audit" in html
        assert "PASS" in html

    def test_distractor_purity_shows_stats(self) -> None:
        results = _make_results()
        results["distractor_purity"] = _make_distractor_purity()
        html = generate_verification_report(results, _make_spec(), _make_manifest())
        assert "Total:" in html
        assert "Flagged:" in html
        assert "Threshold:" in html

    def test_distractor_purity_flagged_episodes(self) -> None:
        results = _make_results()
        purity = _make_distractor_purity()
        purity["summary"] = "fail"
        purity["flagged_count"] = 1
        purity["episodes"][0]["flagged"] = True
        purity["episodes"][0]["max_similarity"] = 0.8
        results["distractor_purity"] = purity
        html = generate_verification_report(results, _make_spec(), _make_manifest())
        assert "FLAGGED" in html

    def test_no_distractor_purity_section_when_absent(self) -> None:
        results = _make_results()
        # No distractor_purity key
        html = generate_verification_report(results, _make_spec(), _make_manifest())
        assert "Distractor Purity" not in html
