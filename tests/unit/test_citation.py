"""Tests for citation extraction and CitationCoverage metric."""
from __future__ import annotations

from lens.agent.harness import _extract_inline_refs
from lens.core.models import (
    AgentAnswer,
    CheckpointResult,
    GroundTruth,
    Question,
    QuestionResult,
    RunResult,
    ScopeResult,
)
from lens.scorer.tier1 import CitationCoverage


def _make_qr(
    required_refs: list[str] | None = None,
    retrieved: list[str] | None = None,
    valid: list[str] | None = None,
) -> QuestionResult:
    return QuestionResult(
        question=Question(
            question_id="test_q",
            scope_id="p1",
            checkpoint_after=10,
            question_type="longitudinal",
            prompt="Test?",
            ground_truth=GroundTruth(
                canonical_answer="canonical",
                required_evidence_refs=required_refs or [],
                key_facts=["fact1"],
            ),
        ),
        answer=AgentAnswer(
            question_id="test_q",
            answer_text="answer",
            tool_calls_made=2,
            total_tokens=100,
            wall_time_ms=50.0,
            refs_cited=retrieved or [],
        ),
        retrieved_ref_ids=retrieved or [],
        valid_ref_ids=valid or [],
    )


def _make_run(question_results: list[QuestionResult]) -> RunResult:
    return RunResult(
        run_id="test",
        adapter="test",
        dataset_version="0.1.0",
        budget_preset="standard",
        scopes=[
            ScopeResult(
                scope_id="p1",
                checkpoints=[
                    CheckpointResult(
                        scope_id="p1",
                        checkpoint=10,
                        question_results=question_results,
                    )
                ],
            )
        ],
    )


class TestExtractInlineRefs:
    """Tests for _extract_inline_refs from harness.py."""

    def test_bracket_format(self):
        text = "The data shows [cascading_failure_01_ep_005] had issues."
        refs = _extract_inline_refs(text)
        assert refs == ["cascading_failure_01_ep_005"]

    def test_ref_id_paren_format(self):
        text = "See (ref_id: insider_threat_05_ep_007) for details."
        refs = _extract_inline_refs(text)
        assert refs == ["insider_threat_05_ep_007"]

    def test_multiple_refs(self):
        text = (
            "Evidence from [clinical_signal_03_ep_010] and "
            "[clinical_signal_03_ep_015] supports this."
        )
        refs = _extract_inline_refs(text)
        assert refs == ["clinical_signal_03_ep_010", "clinical_signal_03_ep_015"]

    def test_deduplication(self):
        text = "[env_drift_04_ep_001] shows ... [env_drift_04_ep_001] confirms"
        refs = _extract_inline_refs(text)
        assert refs == ["env_drift_04_ep_001"]

    def test_no_refs(self):
        text = "No citations here."
        refs = _extract_inline_refs(text)
        assert refs == []

    def test_mixed_formats(self):
        text = (
            "Found in [market_regime_06_ep_003] and "
            "(ref_id: market_regime_06_ep_008)."
        )
        refs = _extract_inline_refs(text)
        assert "market_regime_06_ep_003" in refs
        assert "market_regime_06_ep_008" in refs
        assert len(refs) == 2

    def test_ref_id_with_space(self):
        text = "(ref_id:  financial_irregularity_02_ep_012)"
        refs = _extract_inline_refs(text)
        assert refs == ["financial_irregularity_02_ep_012"]

    def test_does_not_match_random_brackets(self):
        text = "The value [42] was unexpected. See [notes]."
        refs = _extract_inline_refs(text)
        assert refs == []


class TestCitationCoverage:
    """Tests for CitationCoverage metric."""

    def test_full_coverage(self):
        qr = _make_qr(
            required_refs=["ep_001", "ep_002"],
            retrieved=["ep_001", "ep_002", "ep_003"],
            valid=["ep_001", "ep_002"],
        )
        result = _make_run([qr])
        mr = CitationCoverage().compute(result)
        assert mr.value == 1.0

    def test_partial_coverage(self):
        qr = _make_qr(
            required_refs=["ep_001", "ep_002"],
            retrieved=["ep_001"],
            valid=["ep_001"],
        )
        result = _make_run([qr])
        mr = CitationCoverage().compute(result)
        assert mr.value == 0.5

    def test_no_coverage(self):
        qr = _make_qr(
            required_refs=["ep_001", "ep_002"],
            retrieved=["ep_003"],
            valid=["ep_003"],
        )
        result = _make_run([qr])
        mr = CitationCoverage().compute(result)
        assert mr.value == 0.0

    def test_no_required_refs(self):
        """No required refs → score 1.0."""
        qr = _make_qr(required_refs=[], retrieved=["ep_001"])
        result = _make_run([qr])
        mr = CitationCoverage().compute(result)
        assert mr.value == 1.0

    def test_empty_run(self):
        result = _make_run([])
        mr = CitationCoverage().compute(result)
        assert mr.value == 0.0

    def test_valid_ref_ids_also_count(self):
        """Coverage uses union of retrieved_ref_ids and valid_ref_ids."""
        qr = _make_qr(
            required_refs=["ep_001", "ep_002"],
            retrieved=["ep_001"],  # only ep_001 in retrieved
            valid=["ep_002"],      # ep_002 only in valid
        )
        result = _make_run([qr])
        mr = CitationCoverage().compute(result)
        assert mr.value == 1.0  # both covered via union

    def test_metric_properties(self):
        metric = CitationCoverage()
        assert metric.name == "citation_coverage"
        assert metric.tier == 1

    def test_registered(self):
        from lens.scorer.registry import list_metrics
        metrics = list_metrics()
        assert "citation_coverage" in metrics
