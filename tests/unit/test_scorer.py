from __future__ import annotations

from lens.core.models import (
    AgentAnswer,
    CheckpointResult,
    GroundTruth,
    MetricResult,
    PersonaResult,
    Question,
    QuestionResult,
    RunResult,
)
from lens.scorer.aggregate import compute_composite
from lens.scorer.registry import list_metrics
from lens.scorer.tier1 import (
    BudgetCompliance,
    EvidenceCoverage,
    EvidenceGrounding,
    FactRecall,
)
from lens.scorer.tier2 import (
    AnswerQuality,
    InsightDepth,
    ReasoningQuality,
)
from lens.scorer.tier3 import (
    ActionQuality,
    LongitudinalAdvantage,
)


def _make_qr(
    question_type: str = "longitudinal",
    answer_text: str = "",
    key_facts: list[str] | None = None,
    required_refs: list[str] | None = None,
    retrieved: list[str] | None = None,
    valid: list[str] | None = None,
    budget_violations: list[str] | None = None,
    tool_calls: int = 2,
) -> QuestionResult:
    """Helper to create a QuestionResult for testing."""
    return QuestionResult(
        question=Question(
            question_id="test_q",
            persona_id="p1",
            checkpoint_after=10,
            question_type=question_type,
            prompt="Test?",
            ground_truth=GroundTruth(
                canonical_answer="canonical",
                required_evidence_refs=required_refs or [],
                key_facts=key_facts or [],
            ),
        ),
        answer=AgentAnswer(
            question_id="test_q",
            answer_text=answer_text,
            tool_calls_made=tool_calls,
            total_tokens=100,
            wall_time_ms=50.0,
            budget_violations=budget_violations or [],
            refs_cited=retrieved or [],
        ),
        retrieved_ref_ids=retrieved or [],
        valid_ref_ids=valid or [],
    )


def _make_run(question_results: list[QuestionResult]) -> RunResult:
    """Helper to create a RunResult with given question results."""
    return RunResult(
        run_id="test",
        adapter="test",
        dataset_version="0.1.0",
        budget_preset="standard",
        personas=[
            PersonaResult(
                persona_id="p1",
                checkpoints=[
                    CheckpointResult(
                        persona_id="p1",
                        checkpoint=10,
                        question_results=question_results,
                    )
                ],
            )
        ],
    )


class TestEvidenceGrounding:
    def test_all_valid(self):
        qr = _make_qr(retrieved=["ep_001", "ep_002"], valid=["ep_001", "ep_002"])
        result = _make_run([qr])
        mr = EvidenceGrounding().compute(result)
        assert mr.value == 1.0

    def test_half_valid(self):
        qr = _make_qr(retrieved=["ep_001", "ep_002"], valid=["ep_001"])
        result = _make_run([qr])
        mr = EvidenceGrounding().compute(result)
        assert mr.value == 0.5

    def test_no_refs(self):
        qr = _make_qr(retrieved=[], valid=[])
        result = _make_run([qr])
        mr = EvidenceGrounding().compute(result)
        assert mr.value == 0.0

    def test_empty_run(self):
        result = _make_run([])
        mr = EvidenceGrounding().compute(result)
        assert mr.value == 0.0


class TestFactRecall:
    def test_all_facts_found(self):
        qr = _make_qr(
            answer_text="The pattern_alpha evolved and evidence_fragment was found",
            key_facts=["pattern_alpha", "evidence_fragment"],
        )
        result = _make_run([qr])
        mr = FactRecall().compute(result)
        assert mr.value == 1.0

    def test_no_facts_found(self):
        qr = _make_qr(
            answer_text="Nothing relevant was found",
            key_facts=["pattern_alpha", "evidence_fragment"],
        )
        result = _make_run([qr])
        mr = FactRecall().compute(result)
        assert mr.value == 0.0

    def test_partial_facts(self):
        qr = _make_qr(
            answer_text="Found pattern_alpha but nothing else",
            key_facts=["pattern_alpha", "evidence_fragment"],
        )
        result = _make_run([qr])
        mr = FactRecall().compute(result)
        assert mr.value == 0.5

    def test_case_insensitive(self):
        qr = _make_qr(
            answer_text="PATTERN_ALPHA was observed",
            key_facts=["pattern_alpha"],
        )
        result = _make_run([qr])
        mr = FactRecall().compute(result)
        assert mr.value == 1.0

    def test_empty(self):
        result = _make_run([])
        mr = FactRecall().compute(result)
        assert mr.value == 0.0


class TestEvidenceCoverage:
    def test_full_coverage(self):
        qr = _make_qr(
            required_refs=["ep_001", "ep_002"],
            retrieved=["ep_001", "ep_002", "ep_003"],
        )
        result = _make_run([qr])
        mr = EvidenceCoverage().compute(result)
        assert mr.value == 1.0

    def test_partial_coverage(self):
        qr = _make_qr(
            required_refs=["ep_001", "ep_002"],
            retrieved=["ep_001"],
        )
        result = _make_run([qr])
        mr = EvidenceCoverage().compute(result)
        assert mr.value == 0.5

    def test_no_coverage(self):
        qr = _make_qr(
            required_refs=["ep_001", "ep_002"],
            retrieved=["ep_003"],
        )
        result = _make_run([qr])
        mr = EvidenceCoverage().compute(result)
        assert mr.value == 0.0


class TestBudgetCompliance:
    def test_no_violations(self):
        qr = _make_qr(budget_violations=[])
        result = _make_run([qr])
        mr = BudgetCompliance().compute(result)
        assert mr.value == 1.0

    def test_one_violation(self):
        qr = _make_qr(budget_violations=["turn limit exceeded"])
        result = _make_run([qr])
        mr = BudgetCompliance().compute(result)
        assert mr.value == 0.9

    def test_many_violations(self):
        qr = _make_qr(budget_violations=[f"v{i}" for i in range(15)])
        result = _make_run([qr])
        mr = BudgetCompliance().compute(result)
        assert mr.value == 0.0


class TestAnswerQuality:
    def test_stub_returns_zero(self):
        """answer_quality is an LLM judge stub — always returns 0.0."""
        qr = _make_qr(
            answer_text="Found pattern_alpha and evidence_fragment",
            key_facts=["pattern_alpha", "evidence_fragment"],
        )
        result = _make_run([qr])
        mr = AnswerQuality().compute(result)
        assert mr.value == 0.0
        assert mr.details.get("not_implemented") is True


class TestInsightDepth:
    def test_multi_refs(self):
        qr = _make_qr(retrieved=["ep_001", "ep_002"])
        result = _make_run([qr])
        mr = InsightDepth().compute(result)
        assert mr.value == 1.0

    def test_single_ref(self):
        qr = _make_qr(retrieved=["ep_001"])
        result = _make_run([qr])
        mr = InsightDepth().compute(result)
        assert mr.value == 0.0


class TestReasoningQuality:
    def test_substantive_answer(self):
        qr = _make_qr(
            answer_text="A" * 60,
            tool_calls=3,
        )
        result = _make_run([qr])
        mr = ReasoningQuality().compute(result)
        assert mr.value == 1.0

    def test_short_answer(self):
        qr = _make_qr(answer_text="Short", tool_calls=3)
        result = _make_run([qr])
        mr = ReasoningQuality().compute(result)
        assert mr.value == 0.0

    def test_no_tool_calls(self):
        qr = _make_qr(answer_text="A" * 60, tool_calls=0)
        result = _make_run([qr])
        mr = ReasoningQuality().compute(result)
        assert mr.value == 0.0


class TestLongitudinalAdvantage:
    def test_positive_advantage(self):
        long_qr = _make_qr(
            question_type="longitudinal",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([long_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 1.0  # 1.0 - 0.0

    def test_no_longitudinal_questions(self):
        qr = _make_qr(question_type="null_hypothesis")
        result = _make_run([qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 0.0


class TestActionQuality:
    def test_action_questions(self):
        qr = _make_qr(
            question_type="action_recommendation",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        result = _make_run([qr])
        mr = ActionQuality().compute(result)
        assert mr.value == 1.0

    def test_no_action_questions(self):
        qr = _make_qr(question_type="longitudinal")
        result = _make_run([qr])
        mr = ActionQuality().compute(result)
        assert mr.value == 0.0


class TestMetricRegistry:
    def test_all_tier1_registered(self):
        metrics = list_metrics()
        tier1_names = {
            "evidence_grounding",
            "fact_recall",
            "evidence_coverage",
            "budget_compliance",
        }
        assert tier1_names.issubset(set(metrics.keys()))

    def test_tier2_registered(self):
        metrics = list_metrics()
        assert "answer_quality" in metrics
        assert "insight_depth" in metrics
        assert "reasoning_quality" in metrics

    def test_tier3_registered(self):
        metrics = list_metrics()
        assert "longitudinal_advantage" in metrics
        assert "action_quality" in metrics


class TestCompositeScore:
    def test_composite(self):
        metrics = [
            MetricResult(name="evidence_grounding", tier=1, value=1.0),
            MetricResult(name="fact_recall", tier=1, value=0.8),
            MetricResult(name="evidence_coverage", tier=1, value=0.6),
            MetricResult(name="budget_compliance", tier=1, value=1.0),
            MetricResult(name="answer_quality", tier=2, value=0.7),
            MetricResult(name="insight_depth", tier=2, value=0.5),
            MetricResult(name="reasoning_quality", tier=2, value=0.9),
            MetricResult(name="longitudinal_advantage", tier=3, value=0.3),
            MetricResult(name="action_quality", tier=3, value=0.4),
        ]
        score = compute_composite(metrics)
        assert 0.0 < score < 1.0

    def test_empty_metrics(self):
        assert compute_composite([]) == 0.0
