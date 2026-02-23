from __future__ import annotations

from lens.core.models import (
    AgentAnswer,
    CheckpointResult,
    GroundTruth,
    MetricResult,
    ScopeResult,
    Question,
    QuestionResult,
    RunResult,
)
from lens.scorer.aggregate import TIER1_GATE_THRESHOLDS, compute_composite
from lens.scorer.judge import pairwise_fact_judge
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
            scope_id="p1",
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
        # 1 question, 1 violation → 1 - 1/1 = 0.0
        assert mr.value == 0.0
        assert mr.details["total_violations"] == 1
        assert "total_tokens" in mr.details
        assert "total_wall_time_minutes" in mr.details

    def test_violation_rate(self):
        qrs = [_make_qr(budget_violations=[]) for _ in range(9)]
        qrs.append(_make_qr(budget_violations=["over budget"]))
        result = _make_run(qrs)
        mr = BudgetCompliance().compute(result)
        # 10 questions, 1 violation → 1 - 1/10 = 0.9
        assert mr.value == 0.9
        assert mr.details["total_violations"] == 1
        assert mr.details["violation_rate"] == 0.1


class TestAnswerQuality:
    def test_stub_without_judge(self):
        """Without judge_fn, answer_quality returns 0.0 as stub."""
        qr = _make_qr(
            answer_text="Found pattern_alpha and evidence_fragment",
            key_facts=["pattern_alpha", "evidence_fragment"],
        )
        result = _make_run([qr])
        mr = AnswerQuality().compute(result)
        assert mr.value == 0.0
        assert mr.details.get("not_implemented") is True

    def test_with_judge_candidate_wins(self):
        """With judge that always picks candidate, score is 1.0."""
        qr = _make_qr(
            answer_text="Found pattern_alpha and evidence_fragment",
            key_facts=["pattern_alpha"],
        )
        result = _make_run([qr])
        # Judge always returns the candidate's position
        calls = []

        def judge_fn(prompt):
            calls.append(prompt)
            # Determine which position the candidate is in from the prompt
            # The pairwise judge randomizes, so we check both
            if "Response A:\nFound pattern_alpha" in prompt:
                return "A"
            return "B"

        metric = AnswerQuality(judge_fn=judge_fn)
        mr = metric.compute(result)
        assert mr.value == 1.0
        assert mr.details.get("method") == "pairwise"

    def test_with_judge_via_configure(self):
        """judge_fn injected via configure() works."""
        qr = _make_qr(
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        result = _make_run([qr])

        metric = AnswerQuality()
        assert metric.compute(result).value == 0.0  # stub

        metric.configure(judge_fn=lambda prompt: "TIE")
        mr = metric.compute(result)
        assert mr.value == 0.5  # tie on 1 fact

    def test_no_key_facts_scores_half(self):
        """Questions with no key facts score 0.5 (uninformative)."""
        qr = _make_qr(answer_text="answer", key_facts=[])
        result = _make_run([qr])
        metric = AnswerQuality(judge_fn=lambda p: "A")
        mr = metric.compute(result)
        assert mr.value == 0.5


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

    def test_negative_questions_count_as_synthesis(self):
        """Negative question type should be grouped with synthesis (numerator)."""
        neg_qr = _make_qr(
            question_type="negative",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([neg_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 1.0  # 1.0 - 0.0
        assert mr.details["synthesis_count"] == 1
        assert mr.details["control_count"] == 1

    def test_temporal_questions_count_as_synthesis(self):
        """Temporal question type should be grouped with synthesis (numerator)."""
        temp_qr = _make_qr(
            question_type="temporal",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([temp_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 1.0
        assert mr.details["synthesis_count"] == 1

    def test_counterfactual_questions_count_as_synthesis(self):
        """Counterfactual question type should be grouped with synthesis (numerator)."""
        cf_qr = _make_qr(
            question_type="counterfactual",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([cf_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 1.0
        assert mr.details["synthesis_count"] == 1

    def test_paraphrase_questions_count_as_synthesis(self):
        """Paraphrase question type should be grouped with synthesis (numerator)."""
        para_qr = _make_qr(
            question_type="paraphrase",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([para_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 1.0
        assert mr.details["synthesis_count"] == 1

    def test_distractor_resistance_questions_count_as_synthesis(self):
        """Distractor resistance question type should be grouped with synthesis."""
        dr_qr = _make_qr(
            question_type="distractor_resistance",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([dr_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 1.0
        assert mr.details["synthesis_count"] == 1

    def test_severity_assessment_questions_count_as_synthesis(self):
        """Severity assessment question type should be grouped with synthesis."""
        sev_qr = _make_qr(
            question_type="severity_assessment",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([sev_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 1.0
        assert mr.details["synthesis_count"] == 1

    def test_evidence_sufficiency_questions_count_as_synthesis(self):
        """Evidence sufficiency question type should be grouped with synthesis."""
        es_qr = _make_qr(
            question_type="evidence_sufficiency",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([es_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        assert mr.value == 1.0
        assert mr.details["synthesis_count"] == 1

    def test_mixed_synthesis_types(self):
        """Multiple synthesis question types combined in numerator."""
        long_qr = _make_qr(
            question_type="longitudinal",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        neg_qr = _make_qr(
            question_type="negative",
            answer_text="No facts found here",
            key_facts=["different_fact"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["other_fact"],
        )
        result = _make_run([long_qr, neg_qr, null_qr])
        mr = LongitudinalAdvantage().compute(result)
        # synthesis mean = (1.0 + 0.0) / 2 = 0.5, control mean = 0.0
        assert mr.value == 0.5
        assert mr.details["synthesis_count"] == 2
        assert mr.details["control_count"] == 1


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
            "citation_coverage",
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


class TestTier1Gate:
    """Tier 1 hard gate — composite is 0.0 if any gated metric fails."""

    def test_gate_passes(self):
        """All gated metrics above threshold — normal composite."""
        metrics = [
            MetricResult(name="evidence_grounding", tier=1, value=0.8),
            MetricResult(name="budget_compliance", tier=1, value=1.0),
            MetricResult(name="fact_recall", tier=1, value=0.5),
        ]
        score = compute_composite(metrics)
        assert score > 0.0

    def test_gate_fails_evidence_grounding(self):
        """evidence_grounding below threshold — composite is 0."""
        metrics = [
            MetricResult(name="evidence_grounding", tier=1, value=0.3),
            MetricResult(name="budget_compliance", tier=1, value=1.0),
            MetricResult(name="fact_recall", tier=1, value=1.0),
            MetricResult(name="answer_quality", tier=2, value=1.0),
        ]
        score = compute_composite(metrics)
        assert score == 0.0

    def test_budget_compliance_not_gated(self):
        """budget_compliance is observational — low value does NOT zero composite."""
        metrics = [
            MetricResult(name="evidence_grounding", tier=1, value=1.0),
            MetricResult(name="budget_compliance", tier=1, value=0.2),
            MetricResult(name="answer_quality", tier=2, value=1.0),
        ]
        score = compute_composite(metrics)
        assert score > 0.0

    def test_gate_custom_thresholds(self):
        """Custom gate thresholds override defaults."""
        metrics = [
            MetricResult(name="evidence_grounding", tier=1, value=0.3),
            MetricResult(name="fact_recall", tier=1, value=0.5),
        ]
        # Default gate would fail (evidence_grounding 0.3 < 0.5)
        assert compute_composite(metrics) == 0.0
        # Custom gate with lower threshold passes
        score = compute_composite(
            metrics, gate_thresholds={"evidence_grounding": 0.2}
        )
        assert score > 0.0

    def test_gate_disabled(self):
        """Empty gate_thresholds disables gating entirely."""
        metrics = [
            MetricResult(name="evidence_grounding", tier=1, value=0.1),
            MetricResult(name="budget_compliance", tier=1, value=0.1),
        ]
        score = compute_composite(metrics, gate_thresholds={})
        assert score > 0.0

    def test_gate_metric_not_present(self):
        """Gate metric not in results — gate does not trigger."""
        metrics = [
            MetricResult(name="budget_compliance", tier=1, value=0.5),
        ]
        score = compute_composite(metrics)
        assert score > 0.0

    def test_default_gate_thresholds(self):
        """Default gate thresholds only gate evidence_grounding (budget is observational)."""
        assert TIER1_GATE_THRESHOLDS == {
            "evidence_grounding": 0.5,
        }


class TestPairwiseFactJudge:
    """Tests for the pairwise LLM judge function."""

    def test_candidate_wins_all(self):
        """Judge always picks candidate — win_rate 1.0."""
        def judge_fn(prompt):
            # Candidate may be A or B — we look for its text
            if "candidate answer" in prompt.split("Response A:")[1].split("Response B:")[0]:
                return "A"
            return "B"

        # Use a simpler approach: judge always says A, candidate position varies
        # but we just check the aggregate score
        win_rate, details = pairwise_fact_judge(
            candidate_answer="The pattern is X",
            reference_answer="No findings",
            key_facts=["pattern X"],
            question="What pattern?",
            judge_fn=lambda p: "A" if "The pattern is X" in p.split("Response A:")[1].split("Response B:")[0] else "B",
        )
        assert win_rate == 1.0

    def test_reference_wins_all(self):
        """Judge always picks reference — win_rate 0.0."""
        win_rate, details = pairwise_fact_judge(
            candidate_answer="No findings",
            reference_answer="The pattern is X",
            key_facts=["pattern X"],
            question="What pattern?",
            judge_fn=lambda p: "A" if "The pattern is X" in p.split("Response A:")[1].split("Response B:")[0] else "B",
        )
        assert win_rate == 0.0

    def test_all_ties(self):
        """Judge always says TIE — win_rate 0.5."""
        win_rate, details = pairwise_fact_judge(
            candidate_answer="answer a",
            reference_answer="answer b",
            key_facts=["fact1", "fact2"],
            question="What?",
            judge_fn=lambda p: "TIE",
        )
        assert win_rate == 0.5
        assert all(d["winner"] == "tie" for d in details)
        assert all(d["fact_score"] == 0.5 for d in details)

    def test_position_is_randomized(self):
        """Candidate should appear as both A and B across facts."""
        positions = []

        def judge_fn(prompt):
            positions.append("A")  # always return A
            return "A"

        _, details = pairwise_fact_judge(
            candidate_answer="answer",
            reference_answer="other",
            key_facts=[f"fact_{i}" for i in range(20)],
            question="What?",
            judge_fn=judge_fn,
        )
        # With 20 facts and seed=42, candidate should appear in both positions
        candidate_positions = [d["candidate_position"] for d in details]
        assert "A" in candidate_positions
        assert "B" in candidate_positions

    def test_empty_facts(self):
        """No facts → win_rate 0.5 (uninformative), empty details."""
        win_rate, details = pairwise_fact_judge(
            candidate_answer="answer",
            reference_answer="other",
            key_facts=[],
            question="What?",
            judge_fn=lambda p: "A",
        )
        assert win_rate == 0.5
        assert details == []

    def test_details_structure(self):
        """Each detail has the expected fields."""
        _, details = pairwise_fact_judge(
            candidate_answer="answer",
            reference_answer="other",
            key_facts=["fact1"],
            question="What?",
            judge_fn=lambda p: "TIE",
        )
        assert len(details) == 1
        d = details[0]
        assert d["fact"] == "fact1"
        assert d["winner"] == "tie"
        assert d["verdict_raw"] == "TIE"
        assert d["candidate_position"] in ("A", "B")
        assert d["fact_score"] == 0.5

    def test_seed_reproducibility(self):
        """Same seed produces same position assignments."""
        results_1 = pairwise_fact_judge(
            "a", "b", ["f1", "f2", "f3"], "q",
            judge_fn=lambda p: "A", seed=123,
        )[1]
        results_2 = pairwise_fact_judge(
            "a", "b", ["f1", "f2", "f3"], "q",
            judge_fn=lambda p: "A", seed=123,
        )[1]
        pos_1 = [d["candidate_position"] for d in results_1]
        pos_2 = [d["candidate_position"] for d in results_2]
        assert pos_1 == pos_2

    def test_different_seeds_differ(self):
        """Different seeds produce different position assignments."""
        results_1 = pairwise_fact_judge(
            "a", "b", [f"f{i}" for i in range(10)], "q",
            judge_fn=lambda p: "TIE", seed=1,
        )[1]
        results_2 = pairwise_fact_judge(
            "a", "b", [f"f{i}" for i in range(10)], "q",
            judge_fn=lambda p: "TIE", seed=2,
        )[1]
        pos_1 = [d["candidate_position"] for d in results_1]
        pos_2 = [d["candidate_position"] for d in results_2]
        assert pos_1 != pos_2
