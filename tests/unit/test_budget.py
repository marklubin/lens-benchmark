from __future__ import annotations

import pytest

from lens.agent.budget_enforcer import BudgetEnforcement, BudgetViolation, QuestionBudget


class TestQuestionBudget:
    def test_defaults(self):
        budget = QuestionBudget()
        assert budget.max_turns == 10
        assert budget.max_total_tool_calls == 20
        assert budget.max_agent_tokens == 8192

    def test_custom(self):
        budget = QuestionBudget(max_turns=5, max_total_tool_calls=10)
        assert budget.max_turns == 5
        assert budget.max_total_tool_calls == 10


class TestBudgetEnforcement:
    def test_check_turn_within_limit(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_turns=5))
        enforcer.record_turn()  # turns_used = 1
        enforcer.check_turn()  # 1 < 5, should not raise

    def test_check_turn_exceeds_limit(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_turns=2))
        enforcer.record_turn()
        enforcer.record_turn()  # turns_used = 2
        with pytest.raises(BudgetViolation, match="Turn limit exceeded"):
            enforcer.check_turn()  # 2 >= 2
        assert len(enforcer.violations) == 1

    def test_check_tool_call_within_limit(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_total_tool_calls=3))
        enforcer.record_tool_call()
        enforcer.record_tool_call()
        enforcer.check_tool_call()  # 2 < 3, should not raise

    def test_check_tool_call_exceeds_limit(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_total_tool_calls=2))
        enforcer.record_tool_call()
        enforcer.record_tool_call()
        with pytest.raises(BudgetViolation, match="Tool call limit exceeded"):
            enforcer.check_tool_call()

    def test_check_tokens_within_limit(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_agent_tokens=1000))
        enforcer.check_tokens(500)  # Should not raise
        assert enforcer.total_tokens == 500

    def test_check_tokens_exceeds_limit(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_agent_tokens=1000))
        enforcer.check_tokens(500)
        with pytest.raises(BudgetViolation, match="Token limit exceeded"):
            enforcer.check_tokens(600)

    def test_check_payload_records_warning_not_violation(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_payload_bytes=100))
        enforcer.check_payload(200)  # Should warn but not raise or count as violation
        assert len(enforcer.violations) == 0
        assert len(enforcer.warnings) == 1
        assert enforcer.total_payload_bytes == 200

    def test_check_latency_records_violation(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_latency_per_call_ms=100))
        enforcer.check_latency(50)  # within limit
        assert len(enforcer.violations) == 0
        enforcer.check_latency(200)  # exceeds limit
        assert len(enforcer.violations) == 1

    def test_record_turn(self):
        enforcer = BudgetEnforcement(QuestionBudget())
        assert enforcer.turns_used == 0
        enforcer.record_turn()
        assert enforcer.turns_used == 1

    def test_is_exhausted_turns(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_turns=2))
        assert not enforcer.is_exhausted
        enforcer.record_turn()
        enforcer.record_turn()
        assert enforcer.is_exhausted

    def test_is_exhausted_tool_calls(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_total_tool_calls=1))
        enforcer.record_tool_call()
        assert enforcer.is_exhausted

    def test_is_exhausted_tokens(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_agent_tokens=100))
        enforcer.record_tokens(100)
        assert enforcer.is_exhausted

    def test_summary(self):
        enforcer = BudgetEnforcement(QuestionBudget(max_turns=5))
        enforcer.record_turn()
        enforcer.record_tool_call()
        enforcer.record_tokens(50)
        summary = enforcer.summary()
        assert summary["turns_used"] == 1
        assert summary["tool_calls_used"] == 1
        assert summary["total_tokens"] == 50
        assert summary["violations"] == []
        assert summary["warnings"] == []
        assert not summary["is_exhausted"]
