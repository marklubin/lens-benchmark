from __future__ import annotations

import json
from pathlib import Path

import pytest

from lens.artifacts.bundle import load_run_result
from lens.core.models import GroundTruth, Question
from lens.human.assembler import build_run_result, write_artifacts
from lens.human.state import HumanAnswerRecord, HumanBenchmarkState, ScopeProgress


def _make_questions() -> list[Question]:
    return [
        Question(
            question_id="q1",
            scope_id="scope_a",
            checkpoint_after=5,
            question_type="longitudinal",
            prompt="What happened?",
            ground_truth=GroundTruth(
                canonical_answer="answer1",
                required_evidence_refs=["ep1"],
                key_facts=["fact1"],
            ),
        ),
        Question(
            question_id="q2",
            scope_id="scope_a",
            checkpoint_after=10,
            question_type="null_hypothesis",
            prompt="Detail?",
            ground_truth=GroundTruth(
                canonical_answer="answer2",
                required_evidence_refs=["ep2"],
                key_facts=["fact2"],
            ),
        ),
        Question(
            question_id="q3",
            scope_id="scope_b",
            checkpoint_after=3,
            question_type="action_recommendation",
            prompt="What next?",
            ground_truth=GroundTruth(
                canonical_answer="answer3",
                required_evidence_refs=["ep3"],
                key_facts=["fact3"],
            ),
        ),
    ]


def _make_completed_state() -> HumanBenchmarkState:
    return HumanBenchmarkState(
        run_id="human_run_001",
        dataset_path="/test/ds.json",
        dataset_version="1.0-test",
        current_scope_index=2,
        scope_order=["scope_a", "scope_b"],
        scope_progress={
            "scope_a": ScopeProgress(
                scope_id="scope_a",
                episodes_revealed=10,
                total_episodes=10,
                checkpoints_completed=[5, 10],
                answers=[
                    HumanAnswerRecord(
                        question_id="q1",
                        scope_id="scope_a",
                        checkpoint=5,
                        answer_text="Human answer for q1",
                        refs_cited=["scope_a_ep001", "scope_a_ep003"],
                        wall_time_ms=15000.0,
                        answered_at="2024-06-01T12:00:00Z",
                    ),
                    HumanAnswerRecord(
                        question_id="q2",
                        scope_id="scope_a",
                        checkpoint=10,
                        answer_text="Human answer for q2",
                        refs_cited=["scope_a_ep005"],
                        wall_time_ms=22000.0,
                        answered_at="2024-06-01T12:05:00Z",
                    ),
                ],
            ),
            "scope_b": ScopeProgress(
                scope_id="scope_b",
                episodes_revealed=5,
                total_episodes=5,
                checkpoints_completed=[3],
                answers=[
                    HumanAnswerRecord(
                        question_id="q3",
                        scope_id="scope_b",
                        checkpoint=3,
                        answer_text="Human answer for q3",
                        refs_cited=["scope_b_ep001", "scope_b_ep002"],
                        wall_time_ms=18000.0,
                        answered_at="2024-06-01T12:10:00Z",
                    ),
                ],
            ),
        },
        is_complete=True,
    )


class TestBuildRunResult:
    def test_produces_valid_run_result(self):
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)

        assert result.run_id == "human_run_001"
        assert result.adapter == "human"
        assert result.dataset_version == "1.0-test"
        assert result.budget_preset == "human"

    def test_correct_scope_count(self):
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)
        assert len(result.scopes) == 2

    def test_scope_a_has_two_checkpoints(self):
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)

        scope_a = [s for s in result.scopes if s.scope_id == "scope_a"][0]
        assert len(scope_a.checkpoints) == 2
        assert scope_a.checkpoints[0].checkpoint == 5
        assert scope_a.checkpoints[1].checkpoint == 10

    def test_question_result_fields(self):
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)

        scope_a = [s for s in result.scopes if s.scope_id == "scope_a"][0]
        qr = scope_a.checkpoints[0].question_results[0]

        assert qr.question.question_id == "q1"
        assert qr.answer.answer_text == "Human answer for q1"
        assert qr.answer.turns == []
        assert qr.answer.tool_calls_made == 0
        assert qr.answer.total_tokens == 0
        assert qr.answer.wall_time_ms == 15000.0
        assert qr.answer.refs_cited == ["scope_a_ep001", "scope_a_ep003"]
        assert qr.valid_ref_ids == ["scope_a_ep001", "scope_a_ep003"]
        assert qr.retrieved_ref_ids == ["scope_a_ep001", "scope_a_ep003"]

    def test_scope_b_checkpoint(self):
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)

        scope_b = [s for s in result.scopes if s.scope_id == "scope_b"][0]
        assert len(scope_b.checkpoints) == 1
        assert scope_b.checkpoints[0].checkpoint == 3
        assert len(scope_b.checkpoints[0].question_results) == 1

    def test_empty_answers_produce_no_checkpoints(self):
        state = _make_completed_state()
        state.scope_progress["scope_b"].answers = []
        questions = _make_questions()
        result = build_run_result(state, questions)

        scope_b = [s for s in result.scopes if s.scope_id == "scope_b"][0]
        assert len(scope_b.checkpoints) == 0


class TestWriteArtifacts:
    def test_creates_standard_directory_structure(self, tmp_path: Path):
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)
        out = write_artifacts(result, state, tmp_path)

        assert out == tmp_path / "human_run_001"
        assert (out / "run_manifest.json").exists()
        assert (out / "human_state.json").exists()
        assert (out / "scopes" / "scope_a" / "checkpoint_5" / "question_results.json").exists()
        assert (out / "scopes" / "scope_a" / "checkpoint_10" / "question_results.json").exists()
        assert (out / "scopes" / "scope_b" / "checkpoint_3" / "question_results.json").exists()

    def test_manifest_content(self, tmp_path: Path):
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)
        out = write_artifacts(result, state, tmp_path)

        manifest = json.loads((out / "run_manifest.json").read_text())
        assert manifest["run_id"] == "human_run_001"
        assert manifest["adapter"] == "human"
        assert manifest["dataset_version"] == "1.0-test"
        assert manifest["budget_preset"] == "human"

    def test_question_results_content(self, tmp_path: Path):
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)
        out = write_artifacts(result, state, tmp_path)

        qr_path = out / "scopes" / "scope_a" / "checkpoint_5" / "question_results.json"
        data = json.loads(qr_path.read_text())
        assert len(data) == 1
        assert data[0]["answer"]["question_id"] == "q1"
        assert data[0]["answer"]["answer_text"] == "Human answer for q1"
        assert data[0]["answer"]["turns"] == []
        assert data[0]["answer"]["total_tokens"] == 0

    def test_compatible_with_load_run_result(self, tmp_path: Path):
        """Artifacts must be loadable by the standard load_run_result function."""
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)
        out = write_artifacts(result, state, tmp_path)

        loaded = load_run_result(out)
        assert loaded.run_id == "human_run_001"
        assert loaded.adapter == "human"
        assert loaded.budget_preset == "human"
        assert len(loaded.scopes) == 2

        scope_a = [s for s in loaded.scopes if s.scope_id == "scope_a"][0]
        assert len(scope_a.checkpoints) == 2
        # load_run_result sorts dirs lexicographically: checkpoint_10 < checkpoint_5
        cp5 = [c for c in scope_a.checkpoints if c.checkpoint == 5][0]
        assert cp5.question_results[0].answer.answer_text == "Human answer for q1"

    def test_idempotent_write(self, tmp_path: Path):
        """Writing artifacts twice doesn't corrupt anything."""
        state = _make_completed_state()
        questions = _make_questions()
        result = build_run_result(state, questions)
        write_artifacts(result, state, tmp_path)
        out = write_artifacts(result, state, tmp_path)

        loaded = load_run_result(out)
        assert loaded.run_id == "human_run_001"
        assert len(loaded.scopes) == 2
