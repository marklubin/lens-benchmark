"""End-to-end integration test for the human benchmark harness.

Starts the HTTP server, drives through a full benchmark via the API,
then verifies the output is compatible with `lens score`.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path

import pytest

from lens.artifacts.bundle import load_run_result
from lens.datasets.loader import load_dataset, load_episodes, load_questions
from lens.human.frontend import generate_human_ui_html
from lens.human.server import HumanBenchmarkHandler
from lens.human.state import HumanBenchmarkState, discover_checkpoints


@pytest.fixture
def smoke_data():
    path = Path(__file__).parent.parent / "src" / "lens" / "datasets" / "smoke" / "smoke_dataset.json"
    # Adjust if run from project root
    if not path.exists():
        path = Path("src/lens/datasets/smoke/smoke_dataset.json")
    return load_dataset(str(path))


@pytest.fixture
def human_server(smoke_data, tmp_path):
    """Start a human benchmark server on a random port, yield base URL, shut down after."""
    run_id = "test_human_e2e"
    output_dir = tmp_path / "output"
    dataset_path = "smoke_dataset.json"

    # Initialize state
    state = HumanBenchmarkState.initialize(run_id, dataset_path, smoke_data)
    state_dir = output_dir / run_id
    state_dir.mkdir(parents=True, exist_ok=True)
    state.save(state_dir / "human_state.json")

    # Configure handler
    questions = load_questions(smoke_data)
    episodes_by_scope = load_episodes(smoke_data)

    HumanBenchmarkHandler.dataset = smoke_data
    HumanBenchmarkHandler.questions = questions
    HumanBenchmarkHandler.episodes_by_scope = episodes_by_scope
    HumanBenchmarkHandler.output_dir = output_dir
    HumanBenchmarkHandler.run_id = run_id
    HumanBenchmarkHandler.html_page = generate_human_ui_html()

    # Use port 0 to let the OS pick a free port
    server = HTTPServer(("127.0.0.1", 0), HumanBenchmarkHandler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}", output_dir, run_id

    server.shutdown()
    server.server_close()


def _get(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


class TestHumanBenchmarkE2E:
    """Drive a full benchmark through the HTTP API."""

    def test_serves_html_page(self, human_server):
        base_url, _, _ = human_server
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode()
        assert "<!DOCTYPE html>" in html
        assert "LENS Human Benchmark" in html

    def test_initial_state(self, human_server):
        base_url, _, run_id = human_server
        state = _get(f"{base_url}/api/state")
        assert state["run_id"] == run_id
        assert state["is_complete"] is False
        assert len(state["scope_order"]) == 2

    def test_episodes_initially_empty(self, human_server):
        base_url, _, _ = human_server
        data = _get(f"{base_url}/api/episodes")
        assert data["episodes"] == []

    def test_next_episode_reveals_one(self, human_server):
        base_url, _, _ = human_server
        data = _post(f"{base_url}/api/next-episode")
        assert "episode" in data
        assert data["episodes_revealed"] == 1
        assert data["checkpoint_triggered"] is False

        # Episodes endpoint should now return 1
        eps = _get(f"{base_url}/api/episodes")
        assert len(eps["episodes"]) == 1

    def test_full_benchmark_flow(self, human_server):
        """Drive through ALL episodes and questions, then finish and verify artifacts."""
        base_url, output_dir, run_id = human_server

        # Get initial state to know scope structure
        state = _get(f"{base_url}/api/state")
        scope_order = state["scope_order"]

        for scope_id in scope_order:
            sp = state["scope_progress"][scope_id]
            total = sp["total_episodes"]

            for ep_num in range(total):
                data = _post(f"{base_url}/api/next-episode")
                assert "episode" in data

                if data["checkpoint_triggered"] and data["pending_questions"]:
                    for q in data["pending_questions"]:
                        # Get episode list to cite some refs
                        eps_data = _get(f"{base_url}/api/episodes")
                        ep_ids = [e["episode_id"] for e in eps_data["episodes"]]
                        # Cite first and last revealed episode
                        refs = [ep_ids[0], ep_ids[-1]] if len(ep_ids) > 1 else ep_ids

                        result = _post(f"{base_url}/api/submit-answer", {
                            "question_id": q["question_id"],
                            "answer_text": f"Human test answer for {q['question_id']}",
                            "refs_cited": refs,
                            "wall_time_ms": 5000,
                        })
                        assert result["status"] == "ok"

            # Refresh state for next scope
            state = _get(f"{base_url}/api/state")

        # Should be complete now
        state = _get(f"{base_url}/api/state")
        assert state["is_complete"] is True

        # Finish and write artifacts
        finish = _post(f"{base_url}/api/finish")
        assert finish["status"] == "ok"
        out_path = Path(finish["output_path"])
        assert out_path.exists()

        # Verify artifacts are loadable by the standard pipeline
        loaded = load_run_result(out_path)
        assert loaded.run_id == run_id
        assert loaded.adapter == "human"
        assert loaded.budget_preset == "human"
        assert len(loaded.scopes) == 2

        # Verify all questions got answers
        all_qr = []
        for scope in loaded.scopes:
            for cp in scope.checkpoints:
                all_qr.extend(cp.question_results)
        assert len(all_qr) == 6  # smoke dataset has 6 questions

        # Verify answer properties (human-specific)
        for qr in all_qr:
            assert qr.answer.turns == []
            assert qr.answer.tool_calls_made == 0
            assert qr.answer.total_tokens == 0
            assert qr.answer.wall_time_ms == 5000
            assert len(qr.answer.refs_cited) >= 1
            assert qr.valid_ref_ids == qr.answer.refs_cited

    def test_submit_rejects_empty_answer(self, human_server):
        base_url, _, _ = human_server
        # Advance to a checkpoint first
        state = _get(f"{base_url}/api/state")
        scope_id = state["scope_order"][0]
        total = state["scope_progress"][scope_id]["total_episodes"]

        # Find first checkpoint
        questions = load_questions(HumanBenchmarkHandler.dataset)
        checkpoints = discover_checkpoints(questions, scope_id)
        first_cp = checkpoints[0] if checkpoints else total

        for _ in range(first_cp):
            data = _post(f"{base_url}/api/next-episode")

        # Try to submit with empty answer
        if data.get("pending_questions"):
            q = data["pending_questions"][0]
            try:
                _post(f"{base_url}/api/submit-answer", {
                    "question_id": q["question_id"],
                    "answer_text": "",
                    "refs_cited": ["ep1"],
                    "wall_time_ms": 100,
                })
                assert False, "Should have returned 400"
            except urllib.error.HTTPError as e:
                assert e.code == 400

    def test_finish_rejects_incomplete(self, human_server):
        base_url, _, _ = human_server
        try:
            _post(f"{base_url}/api/finish")
            assert False, "Should have returned 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
