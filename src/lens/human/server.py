from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from lens.core.models import Question
from lens.datasets.loader import load_episodes, load_questions
from lens.human.assembler import build_run_result, write_artifacts
from lens.human.frontend import generate_human_ui_html
from lens.human.state import (
    HumanAnswerRecord,
    HumanBenchmarkState,
    discover_checkpoints,
    pending_questions_at_checkpoint,
)


class HumanBenchmarkHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the human benchmark UI.

    Immutable config is set on the class before the server starts:
      dataset, questions, episodes_by_scope, output_dir, run_id, html_page
    """

    # These are set on the class by serve()
    dataset: dict
    questions: list[Question]
    episodes_by_scope: dict[str, list]
    output_dir: Path
    run_id: str
    html_page: str

    def _state_path(self) -> Path:
        return self.output_dir / self.run_id / "human_state.json"

    def _load_state(self) -> HumanBenchmarkState:
        return HumanBenchmarkState.load(self._state_path())

    def _save_state(self, state: HumanBenchmarkState) -> None:
        state.save(self._state_path())

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(self.html_page)
        elif self.path == "/api/state":
            self._handle_get_state()
        elif self.path == "/api/episodes":
            self._handle_get_episodes()
        else:
            self._send_error_json(404, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/next-episode":
            self._handle_next_episode()
        elif self.path == "/api/submit-answer":
            self._handle_submit_answer()
        elif self.path == "/api/finish":
            self._handle_finish()
        else:
            self._send_error_json(404, "Not found")

    def _handle_get_state(self) -> None:
        state = self._load_state()
        self._send_json({
            "run_id": state.run_id,
            "dataset_version": state.dataset_version,
            "current_scope_index": state.current_scope_index,
            "current_scope_id": state.current_scope_id,
            "scope_order": state.scope_order,
            "scope_progress": {
                k: v.to_dict() for k, v in state.scope_progress.items()
            },
            "is_complete": state.is_complete,
        })

    def _handle_get_episodes(self) -> None:
        """Return all episodes revealed so far for the current scope."""
        state = self._load_state()
        sid = state.current_scope_id
        if sid is None:
            self._send_json({"episodes": []})
            return

        progress = state.scope_progress[sid]
        scope_episodes = sorted(
            self.episodes_by_scope.get(sid, []),
            key=lambda e: e.timestamp,
        )
        revealed = scope_episodes[:progress.episodes_revealed]
        self._send_json({
            "episodes": [e.to_dict() for e in revealed],
        })

    def _handle_next_episode(self) -> None:
        state = self._load_state()
        sid = state.current_scope_id

        if sid is None or state.is_complete:
            self._send_error_json(400, "Benchmark already complete")
            return

        progress = state.scope_progress[sid]
        scope_episodes = sorted(
            self.episodes_by_scope.get(sid, []),
            key=lambda e: e.timestamp,
        )

        if progress.episodes_revealed >= progress.total_episodes:
            self._send_error_json(400, "All episodes already revealed for this scope")
            return

        # Reveal next episode
        progress.episodes_revealed += 1
        ep = scope_episodes[progress.episodes_revealed - 1]
        self._save_state(state)

        # Check if this episode count triggers a checkpoint
        ep_count = progress.episodes_revealed
        checkpoints = discover_checkpoints(self.questions, sid)

        checkpoint_triggered = False
        pending: list[dict] = []

        if ep_count in checkpoints:
            answered_ids = {a.question_id for a in progress.answers}
            pq = pending_questions_at_checkpoint(
                self.questions, sid, ep_count, answered_ids
            )
            if pq:
                checkpoint_triggered = True
                pending = [q.to_dict() for q in pq]

        # If all episodes revealed and no checkpoint triggered, check if
        # we need to auto-advance to next scope
        response: dict = {
            "episode": ep.to_dict(),
            "checkpoint_triggered": checkpoint_triggered,
            "pending_questions": pending,
            "episodes_revealed": progress.episodes_revealed,
            "total_episodes": progress.total_episodes,
        }

        self._send_json(response)

    def _handle_submit_answer(self) -> None:
        body = self._read_body()
        question_id = body.get("question_id")
        answer_text = body.get("answer_text", "").strip()
        refs_cited = body.get("refs_cited", [])
        wall_time_ms = body.get("wall_time_ms", 0)

        if not question_id or not answer_text:
            self._send_error_json(400, "question_id and answer_text required")
            return

        # Find the question to get scope_id and checkpoint
        q_match = [q for q in self.questions if q.question_id == question_id]
        if not q_match:
            self._send_error_json(400, f"Unknown question_id: {question_id}")
            return
        question = q_match[0]

        state = self._load_state()
        progress = state.scope_progress.get(question.scope_id)
        if not progress:
            self._send_error_json(400, f"Unknown scope: {question.scope_id}")
            return

        record = HumanAnswerRecord(
            question_id=question_id,
            scope_id=question.scope_id,
            checkpoint=question.checkpoint_after,
            answer_text=answer_text,
            refs_cited=refs_cited,
            wall_time_ms=wall_time_ms,
            answered_at=datetime.now(timezone.utc).isoformat(),
        )
        progress.answers.append(record)

        # Check if this checkpoint is now complete
        answered_ids = {a.question_id for a in progress.answers}
        remaining = pending_questions_at_checkpoint(
            self.questions, question.scope_id, question.checkpoint_after, answered_ids
        )
        if not remaining:
            if question.checkpoint_after not in progress.checkpoints_completed:
                progress.checkpoints_completed.append(question.checkpoint_after)

            # Check if scope is done (all episodes revealed + all checkpoints answered)
            all_cps = discover_checkpoints(self.questions, question.scope_id)
            if all(cp in progress.checkpoints_completed for cp in all_cps):
                # If all episodes also revealed, advance to next scope
                if progress.episodes_revealed >= progress.total_episodes:
                    state.current_scope_index += 1
                    if state.current_scope_index >= len(state.scope_order):
                        state.is_complete = True

        self._save_state(state)
        self._send_json({"status": "ok", "is_complete": state.is_complete})

    def _handle_finish(self) -> None:
        state = self._load_state()
        if not state.is_complete:
            self._send_error_json(400, "Benchmark not yet complete")
            return

        result = build_run_result(state, self.questions)
        out = write_artifacts(result, state, self.output_dir)
        self._send_json({"status": "ok", "output_path": str(out)})

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet down the default stderr logging
        pass


def serve(
    dataset: dict,
    output_dir: Path,
    run_id: str,
    port: int = 8000,
) -> None:
    """Start the human benchmark HTTP server."""
    questions = load_questions(dataset)
    episodes_by_scope = load_episodes(dataset)

    # Set class-level config
    HumanBenchmarkHandler.dataset = dataset
    HumanBenchmarkHandler.questions = questions
    HumanBenchmarkHandler.episodes_by_scope = episodes_by_scope
    HumanBenchmarkHandler.output_dir = output_dir
    HumanBenchmarkHandler.run_id = run_id
    HumanBenchmarkHandler.html_page = generate_human_ui_html()

    server = HTTPServer(("0.0.0.0", port), HumanBenchmarkHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
