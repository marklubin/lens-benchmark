"""State store for V2 benchmark execution.

SQLite-backed persistence for manifests, events, answers, and scores.
Provides the resume and replay primitives that T005/T013 orchestrators
will call.

Design:
    - WAL mode for concurrent read safety
    - All writes are immediate (no batching) — correctness over throughput
    - Manifests stored as JSON blobs with indexed keys
    - Events append-only, never mutated
    - Answers keyed by (run_id, question_id) — atomic unit of resume
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.schemas import (
    BankManifest,
    Event,
    EventType,
    PolicyManifest,
    RunManifest,
    ScoreRecord,
    StudyManifest,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS studies (
    study_id   TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_manifest_id TEXT PRIMARY KEY,
    policy_id          TEXT NOT NULL,
    data               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS banks (
    bank_manifest_id TEXT PRIMARY KEY,
    study_id         TEXT NOT NULL,
    scope_id         TEXT NOT NULL,
    checkpoint_id    TEXT NOT NULL,
    status           TEXT NOT NULL,
    data             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_banks_study_scope
    ON banks(study_id, scope_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    study_id            TEXT NOT NULL,
    scope_id            TEXT NOT NULL,
    policy_id           TEXT NOT NULL,
    status              TEXT NOT NULL,
    data                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_study_scope_policy
    ON runs(study_id, scope_id, policy_id);

CREATE TABLE IF NOT EXISTS events (
    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT UNIQUE NOT NULL,
    timestamp  TEXT NOT NULL,
    study_id   TEXT NOT NULL,
    run_id     TEXT,
    event_type TEXT NOT NULL,
    data       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run
    ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_study_type
    ON events(study_id, event_type);

CREATE TABLE IF NOT EXISTS answers (
    run_id        TEXT NOT NULL,
    question_id   TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    data          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (run_id, question_id)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id      TEXT NOT NULL,
    question_id TEXT NOT NULL,
    data        TEXT NOT NULL,
    scored_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, question_id)
);
"""


class StateStore:
    """SQLite-backed state store for benchmark execution."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA_SQL)

        # Check schema version
        cur.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(_SCHEMA_VERSION)),
        )
        self._conn.commit()

        row = cur.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row and int(row[0]) != _SCHEMA_VERSION:
            raise RuntimeError(
                f"State store schema version mismatch: "
                f"expected {_SCHEMA_VERSION}, got {row[0]}"
            )

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Studies
    # ------------------------------------------------------------------

    def save_study(self, study: StudyManifest) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO studies (study_id, data, created_at) VALUES (?, ?, ?)",
            (study.study_id, study.model_dump_json(), study.created_at.isoformat()),
        )
        self._conn.commit()

    def get_study(self, study_id: str) -> StudyManifest | None:
        row = self._conn.execute(
            "SELECT data FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        if row is None:
            return None
        return StudyManifest.model_validate_json(row[0])

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------

    def save_policy(self, policy: PolicyManifest) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO policies (policy_manifest_id, policy_id, data) VALUES (?, ?, ?)",
            (policy.policy_manifest_id, policy.policy_id, policy.model_dump_json()),
        )
        self._conn.commit()

    def get_policy(self, policy_manifest_id: str) -> PolicyManifest | None:
        row = self._conn.execute(
            "SELECT data FROM policies WHERE policy_manifest_id = ?",
            (policy_manifest_id,),
        ).fetchone()
        if row is None:
            return None
        return PolicyManifest.model_validate_json(row[0])

    # ------------------------------------------------------------------
    # Banks
    # ------------------------------------------------------------------

    def save_bank(self, bank: BankManifest) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO banks "
            "(bank_manifest_id, study_id, scope_id, checkpoint_id, status, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                bank.bank_manifest_id,
                bank.study_id,
                bank.scope_id,
                bank.checkpoint_id,
                bank.status.value,
                bank.model_dump_json(),
            ),
        )
        self._conn.commit()

    def get_bank(self, bank_manifest_id: str) -> BankManifest | None:
        row = self._conn.execute(
            "SELECT data FROM banks WHERE bank_manifest_id = ?",
            (bank_manifest_id,),
        ).fetchone()
        if row is None:
            return None
        return BankManifest.model_validate_json(row[0])

    def get_banks_for_scope(
        self, study_id: str, scope_id: str
    ) -> list[BankManifest]:
        rows = self._conn.execute(
            "SELECT data FROM banks WHERE study_id = ? AND scope_id = ? "
            "ORDER BY checkpoint_id",
            (study_id, scope_id),
        ).fetchall()
        return [BankManifest.model_validate_json(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def save_run(self, run: RunManifest) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, study_id, scope_id, policy_id, status, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run.run_id,
                run.study_id,
                run.scope_id,
                run.policy_id,
                run.status.value,
                run.model_dump_json(),
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> RunManifest | None:
        row = self._conn.execute(
            "SELECT data FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunManifest.model_validate_json(row[0])

    def get_runs_for_study(self, study_id: str) -> list[RunManifest]:
        rows = self._conn.execute(
            "SELECT data FROM runs WHERE study_id = ? ORDER BY run_id",
            (study_id,),
        ).fetchall()
        return [RunManifest.model_validate_json(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # Events (append-only)
    # ------------------------------------------------------------------

    def append_event(self, event: Event) -> None:
        self._conn.execute(
            "INSERT INTO events (event_id, timestamp, study_id, run_id, event_type, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.timestamp.isoformat(),
                event.study_id,
                event.run_id,
                event.event_type.value,
                event.model_dump_json(),
            ),
        )
        self._conn.commit()

    def get_events(
        self,
        *,
        study_id: str | None = None,
        run_id: str | None = None,
        event_type: EventType | None = None,
    ) -> list[Event]:
        clauses: list[str] = []
        params: list[Any] = []
        if study_id is not None:
            clauses.append("study_id = ?")
            params.append(study_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type.value)

        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT data FROM events WHERE {where} ORDER BY rowid",
            params,
        ).fetchall()
        return [Event.model_validate_json(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # Answers
    # ------------------------------------------------------------------

    def save_answer(
        self,
        run_id: str,
        question_id: str,
        checkpoint_id: str,
        answer: dict[str, Any],
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO answers "
            "(run_id, question_id, checkpoint_id, data, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                question_id,
                checkpoint_id,
                json.dumps(answer),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def get_answer(self, run_id: str, question_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data, checkpoint_id FROM answers WHERE run_id = ? AND question_id = ?",
            (run_id, question_id),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["checkpoint_id"] = row[1]
        return data

    def get_answers(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data FROM answers WHERE run_id = ? ORDER BY checkpoint_id, question_id",
            (run_id,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def save_score(self, score: ScoreRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scores "
            "(run_id, question_id, data, scored_at) VALUES (?, ?, ?, ?)",
            (
                score.run_id,
                score.question_id,
                score.model_dump_json(),
                score.scored_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_score(self, run_id: str, question_id: str) -> ScoreRecord | None:
        row = self._conn.execute(
            "SELECT data FROM scores WHERE run_id = ? AND question_id = ?",
            (run_id, question_id),
        ).fetchone()
        if row is None:
            return None
        return ScoreRecord.model_validate_json(row[0])

    def get_scores(self, run_id: str) -> list[ScoreRecord]:
        rows = self._conn.execute(
            "SELECT data FROM scores WHERE run_id = ? ORDER BY question_id",
            (run_id,),
        ).fetchall()
        return [ScoreRecord.model_validate_json(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # Resume queries
    # ------------------------------------------------------------------

    def get_completed_questions(self, run_id: str) -> set[str]:
        """Return the set of question_ids that have saved answers for a run."""
        rows = self._conn.execute(
            "SELECT question_id FROM answers WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {r[0] for r in rows}

    def get_completed_families(self, bank_manifest_id: str) -> set[str]:
        """Return the set of artifact families with non-null hashes in a bank.

        A family with a non-null hash in artifact_families is considered
        completed. This is the resume signal for bank builds.
        """
        bank = self.get_bank(bank_manifest_id)
        if bank is None:
            return set()
        return {
            family
            for family, hash_val in bank.artifact_families.items()
            if hash_val is not None
        }

    def is_question_completed(self, run_id: str, question_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM answers WHERE run_id = ? AND question_id = ?",
            (run_id, question_id),
        ).fetchone()
        return row is not None


class EventWriter:
    """Convenience wrapper for emitting events with shared context."""

    def __init__(self, store: StateStore, study_id: str) -> None:
        self._store = store
        self._study_id = study_id

    def emit(
        self,
        event_type: EventType,
        *,
        run_id: str | None = None,
        scope_id: str | None = None,
        policy_id: str | None = None,
        bank_manifest_id: str | None = None,
        attempt: int | None = None,
        config_hash: str | None = None,
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Event:
        event = Event(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            study_id=self._study_id,
            run_id=run_id,
            scope_id=scope_id,
            policy_id=policy_id,
            bank_manifest_id=bank_manifest_id,
            event_type=event_type,
            attempt=attempt,
            config_hash=config_hash,
            input_refs=input_refs or [],
            output_refs=output_refs or [],
            payload=payload or {},
            error=error,
        )
        self._store.append_event(event)
        return event
