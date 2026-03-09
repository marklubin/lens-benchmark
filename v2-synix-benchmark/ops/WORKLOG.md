# Worklog

Append-only operational log for meaningful progress, blockers, and verification results.

## 2026-03-06

- Created V2 workspace scaffold.
- Added process guardrails in `CLAUDE.md`, `AGENTS.md`, and `AGENT.md`.
- Added benchmark spec, execution plan, and scoring v2 docs.
- Added workboard, task template, risk register, decision log, runbook, and initial task backlog.
- Reframed V2 around checkpoint-scoped artifact-bank compilation and runtime-policy evaluation.
- Updated the benchmark spec, execution plan, runbook, workboard, risk register, decision log, and task backlog to match the new model.
- No implementation code exists yet in `src/`.
- Froze the Synix versus LENS ownership split and converted downstream tasks from local artifact-family implementation to Synix integration work.
- Added `T013` for benchmark integration with the Synix runtime and tool API.
- Created `ops/SYNIX_UPSTREAM_TRACKER.md` with the sequential Synix-first DAG and issue mapping.
- Created Synix epic `#88` plus child issues `#81`, `#82`, `#83`, `#84`, `#85`, `#86`, and `#87`.
- Amended Synix issues `#34`, `#15`, `#10`, and `#60` to add sequencing, e2e, docs, and demo-follow-on requirements.
- Finalized the projection contract around sealed bank manifests and named projection handles, then reconciled downstream task dependencies with `T013` and the upstream tracker.

## 2026-03-08

- Eliminated Synix issue #81 (checkpoint projections + sealed bank manifests) as a platform requirement. Checkpoint isolation is handled entirely by LENS pipeline logic: one projection per checkpoint prefix, using label filtering over the existing projection/release model from PR #92.
- Recorded decision D014.
- Rewrote `ops/SYNIX_UPSTREAM_TRACKER.md` to reflect the simplified blocker DAG (#81 removed), current PR #92 status, and the checkpoint isolation design.
- Updated workboard to remove #81 from blocker notes and reflect PR #92 as the active critical-path item.
- Cleaned up stale git worktree (`worktree-new-scopes-10-12`) — branch was already merged into main.
- T003 partial: Implemented `ResponseCache` (SQLite WAL, content-addressed, thread-safe with lock) and `ModalBroker` (cache-through LLM + embedding calls, retry with backoff, token/cost accounting). 31 unit tests passing. Parameterized `min_containers` in `infra/modal/llm_server.py` via `LENS_MIN_CONTAINERS` env var (default 0). Embedding model unchanged (gte-modernbert-base). Design doc at `docs/plans/2026-03-08-modal-infra-and-cache-design.md`.
