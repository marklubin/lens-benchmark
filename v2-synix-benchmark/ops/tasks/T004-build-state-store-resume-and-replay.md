# T004 - Implement state store, bank or run resume, and replay

status: done
priority: P0
phase: Runtime
owner: claude
created: 2026-03-06
updated: 2026-03-10
depends_on: [T003]
blocks: [T005, T010]

## Purpose

Prevent wasted inference by making every bank build and every run resumable and replayable.

## Scope

In scope:

- `state.sqlite`
- event writer
- bank-build step tracking
- run-step tracking
- resume logic
- replay path without new inference

Out of scope:

- artifact-family logic beyond what is needed to test runtime behavior

## Deliverables

- state-store implementation
- resume command or path
- replay command or path
- failure-injection tests

## Files Or Areas Owned

- runtime execution modules
- tests for resume and replay

## Implementation Plan

1. define build-step and run-step state model
2. persist execution progress
3. resume incomplete steps only
4. replay outputs from stored artifacts

## Verification

- intentionally fail a bank build or run midway
- resume without repeating completed model calls
- replay a completed run with Modal disabled

## Done Criteria

- serious runs can proceed without fear of losing partial progress

## Risks

- hidden recomputation despite apparent resume support

## Verification Record

- 44 unit tests in `tests/test_state.py` — all passing
- CRUD verified for all 6 manifest/record types (study, policy, bank, run, answers, scores)
- Event log: append-only, order-preserved, filtered by study/run/event_type, duplicate rejection
- Resume: completed questions tracked via answers table, completed families via bank artifact_families
- Failure injection: run crash mid-checkpoint and bank crash mid-build both resume correctly
- Replay: completed run's answers and scores loadable for re-scoring without new inference
- WAL mode enabled, idempotent schema init, parent directory auto-creation
- 142 total tests passing across all modules

## Handoff

State store and event writer in `src/bench/state.py`. Import:

```python
from bench.state import StateStore, EventWriter
```

Key APIs for downstream tasks:
- `StateStore(db_path)` — all persistence
- `store.is_question_completed(run_id, question_id)` — resume gate for T005/T013
- `store.get_completed_families(bank_manifest_id)` — resume gate for bank builds
- `EventWriter(store, study_id)` — convenience emitter with auto ID/timestamp
