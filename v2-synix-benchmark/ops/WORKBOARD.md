# Workboard

## Status Legend

- `proposed`
- `ready`
- `in_progress`
- `blocked`
- `review`
- `done`
- `wont_do`

## Active Priorities

1. Freeze the artifact-bank benchmark spec and process.
2. Build schemas, broker, cache, and replay before any expensive study work.
3. Prove checkpoint isolation, resume, and replay before serious runs.
4. Build the base artifact-bank compiler before derived artifact families.
5. Keep the initial policy matrix small and auditable.

## Task Board

| ID | Title | Status | Priority | Phase | Depends On | Owner |
|----|-------|--------|----------|-------|------------|-------|
| [T000](./tasks/T000-bootstrap-v2-workspace.md) | Bootstrap v2 workspace and process docs | done | P0 | Program | - | codex |
| [T001](./tasks/T001-freeze-v2-benchmark-spec.md) | Freeze scope policy, runtime policy set, artifact-bank model, and scoring v2 | ready | P0 | Program | T000 | unassigned |
| [T002](./tasks/T002-design-manifest-and-event-schemas.md) | Design study, policy, bank, run, event, and score schemas | ready | P0 | Runtime | T001 | unassigned |
| [T003](./tasks/T003-build-modal-broker-and-cache.md) | Implement Modal broker, cache, and idempotent call layer | ready | P0 | Runtime | T002 | unassigned |
| [T004](./tasks/T004-build-state-store-resume-and-replay.md) | Implement state store, bank or run resume, and replay | ready | P0 | Runtime | T003 | unassigned |
| [T005](./tasks/T005-implement-artifact-bank-base.md) | Implement checkpoint-scoped artifact-bank compiler for raw, chunk, and search artifacts | ready | P0 | Artifact | T004 | unassigned |
| [T006](./tasks/T006-implement-scoring-v2.md) | Implement simplified scoring and audit path | ready | P0 | Scoring | T002 | unassigned |
| [T007](./tasks/T007-implement-core-artifact-family.md) | Implement core-memory artifact family and policy exposure | ready | P1 | Artifact | T005 | unassigned |
| [T008](./tasks/T008-implement-summary-artifact-family.md) | Implement summary artifact family and policy exposure | ready | P1 | Artifact | T005 | unassigned |
| [T009](./tasks/T009-implement-graph-artifact-family.md) | Implement graph artifact family and policy exposure | ready | P1 | Artifact | T005 | unassigned |
| [T010](./tasks/T010-run-smoke-pilot.md) | Run 2-scope artifact-bank smoke pilot and calibrate cost/runtime | proposed | P0 | Study | T004,T005,T006,T007,T008 | unassigned |
| [T011](./tasks/T011-run-feature-screening-study.md) | Run 4-scope runtime-policy screening study | proposed | P1 | Study | T007,T008,T009,T010 | unassigned |
| [T012](./tasks/T012-freeze-main-study-and-reporting.md) | Freeze main-study matrix and reproducible reporting package | proposed | P1 | Study | T011 | unassigned |

## Blockers

None recorded yet.

## Notes

- Do not start `T010` before replay, resume, and checkpoint-isolation checks are proven.
- `T007`, `T008`, and `T009` can proceed in parallel once `T005` is complete.
- Do not start `T011` until compilation cost and policy-run cost are known from the smoke pilot.
