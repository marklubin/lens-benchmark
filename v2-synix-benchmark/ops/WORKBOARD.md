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

1. Complete the Synix platform milestone before starting LENS runtime integration.
2. Land immutable snapshots, checkpointed banks, and the default Python-local runtime/tool API in Synix sequentially.
3. Require unit coverage, at least one automated e2e test, docs updates, and a demo-extension note for every Synix feature.
4. Keep LENS focused on manifests, Modal broker, state/replay, scoring, and study execution until Synix contracts stabilize.
5. Do not start study work before checkpoint isolation, resume, replay, and bank reuse are proven.

## Task Board

| ID | Title | Status | Priority | Phase | Depends On | Owner |
|----|-------|--------|----------|-------|------------|-------|
| [T000](./tasks/T000-bootstrap-v2-workspace.md) | Bootstrap v2 workspace and process docs | done | P0 | Program | - | codex |
| [T001](./tasks/T001-freeze-v2-benchmark-spec.md) | Freeze scope policy, runtime policy set, artifact-bank model, and scoring v2 | done | P0 | Program | T000 | codex |
| [T002](./tasks/T002-design-manifest-and-event-schemas.md) | Design study, policy, bank, run, event, and score schemas | ready | P0 | Runtime | T001 | unassigned |
| [T003](./tasks/T003-build-modal-broker-and-cache.md) | Implement Modal broker, cache, and idempotent call layer | ready | P0 | Runtime | T002 | unassigned |
| [T004](./tasks/T004-build-state-store-resume-and-replay.md) | Implement state store, bank or run resume, and replay | ready | P0 | Runtime | T003 | unassigned |
| [T005](./tasks/T005-implement-artifact-bank-base.md) | Integrate Synix checkpoint banks, chunk artifacts, and layered search into the benchmark base bank path | ready | P0 | Artifact | T004 | unassigned |
| [T013](./tasks/T013-integrate-synix-runtime-policy-layer.md) | Integrate the Synix runtime/tool API into benchmark policy execution | ready | P0 | Runtime | T005 | unassigned |
| [T006](./tasks/T006-implement-scoring-v2.md) | Implement simplified scoring and audit path | ready | P0 | Scoring | T002,T005 | unassigned |
| [T007](./tasks/T007-implement-core-artifact-family.md) | Integrate the Synix core-memory family into benchmark policies | ready | P1 | Artifact | T005,T013 | unassigned |
| [T008](./tasks/T008-implement-summary-artifact-family.md) | Integrate the Synix summary family into benchmark policies | ready | P1 | Artifact | T005,T013 | unassigned |
| [T009](./tasks/T009-implement-graph-artifact-family.md) | Integrate the Synix graph family into benchmark policies | ready | P1 | Artifact | T005,T013 | unassigned |
| [T010](./tasks/T010-run-smoke-pilot.md) | Run 2-scope artifact-bank smoke pilot and calibrate cost/runtime | proposed | P0 | Study | T004,T005,T006,T007,T008,T013 | unassigned |
| [T011](./tasks/T011-run-feature-screening-study.md) | Run 4-scope runtime-policy screening study | proposed | P1 | Study | T006,T007,T008,T009,T010,T013 | unassigned |
| [T012](./tasks/T012-freeze-main-study-and-reporting.md) | Freeze main-study matrix and reproducible reporting package | proposed | P1 | Study | T011 | unassigned |
| [T014](./tasks/T014-refresh-synix-execution-model-tracker.md) | Refresh Synix execution-model tracker and closeout design | done | P0 | Program | T001 | codex |

## Blockers

None recorded yet.

## Notes

- Track the upstream Synix milestone in `ops/SYNIX_UPSTREAM_TRACKER.md` and do not treat `T005` or later as locally unblocked until the relevant upstream issues are closed.
- The Synix platform milestone is sequential and upstream-first. LENS tasks `T005` and later consume released Synix contracts rather than reimplementing those platform features locally.
- `ready` on downstream tasks means the local task definition is frozen; execution still waits on the mapped upstream Synix issues in `ops/SYNIX_UPSTREAM_TRACKER.md`.
- Do not start `T005` before the upstream Synix milestone covers immutable snapshots, checkpointed banks, the Python-local runtime/tool API, and the built-in chunk family.
- Do not start `T007`, `T008`, or `T009` before the corresponding Synix built-in family exists and `T013` has landed the benchmark runtime integration path.
- Do not start `T010` before replay, resume, checkpoint isolation, and bank reuse checks are proven.
- Do not start `T011` until compilation cost and policy-run cost are known from the smoke pilot.
