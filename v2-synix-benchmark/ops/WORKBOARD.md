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

1. Land PR #92 (projection release v2) and PR #93 (SDK) in Synix — these are the only two critical-path blockers.
2. Start T002 (schemas) and T004 (state/resume/replay) now — no Synix dependency.
3. Synix issues #82–#86 are upstream cleanup, not LENS blockers. LENS implements equivalent functionality using the SDK + generic transforms.
4. Checkpoint isolation is handled in the LENS pipeline definition (D014) — no Synix platform work needed.
5. Do not start study work before resume, replay, and bank reuse are proven.
6. V2 first pass: 7 policies (null, base, core, core_maintained, core_structured, core_faceted, summary). See `docs/plans/memory-strategy-ablation.md`.

## Task Board

| ID | Title | Status | Priority | Phase | Depends On | Owner |
|----|-------|--------|----------|-------|------------|-------|
| [T000](./tasks/T000-bootstrap-v2-workspace.md) | Bootstrap v2 workspace and process docs | done | P0 | Program | - | codex |
| [T001](./tasks/T001-freeze-v2-benchmark-spec.md) | Freeze scope policy, runtime policy set, artifact-bank model, and scoring v2 | done | P0 | Program | T000 | codex |
| [T002](./tasks/T002-design-manifest-and-event-schemas.md) | Design study, policy, bank, run, event, and score schemas | ready | P0 | Runtime | T001 | unassigned |
| [T003](./tasks/T003-build-modal-broker-and-cache.md) | Implement Modal broker, cache, and idempotent call layer | done | P0 | Runtime | T002 | claude |
| [T004](./tasks/T004-build-state-store-resume-and-replay.md) | Implement state store, bank or run resume, and replay | ready | P0 | Runtime | T003 | unassigned |
| [T005](./tasks/T005-implement-artifact-bank-base.md) | Integrate Synix checkpoint banks, chunk artifacts, and layered search into the benchmark base bank path | ready | P0 | Artifact | T004, Synix #92+#93 | unassigned |
| [T013](./tasks/T013-integrate-synix-runtime-policy-layer.md) | Integrate the Synix SDK into benchmark policy execution (search + artifact access) | ready | P0 | Runtime | T005 | unassigned |
| [T006](./tasks/T006-implement-scoring-v2.md) | Implement simplified scoring and audit path | ready | P0 | Scoring | T002,T005 | unassigned |
| [T007](./tasks/T007-implement-core-artifact-family.md) | Implement core-memory policies using FoldSynthesis (core, core_maintained, core_structured, core_faceted) | ready | P1 | Artifact | T005,T013 | unassigned |
| [T008](./tasks/T008-implement-summary-artifact-family.md) | Implement summary policy using GroupSynthesis + ReduceSynthesis | ready | P1 | Artifact | T005,T013 | unassigned |
| [T009](./tasks/T009-implement-graph-artifact-family.md) | Integrate the Synix graph family into benchmark policies | deferred | P2 | Artifact | T005,T013 | unassigned |
| [T010](./tasks/T010-run-smoke-pilot.md) | Run 2-scope artifact-bank smoke pilot and calibrate cost/runtime | proposed | P0 | Study | T004,T005,T006,T007,T008,T013 | unassigned |
| [T011](./tasks/T011-run-feature-screening-study.md) | Run 4-scope runtime-policy screening study (56 configs) | proposed | P1 | Study | T006,T007,T008,T010,T013 | unassigned |
| [T012](./tasks/T012-freeze-main-study-and-reporting.md) | Freeze main-study matrix and reproducible reporting package | proposed | P1 | Study | T011 | unassigned |
| [T014](./tasks/T014-refresh-synix-execution-model-tracker.md) | Refresh Synix execution-model tracker and closeout design | done | P0 | Program | T001 | codex |

## Blockers

| Blocker | Status | Unblocks |
|---------|--------|----------|
| Synix PR #92 (projection release v2) | in progress | T005 |
| Synix PR #93 (Python SDK v0.1.0) | in review | T005, T013 |

## Notes

- **2026-03-09: Blocker chain collapse.** Synix issues #82 (runtime API), #83 (chunks), #84 (summaries), #85 (core-memory) are no longer LENS blockers. The SDK provides the runtime query interface, and generic transforms (FoldSynthesis, GroupSynthesis, ReduceSynthesis, MapSynthesis) provide the memory strategy primitives. LENS implements chunking, core memory, and summaries locally using these generic transforms with policy-specific prompts. See `ops/SYNIX_UPSTREAM_TRACKER.md` for details.
- Track the upstream Synix milestone in `ops/SYNIX_UPSTREAM_TRACKER.md`.
- `ready` on downstream tasks means the local task definition is frozen; execution of T005+ waits on Synix PRs #92 and #93.
- Checkpoint isolation is LENS pipeline logic (D014), not a Synix blocker.
- `T009` (graph) is deferred from v2 first pass — open design questions. See `docs/plans/deferred-memory-strategies.md`.
- Do not start `T010` before replay, resume, and bank reuse checks are proven.
- Do not start `T011` until compilation cost and policy-run cost are known from the smoke pilot.
- T002 and T004 are actionable now — no Synix dependency.
