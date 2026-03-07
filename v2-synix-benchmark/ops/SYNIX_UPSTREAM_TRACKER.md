# Synix Upstream Tracker

This file tracks the Synix platform milestone that must land before the benchmark starts local runtime integration work in earnest.

Epic: [#88](https://github.com/marklubin/synix/issues/88) `LENS support: checkpointed artifact banks and runtime tools`

Reference design note: [ops/SYNIX_EXECUTION_MODEL.md](./SYNIX_EXECUTION_MODEL.md)

## Current Benchmark Read

Assume the current upstream baseline includes:

- build-time search capability via `SearchSurface`
- explicit transform access via `TransformContext` and `ctx.search(...)`
- canonical search outputs via `SynixSearch`

Do not treat mutable local directories as part of the stable Synix contract. The remaining milestone should be planned around immutable builds, explicit releases, sealed banks, and the mounted runtime surface.

## Sequential Remaining DAG

```text
[#34 Snapshot/projection/release closeout]
    |
    v
[#81 Checkpoint projections + sealed bank manifests]
    |
    v
[#82 Python-local runtime/tool API, including retrieval over named search surfaces]
    |
    v
[#83 Built-in chunk artifact family]
    |
    +-------------------+--------------------+
    |                   |                    |
    v                   v                    v
[#84 Summary family] [#85 Core-memory family] [#86 Graph family]
    |                   |                    |
    +-------------------+--------------------+
                        |
                        v
[#60 Typed schemas closeout]
                        |
                        v
[#87 Mesh/API parity]
```

## Landed Foundation Versus Remaining Blockers

### Landed enough to stop reopening

- `SearchSurface`: build-time search dependency declaration
- `TransformContext` and `ctx.search(...)`: explicit transform-side search interface
- `SynixSearch`: canonical search output contract
- the first immutable snapshot substrate: object store, build refs, and run refs

### Remaining blockers in order

1. finish snapshot or projection or release closeout so builds only commit immutable state and releases become explicit
2. emit sealed checkpoint banks and immutable bank manifests
3. expose one Python-local runtime or tool API over those sealed banks, including search or retrieval
4. land the built-in chunk family that all downstream memory families will anchor to
5. land built-in summary, core-memory, and graph families
6. close out typed schemas and only then worry about mesh parity

## Feature Checklist

| Order | Issue | Status In Benchmark View | Purpose | Unblocks In LENS | Demo Or Template Follow-On |
|------:|-------|--------------------------|---------|------------------|----------------------------|
| 1 | [#34](https://github.com/marklubin/synix/issues/34) | partially landed, closeout still open | immutable snapshots plus canonical projection capture, release refs, receipts, diff, and revert | T005, T013, all later bank work | extend `05-batch-build` |
| 2 | [#81](https://github.com/marklubin/synix/issues/81) | open blocker | checkpoint projections and sealed bank manifests | T005 and checkpoint-isolation verification | future `checkpointed-memory-bank` demo |
| 3 | [#15](https://github.com/marklubin/synix/issues/15) | materially landed | explicit build-time search dependencies in the DAG | prerequisite already satisfied for T005 and T013 design | topical-search style demo refresh |
| 4 | [#10](https://github.com/marklubin/synix/issues/10) | fold into #82 runtime surface | retrieval over named search surfaces as part of the mounted runtime contract | T013 and downstream family retrieval | extend `02-tv-returns` |
| 5 | [#82](https://github.com/marklubin/synix/issues/82) | open blocker | Python-local runtime or tool API over sealed banks, including retrieval | T013, T007, T008, T009 | future `agent-tool-runtime` demo |
| 6 | [#83](https://github.com/marklubin/synix/issues/83) | open blocker | built-in chunk family with stable IDs and provenance anchors | T005 and all derived-family integrations | revisit template issue `#42` |
| 7 | [#84](https://github.com/marklubin/synix/issues/84) | open blocker | built-in summary family | T008, T010, T011 | extend `01-chatbot-export-synthesis` |
| 8 | [#85](https://github.com/marklubin/synix/issues/85) | open blocker | built-in core-memory family | T007, T010, T011 | future `persistent-user-memory` demo |
| 9 | [#86](https://github.com/marklubin/synix/issues/86) | open blocker | built-in graph family with source-backed graph retrieval | T009, T011 | future `graph-memory-investigation` demo |
| 10 | [#60](https://github.com/marklubin/synix/issues/60) | open follow-on | typed schemas for built-ins and tool payloads | schema and runtime payload stabilization | template docs should cite schemas |
| 11 | [#87](https://github.com/marklubin/synix/issues/87) | open follow-on | Mesh or HTTP parity after local runtime is stable | optional follow-on after first benchmark milestone | extend an existing mesh demo |

## Definition Of Done Per Synix Feature

Every issue above must land with:

- design locked against actual Synix primitives and examples
- implementation complete
- unit tests complete
- at least one automated end-to-end test
- documentation updates
- a recorded demo or template follow-on note

For the snapshot or projection or release closeout specifically, the benchmark should consider the feature done only when the upstream design matches `ops/SYNIX_EXECUTION_MODEL.md`:

- `synix build` commits immutable state only
- `synix release` is the only durable materialization command
- immutable refs are inspectable without relying on a mutable build directory
- release refs, receipts, diff, and revert are all first-class
- checkpoint releases are suitable substrate for sealed bank manifests

## LENS Follow-On After The Synix Milestone

Once the required Synix issues are closed, execute the benchmark-side DAG:

```text
[T001] -> [T002] -> [T003] -> [T004] -> [T005] -> [T013]
                                                |        \
                                                v         v
                                              [T006]   [T007/T008/T009]
                                                 \        /
                                                  v      v
                                                   [T010]
                                                     |
                                                     v
                                                   [T011]
                                                     |
                                                     v
                                                   [T012]
```
