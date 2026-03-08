# Synix Upstream Tracker

This file tracks the Synix platform milestone that must land before the benchmark starts local runtime integration work in earnest.

Epic: [#88](https://github.com/marklubin/synix/issues/88) `LENS support: checkpointed artifact banks and runtime tools`

Reference design note: [ops/SYNIX_EXECUTION_MODEL.md](./SYNIX_EXECUTION_MODEL.md)

## Current Status (2026-03-08)

PR [#92](https://github.com/marklubin/synix/pull/92) is in progress. This is the critical-path PR — it implements the projection release v2 RFC and removes `build/` as a platform concept.

#81 (checkpoint projections) has been eliminated as a Synix platform requirement. Checkpoint isolation is implemented entirely in the LENS pipeline definition by filtering which artifact labels are included in each projection. See D014 in `ops/DECISIONS.md`.

## Checkpoint Isolation Design

Checkpoints are LENS domain logic, not a Synix platform feature. The pipeline defines one projection per checkpoint, each scoped to an episode prefix:

```python
for cp_id, max_ep in [("cp01", 6), ("cp02", 12), ("cp03", 16), ("cp04", 20)]:
    pipeline.add(SearchIndex(
        f"bank-{cp_id}",
        sources=[episodes, chunks, core, summaries, graph],
        filter=EpisodePrefix(max_ordinal=max_ep),
        search=["fulltext", "semantic"],
        embedding_config={...},
    ))
```

One `synix build` compiles all artifacts from the full corpus. Each projection declares its input artifacts by label convention (e.g., artifacts derived from episodes 1-N). Each `synix release` materializes one projection into a sealed release with a receipt. The release receipt is the bank manifest.

No Synix-side checkpoint concept is needed.

## Revised Sequential DAG

```text
[#34 / PR #92 — projection release v2]       ← IN PROGRESS
    |
    v
[#82 Python-local runtime/tool API]
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

Compared to the prior DAG, #81 is removed entirely. The first blocker is PR #92.

## Landed Foundation

- `SearchSurface`: build-time search dependency declaration
- `TransformContext` and `ctx.search(...)`: explicit transform-side search interface
- `SynixSearch`: canonical search output contract
- first immutable snapshot substrate: object store, build refs, run refs

## Remaining Blockers In Order

1. **PR #92 / #34** (IN PROGRESS): projection release v2 — immutable builds, explicit release lifecycle, SnapshotView, release-aware search, no `build/` directory. 1653 tests passing, 5 demos green. ~9K additions, ~15K deletions.
2. **#82**: Python-local runtime/tool API over sealed banks, including retrieval over named search surfaces. First integration path for LENS policies.
3. **#83**: built-in chunk artifact family with stable IDs and provenance anchors. Required substrate for all derived families.
4. **#84/#85/#86** (parallel): built-in summary, core-memory, and graph families.
5. **#60**: typed schemas closeout — follow-on.
6. **#87**: mesh/API parity — follow-on.

## Feature Checklist

| Order | Issue | Status | Purpose | Unblocks In LENS |
|------:|-------|--------|---------|------------------|
| 1 | [PR #92](https://github.com/marklubin/synix/pull/92) / [#34](https://github.com/marklubin/synix/issues/34) | in progress | immutable snapshots, explicit release lifecycle, projection declarations, release receipts | T005, T013, all later bank work |
| 2 | [#82](https://github.com/marklubin/synix/issues/82) | open blocker | Python-local runtime/tool API over sealed banks | T013, T007, T008, T009 |
| 3 | [#83](https://github.com/marklubin/synix/issues/83) | open blocker | built-in chunk family with stable IDs and provenance | T005, all derived families |
| 4 | [#84](https://github.com/marklubin/synix/issues/84) | open blocker | built-in summary family | T008, T010, T011 |
| 5 | [#85](https://github.com/marklubin/synix/issues/85) | open blocker | built-in core-memory family | T007, T010, T011 |
| 6 | [#86](https://github.com/marklubin/synix/issues/86) | open blocker | built-in graph family | T009, T011 |
| 7 | [#60](https://github.com/marklubin/synix/issues/60) | open follow-on | typed schemas | schema stabilization |
| 8 | [#87](https://github.com/marklubin/synix/issues/87) | open follow-on | mesh/HTTP parity | optional |

Removed:

| Issue | Reason |
|-------|--------|
| [#81](https://github.com/marklubin/synix/issues/81) | checkpoint projections are LENS pipeline logic, not a platform feature — handled by per-checkpoint projection declarations in the pipeline definition over the existing PR #92 model |
| [#15](https://github.com/marklubin/synix/issues/15) | materially landed, prerequisite satisfied |
| [#10](https://github.com/marklubin/synix/issues/10) | folded into #82 |

## Definition Of Done Per Synix Feature

Every issue above must land with:

- design locked against actual Synix primitives and examples
- implementation complete
- unit tests complete
- at least one automated end-to-end test
- documentation updates
- a recorded demo or template follow-on note

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
