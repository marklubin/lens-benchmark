# Synix Upstream Tracker

This file tracks the Synix platform milestone that must land before the benchmark starts local runtime integration work in earnest.

Epic: [#88](https://github.com/marklubin/synix/issues/88) `LENS support: checkpointed artifact banks and runtime tools`

Reference design note: [ops/SYNIX_EXECUTION_MODEL.md](./SYNIX_EXECUTION_MODEL.md)

## Current Status (2026-03-09)

PR [#92](https://github.com/marklubin/synix/pull/92) is in progress — projection release v2.
PR [#93](https://github.com/marklubin/synix/pull/93) is in review — Python SDK v0.1.0.

**Key realization (2026-03-09):** The SDK (PR #93) provides both the runtime API and the programmatic build/release/search interface that LENS needs. Combined with Synix's existing generic transforms (`FoldSynthesis`, `GroupSynthesis`, `ReduceSynthesis`, `MapSynthesis`), LENS can implement all 7 first-pass memory policies without waiting for dedicated built-in families (#82–#86). Those issues become upstream cleanup, not LENS blockers.

## Critical-Path Blockers (LENS is blocked until these land)

```text
[PR #92 — projection release v2]       ← IN PROGRESS
    |
    v
[PR #93 — Python SDK v0.1.0]           ← IN REVIEW
```

**That's it.** Once #92 and #93 merge, LENS can start T005.

## Why The Blocker Chain Collapsed

Previously, LENS waited on 6 sequential Synix issues (#92 → #82 → #83 → #84 → #85 → #86). Three insights eliminated 4 of them:

1. **SDK IS the runtime API (#82).** `Release.search()`, `Release.artifact()`, and `SearchHandle` provide the exact query interface policies need. No separate runtime/tool API required.

2. **Generic transforms ARE the memory families (#84, #85).** FoldSynthesis = core memory. GroupSynthesis + ReduceSynthesis = summaries. These already exist in `synix.transforms`. The "built-in families" would add opinionated memory-specific prompts, but LENS defines its own prompts per policy anyway.

3. **Chunking is a trivial custom transform (#83).** A non-LLM Transform that splits episodes by token window with overlap. ~30 lines LENS-side. Synix's built-in chunk family would add stable IDs and provenance anchors, but LENS can implement equivalent chunking locally.

## Checkpoint Isolation Design

Unchanged. Checkpoints are LENS domain logic, not a Synix platform feature. The pipeline defines one projection per checkpoint, each scoped to an episode prefix:

```python
for cp_id, max_ep in [("cp01", 6), ("cp02", 12), ("cp03", 16), ("cp04", 20)]:
    pipeline.add(SearchIndex(
        f"bank-{cp_id}",
        sources=[episodes, chunks, core, summaries],
        filter=EpisodePrefix(max_ordinal=max_ep),
        search=["fulltext", "semantic"],
        embedding_config={...},
    ))
```

One `synix build` compiles all artifacts from the full corpus. Each projection declares its input artifacts by label convention. Each `synix release` materializes one projection into a sealed release with a receipt. The release receipt is the bank manifest.

## Landed Foundation

- `SearchSurface`: build-time search dependency declaration
- `TransformContext` and `ctx.search(...)`: explicit transform-side search interface
- `SynixSearch`: canonical search output contract
- First immutable snapshot substrate: object store, build refs, run refs
- Generic transforms: `MapSynthesis`, `GroupSynthesis`, `ReduceSynthesis`, `FoldSynthesis`, `Merge`

## Feature Checklist

### Critical Path (LENS blockers)

| Order | Issue | Status | Purpose | Unblocks In LENS |
|------:|-------|--------|---------|------------------|
| 1 | [PR #92](https://github.com/marklubin/synix/pull/92) / [#34](https://github.com/marklubin/synix/issues/34) | in progress | immutable snapshots, explicit release lifecycle, projection declarations, release receipts | T005, all bank work |
| 2 | [PR #93](https://github.com/marklubin/synix/pull/93) | in review | Python SDK — programmatic build/release/search, `Project`, `Release`, `SearchHandle` | T005, T013, T007, T008 |

### Upstream Cleanup (nice-to-have, not LENS blockers)

These improve the Synix platform but LENS can work around their absence using generic transforms and the SDK.

| Issue | Status | Purpose | LENS Workaround |
|-------|--------|---------|-----------------|
| [#82](https://github.com/marklubin/synix/issues/82) | open | Python-local runtime/tool API | SDK `Release.search()` / `Release.artifact()` |
| [#83](https://github.com/marklubin/synix/issues/83) | open | built-in chunk family | Custom `Transform` in LENS pipeline (~30 lines) |
| [#84](https://github.com/marklubin/synix/issues/84) | open | built-in summary family | `GroupSynthesis` + `ReduceSynthesis` with LENS prompts |
| [#85](https://github.com/marklubin/synix/issues/85) | open | built-in core-memory family | `FoldSynthesis` with LENS prompts |
| [#86](https://github.com/marklubin/synix/issues/86) | open | built-in graph family | Deferred from V2 first pass (T009) |
| [#60](https://github.com/marklubin/synix/issues/60) | open | typed schemas closeout | Not needed for V2 first pass |
| [#87](https://github.com/marklubin/synix/issues/87) | open | mesh/HTTP parity | Not needed — LENS runs Python-local |

### Removed From Tracker

| Issue | Reason |
|-------|--------|
| [#81](https://github.com/marklubin/synix/issues/81) | checkpoint projections are LENS pipeline logic (D014) |
| [#15](https://github.com/marklubin/synix/issues/15) | materially landed |
| [#10](https://github.com/marklubin/synix/issues/10) | folded into #82 (now non-blocking) |

## LENS-Side Implementation Plan

Once #92 and #93 land, all memory strategies are implementable LENS-side:

| Policy | Synix Primitives (existing) | LENS-Side Work |
|--------|---------------------------|----------------|
| null | — | No artifacts, question-only prompt |
| policy_base | `MapSynthesis` (chunker) + `SearchSurface` | Custom chunk transform, search surface config |
| policy_core | `FoldSynthesis` | Fold prompt for free-form core memory |
| policy_core_maintained | `FoldSynthesis` + `MapSynthesis` | Fold prompt + refinement prompt |
| policy_core_structured | `FoldSynthesis` | Fold prompt for structured observations |
| policy_core_faceted | 4× `FoldSynthesis` | 4 facet prompts (entity/relation/event/cause) |
| policy_summary | `GroupSynthesis` + `ReduceSynthesis` | Group/reduce prompts for hierarchical summaries |

Runtime query path for all policies: `project.release(name).search(query)` and `project.release(name).artifact(label)` via the SDK.

## LENS Follow-On DAG

```text
[T001 done] -> [T002] -> [T003 done] -> [T004] -> [T005] -> [T013]
                                                      |         \
                                                      v          v
                                                   [T006]    [T007/T008]
                                                      \         /
                                                       v       v
                                                        [T010]
                                                          |
                                                          v
                                                        [T011]
                                                          |
                                                          v
                                                        [T012]
```

T002 and T004 are actionable now (no Synix dependency).
T005+ waits only on PR #92 + #93.

## Definition Of Done Per Synix Feature

Every critical-path issue above must land with:

- design locked against actual Synix primitives and examples
- implementation complete
- unit tests complete
- at least one automated end-to-end test
- documentation updates
- a recorded demo or template follow-on note
