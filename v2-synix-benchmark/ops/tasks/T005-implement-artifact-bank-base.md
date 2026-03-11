# T005 - Integrate Synix checkpoint banks, chunk artifacts, and layered search into the benchmark base bank path

status: done
priority: P0
phase: Artifact
owner: claude
created: 2026-03-06
updated: 2026-03-10
depends_on: [T004]
blocks: [T013, T007, T008, T009, T010]

## Purpose

Wire the benchmark to the released Synix base-bank substrate rather than rebuilding that compiler in this repository.

## Scope

In scope:

- benchmark-side configuration for Synix scope and checkpoint builds
- integration with Synix chunk artifacts
- integration with Synix layered search and ref resolution
- benchmark-side selection of checkpoint-scoped sealed bank manifests
- benchmark-side lookup of named `raw`, `chunk`, and `layered_search` projections
- provenance-preserving retrieval through Synix runtime surfaces

Out of scope:

- implementing the Synix snapshot, checkpoint-bank, chunk, or search primitives themselves
- core-memory artifacts
- summary artifacts
- graph artifacts

## Deliverables

- base-bank integration path over released Synix artifacts
- benchmark-side bank manifest wiring
- provenance tests
- citation-resolution tests
- checkpoint-isolation tests

## Files Or Areas Owned

- `src/` benchmark bank integration and retrieval modules
- related tests

## Implementation Plan

1. wire the benchmark manifests to Synix scope and checkpoint build outputs
2. mount the released Synix chunk and layered-search artifacts
3. expose base retrieval handles through the benchmark runtime integration layer over named projection handles
4. resolve refs back to canonical evidence without reading Synix internals directly

## Verification

- search returns stable refs through the Synix runtime API and named projection path
- retrieve resolves cited ids
- provenance is preserved back to raw episodes
- no checkpoint bank includes future episodes

## Done Criteria

- the benchmark base bank path works end to end on a smoke-test scope over released Synix artifacts and sealed manifests

## Risks

- upstream Synix contracts may still shift before the platform milestone closes

## Handoff

- waiting on the Synix upstream milestone
