# T005 - Implement checkpoint-scoped artifact-bank compiler for raw, chunk, and search artifacts

status: ready
priority: P0
phase: Artifact
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T004]
blocks: [T007, T008, T009, T010]

## Purpose

Build the base artifact bank that all runtime policies depend on.

## Scope

In scope:

- raw evidence ingestion
- chunking
- FTS indexes
- embedding indexes
- RRF-ready hybrid retrieval
- provenance-preserving retrieval and ref resolution
- checkpoint-scoped bank snapshot metadata

Out of scope:

- core-memory artifacts
- summary artifacts
- graph artifacts

## Deliverables

- base artifact-bank compiler
- retrieval tools over the base bank
- provenance tests
- citation-resolution tests
- checkpoint-isolation tests

## Files Or Areas Owned

- `src/` artifact compiler and retrieval modules
- related tests

## Implementation Plan

1. ingest raw episodes by checkpoint prefix
2. materialize chunks and indexes
3. expose hybrid search over the compiled bank
4. resolve refs back to canonical evidence

## Verification

- search returns stable refs
- retrieve resolves cited ids
- provenance is preserved back to raw episodes
- no bank snapshot includes future episodes

## Done Criteria

- the base artifact bank works end to end on a smoke-test scope

## Risks

- provenance shortcuts that break later scoring

## Handoff

- to be filled by owner
