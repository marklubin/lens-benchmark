# T009 - Implement graph artifact family and policy exposure

status: ready
priority: P1
phase: Artifact
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T005]
blocks: [T011]

## Purpose

Implement graph-based memory artifacts as a feature family on the shared substrate.

## Scope

In scope:

- entity extraction
- relation extraction
- graph artifact indexing
- graph-informed retrieval back to source evidence
- runtime exposure under `policy_graph`

Out of scope:

- external graph databases
- complex temporal graph research features not needed for V1

## Deliverables

- graph artifact compiler
- `policy_graph` manifest and runtime exposure
- tests for graph provenance and retrieval

## Files Or Areas Owned

- graph artifact modules
- related tests

## Implementation Plan

1. define graph artifact model
2. implement extraction and indexing path
3. implement retrieval path over graph artifacts plus source evidence
4. ensure compilation is checkpoint-scoped and resumable

## Verification

- graph-derived refs preserve source-evidence lineage
- graph compilation respects cache and resume semantics
- policy execution does not rebuild graph artifacts

## Done Criteria

- `policy_graph` is runnable in feature-screening studies

## Risks

- graph retrieval semantics may not justify the additional build cost

## Handoff

- to be filled by owner
