# T008 - Implement summary artifact family and policy exposure

status: ready
priority: P1
phase: Artifact
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T005]
blocks: [T010, T011]

## Purpose

Implement summary-based memory artifacts with provenance-preserving checkpoint compilation.

## Scope

In scope:

- summary artifact format
- checkpoint or hierarchical summary generation
- runtime exposure of summary artifacts under `policy_summary`

Out of scope:

- summary-only systems with no provenance path

## Deliverables

- summary artifact compiler
- `policy_summary` manifest and runtime exposure
- tests for summary provenance and compilation behavior

## Files Or Areas Owned

- summary artifact modules
- related tests

## Implementation Plan

1. define summary artifact model
2. implement checkpoint-scoped compilation path
3. ensure summary outputs preserve links to source evidence

## Verification

- summary refs resolve to source evidence chains
- summary compilation is cached and resumable
- policy execution does not rebuild summary artifacts

## Done Criteria

- `policy_summary` is runnable in pilot studies

## Risks

- summary compression may lose critical evidence details

## Handoff

- to be filled by owner
