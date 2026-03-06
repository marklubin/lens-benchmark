# T007 - Implement core-memory artifact family and policy exposure

status: ready
priority: P1
phase: Artifact
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T005]
blocks: [T010, T011]

## Purpose

Implement a maintained core-memory artifact family that approximates the useful part of Letta-like memory without internal-agent confounds.

## Scope

In scope:

- core-memory artifact format
- checkpoint-scoped compilation or maintenance job
- runtime exposure of core-memory artifacts under `policy_core`

Out of scope:

- internal answering agents
- multi-agent coordination

## Deliverables

- core-memory artifact compiler
- `policy_core` manifest and runtime exposure
- tests for provenance and maintenance updates

## Files Or Areas Owned

- core artifact modules
- related tests

## Implementation Plan

1. define core-memory artifact and source links
2. implement checkpoint-scoped update path
3. expose retrieval of core state plus raw evidence

## Verification

- maintenance updates preserve provenance
- agent can retrieve core-memory refs and underlying evidence refs
- policy execution does not rebuild core artifacts

## Done Criteria

- `policy_core` is runnable in pilot studies

## Risks

- core-memory abstraction loses too much evidence detail

## Handoff

- to be filled by owner
