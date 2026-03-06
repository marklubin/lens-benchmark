# T002 - Design study, policy, bank, run, event, and score schemas

status: ready
priority: P0
phase: Runtime
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T001]
blocks: [T003, T004, T006]

## Purpose

Define the canonical schemas that all builds, runs, scores, and reports will use.

## Scope

In scope:

- study manifest schema
- policy manifest schema
- artifact-bank manifest schema
- run manifest schema
- event schema
- score record schema

Out of scope:

- implementation of runtime services

## Deliverables

- finalized schemas under `schemas/`
- schema validation strategy
- docs describing required fields and invariants

## Files Or Areas Owned

- `schemas/`
- runtime schema docs if needed

## Implementation Plan

1. convert examples into frozen schemas
2. define required versus optional fields
3. define checkpoint-isolation fields and provenance requirements
4. define schema validation path in tests and runtime

## Verification

- validate sample files against chosen schema format
- ensure schema supports replay and report-generation needs

## Done Criteria

- runtime tasks can build directly against the schemas

## Risks

- under-specified schema forces refactors later

## Handoff

- to be filled by owner
