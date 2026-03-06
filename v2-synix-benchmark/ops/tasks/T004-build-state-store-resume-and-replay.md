# T004 - Implement state store, bank or run resume, and replay

status: ready
priority: P0
phase: Runtime
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T003]
blocks: [T005, T010]

## Purpose

Prevent wasted inference by making every bank build and every run resumable and replayable.

## Scope

In scope:

- `state.sqlite`
- event writer
- bank-build step tracking
- run-step tracking
- resume logic
- replay path without new inference

Out of scope:

- artifact-family logic beyond what is needed to test runtime behavior

## Deliverables

- state-store implementation
- resume command or path
- replay command or path
- failure-injection tests

## Files Or Areas Owned

- runtime execution modules
- tests for resume and replay

## Implementation Plan

1. define build-step and run-step state model
2. persist execution progress
3. resume incomplete steps only
4. replay outputs from stored artifacts

## Verification

- intentionally fail a bank build or run midway
- resume without repeating completed model calls
- replay a completed run with Modal disabled

## Done Criteria

- serious runs can proceed without fear of losing partial progress

## Risks

- hidden recomputation despite apparent resume support

## Handoff

- to be filled by owner
