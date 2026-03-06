# T006 - Implement simplified scoring and audit path

status: ready
priority: P0
phase: Scoring
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T002]
blocks: [T010, T011, T012]

## Purpose

Replace the previous complex scorer with the new three-metric scoring pipeline.

## Scope

In scope:

- fact scorer
- citation validator
- evidence-support scorer
- primary composite calculation
- rescoring from saved answers
- audit export

Out of scope:

- old legacy scoring metrics except optional diagnostics

## Deliverables

- scoring pipeline implementation
- score-record artifacts
- tests and sample audit export

## Files Or Areas Owned

- scoring modules in `src/`
- scoring tests in `tests/`

## Implementation Plan

1. implement fact scoring
2. implement citation-validity checks
3. implement evidence-support prompt and parser
4. add score aggregation and export

## Verification

- scoring runs from saved answers only
- invalid citations are handled mechanically
- score records validate against schema

## Done Criteria

- a completed run can be rescored without regenerating answers

## Risks

- evidence-support prompt becomes overly complex again

## Handoff

- to be filled by owner
