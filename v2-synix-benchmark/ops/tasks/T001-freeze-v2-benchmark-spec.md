# T001 - Freeze scope policy, runtime policy set, artifact-bank model, and scoring v2

status: ready
priority: P0
phase: Program
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T000]
blocks: [T002, T003, T005, T006]

## Purpose

Convert the current planning docs into a frozen benchmark policy so implementation does not thrash.

## Scope

In scope:

- final scope policy
- final main runtime policy set
- final artifact-bank model
- final scoring v2 policy
- study-size guidance for pilot and main study

Out of scope:

- coding runtime or artifact-family features

## Deliverables

- finalized edits to benchmark and process docs
- explicit default study matrix
- explicit decisions recorded in `ops/DECISIONS.md`

## Files Or Areas Owned

- `docs/BENCHMARK_SPEC.md`
- `docs/SCORING_V2.md`
- `ops/DECISIONS.md`

## Implementation Plan

1. resolve open questions on minimum scopes and policy set
2. freeze the checkpoint-scoped artifact-bank model
3. choose default pilot and default main-study matrices
4. record any scope, policy, or scoring changes as decisions

## Verification

- docs are internally consistent
- workboard and benchmark docs agree on policy terminology

## Done Criteria

- implementation can proceed without further benchmark-policy ambiguity

## Risks

- delaying freeze invites more churn

## Handoff

- to be filled by owner
