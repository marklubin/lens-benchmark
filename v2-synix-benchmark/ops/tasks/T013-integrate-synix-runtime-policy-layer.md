# T013 - Integrate the Synix runtime/tool API into benchmark policy execution

status: ready
priority: P0
phase: Runtime
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T005]
blocks: [T007, T008, T009, T010, T011]

## Purpose

Mount the released Synix Python-local runtime and tool API inside the benchmark runner so policies operate on sealed banks and named projections through stable platform contracts.

## Scope

In scope:

- benchmark-side runtime mount over one sealed Synix bank manifest
- policy loader and tool gating for benchmark policies over named projection handles
- null and base policy integration
- event and cost-accounting hooks around runtime tool use

Out of scope:

- implementing the Synix runtime or tool API itself
- derived-family policy integration
- scoring logic

## Deliverables

- benchmark runtime wrapper over the Synix tool surface and sealed-manifest contract
- policy loader and gating rules
- `null` and `policy_base` runtime integration
- tests for tool visibility, bank reuse, and event logging

## Files Or Areas Owned

- `src/` benchmark runtime and policy modules
- runtime integration tests
- policy manifest examples

## Implementation Plan

1. mount one sealed Synix bank manifest and its named projection handles into the benchmark runtime
2. implement benchmark policy gating over the Synix tool surface
3. wire event and cost accounting around policy execution
4. verify the same bank can be reused across multiple policy runs

## Verification

- repeated policy runs reuse the same bank without rebuilds
- policy tool visibility matches the active policy manifest
- event and cost records can be replayed without new inference

## Done Criteria

- the benchmark runner can execute `null` and `policy_base` against a sealed Synix bank through the released runtime API

## Risks

- runtime integration may couple too closely to Synix internals if the tool API is underspecified

## Handoff

- waiting on the Synix upstream milestone
