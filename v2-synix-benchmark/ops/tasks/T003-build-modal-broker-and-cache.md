# T003 - Implement Modal broker, cache, and idempotent call layer

status: ready
priority: P0
phase: Runtime
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T002]
blocks: [T004, T005, T006, T010]

## Purpose

Build the only permitted inference path for the benchmark and make all model calls cacheable and attributable.

## Scope

In scope:

- Modal chat broker
- Modal embedding broker
- idempotency keys
- cache lookup and writeback
- raw response capture
- token and cost accounting

Out of scope:

- state-machine resume logic
- artifact-family or policy-specific model usage

## Deliverables

- broker module
- cache storage format
- tests for cache hit, miss, and idempotency

## Files Or Areas Owned

- runtime broker modules in `src/`
- related tests in `tests/`

## Implementation Plan

1. implement a single broker interface
2. define cache-key generation
3. persist normalized and raw responses
4. write tests for repeated identical calls

## Verification

- same call twice returns cached result on second execution
- cost and token metadata are preserved
- failures are recorded and surfaced

## Done Criteria

- all model usage can be routed through this layer

## Risks

- leaking direct provider calls into artifact or scoring modules

## Handoff

- to be filled by owner
