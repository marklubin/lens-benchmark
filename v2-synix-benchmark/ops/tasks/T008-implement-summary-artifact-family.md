# T008 - Integrate the Synix summary family into benchmark policies

status: ready
priority: P1
phase: Artifact
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T005, T013]
blocks: [T010, T011]

## Purpose

Integrate the released Synix summary family into the benchmark runtime and policy manifests without rebuilding that artifact family locally.

## Scope

In scope:

- benchmark-side configuration for the Synix summary family
- runtime exposure of the Synix `summaries` projection under `policy_summary`
- provenance-preserving retrieval from the benchmark runtime through the Synix tool API

Out of scope:

- implementing the Synix summary artifact family itself
- summary-only systems with no provenance path

## Deliverables

- `policy_summary` manifest and runtime exposure
- tests for summary provenance and runtime access
- benchmark-side configuration for the Synix summary family

## Files Or Areas Owned

- benchmark policy and runtime integration modules
- related tests

## Implementation Plan

1. configure the Synix summary family for the benchmark bank
2. expose summary retrieval under `policy_summary` through the named projection handle
3. ensure summary outputs preserve links to source evidence through the Synix tool API

## Verification

- summary refs resolve to source evidence chains
- summary compilation is cached and resumable
- policy execution does not rebuild summary artifacts

## Done Criteria

- `policy_summary` is runnable in pilot studies over released Synix summary artifacts

## Risks

- upstream summary semantics may need one benchmark-specific configuration pass

## Handoff

- waiting on the Synix upstream milestone and `T013`
