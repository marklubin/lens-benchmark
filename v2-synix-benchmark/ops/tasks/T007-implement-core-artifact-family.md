# T007 - Integrate the Synix core-memory family into benchmark policies

status: ready
priority: P1
phase: Artifact
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T005, T013]
blocks: [T010, T011]

## Purpose

Integrate the released Synix core-memory family into the benchmark runtime and policy manifests without rebuilding that artifact family locally.

## Scope

In scope:

- benchmark-side configuration for the Synix core-memory family
- runtime exposure of the Synix `core_memory` projection under `policy_core`
- provenance-preserving retrieval from the benchmark runtime through the Synix tool API

Out of scope:

- implementing the Synix core-memory artifact family itself
- internal answering agents
- multi-agent coordination

## Deliverables

- `policy_core` manifest and runtime exposure
- tests for provenance and runtime access
- benchmark-side configuration for the Synix core-memory family

## Files Or Areas Owned

- benchmark policy and runtime integration modules
- related tests

## Implementation Plan

1. configure the Synix core-memory family for the benchmark bank
2. expose core-memory retrieval under `policy_core` through the named projection handle
3. verify retrieval returns both core-memory refs and underlying evidence refs

## Verification

- maintenance updates preserve provenance
- agent can retrieve core-memory refs and underlying evidence refs
- policy execution does not rebuild core artifacts

## Done Criteria

- `policy_core` is runnable in pilot studies over released Synix core-memory artifacts

## Risks

- upstream core-memory semantics may need one benchmark-specific configuration pass

## Handoff

- waiting on the Synix upstream milestone and `T013`
