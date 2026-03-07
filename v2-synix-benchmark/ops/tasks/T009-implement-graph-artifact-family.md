# T009 - Integrate the Synix graph family into benchmark policies

status: ready
priority: P1
phase: Artifact
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T005, T013]
blocks: [T011]

## Purpose

Integrate the released Synix graph family into the benchmark runtime and policy manifests without rebuilding that artifact family locally.

## Scope

In scope:

- benchmark-side configuration for the Synix graph family
- runtime exposure of the Synix `graph` projection under `policy_graph`
- graph-informed retrieval back to source evidence through the Synix tool API

Out of scope:

- implementing the Synix graph artifact family itself
- external graph databases
- complex temporal graph research features not needed for V1

## Deliverables

- `policy_graph` manifest and runtime exposure
- tests for graph provenance and retrieval
- benchmark-side configuration for the Synix graph family

## Files Or Areas Owned

- benchmark policy and runtime integration modules
- related tests

## Implementation Plan

1. configure the Synix graph family for the benchmark bank
2. expose graph retrieval under `policy_graph` through the named projection handle
3. ensure graph hits resolve back to source evidence through the Synix tool API
4. verify the benchmark runtime does not trigger graph rebuilds

## Verification

- graph-derived refs preserve source-evidence lineage
- graph compilation respects cache and resume semantics
- policy execution does not rebuild graph artifacts

## Done Criteria

- `policy_graph` is runnable in feature-screening studies over released Synix graph artifacts

## Risks

- upstream graph semantics may need one benchmark-specific configuration pass

## Handoff

- waiting on the Synix upstream milestone and `T013`
