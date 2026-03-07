# T010 - Run 2-scope artifact-bank smoke pilot and calibrate cost/runtime

status: proposed
priority: P0
phase: Study
owner: unassigned
created: 2026-03-06
updated: 2026-03-06
depends_on: [T004, T005, T006, T007, T008, T013]
blocks: [T011, T012]

## Purpose

Run the smallest meaningful study to validate build, runtime, and scoring behavior and establish cost per cell.

## Scope

In scope:

- small pilot study
- artifact-bank and runtime validation
- scoring validation
- cost and latency export

Out of scope:

- final benchmark claims

## Deliverables

- frozen pilot study manifest
- frozen policy manifests
- frozen bank manifests
- completed pilot run artifacts
- compile cost and policy-run cost summary
- notes on runtime failures and fixes needed

## Files Or Areas Owned

- `studies/`
- report exports
- task documentation

## Implementation Plan

1. choose pilot scopes and policies
2. compile or select the required sealed Synix bank manifests
3. run the policy cells
4. verify resume and replay under at least one injected failure
5. export cost and runtime summary

## Verification

- rerun score generation from saved answers only
- replay a completed run without new inference
- compare cache-hit behavior on repeated execution
- verify policy rerun does not trigger bank recompilation

## Done Criteria

- compile cost per checkpoint and runtime cost per cell are known
- no hidden recomputation remains

## Risks

- pilot reveals missing runtime invariants and forces rework

## Handoff

- to be filled by owner
