# T014 - Refresh Synix execution-model tracker and closeout design

status: done
priority: P0
phase: Program
owner: codex
created: 2026-03-06
updated: 2026-03-06
depends_on: [T001]
blocks: []

## Purpose

Update the benchmark-side view of the Synix upstream milestone so it matches the stricter build-versus-release design that should govern the remaining platform work before LENS runtime integration begins.

## Scope

In scope:

- rewrite the benchmark-side Synix execution model around explicit `build` and explicit `release`
- capture the ergonomics shift before and after that design
- document the major user stories that matter for LENS planning
- align the upstream tracker and decision log to the revised model

Out of scope:

- implementing Synix platform code
- changing benchmark runtime or study code
- unblocking downstream tasks beyond clarifying the remaining upstream path

## Deliverables

- benchmark-side Synix execution-model design note
- refreshed upstream tracker aligned to the latest Synix direction
- decision log entry recording the explicit build-versus-release boundary

## Files Or Areas Owned

- ops/SYNIX_EXECUTION_MODEL.md
- ops/SYNIX_UPSTREAM_TRACKER.md
- ops/DECISIONS.md
- ops/WORKBOARD.md
- ops/tasks/T014-refresh-synix-execution-model-tracker.md

## Implementation Plan

1. add a benchmark-side design note for the Synix execution model
2. rewrite the upstream tracker around landed versus remaining Synix work
3. record the build-versus-release decision and update this task with verification

## Verification

- inspect the updated design note for before or after ergonomics and the required user stories
- inspect the upstream tracker for the revised blocker ordering
- run `git diff --check`

## Done Criteria

- the benchmark repo has one clear write-up of the stricter Synix execution model
- the tracker no longer treats mutable build directories as part of the target contract
- the remaining Synix execution critical path is explicit for T005 and T013 planning

## Risks

- benchmark docs may drift from the actual Synix RFC if the upstream design changes again
- over-specifying future Synix ergonomics here could constrain implementation prematurely

## Handoff

- what changed: added `ops/SYNIX_EXECUTION_MODEL.md`, rewrote the upstream tracker around the stricter build-versus-release model, and recorded the new benchmark-side decision in `ops/DECISIONS.md`
- files touched: `ops/SYNIX_EXECUTION_MODEL.md`, `ops/SYNIX_UPSTREAM_TRACKER.md`, `ops/DECISIONS.md`, `ops/WORKBOARD.md`, `ops/tasks/T014-refresh-synix-execution-model-tracker.md`
- verification run: manual inspection of the execution-model note and tracker plus `git diff --check`
- known limitations: this is a benchmark-side design and tracking update only; it does not implement or verify Synix platform code
- suggested next task: continue `T002` locally while treating `T005` and `T013` as blocked on the remaining Synix execution milestone
