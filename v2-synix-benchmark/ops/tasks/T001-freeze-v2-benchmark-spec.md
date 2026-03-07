# T001 - Freeze scope policy, runtime policy set, artifact-bank model, and scoring v2

status: done
priority: P0
phase: Program
owner: codex
created: 2026-03-06
updated: 2026-03-06
depends_on: [T000]
blocks: [T002, T003, T005, T006, T013]

## Purpose

Convert the current planning docs into a frozen benchmark policy and a frozen Synix/LENS ownership split so implementation does not thrash.

## Scope

In scope:

- final scope policy
- final main runtime policy set
- final artifact-bank model
- final scoring v2 policy
- study-size guidance for pilot and main study
- Synix versus LENS ownership boundary
- local execution DAG and task refresh to reflect the Synix-first sequence
- Synix GitHub issue filing and alignment for the required platform features

Out of scope:

- implementing Synix platform features
- coding LENS runtime or artifact-family features

## Deliverables

- finalized edits to benchmark and process docs
- explicit default study matrix
- explicit decisions recorded in `ops/DECISIONS.md`
- refreshed local task graph and ownership notes
- filed or amended Synix GitHub issues for the upstream milestone

## Files Or Areas Owned

- `docs/BENCHMARK_SPEC.md`
- `docs/EXECUTION_PLAN.md`
- `docs/SCORING_V2.md`
- `ops/DECISIONS.md`
- `ops/RUNBOOK.md`
- `ops/WORKBOARD.md`
- `ops/tasks/T005-implement-artifact-bank-base.md`
- `ops/tasks/T007-implement-core-artifact-family.md`
- `ops/tasks/T008-implement-summary-artifact-family.md`
- `ops/tasks/T009-implement-graph-artifact-family.md`
- `ops/tasks/T010-run-smoke-pilot.md`
- `ops/tasks/T011-run-feature-screening-study.md`
- `ops/tasks/T013-integrate-synix-runtime-policy-layer.md`
- `ops/SYNIX_UPSTREAM_TRACKER.md`
- `ops/WORKLOG.md`

## Implementation Plan

1. freeze the Synix/LENS ownership split and the checkpoint-scoped artifact-bank model
2. refresh the local task graph to match the sequential Synix-first execution order
3. choose the default pilot and default main-study matrices
4. record scope, policy, scoring, and runtime-boundary decisions
5. file or amend the Synix GitHub issues required for the upstream milestone

## Verification

- docs are internally consistent
- workboard and benchmark docs agree on policy terminology
- workboard and task files agree on the Synix/LENS split
- Synix GitHub issues exist for every required upstream feature

## Done Criteria

- implementation can proceed without further benchmark-policy ambiguity
- the Synix upstream milestone is explicitly tracked

## Risks

- delaying freeze invites more churn
- upstream issue scope may drift if the contracts are not written clearly

## Handoff

- froze the Synix versus LENS ownership split in the benchmark docs and execution plan
- refreshed the local workboard and downstream task scopes to treat Synix features as upstream platform work
- created `T013` for Synix runtime and tool integration
- created `ops/SYNIX_UPSTREAM_TRACKER.md` and linked the required Synix issue set to the local DAG
- finalized the benchmark-facing projection contract around sealed manifests and named projection handles
- reconciled the runbook language with sealed bank manifests and manifest-based reuse
- amended Synix issues `#34`, `#15`, `#10`, and `#60` and created Synix issues `#81` through `#88` to track the upstream milestone
- verification run: inspected the updated local docs and task files, searched the repo with `rg` for projection and dependency mismatches, and fetched the created or amended GitHub issues with `gh api` to confirm bodies and numbering
- known limitations: no Synix platform code is implemented yet; this task only froze the plan and issue structure
- suggested next task: `T002` after the required upstream contracts are stable enough to finalize manifest boundaries
