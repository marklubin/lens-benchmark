# Synix Memory Benchmark V2

This directory is the fresh-start implementation workspace for the new benchmark.

The purpose of V2 is to replace the previous adapter zoo with a controlled benchmark of memory policies implemented on a single substrate.

## Core Thesis

We are not benchmarking products.

We are benchmarking memory features implemented on top of one shared system:

- one raw evidence store
- one retrieval stack
- one provenance model
- one agent loop
- one scoring system
- one execution runtime
- one inference backend policy: Modal only

## Artifact-Bank Model

V2 is built around a checkpoint-scoped **memory artifact bank**.

For each `scope x checkpoint`, Synix compiles the candidate memory artifacts once:

- raw episode artifacts
- chunk artifacts
- search indexes
- core-memory artifacts
- summary artifacts
- graph or entity artifacts

Then benchmark runs are mostly runtime **policies** over that compiled bank:

- which artifact families are visible
- which search surfaces are enabled
- how retrieval results are fused
- what the agent is allowed to retrieve

This is the key cost and reproducibility advantage of the new design:

- expensive memory compilation is shared
- policies do not rebuild the same artifacts repeatedly
- policy runs are cheap to add
- scoring and replay never require recompiling memory

Important constraint:

- artifact banks must be built from the episode prefix available at each checkpoint
- never from the full future scope corpus

## Authoritative Files

- `CLAUDE.md`: implementation guardrails and required workflow
- `AGENTS.md`: multi-agent coordination and work-splitting rules
- `docs/BENCHMARK_SPEC.md`: benchmark design and scope or policy choices
- `docs/EXECUTION_PLAN.md`: phased delivery plan with durations and exit criteria
- `docs/SCORING_V2.md`: simplified scoring methodology
- `ops/WORKBOARD.md`: current outstanding work and status
- `ops/RUNBOOK.md`: end-to-end compile, run, resume, replay, and verification procedure
- `ops/DECISIONS.md`: append-only architectural decisions
- `ops/WORKLOG.md`: append-only progress log
- `ops/tasks/`: one file per tracked work item

## Directory Layout

```text
v2-synix-benchmark/
  CLAUDE.md
  AGENTS.md
  AGENT.md
  README.md
  docs/
  ops/
    tasks/
  schemas/
  src/
  tests/
  studies/
```

## Working Rules

1. No non-trivial implementation work happens without a task file in `ops/tasks/`.
2. No task moves to `done` without verification evidence recorded in the task file and `ops/WORKLOG.md`.
3. No benchmark claim is considered real until it can be regenerated from manifests and cached artifacts.
4. All inference goes through the central Modal broker.
5. Every run must be resumable and replayable.

## Initial Goal

Build the minimum publishable benchmark on a fresh runtime:

- main scopes: `S07-S12`
- extension scopes: `S13-S15`
- optional smoke-test numeric anchor: `S01`
- main runtime policies: `null`, `policy_base`, `policy_core`, `policy_summary`, `policy_graph`

## Status

This workspace currently contains process and planning scaffolding only. It does not yet contain the new runtime implementation.
