# CLAUDE.md - Synix Memory Benchmark V2

## Mission

Build a controlled, scientifically auditable benchmark for longitudinal memory mechanisms on a single Synix-backed substrate.

The benchmark must be:

- reproducible
- resumable
- replayable
- cost-aware
- strict about provenance
- strict about checkpoint isolation
- easy to verify from stored artifacts only

## Non-Negotiable Rules

1. No external product adapters in the V2 core benchmark.
2. No hard gating in the primary score.
3. No internal answering agents.
4. No direct model calls outside the central Modal broker.
5. No silent recovery from correctness-affecting failures.
6. No task may be considered complete without explicit verification.
7. No non-trivial work starts without a task file in `ops/tasks/`.
8. No benchmark result is valid unless it is tied to a study manifest, policy manifests, bank manifests, run manifests, and an event log.
9. No compiled artifact may include evidence from episodes beyond the active checkpoint prefix.

## Required Workflow

### 1. Pick Work From The Board

- Start in `ops/WORKBOARD.md`.
- Only work on tasks marked `ready` unless explicitly unblocked.
- Move exactly one task per worker to `in_progress`.

### 2. Read The Task File

Every task file is authoritative for:

- scope
- deliverables
- dependencies
- files or modules owned
- verification requirements
- definition of done

### 3. Update Status Before Editing

Before making changes:

- update task status to `in_progress`
- add owner
- add start date if missing
- note any dependency assumptions

### 4. Implement Narrowly

Keep each task small and coherent. Prefer one concern per task:

- schemas and manifests
- Modal broker and cache
- state store, resume, replay
- one artifact family compiler
- one runtime policy layer
- one scoring subsystem
- one report subsystem

Do not mix large design changes with broad implementation refactors in one task.

### 5. Verify Before Handoff

Every task must include:

- code-level verification
- behavior-level verification
- failure-mode verification when applicable

At minimum, record:

- commands run
- artifacts produced
- pass or fail result
- unresolved risks

### 6. Update Operational Records

After finishing a task:

- update the task file
- update `ops/WORKBOARD.md`
- append a concise entry to `ops/WORKLOG.md`
- add a decision entry if behavior or architecture changed materially

## Testing Requirements

### Runtime and Infra Tasks

Must include tests for:

- idempotency
- resume behavior
- replay behavior
- failure recovery
- schema validation
- cache correctness

### Artifact-Bank Tasks

Must include tests for:

- checkpoint isolation
- no future leakage
- provenance preservation
- deterministic config behavior
- cache invalidation on config change
- no raw evidence loss

### Runtime Policy Tasks

Must include tests for:

- allowed artifact family enforcement
- stable retrieval behavior
- citation resolution
- no implicit rebuild during policy execution

### Scoring Tasks

Must include tests for:

- stable parsing
- scorer output schema
- citation validity behavior
- fact matching behavior
- re-score without re-inference

## Failure Handling Policy

Fail fast and preserve state.

Allowed behavior:

- record failure event
- persist partial build or run state
- mark bank build or run retryable or terminal
- resume later from completed checkpoints or completed question steps

Disallowed behavior:

- swallowing correctness-affecting errors
- deleting partial state automatically
- recomputing expensive inference without explicit need
- rebuilding artifact families during policy execution unless the manifest hash changed

## Artifact Policy

Every important output must be addressable and attributable:

- study manifest
- policy manifest
- artifact-bank manifest
- run manifest
- event log
- cached model response
- bank snapshot metadata
- answer file
- score file
- report export

If a result cannot be tied back to these artifacts, it is not part of the benchmark.

## Modal Policy

All inference must flow through the shared Modal broker.

The broker is responsible for:

- endpoint selection
- retries
- idempotency keys
- token accounting
- latency accounting
- cost estimation
- raw response capture
- cache lookup and writeback

Direct provider calls from artifact-family modules or scoring modules are forbidden.

## Scope Control

The default V2 scope policy is:

- main benchmark: `S07-S12`
- extension only: `S13-S15`
- optional smoke test: `S01`

Do not expand scope count casually. Any scope-set change requires an entry in `ops/DECISIONS.md`.

## Policy Control

The default main runtime policy set is:

- `null`
- `policy_base`
- `policy_core`
- `policy_summary`
- `policy_graph`

Any additional policy must justify:

- which hypothesis it tests
- why it is not redundant
- what incremental cost it adds

## Artifact Family Control

The canonical V2 artifact families are:

- raw episodes
- chunks
- hybrid search indexes
- core-memory artifacts
- summary artifacts
- graph artifacts

For V1, use one canonical configuration per family unless a decision explicitly approves a variant study.

## Checkpoint Isolation Rule

Artifact banks are compiled per `scope x checkpoint` from the episode prefix available at that checkpoint.

Never compile from the full future scope corpus and then reuse those outputs for earlier checkpoints.

## Definition Of Done

A task is done only when all of the following are true:

1. Deliverables in the task file are complete.
2. Verification steps were executed and recorded.
3. Relevant docs were updated.
4. Workboard status is updated.
5. No unresolved blockers remain unless explicitly accepted.

## Reference Order

When in doubt, follow this order:

1. `ops/tasks/<task>.md`
2. `ops/WORKBOARD.md`
3. `docs/BENCHMARK_SPEC.md`
4. `docs/EXECUTION_PLAN.md`
5. `docs/SCORING_V2.md`
6. `ops/RUNBOOK.md`
7. `schemas/`
