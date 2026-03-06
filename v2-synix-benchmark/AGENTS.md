# AGENTS.md - Coordination Rules

## Purpose

This file defines how multiple agents or contributors should split and execute work in this repository without thrash.

## Core Rule

One task, one owner, one verification record.

No shared implicit work.

## Roles

### Lead

Responsible for:

- sequencing work
- updating the workboard
- resolving dependency ordering
- approving scope changes
- maintaining decision records

### Worker

Responsible for:

- claiming exactly one task at a time
- working only within the task's defined scope
- recording verification evidence
- leaving a clean handoff note

### Reviewer

Responsible for:

- checking that implementation matches task scope
- checking verification evidence
- identifying regressions or missing cases

## Task Claim Protocol

Before a worker edits code or docs for a task, they must:

1. set task status to `in_progress`
2. put their name or agent id in `owner`
3. confirm dependencies are complete or explicitly assumed
4. note the expected files or modules they will touch

## Ownership Boundaries

A worker should not edit files outside the task's declared scope unless:

- the task file is updated first, or
- the change is a clearly necessary mechanical follow-on and is documented in the handoff

If two workers need the same files, split the work differently. Do not rely on informal coordination.

## Handoff Requirements

Every completed task must end with a short handoff section covering:

- what changed
- files touched
- verification run
- known limitations
- suggested next task

## Required Task Format

Each task file must include:

- id
- title
- status
- owner
- priority
- phase
- dependencies
- scope
- deliverables
- verification
- done criteria
- handoff

Use `ops/TASK_TEMPLATE.md`.

## Status Vocabulary

Allowed task statuses:

- `proposed`
- `ready`
- `in_progress`
- `blocked`
- `review`
- `done`
- `wont_do`

Do not invent new status labels.

## Work Sizing Rules

Preferred task size:

- roughly half a day to two days

If a task is larger than that, split it before implementation.

## Dependency Rules

A task may not start if a dependency is still `proposed`, `ready`, `in_progress`, or `blocked`, unless the task file explicitly states the dependency is being mocked or bypassed for a limited purpose.

## Verification Rules

No task moves to `done` without recorded verification.

Verification should include whichever apply:

- unit tests
- integration tests
- replay or resume checks
- schema validation
- manual artifact inspection
- cost-accounting sanity check

## Run Discipline

Study runs must be tracked as first-class work items.

Do not launch an untracked run. Every meaningful run should map back to:

- a task
- a study manifest
- a runbook step

## Decision Discipline

If a change affects any of the following, add an entry to `ops/DECISIONS.md`:

- scoring
- scope set
- runtime policy set
- artifact-family configuration
- manifest schema
- event schema
- replay or cache semantics
- Modal model policy

## Anti-Thrash Rules

Do not do these:

- broad opportunistic refactors during task execution
- editing unrelated docs while you are there
- rerunning the same expensive inference because prior outputs are hard to find
- making benchmark-policy changes without a decision entry
- mixing benchmark design and benchmark result interpretation in the same task
