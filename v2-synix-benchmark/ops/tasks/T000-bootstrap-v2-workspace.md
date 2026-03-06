# T000 - Bootstrap v2 workspace and process docs

status: done
priority: P0
phase: Program
owner: codex
created: 2026-03-06
updated: 2026-03-06
depends_on: []
blocks: [T001]

## Purpose

Create a fresh workspace for the Synix-backed benchmark with explicit process guardrails before any runtime or policy implementation starts.

## Scope

In scope:

- directory structure
- process docs
- benchmark-spec scaffold
- workboard and task template
- initial execution plan

Out of scope:

- runtime code
- artifact compilers
- scoring code

## Deliverables

- `README.md`
- `CLAUDE.md`
- `AGENTS.md`
- `AGENT.md`
- `docs/`
- `ops/`
- `schemas/`

## Files Or Areas Owned

- entire `v2-synix-benchmark/` scaffold

## Implementation Plan

1. create the workspace directories
2. write the initial operating docs
3. seed the task backlog
4. add schema examples

## Verification

- workspace exists with the planned directory structure
- process docs are internally consistent

## Done Criteria

- implementation work can start from a clean controlled workspace

## Risks

- overdesigning the docs before enough benchmark direction exists

## Handoff

- Created the V2 scaffold and seeded the initial process and planning docs.
- Follow with `T001` to freeze the benchmark model before implementation starts.
