# T002 - Design study, policy, bank, run, event, and score schemas

status: done
priority: P0
phase: Runtime
owner: claude
created: 2026-03-06
updated: 2026-03-10
depends_on: [T001]
blocks: [T003, T004, T006]

## Purpose

Define the canonical schemas that all builds, runs, scores, and reports will use.

## Scope

In scope:

- study manifest schema
- policy manifest schema
- artifact-bank manifest schema
- run manifest schema
- event schema
- score record schema

Out of scope:

- implementation of runtime services

## Deliverables

- finalized schemas under `schemas/`
- schema validation strategy
- docs describing required fields and invariants

## Files Or Areas Owned

- `schemas/`
- runtime schema docs if needed

## Implementation Plan

1. convert examples into frozen schemas
2. define required versus optional fields
3. define checkpoint-isolation fields and provenance requirements
4. define schema validation path in tests and runtime

## Verification

- validate sample files against chosen schema format
- ensure schema supports replay and report-generation needs

## Done Criteria

- runtime tasks can build directly against the schemas

## Risks

- under-specified schema forces refactors later

## Verification Record

- 56 unit tests in `tests/test_schemas.py` — all passing
- All 6 schemas roundtrip through JSON serialization
- Required field validation enforced (7 negative tests)
- Enum validation rejects invalid status/event types
- Default values verified (planned status, deterministic seed, source-backed citations)
- Checkpoint isolation fields verified (max_episode_ordinal, source_episode_ids, dataset_hash)
- JSON Schema export for all 6 models → `schemas/*.schema.json`
- Backward-compatible with existing example JSON files in `schemas/`
- 98 total tests passing (including broker, cache, warmup)

## Handoff

Schemas are defined as Pydantic v2 models in `src/bench/schemas.py`. Downstream tasks (T004, T006) can import directly:

```python
from bench.schemas import (
    StudyManifest, PolicyManifest, BankManifest,
    RunManifest, Event, EventType, ScoreRecord,
    BankStatus, RunStatus,
)
```

JSON Schema files exported to `schemas/*.schema.json` for documentation/interop.
