# T010 - Run 2-scope artifact-bank smoke pilot and calibrate cost/runtime

status: done
priority: P0
phase: Study
owner: claude
created: 2026-03-06
updated: 2026-03-11
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

## Results

### S08 × 4 Policies (Gate 6 — Validated)

| Policy | primary | fact_f1 | evidence | citation |
|--------|---------|---------|----------|----------|
| null | 0.050 | 0.100 | 0.000 | 0.000 |
| base | **0.446** | 0.125 | 0.510 | 1.000 |
| core | 0.397 | 0.212 | 0.410 | 0.589 |
| summary | 0.393 | 0.162 | 0.340 | 0.900 |

### Calibrated Costs

- Bank build (4 checkpoints, cached): ~54s
- Bank build (4 checkpoints, uncached): ~16 min
- Agent answering per policy (10 questions): ~2 min
- Scoring per policy (10 questions): ~5-7 min
- Total tokens (full gate 6): 315,877
- Total wall time (gate 6): ~17 min

### Key Findings

1. **Base search provides 9× lift over null baseline** (0.446 vs 0.050).
2. **Core memory slightly hurts vs base** (0.397 vs 0.446) — derived context may add noise when agent has search.
3. **Summary matches core** (0.393) — no incremental value from summary context.
4. **fact_f1 is low across all policies** — agent retrieves evidence but doesn't synthesize the specific conclusions that match key facts. This is a genuine benchmark finding, not a scorer artifact.
5. **Citation validity works** — base policy achieves 1.000 (all refs resolve). Core policy lower (0.589) because core-memory context doesn't have citation refs.
6. **Evidence support works** — real 0.0-1.0 scores correlating with answer quality.
7. **Echoes V1 finding**: simple retrieval beats complex preprocessing under agent-driven queries.

### Verification Checklist

- [x] Rescore from saved answers only (no new inference) — scorer reads answers from state store
- [x] Replay: completed run skips answered questions on restart
- [x] Cache-hit behavior confirmed — re-run with same gate hits broker cache for agent calls
- [x] Policy rerun does not trigger bank recompilation — BankBuilder checks release existence
- [x] Resume under injected failure — kill mid-run, restart picks up from last completed question

### Fixes Applied During Pilot

1. Synix SDK integration fixes (ParseTransform .txt extension, keyword vs fulltext, add_text → add)
2. Agent max_tokens=4096 to prevent infinite thinking
3. Split system prompts (tools vs no-tools) to prevent null policy tool hallucination
4. Search result truncation (1500 chars) to prevent context blowout
5. Broadened citation extraction regex for model-shortened refs
6. Scorer max_tokens=2048 for judge model thinking
7. `/no_think` soft switch in judge prompts
8. `_resolve_ref` / `_resolve_artifact` for citation validation with prefix expansion
9. `default_extra_body` for `enable_thinking: false` per-request

## Handoff

Gate 6 validated. Ready for gate 7 (S08 + S10) or T011 (feature screening).
