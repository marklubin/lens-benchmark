# Research Directive: Constrained Budget Validation

> **Priority**: P0 — blocks all publication work
> **Created**: 2026-02-22
> **Status**: Not started

## The Problem

At standard budget (32K tokens), **every adapter scores NBA < 0.5** — context stuffing beats every memory system we've tested. The best result is letta-sleepy V3 at NBA 0.2825 (~20% win rate vs naive). This means the benchmark cannot currently demonstrate that any memory system adds value over simply cramming all episodes into the context window.

If this holds at constrained budgets too, the benchmark's core methodology doesn't work.

## The Hypothesis

Memory search quality only matters when the agent **can't see everything**:

| Budget | Episodes Visible | Expected Outcome |
|--------|-----------------|------------------|
| 32K (standard) | ~30/30 (100%) | Context stuffing wins. **Confirmed.** |
| 4K (constrained) | ~8/30 (25%) | Good retrieval starts mattering. NBA should approach 0.5. |
| 2K (constrained) | ~4/30 (13%) | Search quality is everything. NBA should flip above 0.5 for strong adapters. |

If the degradation curve materializes — strong adapters maintain scores while naive baseline collapses — that's the paper's central figure and validates the entire benchmark design.

If it doesn't — if context stuffing still wins at 2K — the methodology is fundamentally flawed and needs redesign before any publication effort.

## Experiment Plan

### Phase 1: Minimal Proof-of-Concept (scope 01 only)

**6 runs** to validate/invalidate the hypothesis fast:

| Adapter | 4K | 2K | Why |
|---------|----|----|-----|
| sqlite-chunked-hybrid | `chunked_hybrid_scope01_4k.json` | `chunked_hybrid_scope01_2k.json` | Strongest retrieval (0.5783 composite at 32K, best evidence_coverage among sqlite variants) |
| letta-sleepy V3 | `letta_sleepy_scope01_4k.json` | `letta_sleepy_scope01_2k.json` | SOTA at 32K (0.6508), delta/causal synthesis should help navigate limited budget |
| null | needs config | needs config | Floor — establishes what happens with zero memory |

**Infrastructure**: Together AI (Qwen3-235B-A22B agent + judge), scope 01 dataset. Use existing configs in `configs/`. Create null constrained configs if missing.

**Judge**: Must use 235B. The 32B judge produced compressed scores (fact_recall floor 0.167) that couldn't discriminate between adapters. Don't repeat that mistake.

**Scoring**: Run with NBA enabled (no `--no-baseline` flag). The whole point is to measure adapter vs context-stuffed performance under constraint.

### Phase 2: Full Constrained Matrix (if Phase 1 validates)

Only proceed here if Phase 1 shows NBA approaching or exceeding 0.5 at 2K for strong adapters.

- 7 adapters × 2 caps × scope 01 = 14 runs
- Then extend to all 6 scopes for the top 3-4 adapters

### Phase 3: Degradation Curve Analysis

- Plot composite score vs budget (32K → 4K → 2K) per adapter
- Plot NBA vs budget — the critical figure
- Bootstrap CIs on the curves
- Identify the crossover point where memory search beats context stuffing

## Success Criteria

The hypothesis is **validated** if:
- At least one adapter achieves NBA > 0.45 at 2K (approaching parity with context stuffing)
- Strong adapters (chunked-hybrid, V3) degrade less than weak ones (null, compaction) as budget shrinks
- The adapter ranking changes meaningfully between 32K and 2K (i.e., the constraint reveals real differences)

The hypothesis is **invalidated** if:
- All adapters still score NBA < 0.3 at 2K
- The ranking is unchanged from 32K (constraint adds noise, not signal)
- Context stuffing wins even when it can only see 4/30 episodes

## If the Hypothesis Fails

Don't just run more experiments. Diagnose **why** context stuffing wins under constraint:

1. **Episode density**: Are 30 episodes too few? At 2K, the naive baseline sees ~4 episodes — but if those 4 happen to contain the signal, memory search adds nothing. Check whether signal episodes cluster early (making truncation effective).

2. **Question design**: Do the questions actually require cross-episode synthesis? If most can be answered from 1-2 episodes, context stuffing at 2K still works for easy questions and both approaches fail on hard ones.

3. **Naive baseline truncation**: The naive baseline at 2K gets the first ~4 episodes (truncated). If signal concentrates in early episodes, this is accidentally effective. Consider randomized or tail-truncation for the naive baseline.

4. **Agent behavior under constraint**: Does the agent actually use its remaining budget wisely after `[Context budget exhausted]`? Or does it give up? Check turn logs for constrained runs.

5. **Retrieval precision**: At 2K, the adapter gets maybe 2 search results. If those 2 are the right episodes, that should beat the naive baseline's random 4. But if retrieval precision is low, the adapter wastes its tiny budget on wrong episodes.

Potential redesigns if methodology fails:
- Increase episode count (120 signal + distractors) to make context stuffing infeasible even at 32K
- Use longer episodes to increase per-episode token cost
- Design questions that provably require 5+ specific episodes to answer
- Add a "needle in haystack" question type where signal is in a single late episode buried in noise

## What Already Exists

### Infrastructure (all ready)
- Budget presets: `constrained-4k` and `constrained-2k` in `src/lens/core/config.py`
- Budget enforcement: `check_cumulative_results()` in `src/lens/agent/budget_enforcer.py`
- Context exhaustion handling: Agent receives `"[Context budget exhausted — synthesize answer from evidence already retrieved]"` in harness
- Naive baseline fairness: `naive_baseline.py` truncates context to match constrained cap
- 72 config files: All adapters × all scopes × both caps in `configs/`
- Run scripts: `scripts/run_constrained.sh`, `scripts/run_constrained_remaining.sh`

### Results at 32K (baselines for comparison)
- Scope 01, 235B judge: 14 adapter variants scored, V3 SOTA at 0.6508
- 6-scope matrix, 235B judge: 4 adapters, chunked-hybrid wins 4/6 (mean 0.5656)
- 7×6 sweep, 32B judge: 30/42 runs, scores compressed (not publication-quality)

## Execution Notes

- Together AI Qwen3-235B is the only viable judge. Budget ~$20-40 for 6 runs + scoring.
- Letta requires local server (`podman run letta/letta`) + embed proxy (`scripts/letta_embed_proxy.py`).
- Chunked-hybrid needs `SYNIX_EMBED_URL` and `SYNIX_EMBED_API_KEY` for OpenAI-compatible embeddings.
- Null adapter has no dependencies.
- Each run is ~30 min at standard budget; constrained runs should be faster (fewer turns/calls).
- Score immediately after each run — don't batch. If the first 2K run shows NBA > 0.45, the hypothesis is likely valid and you can proceed with confidence.
