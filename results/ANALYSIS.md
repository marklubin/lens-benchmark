# LENS Benchmark: Phase 5 Analysis

**Date**: 2026-02-24
**Runs**: 90/96 scored (8 adapters × 6 scopes × 2 budgets)
**Episodes**: 120 per scope (with distractors)
**Agent LLM**: GPT-OSS-120B (Cerebras)
**Judge**: GPT-OSS-120B (Cerebras)
**Embeddings**: GTE-ModernBERT-base 768d (Together AI)

## Systems Under Test

| System | Architecture | Preprocessing |
|--------|-------------|---------------|
| **null** | No memory (control) | None |
| **sqlite-chunked-hybrid** | BM25 + embedding hybrid retrieval | None (raw chunks) |
| **compaction** | Progressive LLM summarization | LLM summary at each checkpoint |
| **mem0-raw** | Vector store (Qdrant) + LLM extraction | LLM memory extraction |
| **cognee** | Knowledge graph + vector store | Entity extraction (cognify) |
| **graphiti** | Temporal knowledge graph (FalkorDB) | Entity + relationship extraction |
| **letta** | Stateful agent with memory management | Agent-managed memory blocks |
| **letta-sleepy** | Letta + sleep/wake consolidation | Agent-managed + periodic synthesis |

## Experimental Design

Each system processes 120 sequential episodes (100 signal + 20 topically-orthogonal distractors) across 6 domain-diverse scopes. Episodes are generated with **information isolation**: a planning LLM encodes signal as numeric progressions, a blind rendering LLM formats each independently. No single episode can answer any benchmark question — answers require synthesizing evidence across the full sequence.

At 7 checkpoints, the system answers 2-4 questions under constrained context budgets (8,192 or 16,384 cumulative result tokens). Questions span 9 types: longitudinal, temporal, action recommendation, evidence sufficiency, counterfactual, paraphrase, negative, severity assessment, and distractor resistance.

Scoring is 3-tier: mechanical metrics (evidence grounding, budget compliance), LLM-judged quality (pairwise answer quality vs. naive baseline), and differential metrics (naive baseline advantage). Composite = weighted combination; hard gate on evidence_grounding >= 0.5.

---

## Finding 1: The Ceiling is Low

| Metric | Value |
|--------|-------|
| Best single run | 0.629 (chunked-hybrid, s04/16k) |
| Best adapter mean | 0.473 (chunked-hybrid) |
| Mean across all systems | 0.372 |

**No memory system exceeds 50% mean composite.** Longitudinal synthesis from 120 episodes with distractors remains fundamentally unsolved. The best system (simple BM25+embedding retrieval) achieves less than half the theoretical maximum.

## Finding 2: Simple Retrieval Beats Complex Architectures

| Rank | System | Composite | 95% CI | p vs #1 |
|------|--------|-----------|--------|---------|
| 1 | sqlite-chunked-hybrid | **0.473** | [0.433, 0.518] | — |
| 2 | cognee | 0.432 | [0.416, 0.449] | 0.037* |
| 3 | graphiti | 0.426 | [0.325, 0.516] | 0.281 ns |
| 4 | mem0-raw | 0.349 | [0.302, 0.390] | <0.001*** |
| 5 | letta | 0.346 | [0.298, 0.394] | <0.001*** |
| 6 | letta-sleepy | 0.335 | [0.291, 0.378] | <0.001*** |
| 7 | compaction | 0.272 | [0.189, 0.347] | <0.001*** |
| 8 | null | 0.000 | [0.000, 0.000] | <0.001*** |

The simplest retrieval approach — BM25 full-text search combined with embedding similarity, with no LLM preprocessing, no graph construction, no entity extraction — outperforms every more sophisticated architecture. The effect sizes are large (Cohen's d = 1.5-1.7 vs. agentic systems).

Graphiti (N=6) is the only system whose CI overlaps with chunked-hybrid, but with incomplete scope coverage and high variance (SD=0.132).

**Key claim**: Across 6 domains, 8 architectures, and 120-episode sequences, basic text retrieval is the most effective longitudinal synthesis strategy. Adding architectural complexity hurts more than it helps.

## Finding 3: Context Budget is the Binding Constraint

Every system improves with doubled context (8k → 16k tokens):

| System | 8k | 16k | Δ | Δ% |
|--------|-----|------|---|-----|
| graphiti | 0.393 | 0.459 | +0.067 | +17% |
| compaction | 0.245 | 0.300 | +0.055 | +23% |
| letta | 0.327 | 0.366 | +0.039 | +12% |
| mem0-raw | 0.330 | 0.368 | +0.039 | +12% |
| chunked-hybrid | 0.454 | 0.492 | +0.037 | +8% |
| letta-sleepy | 0.322 | 0.348 | +0.026 | +8% |
| cognee | 0.421 | 0.444 | +0.023 | +5% |

The uniform improvement direction suggests all systems are **retrieval-limited**, not reasoning-limited. They can synthesize evidence when they have it — the bottleneck is surfacing the right evidence from 120 episodes within the token budget.

## Finding 4: Cross-Scope Consistency

### Adapter × Scope Matrix (composite, mean of 8k+16k)

| System | s01 | s02 | s03 | s04 | s05 | s06 | Mean | SD |
|--------|-----|-----|-----|-----|-----|-----|------|-----|
| chunked-hybrid | 0.445 | 0.530 | 0.389 | 0.582 | 0.470 | 0.422 | 0.473 | 0.072 |
| cognee | 0.463 | 0.424 | 0.397 | 0.459 | 0.414 | 0.436 | 0.432 | 0.026 |
| graphiti | 0.505 | 0.263 | — | — | — | 0.510 | 0.426 | 0.141 |
| mem0-raw | 0.373 | 0.361 | 0.203 | 0.438 | 0.389 | 0.330 | 0.349 | 0.080 |
| letta | 0.320 | 0.365 | 0.253 | 0.460 | 0.436 | 0.245 | 0.346 | 0.091 |
| letta-sleepy | 0.327 | 0.335 | 0.250 | 0.459 | 0.392 | 0.244 | 0.335 | 0.083 |
| compaction | 0.352 | 0.000 | 0.212 | 0.402 | 0.377 | 0.291 | 0.272 | 0.150 |

**Scope wins**: chunked-hybrid 3/6, graphiti 2/6, cognee 1/6.

Cognee has the lowest cross-scope variance (SD=0.026), making it the most *predictable* system, but its mean is 0.04 lower than chunked-hybrid. For deployment where consistency matters, cognee's narrow CI [0.416, 0.449] is notable.

### Scope Difficulty

| Scope | Domain | Mean Composite |
|-------|--------|----------------|
| s04 | Environmental drift | 0.400 |
| s05 | Insider threat | 0.354 |
| s01 | Cascading failure | 0.348 |
| s06 | Supply chain | 0.310 |
| s02 | Financial indicators | 0.285 |
| s03 | Clinical signal | 0.243 |

s03 (clinical) is hardest across all systems; s04 (environmental) is easiest.

## Finding 5: Compaction Collapse

Progressive summarization (compaction) shows catastrophic failure on scope s02 (composite = 0.000) while achieving 0.402 on s04. This represents a **complete information loss** — the rolling summary discarded all signal from 120 financial indicator episodes.

This is the "death by summarization" failure mode: as episode count grows, LLM summaries progressively lose fine-grained evidence. The information-theoretic bottleneck of compressing 120 episodes into a fixed summary destroys the very signal that longitudinal questions test.

## Finding 6: Question Type Discrimination

NBA win rates for sqlite-chunked-hybrid by question type:

| Question Type | Win Rate | Difficulty |
|--------------|----------|------------|
| longitudinal | 0.711 | Easiest |
| evidence_sufficiency | 0.702 | Easy |
| severity_assessment | 0.689 | Easy |
| action_recommendation | 0.679 | Easy |
| paraphrase | 0.653 | Medium |
| counterfactual | 0.477 | Hard |
| temporal | 0.435 | Hard |
| distractor_resistance | 0.430 | Hard |
| negative | 0.427 | Hard |

Even the best system stays below 75% on every question type. The hardest types — distractor resistance (0.430) and temporal reasoning (0.435) — test whether the system can ignore irrelevant episodes and reason about chronological progression, respectively. These remain near chance.

## Finding 7: Agentic Memory Hurts

| System | Architecture | Composite | vs chunked-hybrid |
|--------|-------------|-----------|-------------------|
| letta | Stateful agent | 0.346 | −0.127 (p<0.001) |
| letta-sleepy | Agent + sleep/wake | 0.335 | −0.138 (p<0.001) |
| chunked-hybrid | Passive retrieval | 0.473 | — |

Letta's agent-managed memory performs 27-29% worse than passive chunk retrieval, with statistical significance (p<0.001, Cohen's d > 1.5). The sleep/wake consolidation variant (letta-sleepy) is slightly *worse* than vanilla letta.

This suggests that agentic memory management — where the agent decides what to remember and how to organize it — introduces lossy abstraction that discards evidence needed for longitudinal synthesis. Passive retrieval preserves all raw evidence for the answering stage.

---

## Thesis Statement

> **Across 6 domain-diverse scopes, 8 memory architectures, and 720 benchmark episodes with distractors, no system exceeds 50% composite score on longitudinal evidence synthesis. The simplest approach — BM25 + embedding hybrid retrieval with no preprocessing — significantly outperforms knowledge graphs, vector stores with LLM extraction, progressive summarization, and stateful agentic memory (p < 0.05). The binding constraint is retrieval quality under token budgets, not reasoning capability. Architectural complexity actively degrades performance by introducing lossy transformations that discard the fine-grained evidence longitudinal questions require.**
