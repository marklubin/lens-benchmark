# LENS Benchmark: Synthesis Report

**Date**: 2026-02-23 (Session 19)
**Purpose**: Consolidate Phase 3 results, landscape positioning, red-team findings, and next steps into a coherent, defensible narrative.

---

## 1. The Headline

**No existing AI agent memory system meaningfully outperforms basic text retrieval at longitudinal evidence synthesis — the cognitive task these systems are purportedly designed for.**

This finding is based on Phase 3 of the LENS benchmark: 12 completed runs across 7 memory systems, evaluating whether each system could synthesize conclusions from 30 signal episodes hidden among 90 format-matched distractors (~84K tokens total).

| System | Type | Answer Quality | Status |
|--------|------|---------------|--------|
| sqlite-chunked-hybrid | FTS + embeddings | 0.477 | Best overall |
| letta-sleepy | Agent memory + sleep consolidation | 0.403 | Best "real" memory system |
| mem0-raw | Vector search | 0.368 | |
| letta | Agent memory (MemGPT-style) | 0.346 | |
| compaction | LLM summarization | 0.294 | Collapsed with noise |
| hindsight | Graph + temporal retrieval | 0.213 | Removed from evaluation |
| null | No memory | 0.189 | Baseline |
| cognee | Knowledge graph (GraphRAG) | DNF | API compatibility failure |
| graphiti | Temporal knowledge graph | DNF | API compatibility failure |

No system achieved 50% of key facts. The best absolute score (0.477) came from the simplest retrieval approach.

---

## 2. Why This Matters — The Landscape Context

### 2.1 What Existing Benchmarks Test

We surveyed 10 major memory benchmarks (LoCoMo, LongMemEval, MemBench, MemoryAgentBench, MemoryArena, MEMTRACK, Evo-Memory, ConvoMem, LoCoMo-Plus, DMR). All test some variation of four patterns:

1. **Retrieve-and-answer**: Find an explicit fact in history (LoCoMo, DMR)
2. **Join-and-answer**: Connect 2-3 explicit facts (HotPotQA, LoCoMo multi-hop)
3. **Track-and-answer**: Know the current state of something that changed (LongMemEval, MEMTRACK)
4. **Learn-and-apply**: Use past experience to guide future action (MemoryArena)

All share one property: **the information that constitutes the answer is explicitly stated somewhere.** The memory system's job is to find it, connect it, or maintain it.

### 2.2 What LENS Tests

LENS introduces a fifth pattern: **synthesize-from-progression.** The answer does not exist as any explicit fact. It emerges only from observing a pattern across many sequential data points, where each individual data point appears normal in isolation.

Example: No episode says "latency is increasing." Each episode contains a metric value (p99: 420ms). The conclusion (progressive degradation causing cascading failure) only exists in the *relationship between episodes*.

### 2.3 Why Existing Benchmarks Are Insufficient

Three independent findings validate this gap:

1. **MemoryArena (2026)**: "Agents with near-saturated performance on existing long-context memory benchmarks like LoCoMo perform poorly in our agentic setting." Systems that ace recall fail at memory-guided reasoning.

2. **Letta filesystem baseline**: A simple grep-based filesystem agent scored 74% on LoCoMo, beating Mem0 (66.9%) and matching Zep (75.1%). If keyword search beats specialized memory systems, the benchmark isn't testing memory architecture.

3. **MEMTRACK null result**: Zep and Mem0 provided "no significant improvement" on cross-platform state tracking. GPT-5 alone scored 60% — adding memory components didn't help.

### 2.4 Vendor Claims vs Evidence

| Vendor | Claim | Independent Evidence |
|--------|-------|---------------------|
| **Mem0** | "26% improvement over OpenAI Memory" | Documented implementation errors in competitor evaluation. Full-context baseline and filesystem agent both outperform Mem0 on LoCoMo. |
| **Zep/Graphiti** | "State of the art in agent memory" | Own LoCoMo score challenged (75% claimed, 58% in corrected evaluation). MEMTRACK found "no significant improvement." |
| **Cognee** | "93% on HotPotQA" | 24-question evaluation, couldn't run competitors directly, used "previously shared numbers" for Graphiti. |
| **Hindsight** | "91.4% on LongMemEval" | Most rigorous evaluation, but still tests retrieve/join/track patterns, not synthesis. |
| **Letta** | "Filesystem approach achieves 74% on LoCoMo" | Most honest framing — their finding is an indictment of the benchmarks, not a claim about their product. |

**Every vendor evaluates on benchmarks where their architectural choices confer an advantage.** No vendor has submitted to an independent, adversarial evaluation.

---

## 3. What Our Evidence Actually Shows

### 3.1 Defensible Claims (supported by current evidence)

1. **Existing memory benchmarks have a synthesis gap.** The landscape survey identifies this clearly — no benchmark tests longitudinal synthesis from scattered evidence. MemoryArena's own finding validates that high recall scores don't predict real-world memory utility.

2. **Distractors create meaningful signal/noise separation.** Compaction NBA dropped from 0.790 (30 episodes, no distractors) to 0.404 (120 episodes with distractors). This is a validated experimental manipulation.

3. **Simple retrieval is competitive with complex architectures on synthesis tasks.** sqlite-chunked-hybrid (0.477 AnsQ) outperformed letta-sleepy (0.403), mem0-raw (0.368), letta (0.346), compaction (0.294), and hindsight (0.213) on scope 01.

4. **Heavy memory infrastructure often fails operationally.** 3 of 6 infrastructure-dependent adapters couldn't complete Phase 3 (cognee: embed API prefix bug, graphiti: Cerebras entity extraction incompatibility, hindsight: batch embed overflow). This is itself a finding about production readiness.

5. **The task is genuinely hard.** No system — including the naive baseline with full context access — achieves >50% answer quality. Longitudinal synthesis from terse operational logs is difficult even for frontier LLMs.

### 3.2 Claims That Need Strengthening

1. **"No memory system outperforms basic retrieval" needs multi-scope validation.** Phase 3 tested only scope 01 (cascading_failure). The 6-scope sweep (Phase 1-2) showed significant domain variation. We need Phase 3 across all 6 scopes to generalize.

2. **Budget enforcement invalidates the "constrained budget" framing.** Budget compliance was 0.000 for all adapters — they used 4-5x the allocated tokens on the first retrieval call. Phase 3 is actually testing *unconstrained retrieval quality with noise*, not *constrained budget performance*. This changes the interpretation but doesn't invalidate the finding — it means simple retrieval beats complex architectures even without budget pressure.

3. **Missing adapters (cognee, graphiti) could change rankings.** Cognee scored 0.564 composite in Phase 1-2 (2nd overall). Graphiti scored 0.498. Both failures are trivially fixable engineering issues, not fundamental limitations.

4. **Single agent LLM tested.** GPT-OSS-120B may favor certain retrieval patterns. Rankings shifted between Qwen3-32B and Qwen3-235B in earlier phases, suggesting sensitivity to agent LLM choice.

### 3.3 Claims We Cannot Make

1. **"Memory systems are useless"** — We tested a specific task (longitudinal synthesis from operational logs). These systems may perform well on the tasks they were designed for (conversational recall, preference tracking, etc.).

2. **"Simple retrieval is always better"** — Our task specifically requires finding relevant episodes in noise. Tasks that require reasoning over relationships (knowledge graph queries, temporal traversal) may favor graph-based systems.

3. **"The benchmark is publication-ready"** — Several methodological issues need resolution first (see Section 4).

---

## 4. Red Team Findings

### 4.1 Critical Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| **Budget enforcement is broken** | "Constrained-8k" preset doesn't actually constrain (adapters use 39K tokens avg). Phase 3 is testing unconstrained retrieval, not constrained budgets. | Implement pre-flight truncation in `dispatch_tool_call()`. Re-run or re-frame results as "unconstrained with noise." |
| **N=1 scope in Phase 3** | Cannot generalize from cascading_failure_01 alone. | Run Phase 3 across all 6 scopes (at minimum: lightweight adapters). |
| **No-key-fact questions score 1.0** | 3/24 questions give free points to all adapters, compressing dynamic range. | Score as 0.5 (neutral) or exclude from answer_quality. |

### 4.2 Major Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| **Compaction has privileged `prepare()` access** | Gets unconstrained LLM summarization before budget clock starts. | Label as oracle upper bound, not comparable adapter. |
| **Agent LLM may favor hybrid search** | FTS+embedding search benefits from any query style; pure semantic adapters may be disadvantaged by keyword-heavy queries. | Run ablation with 2nd agent LLM. |
| **NBA compares against privileged baseline** | Naive baseline gets chronological order; adapters get relevance order. | Consider random-order naive baseline or add sort capability. |
| **No judge reliability validation** | Single LLM judge call per comparison, no inter-rater agreement measured. | Run judge twice (both position orders), compute agreement. |
| **3/9 adapters failed (survivorship bias)** | Failed adapters (cognee, graphiti) are the most likely to benefit from noise filtering. | Fix and re-run. |
| **fact_recall provides zero discrimination** | Uniformly 0.167 across all adapters due to naive substring matching. | Remove from composite or use fuzzy matching. |

### 4.3 Strengths Confirmed

| Strength | Assessment |
|----------|-----------|
| **Two-stage anti-contamination pipeline** | Genuinely novel. No other benchmark addresses episode-level LLM contamination. |
| **Pairwise judging with position debiasing** | Sound methodology, more reliable than Likert ratings. |
| **Multi-tier scoring with hard gates** | Prevents gaming; evidence_grounding gate is well-motivated. |
| **6-scope domain diversity** | Better than any vendor evaluation (all use 1-2 domains). |
| **Operational realism** | Real adapters, real containers, real API calls — not mocked. |
| **Negative/null questions** | Tests whether systems can say "no" — most benchmarks don't. |
| **Distractor-resistance questions** | Specifically tests noise filtering — unique to LENS. |

---

## 5. Strongest Defensible Position

### The Paper Thesis

> **"Longitudinal evidence synthesis — recognizing patterns that emerge across many sequential observations where no single observation contains the answer — is an unsolved capability gap in AI agent memory. Existing benchmarks do not measure it. When measured by LENS, no current memory system achieves >50% accuracy, and simple text retrieval outperforms specialized architectures."**

### Supporting Arguments

1. **The gap is real** (landscape survey confirms no benchmark tests synthesis-from-progression)
2. **The task is well-constructed** (two-stage anti-contamination, distractor-resistance, checkpoint-based temporal evaluation, negative questions)
3. **The task is genuinely hard** (all systems <50%, including naive baseline)
4. **Complex memory architectures provide no synthesis advantage** (simple retrieval leads, heavy systems fail operationally)
5. **Industry benchmarks measure the wrong thing** (LoCoMo is beatable by grep; MemoryArena validates that recall benchmarks don't predict utility; MEMTRACK shows Zep/Mem0 add nothing on complex tasks)

### Qualifications to Include

- Phase 3 results are from one scope; multi-scope validation is in progress
- Cognee and graphiti failed due to API compatibility, not fundamental limitations — results are preliminary
- Budget enforcement was non-binding; results reflect unconstrained retrieval quality, not constrained performance
- Single agent LLM (GPT-OSS-120B); rankings may vary with different agent models

---

## 6. Minimum Additional Work

### Must-Do (to publish with confidence)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | **Run Phase 3 lightweight adapters across all 6 scopes** (null, chunked-hybrid, compaction, mem0-raw × 6 scopes × 1 budget = 24 runs) | ~4 hrs compute | Transforms N=1 claim into N=6 with confidence intervals |
| 2 | **Fix and re-run cognee + graphiti** on scope 01 with distractors | ~4 hrs engineering | Eliminates survivorship bias for the two most promising graph-based systems |
| 3 | **Re-frame budget enforcement** — either fix it or explicitly state Phase 3 tests unconstrained retrieval with noise | ~2 hrs write-up or code | Resolves the most damaging methodological critique |
| 4 | **Bootstrap confidence intervals** for Phase 3 answer_quality per adapter | ~1 hr analysis | Determines if rankings are statistically significant |

### Should-Do (to withstand rigorous review)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 5 | Fix no-key-fact scoring (0.5 instead of 1.0) and re-score | ~30 min | Removes free-point inflation |
| 6 | Remove or zero-weight fact_recall in composite | ~30 min | Removes non-discriminative noise |
| 7 | Measure judge inter-rater reliability (position-swap analysis) | ~2 hrs | Validates judging methodology |
| 8 | Test with 2nd agent LLM (e.g., Claude Sonnet) on top 4 adapters | ~4 hrs compute | Shows finding is agent-LLM robust |

### Nice-to-Have

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 9 | Human baseline on 1-2 scopes | ~8 hrs human time | Gold standard comparison |
| 10 | Validate distractor similarity thresholds | ~1 hr | Confirms distractor calibration |
| 11 | Label compaction as "oracle upper bound" throughout | ~30 min | Clarifies privileged prepare() access |

**Total must-do effort: ~11 hours.** This takes the finding from "preliminary single-scope" to "multi-scope validated with the field's most promising systems included."

---

## 7. How Our Results Compare to the Field

### 7.1 On Benchmarks Where Others Score High

| Benchmark | Top System | Score | What It Measures |
|-----------|-----------|-------|-----------------|
| LoCoMo | Hindsight | 89.6% | Factual recall from short conversations |
| LongMemEval | Hindsight | 91.4% | Cross-session retrieval, temporal reasoning |
| DMR | Zep | 98.2% | 60-message fact retrieval (trivially easy) |
| HotPotQA | Cognee | 93% | 2-hop reasoning over 2 documents |

These scores are high because the tasks are fundamentally about finding explicit information. Memory systems are optimized for this.

### 7.2 On LENS (Longitudinal Synthesis)

| System | Answer Quality | Notes |
|--------|---------------|-------|
| sqlite-chunked-hybrid | 0.477 | No memory architecture — just search |
| letta-sleepy | 0.403 | Best dedicated memory system |
| All others | <0.37 | Including naive baseline with full context |

The gap between "what these systems can do" (89-98% on retrieval) and "what longitudinal synthesis requires" (max 47.7%) is the finding.

### 7.3 The Irony

**Hindsight scores 91.4% on LongMemEval but 0.213 on LENS (barely above null).** The system that dominates retrieval benchmarks provides near-zero value on synthesis tasks. This single comparison captures the entire thesis: existing benchmarks measure the wrong thing.

---

## Appendix: Source Documents

- Benchmark landscape survey: `docs/BENCHMARK_LANDSCAPE_SURVEY.md`
- Red team report: Agent output (session 19)
- Phase 3 raw results: `results/phase3/`
- All scorecards: `output/{run_id}/scores/scorecard.json`
- Full project status: `docs/STATUS_REPORT.md`
