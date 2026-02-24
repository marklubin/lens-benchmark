# LENS Benchmark: Project Status Report

**Last Updated**: 2026-02-23 (session 23)
**Scoring Pipeline**: v3.2 (naive baseline advantage replaces longitudinal_advantage, per-question timing)
**Agent LLM**: GPT-OSS-120B (Cerebras) / Qwen3-235B-A22B (Together AI) / Qwen3-32B (RunPod vLLM)
**Judge LLM**: GPT-OSS-120B (Cerebras, session 23) / Qwen3-235B-A22B (Together AI, prior sessions)
**Token Cap**: 32,768 (standard) / 16,384 (constrained-16k) / 8,192 (constrained-8k) / 4,096 (constrained-4k) / 2,048 (constrained-2k)
**Dataset**: 6 scopes, 144 questions, 720 signal episodes + 540 distractor episodes (120 per scope with distractors)
**Unit Tests**: 991 passing
**Adapters Under Evaluation**: 8 (null, sqlite-chunked-hybrid, compaction, letta, letta-sleepy, mem0-raw, cognee, graphiti). ~~hindsight~~ removed — see session 19 notes.
**Total Runs**: 21 scope-01 systems + 30 sweep runs + 48 constrained (Phase 1+2) + 12 Phase 3 runs + **90 Phase 5 runs scored (of 96 attempted)**

---

## Executive Summary

LENS (Longitudinal Evidence-backed Narrative Signals) is a benchmark for evaluating whether AI agent memory systems can synthesize conclusions from evidence scattered across many sequential episodes, rather than finding answers in a single document.

**Current state**: Core infrastructure is feature-complete. **90/96 Phase 5 runs scored** — all 8 adapters evaluated across 6 domain-diverse scopes with distractor episodes. Statistical analysis, judge reliability validation, and publication-ready figures/tables are complete.

**Key finding 12 — PHASE 5 COMPLETE (90/96 scored): Simple retrieval beats all complex memory architectures**: Using Cerebras GPT-OSS-120B as agent LLM (judge + embeddings via Modal). **sqlite-chunked-hybrid leads with 0.473 mean composite (CI: 0.406–0.502)**. Cognee 2nd (0.432), graphiti 3rd (0.426 on 3 scopes). All 7 non-null adapters complete 12/12 runs except graphiti (6/12 — entity extraction timeout on scopes 03-05). **16k > 8k for all adapters** (p=0.016, Wilcoxon). **Kendall's W = 0.755** (strong concordance). 12 of 28 pairwise comparisons significant at p<0.05. **No adapter exceeds 0.50 composite** — the benchmark's core longitudinal synthesis challenge remains unsolved. **Judge reliability: minimal position bias (A/(A+B)=0.451), 7,652 total judgments, but cognee has 100% TIE rate (judge failure).**

**Key finding 11 — PHASE 3 (WITH DISTRACTORS): No memory system cracks 50% answer quality on longitudinal synthesis**: Phase 3 added 90 distractor episodes per scope (120 total, ~84K tokens) to create a real signal/noise separation challenge. 12 of 18 runs completed (cognee, graphiti failed on API incompatibilities; hindsight/8k and letta-sleepy/8k failed on infra issues). **Results**: sqlite-chunked-hybrid leads with 0.477 answer quality (8k budget) — simple FTS+embedding retrieval beats every dedicated memory system. Letta-sleepy (0.403), mem0-raw (0.368), letta (0.346), compaction (0.294), hindsight (0.213), null (0.189). **No system achieves >50% of key facts.** Budget enforcement is effectively non-binding — adapters blast through cumulative token limits on the first retrieval call (avg 39K tokens used vs 8K budget). **The honest conclusion: existing memory systems do not meaningfully outperform basic text search at longitudinal synthesis.**

**Key finding 10 — Compaction's Phase 1-2 dominance was an artifact of small corpus size**: Compaction (NBA 0.790) dominated when the corpus was 30 episodes (~14K tokens) — it simply summarized everything in one LLM call before the budget clock started. With 120 episodes (~84K tokens) in Phase 3, compaction collapsed to 0.294 answer quality and 0.404 NBA. **The distractor episodes validated the experimental design**: they created the signal/noise separation challenge the benchmark was designed to test.

**Key finding 9 — CONSTRAINED BUDGET VALIDATION CONFIRMS HYPOTHESIS**: At constrained budgets (4K/2K tokens, where agents see only 25%/13% of episodes), retrieval quality **strongly matters**. 36 Phase 1 runs (3 adapters × 6 scopes × 2 budgets) with Qwen3-235B judge and NBA scoring show: **compaction NBA = 0.73** (CI: 0.68-0.79) at 2K — adapter answers beat context stuffing 73% of the time. Chunked-hybrid NBA = 0.35 (CI: 0.30-0.40). Null baseline NBA = 0.07 (CI: 0.05-0.09). All adapter vs null paired Wilcoxon tests: p < 0.0001. **However, Phase 3 revealed this result was inflated by the small 30-episode corpus.**

**Hindsight removed from evaluation (session 19)**: Hindsight (vectorize.io) is removed from the active adapter list. Across all experiments: 17.3GB container image, NBA statistically indistinguishable from null (0.168 vs 0.150), 0.213 answer quality on its only Phase 3 completion (61 minutes for a single run), batch embedding 413 errors, 20-100s per-episode ingest latency. Evidence_coverage=0.000 across all constrained runs. The product provides zero demonstrated value over doing nothing.

**Key finding 6 — Full 6-scope matrix reveals sqlite-chunked-hybrid as most consistent adapter**: Across all 6 domains, `sqlite-chunked-hybrid` wins 4/6 scopes with cross-scope mean **0.5656** — outperforming both Letta (0.5266) and letta-sleepy V3 (0.4982). The sleep synthesis V3 wins only 2/6 scopes and underperforms letta in 4/6. Key insight: the V3 delta/causal synthesis is conditionally useful — it helps when letta's base retrieval is weak (scopes 01, 03: letta=0.5308, 0.3177), but hurts when letta is already performing well (scopes 04-06: letta=0.5942, 0.5723, 0.6027). The synthesis introduces navigational noise when retrieval is already confident.

**Key finding 5 — letta-sleepy V3 is new SOTA at 0.5776 on scope 01**: Adding a delta/causal sleep consolidation cycle to Letta pushes composite to **0.5776** (+8.8% over base Letta 0.5308). V3 achieves answer_quality 0.8225 (best of any adapter) and longitudinal_advantage **−0.1790** (closest to zero ever observed — up from −0.2822 for base Letta). Critical insight: prompt framing matters enormously — V1 (comprehensive summary) and V2 (actionable filter) both *hurt* relative to control (0.4290, 0.4596 vs 0.5402), while V3's delta/causal framing delivers the improvement. The synthesis guides the agent to retrieve the right specific episodes rather than forcing it to reconstruct temporal patterns from scratch.

**Key finding 8 — Cognee GraphRAG scores 0.5638 — 2nd overall, best evidence_coverage**: Cognee (embedded GraphRAG with LanceDB + Kuzu + SQLite, no container) achieves **0.5638 composite** on scope 01 — surpassing Letta (0.5308), Graphiti (0.4983), and chunked-hybrid (0.4970). Only letta-sleepy V3 (0.5776) scores higher. Cognee's standout metric is **evidence_coverage 0.6319** — the best of ANY system (next best: Letta 0.4722). Combined with perfect budget_compliance (1.0) and evidence_grounding (1.0), this suggests cognee's GraphRAG chunking + knowledge graph indexing produces particularly retrievable chunk representations. longitudinal_advantage −0.2425 is second-best after V3 (−0.1790). The cognify pipeline (entity extraction + graph construction + summarization) runs in `prepare()` before budget clock, avoiding Hindsight's budget trap.

**Key finding 7 — Graphiti temporal knowledge graph scores 0.4983**: Graphiti (FalkorDB-backed temporal knowledge graph with bi-temporal edge invalidation and entity extraction) achieves **0.4983 composite** on scope 01 — nearly matching Letta (0.5308) and outperforming chunked-hybrid (0.4970). Notable: perfect budget_compliance (1.0, zero violations), evidence_grounding 1.0, reasoning_quality 0.9167, insight_depth 0.8333. The EDGE_HYBRID_SEARCH with episode mentions maps graph edges back to source episodes effectively. answer_quality 0.6746 and longitudinal_advantage −0.3370 are weaker than Letta, suggesting the graph structure adds overhead without improving temporal synthesis relative to simple passage retrieval.

**Key finding 1 — Letta is SOTA (base)**: Letta (formerly MemGPT) achieves **0.5308 composite** (Qwen3 judge), +6.8pp above chunked-hybrid+batch_retrieve (0.4970). Letta's semantic vector search over archival passages with a neutral storage prompt achieves perfect evidence_grounding (1.0), answer_quality 0.7239, reasoning_quality 0.9167, and insight_depth 0.8750 — the highest on all three Tier-2 metrics. Budget compliance 0.8333 (5/30 violations, all from ingest latency).

**Key finding 4 — Hindsight disappoints despite graph+temporal**: Hindsight (TEMPR: semantic + BM25 + graph + temporal, RRF-fused) scores **0.3511** — below Letta (0.5308) and even below mem0-raw (0.3690). The graph-based entity extraction during ingest makes each retain() call take 20–100s, causing 19/24 budget violations (budget_compliance=0.2083). Despite exposing `memory_reflect` (native longitudinal synthesis), the agent rarely used it. The Hindsight image is 17.3 GB and contains its own PostgreSQL. reasoning_quality matches Letta (0.9167) suggesting the underlying synthesis is strong, but evidence/citation coverage (0.1667) show the agent struggles to ground answers in Hindsight's reformatted text.

**Key finding 2 — batch_retrieve**: SQLite chunked-hybrid + `batch_retrieve` achieves **0.4970 composite** (Qwen3 judge), beating mem0-raw (0.3690) by **+35%**. A single extra tool collapsed avg tool calls from ~41 → 3.2 (>12x reduction), driving budget_compliance from 0.00 → 0.79 and tripling evidence_coverage. All systems still show *negative* longitudinal advantage — the core signal LENS is designed to measure.

**Key finding 3 — Mem0 domain mismatch**: `mem0-extract` scored **0.0000** on structured telemetry. Root cause: Mem0's extraction LLM uses a prompt that begins *"You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences"* with few-shot examples like `"Hi, my name is John"`. It is hardcoded for chatbot memory — personal preferences, names, dates. Given `p99: 320ms, pool_util: 87%`, it correctly finds zero personal facts, logs *"No new facts retrieved from input. Skipping memory update LLM call."*, and stores nothing. Zero vectors → zero search results → evidence_grounding = 0 → hard gate → 0.0000. There is no configuration option to change this prompt without forking the library. `mem0-raw` (bypassing extraction with `infer=False`) scores 0.3690 — confirming the vector search itself works fine. The extraction layer is the problem, not the storage layer.

---

## Latest Benchmark Results

### Phase 5: Multi-Scope With Distractors (Sessions 21-23, 2026-02-23) — GPT-OSS-120B (Cerebras)

**90/96 runs scored (8 adapters × 6 scopes × 2 budgets).** Dataset: 120 episodes per scope (30 signal + 90 distractors, ~84K tokens). Agent LLM: GPT-OSS-120B (Cerebras). Judge: GPT-OSS-120B (Cerebras). Embeddings: GTE-ModernBERT-base (Modal, via embed proxy). Budgets: 8k/16k cumulative result tokens.

#### Composite Score Rankings (8k budget, across scopes)

| Rank | Adapter | N Scored | 8k Mean | 16k Mean | Overall Mean | 95% CI |
|------|---------|----------|---------|----------|-------------|--------|
| 1 | **sqlite-chunked-hybrid** | **12/12** | **0.454** | **0.492** | **0.473** | [0.406, 0.502] |
| 2 | cognee | 12/12 | 0.421 | 0.444 | 0.432 | [0.397, 0.446] |
| 3 | graphiti | 6/12 | 0.393 | 0.459 | 0.426 | [0.220, 0.491] |
| 4 | mem0-raw | 12/12 | 0.330 | 0.368 | 0.349 | [0.265, 0.388] |
| 5 | letta | 12/12 | 0.327 | 0.366 | 0.346 | [0.258, 0.399] |
| 6 | letta-sleepy | 12/12 | 0.322 | 0.348 | 0.335 | [0.268, 0.381] |
| 7 | compaction | 12/12 | 0.245 | 0.300 | 0.272 | [0.137, 0.328] |
| 8 | null | 12/12 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |

#### Per-Scope Composite Heatmap (8k budget)

| Adapter | S01 | S02 | S03 | S04 | S05 | S06 | Mean |
|---------|-----|-----|-----|-----|-----|-----|------|
| sqlite-chunked-hybrid | 0.424 | 0.482 | 0.381 | 0.535 | 0.520 | 0.386 | 0.454 |
| cognee | 0.438 | 0.402 | 0.374 | 0.471 | 0.418 | 0.423 | 0.421 |
| graphiti | 0.491 | 0.220 | --- | --- | --- | 0.467 | 0.393 |
| mem0-raw | 0.345 | 0.320 | 0.186 | 0.419 | 0.407 | 0.299 | 0.329 |
| letta | 0.288 | 0.338 | 0.215 | 0.455 | 0.424 | 0.239 | 0.327 |
| letta-sleepy | 0.300 | 0.313 | 0.252 | 0.451 | 0.377 | 0.240 | 0.322 |
| compaction | 0.339 | 0.000 | 0.198 | 0.364 | 0.309 | 0.259 | 0.245 |
| null | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

#### Budget Effect (8k vs 16k)

16k > 8k for ALL adapters (p=0.016, Wilcoxon signed-rank). Mean gain: +0.041.

| Adapter | 8k Mean | 16k Mean | Delta |
|---------|---------|----------|-------|
| graphiti | 0.393 | 0.459 | +0.067 |
| compaction | 0.245 | 0.300 | +0.055 |
| letta | 0.327 | 0.366 | +0.039 |
| mem0-raw | 0.330 | 0.368 | +0.039 |
| sqlite-chunked-hybrid | 0.454 | 0.492 | +0.037 |
| letta-sleepy | 0.322 | 0.348 | +0.026 |
| cognee | 0.421 | 0.444 | +0.023 |

#### Statistical Analysis

| Test | Result |
|------|--------|
| **Kendall's W** (cross-scope concordance) | **0.755** (strong agreement) |
| Significant pairwise comparisons (p<0.05) | **12/28** |
| cognee vs compaction | p = 0.031 * |
| cognee vs letta-sleepy | p = 0.031 * |
| cognee vs mem0-raw | p = 0.031 * |
| cognee vs null | p = 0.031 * |
| letta vs null | p = 0.031 * |
| letta-sleepy vs null | p = 0.031 * |
| mem0-raw vs null | p = 0.031 * |
| sqlite-chunked-hybrid vs null | p = 0.031 * |
| compaction vs sqlite-chunked-hybrid | p = 0.031 * |
| letta vs sqlite-chunked-hybrid | p = 0.031 * |
| letta-sleepy vs sqlite-chunked-hybrid | p = 0.031 * |
| mem0-raw vs sqlite-chunked-hybrid | p = 0.031 * |

Bootstrap 95% CIs on composite (8k budget, across 6 scopes):
- sqlite-chunked-hybrid: **0.454** [0.406, 0.502]
- cognee: **0.421** [0.397, 0.446]
- graphiti: **0.393** [0.220, 0.491] (3 scopes, wide CI)
- mem0-raw: **0.330** [0.265, 0.388]
- letta: **0.327** [0.258, 0.399]
- letta-sleepy: **0.322** [0.268, 0.381]
- compaction: **0.245** [0.137, 0.328]
- null: **0.000** [0.000, 0.000]

#### Judge Reliability

| Metric | Value |
|--------|-------|
| Total judge calls | 7,652 |
| Position A wins | 2,503 (32.7%) |
| Position B wins | 3,052 (39.9%) |
| Ties | 2,097 (27.4%) |
| A/(A+B) ratio | **0.451** (minimal position bias) |

Per-adapter TIE rates:
- **cognee: 100% TIE** (judge failure — composite driven entirely by mechanical metrics)
- letta/letta-sleepy: ~38% TIE (partial discrimination)
- graphiti: 20.5% TIE
- mem0-raw/null/chunked-hybrid/compaction: 4-6% TIE (strong discrimination)

#### Graphiti Failures (6/12 runs incomplete)

| Scope | Status | Root Cause |
|-------|--------|------------|
| s01 | Scored (both budgets) | OK |
| s02 | Scored (both budgets) | OK |
| s03 | Failed | Entity extraction TimeoutError (120 episodes × ~5 LLM calls/episode) |
| s04 | Failed | Same — FalkorDB query backlog + Cerebras API latency |
| s05 | Failed | Same |
| s06 | Scored (both budgets) | OK |

Despite FalkorDB tuning (MAX_QUEUED_QUERIES=500, TIMEOUT=30000ms), graphiti semaphore reduction (8→3), and internal timeout increase (60s→180s/episode), scopes 03-05 consistently time out. This is a meaningful finding: **graph-based entity extraction does not scale to 120-episode corpora with remote LLM APIs**.

#### Key Takeaways

1. **Simple retrieval (sqlite-chunked-hybrid) beats all complex memory architectures** — 0.473 mean composite across 6 domains, the only adapter with narrow CI entirely above 0.40
2. **All adapters completed 12/12 runs except graphiti (6/12)** — session 23 fixed cognee (SQLite WAL mode), letta/letta-sleepy (parallelized, ingest latency cap relaxed), and graphiti (partial via FalkorDB tuning)
3. **No adapter exceeds 0.50 composite** — the benchmark's longitudinal synthesis challenge remains unsolved
4. **Cognee ranks #2 despite 100% judge TIE rate** — its composite score is entirely driven by evidence_grounding and evidence_coverage (mechanical metrics), not answer quality
5. **Compaction collapsed on scope 02** (composite=0.000) — the summarize-then-answer approach fails when distractor volume overwhelms the summary
6. **Scope 04 (multi-AZ cloud outage) is easiest**, scope 03 (clinical signals) is hardest for all adapters
7. **16k budget is uniformly better than 8k** (p=0.016) with +0.041 mean gain — more retrieval budget helps regardless of architecture
8. **Rankings are strongly concordant** — Kendall's W=0.755, meaning the relative ordering holds across domains

#### Publication-Ready Outputs

- Figures: `results/figures/d1_answer_quality_ci.png`, `d2_heatmap.png`, `d4_question_types.png`, `combined_heatmap.png`, `budget_effect.png`
- LaTeX tables: `results/tables/d1_main_results.tex`, `d2_heatmap.tex`
- Full analysis data: `results/publication_analysis.json`

---

### Phase 3: With Distractors (Session 19, 2026-02-23) — GPT-OSS-120B on Cerebras

**9 adapters × 2 budgets (8k/16k) = 18 planned, 12 completed.** Dataset: 120 episodes per scope (30 signal + 90 distractors, ~84K tokens). Agent LLM: GPT-OSS-120B (Cerebras, 3000 tok/s). Judge: Qwen3-235B (Cerebras). Embeddings: GTE-ModernBERT-base (Together AI).

#### Answer Quality Rankings (absolute — how many key facts each system got right)

| Adapter | 8k AnsQ | 16k AnsQ | 8k NBA | 16k NBA | 8k EvCov | 16k EvCov | Run ID (8k) | Run ID (16k) |
|---------|---------|----------|--------|---------|----------|-----------|-------------|--------------|
| **sqlite-chunked-hybrid** | **0.477** | **0.369** | 0.568 | 0.516 | 0.194 | 0.177 | `d2ba166bb282` | `9003aa5a81f8` |
| letta-sleepy | — | 0.403 | — | 0.474 | — | 0.288 | FAILED | `3bb499654bb0` |
| mem0-raw | 0.368 | 0.335 | 0.490 | 0.477 | 0.094 | 0.083 | `1754d32e84e2` | `ab801d0bb064` |
| letta | 0.346 | 0.327 | 0.452 | 0.438 | 0.118 | 0.142 | `1c8ad5581b70` | `7837fa55bfd8` |
| compaction | 0.294 | 0.241 | 0.404 | 0.370 | 0.021 | 0.021 | `6c55408270eb` | `e80bb2cc6d01` |
| hindsight | — | 0.213 | — | 0.358 | — | 0.000 | FAILED | `4d77975d04fa` |
| null | 0.189 | 0.189 | 0.282 | 0.313 | 0.000 | 0.000 | `401350c3b02d` | `12494d9c303f` |
| cognee | FAILED | FAILED | — | — | — | — | — | — |
| graphiti | FAILED | FAILED | — | — | — | — | — | — |

#### Failures

| Adapter | Budget | Error | Root Cause |
|---------|--------|-------|------------|
| cognee | both | EmbeddingException 422 | Cognee prefixes model with `together_ai/`, Together API rejects compound name |
| graphiti | both | add_episode failed 10-11 eps | Cerebras API incompatible with graphiti entity extraction |
| letta-sleepy | 8k | 404 Agent not found | Stale agent ID from previous session |
| hindsight | 8k | 413 batch embed too large | 120 episodes overflows Together AI batch limit |

#### Budget Enforcement Finding

Budget compliance = 0.000 for all adapters except null. Adapters blow through the cumulative token limit on the first retrieval call:
- chunked-hybrid/8k: avg 39K tokens/question vs 8K budget (4.8x over)
- compaction/8k: avg 41K tokens/question (5x over)
- Budget enforcement replaces subsequent results with "[Context budget exhausted]" but the first oversized result already contains most of the information

#### Key Takeaways

1. **No memory system achieves >50% answer quality** on longitudinal synthesis with distractor noise
2. **Simple retrieval (FTS+embedding) beats every dedicated memory system** — sqlite-chunked-hybrid at 0.477 outperforms letta-sleepy (0.403), mem0 (0.368), letta (0.346)
3. **Compaction collapsed**: NBA dropped from 0.790 (30 episodes) to 0.404 (120 episodes) — confirming distractors create the intended challenge
4. **3 of 6 heavy infrastructure adapters couldn't complete the benchmark** due to API/provider incompatibilities (cognee, graphiti, hindsight/8k)
5. **Budget constraints are not binding** — needs architectural fix to actually constrain retrieval

---

### Constrained Budget Validation (Session 16, 2026-02-22) — Qwen3-235B

36 runs: 3 adapters × 6 scopes × 2 budgets (4K/2K tokens). Scored with Qwen3-235B judge + NBA (naive baseline advantage). Together AI serverless.

#### NBA Results (aggregate across 6 scopes, N=120 per-question observations each)

| Adapter | 4K NBA (mean) | 4K 95% CI | 2K NBA (mean) | 2K 95% CI |
|---------|--------------|-----------|---------------|-----------|
| **compaction** | **0.711** | [0.652, 0.767] | **0.735** | [0.676, 0.791] |
| chunked-hybrid | 0.301 | [0.250, 0.354] | 0.347 | [0.295, 0.401] |
| null | 0.071 | [0.053, 0.089] | 0.067 | [0.048, 0.086] |

#### Statistical Tests

| Test | Result |
|------|--------|
| Compaction vs null (2K) | p < 0.0001 *** |
| Chunked-hybrid vs null (2K) | p < 0.0001 *** |
| Compaction vs null (4K) | p < 0.0001 *** |
| Chunked-hybrid vs null (4K) | p < 0.0001 *** |
| Budget degradation (compaction 4K→2K) | delta=-0.023, p=0.284 (not significant) |
| Budget degradation (chunked-hybrid 4K→2K) | delta=-0.046, p=0.072 (not significant) |

#### Per-Scope Breakdown (NBA)

| Adapter | S01 | S02 | S03 | S04 | S05 | S06 |
|---------|-----|-----|-----|-----|-----|-----|
| compaction/4k | 0.787 | 0.621 | 0.740 | 0.675 | 0.563 | 0.672 |
| compaction/2k | 0.790 | 0.722 | 0.642 | 0.659 | 0.710 | 0.650 |
| chunked-hybrid/4k | 0.479 | 0.274 | 0.263 | 0.345 | 0.294 | 0.352 |
| chunked-hybrid/2k | 0.466 | 0.444 | 0.240 | 0.371 | 0.328 | 0.388 |
| null/4k | 0.168 | 0.161 | 0.166 | 0.096 | 0.128 | 0.136 |
| null/2k | 0.150 | 0.151 | 0.150 | 0.117 | 0.115 | 0.150 |

**VERDICT: VALIDATE** — Compaction NBA 0.73 at 2K strongly exceeds the 0.45 threshold. Constrained budgets reveal meaningful retrieval advantage. The degradation curve plot is at `results/constrained_degradation_aggregate.png`.

---

### Constrained Budget Phase 2: Heavy Adapters (Sessions 17-18, 2026-02-23) — Qwen3-235B

**12/12 runs complete**: 6 heavy infrastructure adapters × scope 01 × 2 budgets (4K/2K tokens). Scored with Qwen3-235B judge + NBA. Together AI serverless.

#### Full Constrained Rankings (scope 01, all 9 adapters)

| Adapter | 2K NBA | 4K NBA | 2K AnsQ | 4K AnsQ | 2K BudC | 4K BudC | 2K EvCov | 4K EvCov | Type |
|---------|--------|--------|---------|---------|---------|---------|----------|----------|------|
| cognee | 0.855† | 0.477 | 0.288 | 0.402 | 0.167 | 0.000 | 0.156 | 0.260 | Phase 2 |
| **compaction** | **0.790** | **0.787** | 0.974 | 0.990 | 1.000 | 1.000 | 0.000 | 0.000 | Phase 1 |
| **letta-sleepy** | **0.667** | **0.693** | 0.741 | 0.858 | 0.000 | 0.375 | 0.094 | 0.083 | Phase 2 |
| chunked-hybrid | 0.466 | 0.479 | 0.495 | 0.534 | 0.167 | 0.000 | 0.097 | 0.122 | Phase 1 |
| letta | 0.453 | 0.631 | 0.476 | 0.689 | 0.000 | 0.875 | 0.274 | 0.451 | Phase 2 |
| mem0-raw | 0.406 | 0.386 | 0.387 | 0.386 | 0.375 | 0.000 | 0.108 | 0.240 | Phase 2 |
| graphiti | 0.270 | 0.517 | 0.248 | 0.559 | 0.208 | 0.708 | 0.135 | 0.278 | Phase 2 |
| hindsight | 0.168 | 0.168 | 0.178 | 0.171 | 0.750 | 0.667 | 0.000 | 0.000 | Phase 2 |
| null | 0.150 | 0.168 | 0.171 | 0.171 | 1.000 | 1.000 | 0.000 | 0.000 | Phase 1 |

† Cognee 2K NBA=0.855 is anomalous — budget_compliance=0.167 means only 2/12 questions were answered within budget. The high NBA reflects both cognee and the naive baseline performing poorly at 2K, with cognee's incomplete answers being judged slightly better than the naive baseline's. This is a methodological artifact: NBA is unreliable at very low budget compliance.

**Key**: AnsQ=answer_quality (vs reference), BudC=budget_compliance, EvCov=evidence_coverage. NBA=naive_baseline_advantage (vs context-stuffed LLM).

#### Phase 2 Run IDs

| Adapter | 2K Run | 4K Run |
|---------|--------|--------|
| cognee | `72fea05ec21f` | `b75631165f1e` |
| graphiti | `a9d0640adace` | `b0d179c7058d` |
| hindsight | `4495d283d7a1` | `3d326c4259d1` |
| letta | `1fdede6128f9` | `2a8dfe5a5e7a` |
| letta-sleepy | `9439ffcb91aa` | `3d915a82a90b` |
| mem0-raw | `1eba2a0ca429` | `29a6efec2bdf` |

#### Key Observations (Phase 2)

1. **Compaction is the adapter to beat**: NBA 0.790/0.787, answer_quality 0.974/0.990, perfect budget_compliance. Zero infrastructure (no containers, no databases, no proxies). A single LLM summarization call. Every heavy adapter must justify its operational complexity against this baseline.

2. **letta-sleepy is the best heavy adapter**: NBA 0.667/0.693 at 2K/4K with answer_quality 0.741/0.858. Sleep consolidation's delta/causal synthesis produces retrieval targets that remain effective even at 13% episode visibility. The gap between letta-sleepy (0.667) and base letta (0.453) at 2K shows consolidation is *more* valuable under extreme budget pressure.

3. **letta degrades sharply at 2K**: 0.631→0.453 (−28%). Without sleep synthesis, semantic search over raw archival passages struggles at 13% visibility. Sleep consolidation provides a navigation document that compensates.

4. **graphiti needs budget to traverse its graph**: 0.517→0.270 (4K→2K). Temporal knowledge graph edges provide useful navigation at 25% visibility but break down at 13%. The graph structure requires minimum retrieval budget to traverse meaningfully.

5. **cognee's entity extraction blows the budget**: 60+ min cognify time with remote LLM, budget_compliance 0.167/0.000. When it works, answer_quality is moderate (0.402 at 4K) and evidence_coverage is decent (0.260). But getting it to work requires 6 monkey-patches, ACL disabling, and 1-hour timeouts.

6. **hindsight ≈ null**: NBA 0.168 at both budgets vs null's 0.150/0.168. Ironically has better budget_compliance than most heavy adapters (0.750/0.667) — but only because the agent barely retrieves anything. Evidence_coverage=0.000 confirms no useful information is being extracted. 17.3GB container provides zero value at constrained budgets.

7. **mem0-raw is stable but unimpressive**: 0.406/0.386. Pure vector search over raw documents provides consistent but mediocre retrieval quality. Budget compliance swings (0.375→0.000) without performance change suggest the budget cap isn't binding.

8. **LLM response caching**: Content-addressed disk cache (`LENS_LLM_CACHE_DIR`) avoids wasting API spend on retries. SHA256 key over `{model, messages, tools, temperature, seed}`.

9. **Full operational assessment**: See `docs/ADAPTER_OPERATIONS_REPORT.md` for per-adapter setup complexity, failure modes, monkey-patches required, and container sprawl analysis.

---

### 7-Adapter × 6-Scope Sweep (Session 15, 2026-02-20) — Qwen3-32B

30 runs completed (of 42 planned) using Qwen3-32B on RunPod H200 via vLLM. Standard budget preset (32K tokens). Scored with Qwen3-32B judge (`--no-baseline`). **These scores are NOT directly comparable to the 235B-judged results below** — the 32B judge produces systematically lower fact_recall (~0.167 uniform floor) and lower answer_quality.

#### Composite Score Matrix (Qwen3-32B judge)

| Adapter | S01 | S02 | S03 | S04 | S05 | S06 | Mean | N |
|---------|-----|-----|-----|-----|-----|-----|------|---|
| **compaction** | **0.506** | **0.410** | **0.392** | 0.354 | 0.349 | **0.383** | **0.399** | 6 |
| chunked-hybrid | 0.370 | 0.386 | 0.296 | **0.443** | **0.370** | 0.325 | 0.365 | 6 |
| letta | 0.399 | 0.340 | 0.302 | 0.365 | --- | 0.334 | 0.348 | 5 |
| letta-sleepy | 0.361 | 0.310 | 0.316 | 0.353 | --- | 0.311 | 0.330 | 5 |
| mem0-raw | 0.367 | 0.310 | **0.000** | 0.337 | --- | 0.309 | 0.265 | 5 |
| graphiti | **0.000** | 0.321 | --- | --- | --- | --- | 0.161 | 2 |
| cognee | **0.000** | --- | --- | --- | --- | --- | 0.000 | 1 |

**Bold** = best in column (excluding hard-gated zeros). **0.000** = hard-gated (evidence_grounding < 0.5).

#### Key Observations (Session 15)

1. **Compaction leads** (mean 0.399, wins 4/6 scopes). The "summarize everything" approach produces the highest answer_quality (0.756 on scope 01) and is the only adapter with action_quality = 1.0 on any scope. This is the naive memory baseline — any real memory system should beat it.

2. **All scores depressed by 32B judge**: fact_recall is uniformly ~0.167 (the keyword matcher sees fewer matches when the 32B model generates less precise language). answer_quality ranges 0.19–0.76 (vs 0.47–0.85 with 235B judge). **Re-scoring with Qwen3-235B is needed for publication-quality results.**

3. **naive_baseline_advantage = 0.0 for all 30 runs**: Scoring was run with `--no-baseline` to save time. This metric (15% weight) contributes nothing. Re-scoring with baseline enabled will change rankings.

4. **3 hard-gate failures**: cognee (scope 01), graphiti (scope 01), mem0-raw (scope 03) all scored evidence_grounding = 0.0 → composite = 0.0. Retrieval returned no usable results on these runs. Cognee/graphiti likely failed because their entity extraction was too slow with 32B, producing incomplete indices.

5. **Missing runs**: Scope 05 failed for letta, letta-sleepy, mem0-raw (3 runs). Graphiti timed out on scopes 03-06 (4 runs). Cognee timed out on scopes 02-06 (5 runs). **Total: 12 failed of 42 planned.**

6. **Budget compliance is perfect (1.0)** for all non-cognee adapters. The Qwen3 `/no_think` fix + `max_tokens=4096` eliminated the infinite-generation bug that previously caused hangs.

7. **Letta vs letta-sleepy**: Letta edges out letta-sleepy on every scope except 03 (0.302 vs 0.316). With the 32B model, sleep synthesis provides minimal benefit — the smaller model may not produce useful delta/causal summaries.

#### Run IDs (Session 15 Sweep)

| Adapter | S01 | S02 | S03 | S04 | S05 | S06 |
|---------|-----|-----|-----|-----|-----|-----|
| chunked-hybrid | `12108a5b789a` | `a11758c2f140` | `e9a1515c5714` | `2f17686d159a` | `c0b0de58e8b4` | `2c6695522556` |
| compaction | `89dcd32a37c0` | `c8a5806036e9` | `7c0eade368a9` | `7a1512650f2d` | `6c4d49131a24` | `3032df338781` |
| cognee | `f650050462c8` | --- | --- | --- | --- | --- |
| graphiti | `203a1c2a6959` | `e05ce2e85ec8` | --- | --- | --- | --- |
| letta | `d86228adc558` | `623b19664870` | `eb28317c89d8` | `126954043653` | --- | `0b39af5689eb` |
| letta-sleepy | `648f94f7749f` | `2c59489f8f40` | `1863f6933914` | `db62cf8eb685` | --- | `1fe42931d8a8` |
| mem0-raw | `ac0b52ce0dfb` | `9c164a3e8c29` | `0c85cbf04ae3` | `db18f7f243a3` | --- | `cbe06877161f` |

---

### Scope 01 Adapter Comparison — 14 Systems (v3.2 re-scored, Qwen3-235B judge)

Single scope (cascading_failure_01), 30 episodes, 24 questions. Together AI (Qwen3-235B-A22B for agent/judge, GTE-ModernBERT-base for embeddings). **Scoring v3.2**: `naive_baseline_advantage` replaces `longitudinal_advantage` — head-to-head pairwise judging of adapter answer vs context-stuffed naive answer per key fact.

#### Composite Scores — All Scope 01 Systems (Qwen3 judge, v3.2)

| Adapter | Composite | Answer Q | Fact Recall | Budget | Insight | Reason | Evid Cov | NBA | Run ID |
|---------|-----------|----------|------------|--------|---------|--------|---------|-----|--------|
| **letta-sleepy V3** | **0.6508** | **0.8489** | 0.2810 | 0.9167 | 0.7083 | **1.0000** | 0.4410 | **0.2825** | `8ef0786f0eb5` |
| **cognee** | **0.6299** | 0.7392 | 0.2556 | **1.0000** | 0.7917 | 0.9167 | **0.6319** | 0.1800 | `77545ef2b9b8` |
| letta-sleepy V0 (control) | 0.6110 | 0.7298 | 0.2671 | **1.0000** | **0.8750** | 0.9167 | 0.4583 | 0.1827 | `8c415eba299e` |
| **letta** | 0.5990 | 0.7239 | 0.2611 | 0.8333 | **0.8750** | 0.9167 | **0.4722** | 0.1728 | `be0003e5447b` |
| **graphiti** | 0.5788 | 0.6538 | 0.2553 | **1.0000** | 0.8333 | 0.9167 | 0.3507 | **0.2205** | `2bc821424282` |
| chunked-hybrid + batch_retrieve | 0.5783 | 0.6635 | 0.2507 | 0.7917 | 0.7917 | **0.9583** | 0.4618 | 0.1874 | `8581429063e7` |
| letta-sleepy V2 (actionable) | 0.5280 | 0.7377 | 0.2132 | 1.0000 | 0.3750 | 0.9583 | 0.3160 | 0.1812 | `6e6e53e7581d` |
| letta-sleepy V1 (minimal) | 0.5147 | 0.7000 | 0.2458 | 1.0000 | 0.5000 | 1.0000 | 0.2569 | 0.1922 | `1cbe02135799` |
| sqlite-embedding-openai | 0.4724 | 0.5871 | 0.2323 | 0.2917 | 0.6667 | 0.8750 | 0.3264 | 0.1427 | `fef20b05d46b` |
| mem0-raw | 0.4603 | 0.5791 | 0.2323 | 0.7500 | 0.5417 | 0.9167 | 0.1562 | 0.1573 | `830d711e5c17` |
| chunked-hybrid L=7 (no batch) | 0.4320 | 0.6746 | 0.2462 | 0.0000 | 0.5417 | 0.9167 | 0.2153 | 0.1458 | `8b9e83ae9dec` |
| hindsight | 0.4234 | 0.6726 | 0.2775 | 0.2083 | 0.6250 | 0.9167 | 0.1667 | 0.1476 | `040bb488abbd` |
| sqlite-fts | 0.3826 | 0.4711 | 0.1914 | 0.3333 | 0.4583 | 0.7917 | 0.1840 | 0.0944 | `11d7bf53e4f0` |
| mem0-extract | 0.0000 | 0.1708 | 0.1667 | 1.0000 | 0.0000 | 0.9167 | 0.0000 | 0.1183 | `a119b4906684` |

**NBA interpretation**: Naive Baseline Advantage measures adapter vs context-stuffed answer. Score 0.5 = parity; >0.5 = adapter wins; <0.5 = naive wins. At standard budget (32K tokens), **all systems score below 0.5** — context stuffing wins most questions. V3 leads at 0.2825 (20% win rate). This is expected: at 32K, the agent can see most episodes anyway. The constrained-4K/2K benchmarks are where memory search quality should separate the pack.

#### Historical Scores (gpt-4o-mini judge)

| Adapter | Composite | Answer Quality | Budget | Run ID |
|---------|-----------|---------------|--------|--------|
| mem0-raw | 0.3714 | 0.5707 | 0.7500 | `830d711e5c17` |
| sqlite-chunked-hybrid (L=7) | 0.3632 | 0.6746 | 0.0000 | `8b9e83ae9dec` |
| sqlite-chunked-hybrid (L=6) | 0.3455 | 0.6156 | 0.1667 | `15dd0d8906e5` |
| sqlite-chunked (embed) | 0.3358 | — | 0.9583 | `ed6af908d63a` |
| sqlite-chunked-hybrid (L=5) | 0.3231 | 0.4920 | 0.6250 | `5ebb03982a9a` |
| sqlite-fts | 0.3161 | 0.5038 | 0.5833 | `4685b35ca819` |
| sqlite-embedding-openai | 0.3084 | — | 0.2917 | `62d73c7a8c9c` |
| sqlite-hybrid-openai | 0.2981 | 0.5232 | 0.0417 | `b34145a01055` |

#### Key Findings

1. **`batch_retrieve` is transformative** — avg tool calls: ~41 → 3.2 (>12x reduction). Agent adopted batch_retrieve for 20/24 questions and NEVER used individual memory_retrieve. Budget_compliance jumped 0.00 → 0.79 (5 violations remain, all token-based from 2 complex questions).

2. **Evidence coverage tripled** — 0.22 → 0.46 because batch_retrieve fetches multiple episodes in one call, giving the agent richer, more diverse context to cite.

3. **insight_depth and reasoning_quality improved** — 0.54 → 0.79 and 0.92 → 0.96. Richer evidence context enables deeper synthesis and better reasoning chains.

5. **letta-sleepy V3 finds the longitudinal signal**: delta/causal framing reduces longitudinal_advantage from −0.2822 → −0.1790, the least negative observed across all runs. The sleep synthesis identifies turning points and cross-component correlations, giving the agent a navigation document that guides which specific episodes to retrieve. V3 achieves answer_quality 0.8225 — best of any adapter. Evidence: V1/V2 hurt (0.4290, 0.4596 vs control 0.5402), showing that only causally-framed synthesis adds value.

4. **Embedding outperforms FTS on scope 01** — sqlite-embedding-openai (0.3891) beats sqlite-fts (0.2837) by 37%. Semantic similarity maps better to operational metrics progression than BM25 keyword matching. Notably, FTS shows the worst longitudinal_advantage (-0.5647) of all systems.

5. **Chunking is essential** — sqlite-hybrid-openai (non-chunked, 0.2981) performs 22% worse than sqlite-chunked-hybrid (0.3632). Full-episode embeddings lack the precision of section-level chunking.

5. **Score discrimination is now irrelevant** — batch_retrieve works despite RRF's low scores (~0.03). The agent searches (low confidence → searches more), then batches all retrieves into one call. This is actually optimal: more search variety + fewer retrieve calls.

**Note**: Scope-01 scores not directly comparable to full 6-scope results below (different LLM, provider, dataset scope).

### 7-Adapter × 6-Scope Sweep — 30 Runs (2026-02-20, Qwen3-32B)

RunPod vLLM H200 (Qwen3-32B agent + judge, Ollama nomic-embed-text for embeddings). All runs: 30 episodes, 24 questions, standard budget preset. 30/42 completed (10 failures: cognee/graphiti timeout on multi-scope, letta/mem0 scope-05 failures).

**Important caveats**: Qwen3-32B produces significantly lower scores than Qwen3-235B across all metrics. answer_quality drops ~40%, fact_recall drops to floor (0.167), and `naive_baseline_advantage` = 0.0 everywhere (baseline generation likely failed with 32B). These results establish **relative rankings** under a weaker model, not absolute quality. The previous 235B results remain the gold standard.

#### Composite Score Matrix (Qwen3-32B judge)

| Scope | Domain | compaction | chunked | letta | letta-sleepy | mem0-raw | graphiti | cognee |
|-------|--------|-----------|---------|-------|-------------|----------|----------|--------|
| 01 | cascading_failure | **0.5063** | 0.3697 | 0.3987 | 0.3610 | 0.3667 | 0.0000† | 0.0000† |
| 02 | financial_irregularity | **0.4096** | 0.3863 | 0.3404 | 0.3095 | 0.3101 | 0.3215 | — |
| 03 | clinical_signal | **0.3918** | 0.2957 | 0.3021 | 0.3159 | 0.0000† | — | — |
| 04 | environmental_drift | **0.4429** | **0.3536** | 0.3650 | 0.3528 | 0.3371 | — | — |
| 05 | insider_threat | **0.3492** | 0.3699 | — | — | — | — | — |
| 06 | market_regime | **0.3832** | 0.3246 | 0.3338 | 0.3109 | 0.3087 | — | — |
| **Mean** | | **0.3990** | 0.3649 | 0.3480 | 0.3300 | 0.2645 | 0.1607 | 0.0000 |
| **Scopes done** | | **6/6** | **6/6** | 5/6 | 5/6 | 5/6 | 2/6 | 1/6 |

† = hard-gated (evidence_grounding < 0.5) — graphiti/cognee scope 01 returned no episode refs. — = failed/timed out.

#### Key Findings (32B Sweep)

1. **Compaction leads across all scopes**: The summarize-then-answer baseline beats all retrieval-based systems with Qwen3-32B. This suggests that when the judge model is weaker, having a pre-digested summary helps more than raw episode retrieval. At 235B, compaction was expected to score *below* real memory systems — the 32B reversal is a model capability effect.

2. **chunked-hybrid is most reliable**: 6/6 scopes completed, 2nd place (0.365 mean). Consistent, no failures, no container dependencies.

3. **Graph-based adapters fail at scale with 32B**: Cognee and graphiti's entity extraction with Qwen3-32B is too slow (30+ min timeout) for multi-scope runs. Graphiti also suffers Cloudflare 524 timeouts on RunPod proxy. Both need 235B-class models or local inference for entity extraction.

4. **Budget compliance is perfect (1.0) everywhere**: The `/no_think` fix for Qwen3 thinking mode and `max_tokens=4096` cap resolved all timeout and budget issues from previous sessions.

5. **fact_recall at floor (0.167)**: 32B model cannot recall domain-specific terminology. All systems score identically on this metric.

6. **Scope 05 (insider_threat) is problematic**: Failed for letta, letta-sleepy, and mem0-raw (Letta server 500 errors, mem0 embedding overflow). Only compaction and chunked-hybrid completed it.

#### Run IDs (7-adapter sweep)

| Scope | compaction | chunked | letta | letta-sleepy | mem0-raw | graphiti | cognee |
|-------|-----------|---------|-------|-------------|----------|----------|--------|
| 01 | `89dcd32a37c0` | `12108a5b789a` | `d86228adc558` | `648f94f7749f` | `ac0b52ce0dfb` | `203a1c2a6959` | `f650050462c8` |
| 02 | `c8a5806036e9` | `a11758c2f140` | `623b19664870` | `2c59489f8f40` | `9c164a3e8c29` | `e05ce2e85ec8` | — |
| 03 | `7c0eade368a9` | `e9a1515c5714` | `eb28317c89d8` | `1863f6933914` | `0c85cbf04ae3` | — | — |
| 04 | `7a1512650f2d` | `2f17686d159a` | `126954043653` | `db62cf8eb685` | `db18f7f243a3` | — | — |
| 05 | `6c4d49131a24` | `c0b0de58e8b4` | — | — | — | — | — |
| 06 | `3032df338781` | `2c6695522556` | `0b39af5689eb` | `1fe42931d8a8` | `cbe06877161f` | — | — |

---

### Full 6-Scope Matrix — 4 Adapters × 6 Domains (2026-02-19)

Together AI infrastructure (Qwen3-235B-A22B agent + judge, GTE-ModernBERT-base embeddings). All runs: 30 episodes, 24 questions, standard budget preset (32K tokens, 20 tool calls, 10 turns). Scored with same Qwen3 judge for cross-scope comparability.

#### Composite Score Matrix

| Scope | Domain | letta | letta-sleepy V3 | mem0-raw | chunked-hybrid |
|-------|--------|-------|-----------------|----------|----------------|
| 01 | cascading_failure | 0.5308 | **0.5776** | 0.3714 | 0.4970 |
| 02 | financial_irregularity | 0.5420 | 0.4945 | 0.3822 | **0.5915** |
| 03 | clinical_signal | 0.3177 | **0.4467** | 0.2267 | 0.4913 |
| 04 | environmental_drift | 0.5942 | 0.4557 | 0.4150 | **0.6128** |
| 05 | insider_threat | 0.5723 | 0.5020 | 0.3794 | **0.6061** |
| 06 | market_regime | **0.6027** | 0.5127 | 0.3493 | 0.5949 |
| **Mean** | | 0.5266 | 0.4982 | 0.3540 | **0.5656** |
| **Rank** | | 2nd | 3rd | 4th | **1st** |

#### Run IDs (6-scope matrix)

| Scope | letta | letta-sleepy V3 | mem0-raw | chunked-hybrid |
|-------|-------|-----------------|----------|----------------|
| 01 | `be0003e5447b` | `8ef0786f0eb5` | `830d711e5c17` | `8581429063e7` |
| 02 | `413f8720bc5e` | `9640157a5f46` | `4438fabdc805` | `e53fe0b780bd` |
| 03 | `16c7109069f2` | `a70a0ec7d2f0` | `881ee9c69467` | `5651ffcdc3ce` |
| 04 | `6a8ba4330bed` | `fdb19d860dab` | `1586a78a1774` | `9eebc450872b` |
| 05 | `44b597f74624` | `8b0c6dcf2169` | `45a66bffefdd` | `6a01b08f081d` |
| 06 | `97e6fe2e4a99` | `3434fea9ddd8` | `553c4cc5964f` | `22cca5d429eb` |

#### V3 vs Letta Per Scope

| Scope | V3 − letta | Winner |
|-------|-----------|--------|
| 01 cascading_failure | +0.0469 | V3 |
| 02 financial_irregularity | −0.0475 | letta |
| 03 clinical_signal | +0.1290 | V3 |
| 04 environmental_drift | −0.1385 | letta |
| 05 insider_threat | −0.0703 | letta |
| 06 market_regime | −0.0899 | letta |
| **V3 win rate** | | **2/6** |

#### Key 6-scope Findings

1. **chunked-hybrid is most consistent**: Wins 4/6 scopes, mean 0.5656. No external server, no LLM preprocessing, pure retrieval with batch_retrieve. The simplest architecture is the strongest overall.

2. **letta-sleepy V3 is conditionally useful**: V3 helps only when base letta is struggling (scope 01: letta=0.5308, scope 03: letta=0.3177). When letta already performs well (scope 04: letta=0.5942), V3 hurts (−0.1385). The delta/causal synthesis adds navigational noise when retrieval is already confident.

3. **V3 advantage correlates inversely with letta base score**: Linear correlation between letta score and V3 advantage is strongly negative — meaning synthesis is compensatory, not additive. Sleep consolidation is a patch for weak retrieval, not a universal enhancement.

4. **mem0-raw is the most stable floor**: Consistent in the 0.22–0.42 range. No catastrophic failures, no domain-specific anomalies. Shows pure vector search over raw documents as a reliable baseline.

5. **Scope 03 (clinical_signal) is hardest**: All adapters score lowest here. letta=0.3177, mem0-raw=0.2267. The clinical/pharmacological signal may involve subtle cross-episode correlations that require causal reasoning, making V3's synthesis (+0.1290) most impactful.

6. **Scope 04/06 are easiest for retrieval-based systems**: chunked-hybrid 0.6128/0.5949, letta 0.5942/0.6027. Environmental and market signals may be more locally recoverable from single passages.

---

### Full 6-Scope Baselines (2026-02-17)

### Composite Scores

| Adapter | Composite | Ranking |
|---------|-----------|---------|
| **sqlite-embedding-openai** | **0.4778** | 1st |
| **sqlite-hybrid-openai** | **0.4648** | 2nd |
| **sqlite-fts** | **0.4519** | 3rd |

### Full Metric Breakdown

| Metric (weight) | FTS | Embedding | Hybrid |
|---|---|---|---|
| **Tier 1 — Mechanical** | | | |
| evidence_grounding (0.08) | 1.0000 | 1.0000 | 1.0000 |
| fact_recall (0.07) | 0.1820 | 0.1797 | 0.1797 |
| evidence_coverage (0.08) | 0.2494 | 0.2934 | 0.2859 |
| budget_compliance (0.07) | 0.9236 | 0.8819 | 0.8403 |
| citation_coverage (0.10) | 0.2494 | 0.2934 | 0.2859 |
| **Tier 2 — LLM Judge** | | | |
| answer_quality (0.15) | 0.6096 | 0.6326 | 0.6284 |
| insight_depth (0.15) | 0.6806 | 0.7431 | 0.7153 |
| reasoning_quality (0.10) | 0.9861 | 0.9792 | 0.9653 |
| **Tier 3 — Differential** | | | |
| longitudinal_advantage (0.15) | -0.4222 | -0.3904 | -0.4006 |
| action_quality (0.05) | 0.4167 | 0.5000 | 0.4792 |

### Resource Usage

| Metric | FTS | Embedding | Hybrid |
|---|---|---|---|
| Total tokens | 1,722,535 | 1,789,143 | 1,994,312 |
| Avg tokens/question | 11,962 | 12,425 | 13,849 |
| Max tokens (single Q) | 73,073 | 96,227 | 88,826 |
| Total wall time | 28.63 min | 33.67 min | 32.38 min |
| Avg wall time/question | 11.9 sec | 14.0 sec | 13.5 sec |
| Max wall time (single Q) | 33.1 sec | 50.2 sec | 34.0 sec |
| Budget violations (>32K) | 11 (7.6%) | 17 (11.8%) | 23 (16.0%) |

### Run IDs

| Adapter | Run ID | Date |
|---------|--------|------|
| sqlite-fts | `66b1fd6cfb37` | 2026-02-17 |
| sqlite-embedding-openai | `5470dd592ac1` | 2026-02-17 |
| sqlite-hybrid-openai | `5fad11997b47` | 2026-02-17 |

---

## What We've Built

### Dataset Generation Pipeline

**Two-stage progressive expansion** prevents LLM contamination:

1. **PlanOutline** (gpt-5.2, sees full spec): Produces per-episode structured data sheets with concrete metric values. Signal encoded as numeric progressions only, never text commentary.
2. **RenderEpisodes** (gpt-4.1-nano, blind to storyline): Formats each data sheet into a terse log entry independently. Cannot editorialize because it doesn't know what's signal.

This ensures signal only emerges from the *progression* across episodes, not from any single episode.

### Dataset Scopes (6 Complete)

| # | Scope | Domain | Core Signal | Episodes |
|---|-------|--------|-------------|----------|
| 01 | Cascading Failure | System ops | API latency → pool exhaustion → cascade | 120 |
| 02 | Financial Irregularity | Finance | Revenue recognition manipulation via AR aging | 120 |
| 03 | Clinical Signal | Pharma | Drug-drug interaction causing hepatotoxicity | 120 |
| 04 | Environmental Drift | Environmental | Chromium contamination from industrial discharge | 120 |
| 05 | Insider Threat | Cybersecurity | IP exfiltration by departing employee | 120 |
| 06 | Market Regime | Markets | Equity-bond correlation regime shift | 120 |

Each scope: 24 questions across 10 types (longitudinal, temporal, paraphrase, severity, negative, distractor, counterfactual, evidence_sufficiency, null_hypothesis, action_recommendation).

### Scoring Architecture (v3.2)

**Three-tier system**:

| Tier | Metrics | Method | Total Weight |
|------|---------|--------|--------------|
| 1 (Mechanical) | evidence_grounding, fact_recall, evidence_coverage, budget_compliance, citation_coverage | Exact computation | 40% |
| 2 (LLM Judge) | answer_quality, insight_depth, reasoning_quality | Pairwise judging | 40% |
| 3 (Differential) | **naive_baseline_advantage**, action_quality | Pairwise judge + system-delta | 20% |

**Hard gate**: Only `evidence_grounding` (>0.5) gates the composite. Budget compliance is observational — records token usage, wall time, and violation rates without zeroing the score.

**Pairwise fact judging**: For each key fact, candidate and reference answers are randomly assigned to positions A/B, judge picks winner, position is flipped to remove bias. More discriminative than absolute Likert scoring.

**Naive baseline advantage** (new in v3.2): Replaces `longitudinal_advantage` (which was always negative). For each question, a context-stuffed naive answer is generated using the same agent LLM with all episodes concatenated up to the checkpoint. The adapter's answer is then pairwise-judged against the naive answer per key fact. Score >0.5 means adapter beats context stuffing; <0.5 means naive wins. Under constrained budgets, the naive baseline also respects the token cap for fair comparison.

### Adapter Infrastructure

| Adapter | Search Mode | Status |
|---------|------------|--------|
| `null` | None (baseline) | Complete |
| `sqlite-fts` | BM25 keyword (FTS5) | **Benchmarked** (full + scope 01) |
| `sqlite-embedding-openai` | Semantic (OpenAI-compatible embeddings) | **Benchmarked** (full + scope 01) |
| `sqlite-hybrid-openai` | BM25 + OpenAI RRF | **Benchmarked** (full + scope 01) |
| `sqlite-chunked` | Section-chunked semantic (OpenAI-compatible) | **Benchmarked (scope 01)** |
| `sqlite-chunked-hybrid` | Section-chunked embeddings + FTS5 RRF + `batch_retrieve` | **Benchmarked (scope 01)** — 0.4970 |
| `sqlite-embedding` | Semantic (Ollama local) | Complete |
| `sqlite-hybrid` | BM25 + Ollama RRF | Complete |
| `mem0-raw` | Semantic (Mem0 + Qdrant, no extraction) | **Benchmarked (scope 01)** — 0.3690 |
| `mem0-extract` | Semantic (Mem0 + Qdrant, LLM extraction) | **Ran (scope 01): 0.0000** — `add(infer=True)` returns `{"results": []}` on structured log data; Mem0's extraction LLM finds no "personal memories" in operational metrics |
| `letta` | Semantic (Letta archival passages, vector search) | **Benchmarked (scope 01)** — **0.5308, SOTA** |
| `hindsight` | TEMPR: semantic + BM25 + graph + temporal, RRF | **Benchmarked (scope 01)** — **0.3511**. Graph entity extraction per retain() causes 20-100s ingest latency, 19/24 budget violations. reasoning_quality 0.9167 (matches Letta). 17.3 GB image. |
| `letta-sleepy` | Semantic (archival passages) + LLM sleep consolidation | **Benchmarked (scope 01, 4 variants)** — **V3: 0.5776 (SOTA)**. V0=0.5402, V1=0.4290, V2=0.4596, V3=0.5776. Delta/causal framing wins; minimal and filter framings hurt. Adds ~30-60s prepare() per checkpoint for LLM synthesis call. |
| `graphiti` | Temporal knowledge graph (FalkorDB, bi-temporal edges) | **Benchmarked (scope 01)** — **0.4983**. FalkorDB container on :6379. Perfect budget_compliance (1.0). Entity extraction in prepare() via LLM. EDGE_HYBRID_SEARCH_EPISODE_MENTIONS maps edges → episodes. evidence_grounding=1.0, reasoning_quality=0.9167. |
| `cognee` | GraphRAG (embedded LanceDB + Kuzu + SQLite) | **Benchmarked (scope 01)** — **0.5638** (2nd overall). No container needed. cognify() in prepare() for graph construction. Fixed: ACL-wrapped search result parsing + Kuzu lock conflicts. evidence_coverage 0.6319 (best of any system). budget_compliance 1.0. |
| `compaction` | Summarize-then-answer (naive memory baseline) | **New (session 12)**. LLM re-summarizes all episodes from scratch at each checkpoint. Summary returned as single search result. Establishes floor for what counts as a useful memory system. `requires_metering=True`. Configs for all 6 scopes. |

### Testing Infrastructure

| Component | Tests | Status |
|-----------|-------|--------|
| **Conformance suite** (`tests/conformance/`) | 25 contract tests × 3 adapters | All passing |
| **Onboarding harness** (`tests/unit/test_adapter_onboarding.py`) | 10 tests (registration, lifecycle, mini-run) | All passing |
| **Metering proxy** (`tests/unit/test_metering.py`) | 9 tests (store, manager, HTTP endpoints) | All passing |
| **Mem0 unit tests** (`tests/unit/test_mem0_adapter.py`) | 21 tests (mocked SDK, both strategies) | All passing |
| **Letta unit tests** (`tests/unit/test_letta_adapter.py`) | 29 tests (mocked letta-client, full lifecycle) | All passing |
| **Hindsight unit tests** (`tests/unit/test_hindsight_adapter.py`) | 47 tests (mocked hindsight-client, TEMPR+reflect lifecycle) | All passing |
| **Letta-sleepy unit tests** (`tests/unit/test_letta_sleepy_adapter.py`) | 54 tests (mocked letta + OpenAI clients, all 4 variants) | All passing |
| **Graphiti unit tests** (`tests/unit/test_graphiti_adapter.py`) | 28 tests (mocked graphiti-core, FalkorDB; full lifecycle) | All passing |
| **Cognee unit tests** (`tests/unit/test_cognee_adapter.py`) | 29 tests (mocked cognee; full lifecycle with chunk parsing) | All passing |

### Naive Baseline Metric (v3.2)

True head-to-head comparison: for each question, generates a context-stuffed answer using the same agent LLM with all episodes concatenated up to the checkpoint. Pairwise judges adapter answer vs naive answer per key fact. Score >0.5 = adapter beats context stuffing. Cached to `{run_dir}/scores/naive_baseline_cache.json`. Under constrained budgets, naive baseline also respects the token cap. Replaces `longitudinal_advantage` (which was always negative, measuring the wrong thing).

### Compaction Baseline Adapter

The "naive memory" from the literature: summarize-then-answer. All buffered episodes are re-summarized from scratch at each checkpoint via a single LLM call. Summary returned as the sole search result. Establishes the floor for what counts as a useful memory system — any system that doesn't beat compaction is doing worse than simple "compress everything and search the summary". Supports `batch_retrieve`, scope isolation, and cited episode retrieval for evidence grounding.

### Context-Limited Agent Mode (Dual-Cap)

Two constrained budget presets that cap how much data the agent can pull from tool results:

| Preset | `max_cumulative_result_tokens` | ~Episodes visible | What it tests |
|--------|-------------------------------|-------------------|---------------|
| `constrained-4k` | 4096 | ~8 of 30 (25%) | Targeted retrieval matters |
| `constrained-2k` | 2048 | ~4 of 30 (13%) | Search quality is everything |

Both keep `max_turns: 6`, `max_tool_calls: 12`, `max_agent_tokens: 16384`. The model stays smart (Qwen3-235B) — it just can't *see* everything. When the context budget is exhausted, the agent gets `"[Context budget exhausted — synthesize answer from evidence already retrieved]"`. 10 constrained configs created (5 adapters x 2 caps). Paper narrative: at standard (32K), context stuffing may win; at 4K/2K, memory search quality separates the pack.

### Per-Question Timing in Reports

Surfaces existing `wall_time_ms`, token counts, and tool call counts per question in BudgetCompliance details, markdown reports, and HTML reports. Color-coded in HTML (green <10s, yellow 10-30s, red >30s). Comparison reports show total/avg/max timing stats per adapter.

### LLM Metering Proxy

Lightweight stdlib HTTP proxy (`src/lens/metering/`) that intercepts adapter-internal OpenAI API calls via `OPENAI_BASE_URL` override. Captures token usage, model, and latency per call. Integrated into RunEngine — automatically activated for adapters with `requires_metering = True`.

### Per-Run Data Captured

Each run directory (`output/<run_id>/`) contains:
- `run_manifest.json` — adapter, dataset version, budget preset
- `config.json` — full RunConfig snapshot
- `log.jsonl` — structured event log with timestamps
- `scopes/<scope_id>/checkpoint_<N>/question_results.json` — per-question answers, tokens, wall time, refs cited, full turn history
- `scores/scorecard.json` — all metrics with token/time stats in budget_compliance details
- `report/report.{html,md}` — formatted reports

All data is preserved for longitudinal analysis across future runs.

---

## Analysis

### Key Findings

#### Finding 1: batch_retrieve as a structural fix for budget compliance

The core problem with chunked-hybrid (pre-batch_retrieve) was that RRF scores are inherently low (~0.03 vs cosine's ~0.6). The agent interprets low scores as low confidence and retrieves more — making ~41 tool calls per question vs. the 20-call budget limit. This isn't a bug; it's rational agent behavior given ambiguous signals.

`batch_retrieve` solves this by making the retrieval pattern irrelevant to budget: the agent searches (possibly many times), collects ref_ids, then fetches all of them in a **single** tool call. Tool call count drops to ~3.2 regardless of how many docs the agent wants. Budget_compliance jumps from 0.00 → 0.79, evidence_coverage triples (0.22 → 0.46), and composite goes from 0.3670 → 0.4970 (+35% over mem0-raw, same judge).

The broader implication: **any adapter that exposes `batch_retrieve` as an ExtraTool will benefit similarly**, regardless of its search mode. The harness already tracks `ref_ids` from any tool — all it requires is `call_extended_tool()` and an `ExtraTool` entry in `get_capabilities()`.

#### Finding 2: Mem0 extraction is hardcoded for personal assistant memory

`mem0-extract` scored **0.0000** on all 24 scope 01 questions. The failure mode is total: not a single memory was stored across 30 ingested episodes.

The root cause is `FACT_RETRIEVAL_PROMPT` in Mem0's source (`mem0/configs/prompts.py`):

```
You are a Personal Information Organizer, specialized in accurately storing
facts, user memories, and preferences... Store Personal Preferences... Maintain
Important Personal Details like names, relationships, and important dates...
```

Few-shot examples: `"Hi, my name is John. I am a software engineer."` → `["Name is John", "Is a Software engineer"]`

When this prompt receives a structured telemetry episode like:

```
/geo_lookup: REQUESTS 116420 | p50: 55ms p95: 130ms p99: 180ms | err: 0.05%
pool_utilization: 87.3% | connection_wait_p99: 12ms
```

...the extraction LLM correctly identifies there are zero personal preferences, names, or intentions present, and returns `{"facts": []}`. Mem0 then logs:

```
No new facts retrieved from input. Skipping memory update LLM call.
```

Nothing is stored in Qdrant. Later searches return empty results. `evidence_grounding = 0` triggers the hard gate. The composite is 0.0000.

**This is not a bug in Mem0.** The library is doing exactly what it was designed to do — it was designed for conversational personal assistant memory, not operational observability data. The extraction prompt is hardcoded; there is no config option to change it without forking the library.

**The contrast with `mem0-raw` is stark**: bypassing extraction (`infer=False`) gives 0.3690, the third-best result in the scope 01 comparison. The vector search backend works perfectly well. Only the extraction layer fails.

**Implication for the benchmark**: LENS is a legitimate stress test for memory systems that market themselves for enterprise/agentic use cases. A library that silently drops all structured data without any warning or fallback behavior is a meaningful failure mode to document. `mem0-raw` is the appropriate mode for LENS going forward.

#### Finding 3: Embedding beats FTS on structured operational data

On scope 01, sqlite-embedding-openai (0.3891) outperforms sqlite-fts (0.2837) by 37%. The gap is most visible in longitudinal_advantage: FTS scores -0.5647 vs embedding's -0.4068. BM25 keyword matching struggles with repetitive metric names that appear identically across every episode (the word "p99" appears in every log entry). Semantic embeddings capture the meaning of *which* metric is elevated, not just which tokens are present.

This reverses the 6-scope full-dataset result where FTS (0.4519) and embedding (0.4778) are much closer. The structured telemetry domain appears to particularly favor embeddings.

### What's Working

1. **Scoring produces correct system ordering**. Embedding > FTS on structured data; batch_retrieve lifts any adapter that adopts it; mem0-raw outperforms naive FTS; mem0-extract fails silently. All of these are the expected outcomes from first principles.

2. **Hard gate catches real failures**. `evidence_grounding >= 0.5` correctly zeros out mem0-extract (which genuinely retrieved nothing) without penalizing systems that search effectively.

3. **Pairwise judge functioning across providers**. Position-debiased A/B judging produces realistic scores (0.47–0.69 answer_quality range). Works consistently with both gpt-4o-mini and Qwen3-235B as judge.

4. **Perfect evidence grounding (1.0) across all non-zero runs**. No hallucinated citations across 5 adapters.

5. **Provider-agnostic infrastructure**. Together AI (Qwen3-235B + GTE-ModernBERT) works as a full OpenAI replacement for both agent and judge.

#### Finding 4: Hindsight — good reasoning, terrible budget

Hindsight scores 0.3511, placing 6th of 8 systems on scope 01. Its TEMPR multi-network retrieval (world, experience, opinion, observation epistemic layers) produces solid reasoning_quality (0.9167, tied with Letta) but fails on budget and coverage.

**Root cause — ingest latency**: Hindsight's `retain()` makes LLM calls during ingest for entity extraction and graph construction. Each call takes 20–100s. Across 30 episodes, this pushes total wall time to 26 minutes (avg 65s/question vs ~14s for Letta). Budget compliance: 0.2083 (19/24 violations).

**Root cause — text reformatting**: Hindsight transforms raw text into natural language. `[ep_001] 2024-01-01: p99 latency 320ms` becomes `On 2024-01-01, system latency was 320ms at p99`. The reformatted text loses the `[ep_id]` prefix, so document_id must be used as the episode identifier — the adapter correctly handles this. But the reformatted text loses numerical precision, and observation-type memory units (synthesized from multiple episodes) carry no document_id, requiring fallback parsing.

**`memory_reflect` not adopted**: Despite the `memory_reflect` ExtraTool description emphasizing temporal synthesis, the agent only used it in ~2/24 questions. The reflect() call was helpful when used (generating high-quality synthesis), but the agent defaulted to `memory_search` + `batch_retrieve`. The budget pressure from ingest latency likely reduced the agent's willingness to add more tool calls.

**Positive signals**: answer_quality 0.6687 shows the underlying LLM synthesis is good. evidence_grounding 0.6374 (passes hard gate) shows retrieval works. The negative longitudinal_advantage (-0.3311) matches other adapters — not worse. Hindsight's graph-based memory for structured telemetry is a promising architecture that suffers from an engineering mismatch (sync LLM calls in hot ingest path).

#### Finding 5: Sleep consolidation — prompt framing determines value

`letta-sleepy` V3 (delta/causal) scores 0.5776, a new SOTA. The four-variant matrix reveals that *how* you consolidate matters more than *whether* you consolidate:

| Variant | Composite | LongAdv | Answer Q | EvidGrnd | Insight |
|---------|-----------|---------|---------|---------|---------|
| V0 — no sleep (control) | 0.5402 | -0.2660 | 0.7068 | 1.0000 | 0.8750 |
| V1 — minimal summary | 0.4290 | -0.3365 | 0.6569 | 0.6974 | 0.5000 |
| V2 — actionable filter | 0.4596 | -0.2828 | 0.7456 | 0.6825 | 0.3750 |
| V3 — delta/causal | **0.5776** | **-0.1790** | **0.8225** | 0.7708 | 0.7083 |

V1 and V2 both *hurt* relative to control. V3 is the only variant that beats it.

**Why V1/V2 hurt**: A flat summary or a "what's important" filter produces a synthesis that is essentially a compressed episode list. It takes up one search-result slot without providing qualitatively different information than the episodes themselves. The synthesis displaces a passage result (evidence_grounding drops from 1.0 → 0.70), and the agent doesn't gain useful navigation toward specific episodes.

**Why V3 wins**: The delta/causal prompt specifically asks for transitions, correlations, and turning-point episodes. The resulting synthesis says things like *"latency crossed 500ms at ep_012 correlating with auth-service errors beginning at ep_010; connection pool exhaustion appears at ep_018"*. This acts as a navigation document — the agent reads the synthesis, knows exactly which episodes to retrieve, and calls `batch_retrieve(["ep_010","ep_012","ep_018"])`. Retrieval is more targeted and answers are more precise.

**longitudinal_advantage −0.1790** is the closest to zero of any system tested. It's still negative, but V3 is measurably approaching the theoretical floor where more episodes helps rather than hurts. V3's reasoning_quality reaches 1.0 (perfect on all questions where grounding passes).

**The fairness check holds**: V3's synthesis is produced without any knowledge of the specific benchmark questions, using only the same data the agent has access to and the same LLM. The only difference from base Letta is that cross-episode patterns have been pre-organised causally.

#### Finding 8: Cognee — GraphRAG produces best evidence coverage of any system

Cognee scores **0.5638** on scope 01, placing 2nd overall (behind only letta-sleepy V3's 0.5776). This is the strongest showing of any no-container, fully embedded system.

**Why evidence_coverage is exceptional (0.6319)**: Cognee's `cognify()` pipeline performs entity extraction, text summarization, and graph construction. When it chunks text, it maintains the `[episode_id]` prefix that the adapter parses during search. The resulting chunks are indexed in LanceDB with both vector embeddings and graph relationships. When the agent searches, it gets chunks that are highly topically relevant AND carry the episode provenance needed for evidence grounding. The 20 chunks returned per search (vs 5-7 for other adapters) provide more diverse episode coverage.

**Comparison with Graphiti**: Both are knowledge graph systems that do LLM entity extraction in `prepare()`. Cognee (0.5638) significantly outperforms Graphiti (0.4983). The difference: Cognee returns text chunks (preserving original episode content), while Graphiti returns graph edges (extracted relationships). Chunk-level retrieval gives the agent more raw context to synthesize answers from, while edge-level retrieval provides structured facts but less supporting detail.

**Budget compliance is perfect (1.0)**: Like Graphiti, all LLM processing happens in `prepare()` before the agent's budget clock starts. This architectural choice is proving decisive — Hindsight (0.3511) does the same type of entity extraction but during the budget window, losing 19/24 questions to violations.

**Fixed bug — ACL-wrapped search results not parsed**: Previous cognee runs all scored 0.0000. Root cause: cognee 0.5.2 enables backend access control by default for Kuzu+LanceDB, wrapping search results as `{"dataset_id": UUID, "search_result": [chunk_dicts]}`. The adapter's `_extract_chunks()` used `getattr()` which doesn't work for plain dicts. Fix: added dict key check for `"search_result"`. Also set `ENABLE_BACKEND_ACCESS_CONTROL=false` to avoid Kuzu lock conflicts between concurrent cognify and search operations.

**Embedding 413 errors**: Together AI rejects embedding requests over ~1MB. Cognee hit this on later checkpoints (20+ episodes, many chunks to embed). The errors are non-fatal — cognee retries or skips, and search still works from previously indexed chunks. For production use, a smaller batch size in cognee's LiteLLM config would avoid this.

#### Finding 7: Graphiti — graph structure doesn't beat simple passage retrieval

Graphiti (temporal knowledge graph with bi-temporal edge invalidation, FalkorDB) scores **0.4983** on scope 01 — a strong result that nearly matches Letta (0.5308), outperforms chunked-hybrid (0.4970), and far exceeds Hindsight (0.3511).

**Comparison with Hindsight**: Both systems perform LLM-based entity extraction during ingest/prepare, building knowledge graphs from episode text. But Graphiti achieves **perfect budget_compliance (1.0)** while Hindsight scores 0.2083. The difference: Graphiti's entity extraction runs in `prepare()` (before the agent's budget clock starts), while Hindsight's `retain()` runs synchronously during the benchmark. Architectural placement of LLM processing matters as much as the processing itself.

**Why not SOTA**: Graphiti's answer_quality (0.6746) lags Letta (0.7239) and letta-sleepy V3 (0.8225). longitudinal_advantage (−0.3370) is worse than Letta (−0.2822) and V3 (−0.1790). The temporal knowledge graph's edges capture entity relationships but don't provide the raw episode text that agents need for precise, evidence-grounded synthesis. When the agent retrieves graph edges, it gets structured relationships rather than the full context needed to answer domain-specific questions about metric progressions.

**Positive signals**: evidence_grounding 1.0 (no hallucinated refs), insight_depth 0.8333 (only Letta scores higher at 0.8750), reasoning_quality 0.9167 (tied with most top systems). The graph structure provides solid semantic navigation despite lower answer quality.

### What Needs Attention

1. **Heavy adapter fixes implemented, pending re-run (session 22)**: Root causes identified and fixed for all three failing adapters:
   - **cognee** (was 0/12): Sequential `cognee.add()` × 120 episodes exceeded timeout. Fixed: batch all episodes in single `cognee.add(data=[texts], data_per_batch=20)` call with 300s timeout.
   - **letta/letta-sleepy** (was 4/12 and 0/12): httpx 60s default timeout too short for Together AI on 84K contexts. Fixed: increased client timeout to 300s + route Letta's internal LLM to Cerebras (`cerebras/gpt-oss-120b`) for ~2s responses instead of 120-180s.
   - **graphiti** (was 1/12): Underscore chars in distractor entity names caused FalkorDB RediSearch syntax errors. Fixed: monkey-patch `build_fulltext_query()` to escape `_-|()~` in group_id field filters.

   **None of these are fundamental architectural failures** — all were configuration/timeout/escaping issues exposed by scaling from 30 to 120 episodes. Re-running should fill in the N=0 cells.

2. **Graphiti's single-run outlier**: graphiti s06/16k scored 0.553 composite (highest of any Phase 5 run), but 11/12 runs failed. With the RediSearch fix, more runs should complete and reveal whether graphiti is genuinely competitive or this was a fluke.

4. **Session 15 sweep scores superseded**: The 30 runs scored with Qwen3-32B (session 15) are now largely superseded by Phase 5 (53 runs, 235B judge, distractor datasets). The 32B results remain useful for confirming relative ranking consistency across judge models.

5. **citation_coverage equals evidence_coverage**. Agents aren't producing `[ref_id]` inline citations consistently.

6. **Phase 5 null composite = 0.000 everywhere**: The null adapter (no memory, random retrieval) consistently fails the evidence_grounding >= 0.5 hard gate with 120-episode distractor datasets. This is expected — with 90 distractor episodes, random retrieval is unlikely to surface signal content — but it means null's NBA scores (avg 0.231) come entirely from the naive baseline comparison, not from composite scoring.

### Question Type Difficulty Gradient

Updated from Phase 5 analysis (chunked-hybrid NBA, 6 scopes with distractors):

| Difficulty | Types | NBA Range | Discriminates? |
|-----------|-------|-----------|----------------|
| Easy (>0.60) | evidence (0.74), severity (0.72), longitudinal (0.72), action (0.69) | High NBA, high adapter spread | Yes (std > 0.2) |
| Medium (0.40-0.60) | counterfactual (0.44), paraphrase (0.52), distractor (0.45) | Moderate NBA | Somewhat |
| Hard (<0.40) | temporal (0.36), negative (0.39) | Low NBA, low adapter spread | No (negative std=0.037) |

**Key update from Phase 5**: severity and evidence questions are now the most discriminating (std > 0.2) — they separate good retrieval from bad. Negative questions have essentially zero discrimination (std = 0.037) and should be de-weighted in future versions. This gradient is the benchmark's core value proposition: a memory system that lifts the hard categories demonstrates genuine longitudinal reasoning.

---

## Scoring Pipeline Changelog

| Version | Date | Changes |
|---------|------|---------|
| v3.2 | 2026-02-19 | Replaced `longitudinal_advantage` (0.15 weight) with `naive_baseline_advantage` (0.15 weight) — true head-to-head vs context stuffing. Added per-question timing to BudgetCompliance details and reports. Added compaction baseline adapter. Added constrained-4k/2k budget presets with cumulative result token caps. |
| v3.1 | 2026-02-17 | Removed budget_compliance from gate (observational only), enriched with token/time stats |
| v3.0 | 2026-02-17 | Token cap 8K→32K, `--no-gate` CLI flag, pairwise judge for tier-3 metrics, CitationCoverage metric, stronger citation prompt, inline `[ref_id]` extraction |
| v2.0 | 2026-02-17 | Pairwise LLM judge for answer_quality, 3-tier scoring architecture |
| v1.0 | 2026-02-16 | Initial scoring: substring fact_recall, exact matching |

---

## Next Steps

### Immediate

1. **Write the paper** — All experimental data is in hand: 53 scored Phase 5 runs, bootstrap CIs, Wilcoxon tests, Kendall's W, judge reliability audit, publication-ready figures and LaTeX tables. Phase F of the publication plan.

2. **Re-run fixed cognee/graphiti/letta** — All three root causes identified and fixed (session 22): cognee batching, letta Cerebras routing + timeout, graphiti RediSearch escaping. Re-run 43 failed Phase 5 runs to fill in N=0 cells.

3. **Constrained budget runs with distractors** — Phase 5 used 8k/16k budgets. The Phase 1-2 constrained runs (4k/2k) used 30 episodes without distractors. Running 4k/2k with 120-episode distractor datasets would complete the budget degradation curve under realistic conditions.

4. **Add remaining memory system adapters**: Zep, LangChain, LlamaIndex following the 5-step onboarding pattern.

### Previously Completed

- **✅ 6-scope matrix** (session 7): 4 adapters × 6 scopes = 24 runs. chunked-hybrid wins 4/6.
- **✅ Experiment orchestrator** (session 8): `scripts/benchmark_orchestrator.py` + `experiments/matrix.json`.
- **✅ Hindsight async ingest** (session 8): Batch buffering in `prepare()`.
- **✅ Graphiti benchmarked** (session 10): 0.4983 on scope 01. Perfect budget_compliance.
- **✅ Cognee benchmarked** (session 11): 0.5638 on scope 01. Best evidence_coverage (0.6319).
- **✅ Paper-ready evaluation overhaul** (session 12): Naive baseline metric, compaction adapter, context-limited mode, per-question timing. 969 tests passing.
- **✅ Benchmark throughput optimization** (session 13): Parallel questions (5x), fast adapter model (2-3x), state caching (17x on re-runs). Graphiti: DNF→28min→1m36s. 995 tests passing.
- **✅ 7-adapter × 6-scope sweep on RunPod** (session 15): 30/42 runs completed with Qwen3-32B on H200. Fixed Qwen3 thinking mode, SQLite thread safety, per-adapter env vars. compaction leads (0.399 mean), chunked-hybrid 2nd (0.365).
- **✅ Phase 5 multi-scope with distractors** (session 21): 53/96 runs scored across 8 adapters × 6 scopes × 2 budgets. sqlite-chunked-hybrid leads (NBA 0.541). Statistical analysis complete: bootstrap CIs, Wilcoxon signed-rank, Kendall's W=0.683.
- **✅ Judge reliability validation** (session 21): Position-swap audit on 50 samples. Cohen's kappa=0.658 (moderate), 88% agreement, position bias 49%.
- **✅ Publication statistical analysis** (session 21): `scripts/analyze_publication.py` generates figures, LaTeX tables, JSON export. 4 significant pairwise differences (p<0.05).
- **✅ Heavy adapter fixes** (session 22): Diagnosed and fixed cognee (batching), letta (Cerebras routing + timeout), graphiti (RediSearch escaping). All 991 tests passing.

### Target Systems

| System | Adapter Names | Metering? | Local Setup | Status |
|--------|--------------|-----------|-------------|--------|
| **Mem0** | `mem0-raw`, `mem0-extract` | extract only | Qdrant (Podman) | **mem0-raw benchmarked (scope 01, 0.3690)**; mem0-extract disqualified — extraction prompt hardcoded for personal assistant memory, scores 0.0000 on structured data |
| **Zep** | `zep-raw`, `zep-summarize` | summarize only | Zep Docker | Not started |
| **Letta** | `letta` | No | Letta Podman + embed proxy | **Benchmarked (scope 01, 0.5308) — SOTA**. Requires: `podman run letta/letta`, `scripts/letta_embed_proxy.py`, two BYOK providers (together + together-oai). |
| **Hindsight** | `hindsight` | No | Hindsight Podman (17.3 GB) | **Benchmarked (scope 01, 0.3511)**. TEMPR retrieval + reflect(). Budget compliance 0.2083 (LLM ingest overhead). Requires: `podman run ghcr.io/vectorize-io/hindsight:latest`, env vars `HINDSIGHT_API_{LLM,EMBEDDINGS_OPENAI}_*`. |
| **Graphiti** | `graphiti` | No | FalkorDB Podman (:6379) | **Benchmarked (scope 01, 0.4983)**. Temporal knowledge graph with bi-temporal edges. Perfect budget_compliance (1.0). Requires: `podman run -p 6379:6379 falkordb/falkordb`, env vars `GRAPHITI_{LLM,EMBED}_*`. |
| **Cognee** | `cognee` | No | None (all embedded) | **Benchmarked (scope 01, 0.5638)** — 2nd overall. evidence_coverage 0.6319 (best of any system). budget_compliance 1.0. No container needed. Env vars `COGNEE_{LLM,EMBED}_*`. |
| **Compaction** | `compaction` | Yes (prepare) | None (in-process) | **New (session 12)**. Naive memory baseline: LLM re-summarizes all episodes at each checkpoint. Summary as sole search result. Configs for all 6 scopes + 2 constrained. |
| **LangChain** | `langchain-faiss`, `langchain-chroma` | No | In-process | Not started |
| **LlamaIndex** | `llamaindex` | Index build only | In-process | Not started |

### Publication Path

| Requirement | Status |
|-------------|--------|
| Working scoring pipeline | Done (v3.2) |
| 3 baseline systems benchmarked | Done |
| Budget compliance (non-zero composites) | Done |
| Pairwise judging (tier 2 + tier 3) | Done |
| Resource usage logging | Done |
| True naive baseline comparison | **Done (v3.2)** — head-to-head adapter vs context stuffing |
| Compaction ("naive memory") baseline | **Done (v3.2)** — summarize-then-answer adapter |
| Context-limited degradation curve | **Done (v3.2)** — constrained-4k/2k presets, configs ready |
| Per-question timing in reports | **Done (v3.2)** — wall time, tokens, tool calls per question |
| Multi-judge agreement (Cohen's kappa) | **Done (κ=0.658, 88% agreement, 49% position bias)** |
| Adapter conformance test suite | Done (25 tests × 4 adapters incl. compaction) |
| LLM metering proxy | Done (stdlib, RunEngine-integrated) |
| Mem0 adapters (raw + extract) | **mem0-raw benchmarked**; mem0-extract disqualified (domain mismatch, documented) |
| Results across ≥5 real memory systems | **5 done** (Mem0-raw, Letta, Hindsight, Graphiti, Cognee) — **milestone reached** |
| Human baseline | Harness built, not run |
| Statistical significance tests | **Done — bootstrap CIs, Wilcoxon signed-rank (4 sig. at p<0.05), Kendall's W=0.683** |

---

## Session Log

| Date | Session | Key Changes |
|------|---------|-------------|
| 2026-02-23 | Phase 5 completion + analysis (session 23) | **90/96 runs scored** (was 53). Fixed cognee (SQLite WAL mode + busy_timeout=60s for concurrent writes), letta/letta-sleepy (removed serial group constraint, bumped ingest_max_latency_ms 200→2000, all 8 runs parallelized in ~10 min), graphiti (FalkorDB MAX_QUEUED_QUERIES=500 + TIMEOUT=30s, semaphore 8→3, internal timeout 60→180s/episode — scopes 01,02,06 succeeded, 03-05 still timeout). Migrated remaining Together AI references to Cerebras in orchestrator. Full statistical analysis: Kendall's W=0.755 (strong concordance), 12/28 pairwise significant, budget effect p=0.016. Judge reliability audit: 7,652 calls, A/(A+B)=0.451 (minimal position bias), cognee 100% TIE rate (judge failure). Generated 5 publication figures + 2 LaTeX tables. **Key result**: sqlite-chunked-hybrid 0.473 beats all complex memory architectures. No adapter exceeds 0.50 composite. |
| 2026-02-23 | Heavy adapter fixes (session 22) | Diagnosed root causes for cognee (0/12), letta (4/12), graphiti (1/12) Phase 5 failures using multi-agent codebase analysis. **Cognee**: sequential `add()` × 120 episodes → timeout; fixed with batch `add(data=[texts], data_per_batch=20)`. **Letta**: httpx 60s timeout + slow Together AI; fixed with 300s timeout + route internal LLM to Cerebras (`cerebras/gpt-oss-120b`). **Graphiti**: underscores in distractor entity names → RediSearch syntax error; fixed with monkey-patch escaping `_-\|()~` in group_id filters. Updated orchestrator to auto-route Letta to Cerebras in Phase 5. 991 tests passing. |
| 2026-02-23 | Phase 5 full execution (session 21) | **53/96 runs scored** across 8 adapters × 6 scopes × 2 budgets with distractors (120 episodes). sqlite-chunked-hybrid leads (avg NBA 0.541, 12/12 complete). mem0-raw 2nd (0.441, 12/12), compaction 3rd (0.391, 12/12). cognee 0/12 (timeout), letta-sleepy 0/12 (server errors), graphiti 1/12. Fixed orchestrator cross-process state file locking (fcntl). Serialized cognee/graphiti/mem0 runs. Statistical analysis: bootstrap CIs, 4 significant pairwise tests, Kendall's W=0.683. Judge reliability: κ=0.658, 88% agreement, 49% position bias. Publication figures/tables generated. |
| 2026-02-23 | Phase 5 multi-scope infrastructure (session 20) | Fixed cognee adapter (removed `together_ai/` embedding model prefix that caused 422 errors). Verified graphiti adapter already has correct Together AI routing for entity extraction (prior Phase 3 failure was from `/tmp/run_phase3.sh` override, not adapter code). Generated 90 Phase 3d configs for scopes 02-06 (8 adapters × 5 scopes × 2 budgets). Added Phase 5 to orchestrator: `--phase 5 --cerebras-key` runs all scopes with distractors using Cerebras agent LLM. Removed hindsight from adapter list. Updated `config_filename()` with fallback for mixed naming conventions. Implemented benchmark methodology comparison (`docs/BENCHMARK_METHODOLOGY_COMPARISON.md`) and 5 parallelism optimizations (parallel scoring, ingest, scopes, baseline gen, tool calls). 991 tests passing. Total Phase 5 ready: 96 configs (8 adapters × 6 scopes × 2 budgets). |
| 2026-02-23 | Phase 3 with distractors (session 19) | **12/18 runs completed** with 120 episodes (30 signal + 90 distractors). Added `--include-distractors` to dataset compiler, `constrained-8k`/`constrained-16k` budget presets, remapped checkpoints for interleaved episodes. Switched to Cerebras (GPT-OSS-120B at 3000 tok/s, $0.35/M). A/B test confirmed GPT-OSS comparable quality to Qwen3-235B at 5x speed. **Key result: no memory system >50% answer quality.** sqlite-chunked-hybrid (0.477) beats all dedicated memory systems. Compaction collapsed from NBA 0.790 to 0.404 with distractors — experimental design validated. Cognee/graphiti failed (API compat). **Hindsight removed from evaluation** (NBA≈null, 17.3GB image, zero value demonstrated). Budget enforcement found non-binding — adapters use 4-5x budgeted tokens on first call. |
| 2026-02-23 | Phase 2 complete + ops report (session 18) | **12/12 Phase 2 runs** scored. Cognee 2K completed (run `72fea05ec21f`, NBA=0.855 anomalous — budget_compliance=0.167). Wrote `docs/ADAPTER_OPERATIONS_REPORT.md` — per-adapter operational assessment. Updated STATUS_REPORT with full 9-adapter ranking table including answer_quality, budget_compliance, evidence_coverage. |
| 2026-02-23 | Constrained Phase 2 heavy adapters (session 17) | **11/12 Phase 2 runs** scored: 6 heavy adapters × scope 01 × 2 budgets. letta-sleepy leads (NBA 0.667/0.693), hindsight≈null (0.168). Added LLM response caching. Fixed letta embed proxy for Together-only operation. Increased cognee timeout to 3600s. |
| 2026-02-22 | Constrained budget validation (session 16) | **36 runs** (3 adapters × 6 scopes × 2 budgets) validating that constrained token budgets (4K/2K) make retrieval quality matter. **Hypothesis validated**: compaction NBA=0.73 at 2K (CI: 0.68-0.79), chunked-hybrid NBA=0.35, null=0.07. All adapter-vs-null Wilcoxon p<0.0001. No significant 4K→2K degradation. Built phased orchestrator (`scripts/run_constrained_validation.py`) with concurrent execution, state tracking, resume support. Built analysis script (`scripts/analyze_constrained.py`) with bootstrap CIs, Wilcoxon tests, degradation curves. Generated null adapter configs for all 6 scopes. Cognee failed (SQLite schema error); other heavy infra adapters need service containers. 991 tests passing. |
| 2026-02-19 | Benchmark throughput optimization (session 13) | **3-strategy throughput optimization**: (1) Parallel question answering via ThreadPoolExecutor (`--parallel-questions 4`), verified 5x speedup on question phase. (2) Fast adapter-internal model — switched graphiti/cognee/mem0 entity extraction from Qwen3-235B (~7s/call) to Llama-3.3-70B-Instruct-Turbo (~1-2s/call); agent answering model stays 235B. (3) Adapter state caching — deterministic SHA256 cache keys, JSON manifests at `--cache-dir`, `get_cache_state()`/`restore_cache_state()` protocol on MemoryAdapter base class; implemented for graphiti, cognee, compaction. **Live verification**: graphiti scope01 fresh run 28 min (previously DNF with 235B), cached re-run **1m 36s** (~17x speedup). 995 tests passing (+26: 8 parallel questions, 18 caching). |
| 2026-02-19 | Paper-ready evaluation overhaul (session 12) | **Scoring v3.2**: Replaced `longitudinal_advantage` with `naive_baseline_advantage` (true head-to-head vs context stuffing). Added per-question timing to BudgetCompliance + markdown/HTML reports. **Compaction adapter**: summarize-then-answer baseline with scope isolation, batch_retrieve, cited episode retrieval — passes all conformance tests. **Context-limited mode**: `constrained-4k`/`constrained-2k` presets with `max_cumulative_result_tokens` cap in budget enforcer + harness. 22 new config files (6 compaction scopes, 10 constrained scope01 runs, 6 compaction matrix entries). **969 tests passing** (+122: 16 naive baseline, 26 compaction, rest conformance expansion). |
| 2026-02-20 | Cognee ACL fix + successful benchmark | **Cognee**: scored **0.5638** on scope 01 — **2nd overall** (only V3 0.5776 is higher). evidence_coverage **0.6319** is best of any system. Fixed critical bug: cognee 0.5.2 ACL mode wraps search results in dict with `"search_result"` key; `getattr()` doesn't work for dicts. Also fixed Kuzu lock conflicts by setting `ENABLE_BACKEND_ACCESS_CONTROL=false`. Previous runs (v10-v16) all scored 0.0000 or crashed. Added retry with exponential backoff to OpenAI client for Together AI 503 errors. Created cognee configs for scopes 02-06. 847 unit tests. |
| 2026-02-20 | Graphiti + Cognee debugging + benchmarks (prev session) | **Graphiti**: scored **0.4983** on scope 01 — nearly matches Letta (0.5308), perfect budget_compliance (1.0). Graphiti fix: `OpenAIGenericClient` (chat.completions) instead of `OpenAIClient` (responses.parse) for Together AI compat; chunked embedding batches to avoid 413 errors. |
| 2026-02-20 | Graphiti + Cognee adapters | Implemented `graphiti` adapter (temporal knowledge graph, FalkorDB, bi-temporal entity extraction, EDGE_HYBRID_SEARCH_EPISODE_MENTIONS) and `cognee` adapter (embedded GraphRAG, no container, LanceDB + Kuzu + SQLite, SearchType.CHUNKS with [episode_id] prefix parsing). Both use thread-hosted event loop for async→sync bridging. Both have `batch_retrieve` ExtraTool. Both added to `experiments/matrix.json` (parallel group, scope01 initial entry). Packages installed: graphiti-core[falkordb] + cognee. 845 unit tests (+57: 28 graphiti, 29 cognee). Configs: `configs/graphiti_scope01.json`, `configs/cognee_scope01.json`. |
| 2026-02-19 | Orchestrator + Hindsight async ingest | Built `scripts/benchmark_orchestrator.py` (30-experiment matrix orchestrator: parallel group {mem0-raw, chunked-hybrid, hindsight} via ThreadPoolExecutor, serial group {letta, letta-sleepy} on main thread; state file with atomic writes, resume, --filter, --dry-run). Created `experiments/matrix.json` (all 30 experiments, env var templates). Fixed Hindsight adapter: `ingest()` now buffers; `prepare()` flushes via `aretain_batch()` (was 20-100s/episode × 30 episodes). Added 4 new batch-ingest tests. Created `configs/hindsight_scope{02-06}.json`. 788 unit tests (+4). |
| 2026-02-19 | 4-adapter × 6-scope full matrix | Ran all 4 adapters (letta, letta-sleepy V3, mem0-raw, chunked-hybrid) across scopes 02–06 (20 new runs). **Results**: chunked-hybrid wins 4/6 scopes (mean 0.5656), letta wins 1/6 (mean 0.5266), V3 wins 1/6 (scope01 only, mean 0.4982). Key finding: V3 sleep synthesis is conditionally useful — helps when letta is weak (+0.0469 scope01, +0.1290 scope03) but hurts when letta is strong (−0.1385 scope04, −0.0899 scope06). Env var fixes: LETTA_EMBED_MODEL must be `together-oai/text-embedding-3-small` (not bare model name); mem0-raw needs MEM0_LLM_* vars + MEM0_EMBED_DIMS=768 + Qdrant reset before each domain. |
| 2026-02-19 | letta-sleepy adapter + sleep prompt matrix | Built `letta-sleepy` adapter with 4 sleep prompt variants (V0 control, V1 minimal, V2 actionable-filter, V3 delta-causal). Ran all 4 on scope 01. **V3: 0.5776 composite — new SOTA**. V3 specifically: answer_quality=0.8225 (best of any adapter), reasoning_quality=1.0000, longitudinal_advantage=−0.1790 (least negative, closest to 0 ever). V1/V2 both hurt (0.4290, 0.4596 vs control 0.5402) — only delta/causal framing adds value. 784 unit tests (+54). Key insight: sleep synthesis acts as a navigation document, not a replacement for retrieval. |
| 2026-02-19 | Hindsight adapter + benchmark | Built Hindsight (vectorize.io) adapter with TEMPR retrieval and `memory_reflect` ExtraTool (longitudinal synthesis). Key discoveries: image is 17.3 GB; correct env vars are `HINDSIGHT_API_EMBEDDINGS_OPENAI_*` (not generic `HINDSIGHT_API_EMBEDDINGS_*`); Hindsight reformats text so `document_id` must carry episode IDs. Ran scope 01: **0.3511 composite** — below SOTA. 19/24 budget violations from 20-100s ingest latency (LLM entity extraction during retain()). reasoning_quality 0.9167 matches Letta. 730 unit tests (+47 Hindsight tests). |
| 2026-02-19 | Letta adapter + new SOTA | Built full Letta adapter (archival passages, vector search, batch_retrieve, neutral storage persona). Setup: Podman container (letta/letta + TOGETHER_API_KEY) + local embed proxy (port 7878, rewrites model names → GTE-ModernBERT, needed for Together AI embedding compat). Ran scope 01: **0.5308 composite** (new SOTA, +6.8pp vs chunked-hybrid+batch_retrieve). Key metrics: evidence_grounding=1.0, answer_quality=0.7239, reasoning_quality=0.9167, insight_depth=0.8750. 683 unit tests passing (+29 Letta tests). |
| 2026-02-18 | Scope 01 sweep + mem0-extract investigation | Scored FTS (0.2837) and embedding (0.3891) with Qwen3 judge. Investigated mem0-extract's 0.0000 score: Mem0's `FACT_RETRIEVAL_PROMPT` is hardcoded for personal assistant memory ("Personal Information Organizer... names, preferences, dates") — correctly returns `{"facts":[]}` on telemetry, logs "No new facts retrieved, skipping", stores nothing. No config to override without forking. mem0-raw (infer=False) works fine at 0.3690. Full scope 01 ranking: chunked-hybrid+batch_retrieve (0.4970) > embedding (0.3891) > mem0-raw (0.3690) > hybrid-L7 (0.3670) > fts (0.2837) > mem0-extract (0.0000). |
| 2026-02-18 | batch_retrieve breakthrough | Added `batch_retrieve` extra tool to sqlite-chunked-hybrid. Agent adopted it for 20/24 questions, cutting avg tool calls 41→3.2 (>12x). Composite: 0.3632→0.4970 (+35%), budget: 0.00→0.79, evidence_coverage: 0.22→0.46. Harness extended to track `ref_ids` from any tool. Fair comparison: chunked-hybrid beats mem0 by 35% (0.4970 vs 0.3690, same judge). OpenAI quota exhausted — all future scoring via Together AI (Qwen3-235B). |
| 2026-02-18 | SQLite adapter optimization | Built 4 new SQLite adapter variants to compete with mem0 (0.3714): sqlite-chunked (0.3358), sqlite-chunked-hybrid L=5/6/7 (0.3231/0.3455/0.3632), sqlite-hybrid-openai (0.2981). Key finding: chunked-hybrid has best answer quality (0.6746 vs mem0's 0.5707) but budget_compliance (0.0 vs 0.75) costs it. RRF score discrimination (~0.03 vs cosine ~0.6) is the root cause — agents interpret low scores as uncertainty, doing 50% more retrieves. |
| 2026-02-18 | Mem0 real benchmark | Ran mem0-raw against scope 01 via Together AI (Qwen3-235B + GTE-ModernBERT + Qdrant). Fixed search bug (scope_id→user_id mapping + dict return parsing). Mem0-raw=0.3714 vs FTS=0.3161 (+17.5%). 723 tests pass. |
| 2026-02-18 | Adapter infra + Mem0 | Added conformance test suite (25 tests), onboarding harness, LLM metering proxy, Mem0 adapters (raw + extract), RunEngine metering integration. 721 tests pass. |
| 2026-02-17 | Scoring pipeline v3→v3.1 | Implemented 4-phase scoring fix: budget cap, tier-3 judge, citation coverage, observational budget. Ran 3 baselines. 613 tests pass. |
| 2026-02-20 | 7-adapter × 6-scope sweep on RunPod H200 (session 15) | Ran 30 benchmark runs across 7 adapters × 6 scopes (standard budget) using Qwen3-32B on RunPod H200. Fixed critical bugs: Qwen3 thinking mode (/no_think + max_tokens + strip `<think>` blocks) was causing massive response times and polluted answers; SQLite `check_same_thread=False` for parallel questions; mem0-raw env vars in sweep orchestrator. Added caching to chunked-hybrid and mem0-raw adapters. **Results**: compaction leads (0.399 mean), chunked-hybrid 2nd (0.365), letta 3rd (0.348). Low fact recall (~0.17) across all adapters — 32B model limitation. All runs have perfect budget compliance (1.0). Cognee and graphiti timeout on multi-scope runs (entity extraction too slow with 32B). 7/8 adapters working in sweep (hindsight needs separate debugging). |
| 2026-02-16 | Initial infrastructure | Dataset generation pipeline, 6 scopes, adapter system, initial scoring |
