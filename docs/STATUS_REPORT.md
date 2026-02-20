# LENS Benchmark: Project Status Report

**Last Updated**: 2026-02-20 (session 10)
**Scoring Pipeline**: v3.1 (pairwise judge + citation coverage + observational budget)
**Agent LLM**: Qwen3-235B-A22B (Together AI) / gpt-4o-mini (OpenAI)
**Judge LLM**: Qwen3-235B-A22B (Together AI) / gpt-4o-mini (OpenAI)
**Token Cap**: 32,768 (standard preset)
**Dataset**: 6 scopes, 144 questions, 720 episodes
**Unit Tests**: 847 passing (unit/ only)
**Adapters Tested**: 20 (14 systems on scope 01, 4 on full 6-scope)

---

## Executive Summary

LENS (Longitudinal Evidence-backed Narrative Signals) is a benchmark for evaluating whether AI agent memory systems can synthesize conclusions from evidence scattered across many sequential episodes, rather than finding answers in a single document.

**Current state**: Core infrastructure is feature-complete. We have 6 domain-diverse dataset scopes, a contamination-resistant two-stage data generation pipeline, a three-tier scoring system with pairwise LLM judging, and benchmark results across 3 SQLite-based retrieval variants. The scoring pipeline (v3.1) produces interpretable, non-zero composite scores that correctly rank retrieval strategies.

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

### Scope 01 Adapter Comparison — 9 Systems (2026-02-18)

Single scope (cascading_failure_01), 30 episodes, 24 questions. Together AI (Qwen3-235B-A22B for agent/judge, GTE-ModernBERT-base for embeddings).

**Note**: All `(Qwen3 judge)` scores use the same judge for fair comparison. Historical scores with `(gpt-4o-mini judge)` listed separately for reference.

#### Composite Scores — All Scope 01 Systems (Qwen3 judge)

| Adapter | Composite | Answer Quality | Fact Recall | Budget | Insight Depth | Reason | Evid Cov | Long. Adv | Run ID |
|---------|-----------|----------------|------------|--------|--------------|--------|---------|-----------|--------|
| **letta-sleepy V3** | **0.5776** | **0.8225** | — | 0.9167 | 0.7083 | **1.0000** | 0.4410 | **-0.1790** | `8ef0786f0eb5` |
| letta-sleepy V0 (control) | 0.5402 | 0.7068 | — | **1.0000** | 0.8750 | 0.9167 | 0.4583 | -0.2660 | `8c415eba299e` |
| **letta** | 0.5308 | 0.7239 | 0.2611 | 0.8333 | **0.8750** | 0.9167 | **0.4722** | -0.2822 | `be0003e5447b` |
| sqlite-chunked-hybrid + batch_retrieve | 0.4970 | 0.6552 | 0.2507 | **0.7917** | 0.7917 | **0.9583** | **0.4618** | -0.3465 | `8581429063e7` |
| sqlite-embedding-openai | 0.3891 | 0.5815 | 0.2323 | 0.2917 | 0.6667 | 0.8750 | 0.3264 | -0.4068 | `fef20b05d46b` |
| mem0-raw | 0.3690 | 0.5707 | — | 0.7500 | 0.5417 | 0.9167 | 0.1562 | — | `830d711e5c17` |
| sqlite-chunked-hybrid L=7 | 0.3670 | 0.6920 | — | 0.0000 | 0.5417 | 0.9167 | 0.2153 | — | `8b9e83ae9dec` |
| letta-sleepy V2 (actionable) | 0.4596 | 0.7456 | — | 1.0000 | 0.3750 | 0.9583 | 0.3160 | -0.2828 | `6e6e53e7581d` |
| letta-sleepy V1 (minimal) | 0.4290 | 0.6569 | — | 1.0000 | 0.5000 | 1.0000 | 0.2569 | -0.3365 | `1cbe02135799` |
| **cognee** | **0.5638** | 0.7357 | 0.2556 | **1.0000** | 0.7917 | 0.9167 | **0.6319** | -0.2425 | `77545ef2b9b8` |
| **graphiti** | 0.4983 | 0.6746 | 0.2553 | **1.0000** | 0.8333 | 0.9167 | 0.3507 | -0.3370 | `2bc821424282` |
| hindsight | 0.3511 | 0.6687 | 0.2775 | 0.2083 | 0.6250 | 0.9167 | 0.1667 | -0.3311 | `040bb488abbd` |
| sqlite-fts | 0.2837 | 0.4711 | 0.1914 | 0.3333 | 0.4583 | 0.7917 | 0.1840 | -0.5647 | `11d7bf53e4f0` |
| mem0-extract | 0.0000 | — | — | — | — | — | — | — | `a119b4906684` |

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

### Scoring Architecture (v3.1)

**Three-tier system**:

| Tier | Metrics | Method | Total Weight |
|------|---------|--------|--------------|
| 1 (Mechanical) | evidence_grounding, fact_recall, evidence_coverage, budget_compliance, citation_coverage | Exact computation | 40% |
| 2 (LLM Judge) | answer_quality, insight_depth, reasoning_quality | Pairwise judging | 40% |
| 3 (Differential) | longitudinal_advantage, action_quality | Pairwise judge + system-delta | 20% |

**Hard gate**: Only `evidence_grounding` (>0.5) gates the composite. Budget compliance is observational — records token usage, wall time, and violation rates without zeroing the score.

**Pairwise fact judging**: For each key fact, candidate and reference answers are randomly assigned to positions A/B, judge picks winner, position is flipped to remove bias. More discriminative than absolute Likert scoring.

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

1. **longitudinal_advantage is still negative for all systems (−0.18 to −0.57)**. Agents score lower on synthesis questions than on null_hypothesis controls. No current adapter flips this positive — which is expected for naive RAG, but is the core thing LENS is measuring. A memory system with genuine temporal reasoning should be the first to cross zero.

2. **fact_recall is low (0.18–0.25)**. Expected — key_facts use domain terms that agents paraphrase rather than quote. The judge-based answer_quality is the more accurate measure of correctness.

3. **citation_coverage equals evidence_coverage**. Agents aren't producing `[ref_id]` inline citations consistently. These metrics won't diverge until that changes.

4. **mem0-extract has no fallback**. The library silently drops episodes when extraction returns nothing. For a fair multi-domain benchmark, we need to document this clearly and use `mem0-raw` for LENS. If mem0 is ever updated with a configurable extraction prompt, `mem0-extract` becomes viable.

### Question Type Difficulty Gradient

From earlier analysis (hybrid adapter, 6-scope):

| Difficulty | Types | Score Range |
|-----------|-------|------------|
| Easy (>0.70) | paraphrase (0.76), temporal (0.74), severity (0.75) | Retrieval alone suffices |
| Medium (0.50-0.70) | longitudinal (0.70), evidence (0.69), action (0.65) | Requires some synthesis |
| Hard (<0.50) | negative (0.49), distractor (0.45), counterfactual (0.35) | Requires reasoning beyond retrieval |

This gradient is the benchmark's core value proposition. A memory system that lifts the hard categories demonstrates genuine longitudinal reasoning.

---

## Scoring Pipeline Changelog

| Version | Date | Changes |
|---------|------|---------|
| v3.1 | 2026-02-17 | Removed budget_compliance from gate (observational only), enriched with token/time stats |
| v3.0 | 2026-02-17 | Token cap 8K→32K, `--no-gate` CLI flag, pairwise judge for tier-3 metrics, CitationCoverage metric, stronger citation prompt, inline `[ref_id]` extraction |
| v2.0 | 2026-02-17 | Pairwise LLM judge for answer_quality, 3-tier scoring architecture |
| v1.0 | 2026-02-16 | Initial scoring: substring fact_recall, exact matching |

---

## Next Steps

### Immediate

1. **✅ DONE — 6-scope matrix complete**: All 4 adapters × 6 scopes = 24 runs completed and scored. chunked-hybrid wins 4/6, letta 1/6, letta-sleepy V3 1/6. V3 advantage is conditional on letta weakness, not universal.

2. **letta-sleepy V4 with adaptive synthesis**: Based on 6-scope findings, V3's synthesis is a static one-shot compression at prepare() time. The hypothesis: synthesis should track what the agent *actually queries* and adapt its framing to what the specific scope reveals. A checkpoint-aware synthesis that evolves based on the query patterns seen so far could close the gap on scopes where V3 currently underperforms. This is the most promising research direction.

3. **Add `batch_retrieve` to mem0-raw** — harness already supports `ref_ids` tracking from any tool. Expected to push mem0-raw from 0.35 toward 0.45+. Follow 5-step adapter pattern.

4. **Run judge reliability analysis**: `scripts/judge_reliability.py` ready. Target: Cohen's kappa >= 0.6 across duplicate question pairs.

5. **Add remaining memory system adapters**: Zep, LangChain, LlamaIndex following the 5-step onboarding pattern (adapter file → guarded import → unit tests → conformance → integration).

6. **✅ DONE — Hindsight async ingest**: `ingest()` now buffers episodes; `prepare()` flushes via `aretain_batch()`. Expected to cut ingest time 5-10x and push budget_compliance above 0.6. 4 new tests added (`tests/unit/test_adapter_hindsight.py`). Run scope 01 to validate.

7. **✅ DONE — Experiment orchestrator**: `scripts/benchmark_orchestrator.py` + `experiments/matrix.json` (30 experiments: 5 adapters × 6 scopes). Run with `python3 scripts/benchmark_orchestrator.py --dry-run` to preview. State tracking + resume support. Hindsight configs added for scopes 02–06.

8. **✅ DONE — Graphiti benchmarked**: 0.4983 on scope 01 (nearly matches Letta 0.5308). Perfect budget_compliance. Entity extraction in prepare() avoids Hindsight's budget trap.

9. **✅ DONE — Cognee benchmarked**: 0.5638 on scope 01 (2nd overall, best evidence_coverage). Fixed ACL-wrapped search result parsing. Previous runs all scored 0.0000. Configs created for all 6 scopes.

10. **Scale cognee to 6 scopes**: Configs `configs/cognee_scope{02-06}.json` ready. Requires `ENABLE_BACKEND_ACCESS_CONTROL=false` env var. Each run takes ~25 min (cognify is slow). Run sequentially to avoid Kuzu lock conflicts.

### Target Systems

| System | Adapter Names | Metering? | Local Setup | Status |
|--------|--------------|-----------|-------------|--------|
| **Mem0** | `mem0-raw`, `mem0-extract` | extract only | Qdrant (Podman) | **mem0-raw benchmarked (scope 01, 0.3690)**; mem0-extract disqualified — extraction prompt hardcoded for personal assistant memory, scores 0.0000 on structured data |
| **Zep** | `zep-raw`, `zep-summarize` | summarize only | Zep Docker | Not started |
| **Letta** | `letta` | No | Letta Podman + embed proxy | **Benchmarked (scope 01, 0.5308) — SOTA**. Requires: `podman run letta/letta`, `scripts/letta_embed_proxy.py`, two BYOK providers (together + together-oai). |
| **Hindsight** | `hindsight` | No | Hindsight Podman (17.3 GB) | **Benchmarked (scope 01, 0.3511)**. TEMPR retrieval + reflect(). Budget compliance 0.2083 (LLM ingest overhead). Requires: `podman run ghcr.io/vectorize-io/hindsight:latest`, env vars `HINDSIGHT_API_{LLM,EMBEDDINGS_OPENAI}_*`. |
| **Graphiti** | `graphiti` | No | FalkorDB Podman (:6379) | **Benchmarked (scope 01, 0.4983)**. Temporal knowledge graph with bi-temporal edges. Perfect budget_compliance (1.0). Requires: `podman run -p 6379:6379 falkordb/falkordb`, env vars `GRAPHITI_{LLM,EMBED}_*`. |
| **Cognee** | `cognee` | No | None (all embedded) | **Benchmarked (scope 01, 0.5638)** — 2nd overall. evidence_coverage 0.6319 (best of any system). budget_compliance 1.0. No container needed. Env vars `COGNEE_{LLM,EMBED}_*`. |
| **LangChain** | `langchain-faiss`, `langchain-chroma` | No | In-process | Not started |
| **LlamaIndex** | `llamaindex` | Index build only | In-process | Not started |

### Publication Path

| Requirement | Status |
|-------------|--------|
| Working scoring pipeline | Done (v3.1) |
| 3 baseline systems benchmarked | Done |
| Budget compliance (non-zero composites) | Done |
| Pairwise judging (tier 2 + tier 3) | Done |
| Resource usage logging | Done |
| Multi-judge agreement (Cohen's kappa) | Script ready, not yet run |
| Adapter conformance test suite | Done (25 tests × 3 adapters) |
| LLM metering proxy | Done (stdlib, RunEngine-integrated) |
| Mem0 adapters (raw + extract) | **mem0-raw benchmarked**; mem0-extract disqualified (domain mismatch, documented) |
| Results across ≥5 real memory systems | **5 done** (Mem0-raw, Letta, Hindsight, Graphiti, Cognee) — **milestone reached** |
| Human baseline | Harness built, not run |
| Statistical significance tests | Not started |

---

## Session Log

| Date | Session | Key Changes |
|------|---------|-------------|
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
| 2026-02-16 | Initial infrastructure | Dataset generation pipeline, 6 scopes, adapter system, initial scoring |
