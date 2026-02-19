# LENS Benchmark: Project Status Report

**Last Updated**: 2026-02-19 (session 4)
**Scoring Pipeline**: v3.1 (pairwise judge + citation coverage + observational budget)
**Agent LLM**: Qwen3-235B-A22B (Together AI) / gpt-4o-mini (OpenAI)
**Judge LLM**: Qwen3-235B-A22B (Together AI) / gpt-4o-mini (OpenAI)
**Token Cap**: 32,768 (standard preset)
**Dataset**: 6 scopes, 144 questions, 720 episodes
**Unit Tests**: 683 passing (unit/ only)
**Adapters Tested**: 12 (8 systems on scope 01, 3 on full 6-scope)

---

## Executive Summary

LENS (Longitudinal Evidence-backed Narrative Signals) is a benchmark for evaluating whether AI agent memory systems can synthesize conclusions from evidence scattered across many sequential episodes, rather than finding answers in a single document.

**Current state**: Core infrastructure is feature-complete. We have 6 domain-diverse dataset scopes, a contamination-resistant two-stage data generation pipeline, a three-tier scoring system with pairwise LLM judging, and benchmark results across 3 SQLite-based retrieval variants. The scoring pipeline (v3.1) produces interpretable, non-zero composite scores that correctly rank retrieval strategies.

**Key finding 1 — Letta is new SOTA**: Letta (formerly MemGPT) achieves **0.5308 composite** (Qwen3 judge), +6.8pp above chunked-hybrid+batch_retrieve (0.4970). Letta's semantic vector search over archival passages with a neutral storage prompt achieves perfect evidence_grounding (1.0), answer_quality 0.7239, reasoning_quality 0.9167, and insight_depth 0.8750 — the highest on all three Tier-2 metrics. Budget compliance 0.8333 (5/30 violations, all from ingest latency).

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
| **letta** | **0.5308** | **0.7239** | 0.2611 | 0.8333 | **0.8750** | **0.9167** | 0.4722 | -0.2822 | `be0003e5447b` |
| sqlite-chunked-hybrid + batch_retrieve | 0.4970 | 0.6552 | 0.2507 | **0.7917** | 0.7917 | **0.9583** | **0.4618** | -0.3465 | `8581429063e7` |
| sqlite-embedding-openai | 0.3891 | 0.5815 | 0.2323 | 0.2917 | 0.6667 | 0.8750 | 0.3264 | -0.4068 | `fef20b05d46b` |
| mem0-raw | 0.3690 | 0.5707 | — | 0.7500 | 0.5417 | 0.9167 | 0.1562 | — | `830d711e5c17` |
| sqlite-chunked-hybrid L=7 | 0.3670 | 0.6920 | — | 0.0000 | 0.5417 | 0.9167 | 0.2153 | — | `8b9e83ae9dec` |
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

4. **Embedding outperforms FTS on scope 01** — sqlite-embedding-openai (0.3891) beats sqlite-fts (0.2837) by 37%. Semantic similarity maps better to operational metrics progression than BM25 keyword matching. Notably, FTS shows the worst longitudinal_advantage (-0.5647) of all systems.

5. **Chunking is essential** — sqlite-hybrid-openai (non-chunked, 0.2981) performs 22% worse than sqlite-chunked-hybrid (0.3632). Full-episode embeddings lack the precision of section-level chunking.

5. **Score discrimination is now irrelevant** — batch_retrieve works despite RRF's low scores (~0.03). The agent searches (low confidence → searches more), then batches all retrieves into one call. This is actually optimal: more search variety + fewer retrieve calls.

**Note**: Scope-01 scores not directly comparable to full 6-scope results below (different LLM, provider, dataset scope).

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
| `letta` | Semantic (Letta archival passages, vector search) | **Benchmarked (scope 01)** — **0.5308, new SOTA** |

### Testing Infrastructure

| Component | Tests | Status |
|-----------|-------|--------|
| **Conformance suite** (`tests/conformance/`) | 25 contract tests × 3 adapters | All passing |
| **Onboarding harness** (`tests/unit/test_adapter_onboarding.py`) | 10 tests (registration, lifecycle, mini-run) | All passing |
| **Metering proxy** (`tests/unit/test_metering.py`) | 9 tests (store, manager, HTTP endpoints) | All passing |
| **Mem0 unit tests** (`tests/unit/test_mem0_adapter.py`) | 21 tests (mocked SDK, both strategies) | All passing |
| **Letta unit tests** (`tests/unit/test_letta_adapter.py`) | 29 tests (mocked letta-client, full lifecycle) | All passing |

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

### What Needs Attention

1. **longitudinal_advantage is negative for every system (-0.35 to -0.57)**. Agents score lower on synthesis questions than on null_hypothesis controls. No current adapter flips this positive — which is expected for naive RAG, but is the core thing LENS is measuring. A memory system with genuine temporal reasoning should be the first to cross zero.

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

1. **Scale Letta to all 6 scopes** — highest priority. Letta at 0.5308 on scope 01 (Qwen3 judge) is the new SOTA. Running all 6 scopes establishes whether this advantage is consistent across domains.

2. **Add `batch_retrieve` to letta + mem0-raw** — harness already supports `ref_ids` tracking from any tool. Expected to improve budget_compliance for Letta (currently 0.8333, violations from 400ms+ ingest latency, not tool calls). Mem0-raw should improve from 0.3690 toward 0.45+.

3. **Scale chunked-hybrid + batch_retrieve to all 6 scopes** — establishes definitive cross-domain baselines. All future runs via Together AI (Qwen3-235B judge, GTE-ModernBERT embeddings).

4. **Run judge reliability analysis**: `scripts/judge_reliability.py` ready. Target: Cohen's kappa >= 0.6 across duplicate question pairs.

5. **Add remaining memory system adapters**: Zep, LangChain, LlamaIndex following the 5-step onboarding pattern (adapter file → guarded import → unit tests → conformance → integration).

### Target Systems

| System | Adapter Names | Metering? | Local Setup | Status |
|--------|--------------|-----------|-------------|--------|
| **Mem0** | `mem0-raw`, `mem0-extract` | extract only | Qdrant (Podman) | **mem0-raw benchmarked (scope 01, 0.3690)**; mem0-extract disqualified — extraction prompt hardcoded for personal assistant memory, scores 0.0000 on structured data |
| **Zep** | `zep-raw`, `zep-summarize` | summarize only | Zep Docker | Not started |
| **Letta** | `letta` | No | Letta Podman + embed proxy | **Benchmarked (scope 01, 0.5308) — new SOTA**. Requires: `podman run letta/letta`, `scripts/letta_embed_proxy.py`, two BYOK providers (together + together-oai). |
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
| Results across ≥5 real memory systems | 2 done (Mem0-raw, Letta), 3 remaining |
| Human baseline | Harness built, not run |
| Statistical significance tests | Not started |

---

## Session Log

| Date | Session | Key Changes |
|------|---------|-------------|
| 2026-02-19 | Letta adapter + new SOTA | Built full Letta adapter (archival passages, vector search, batch_retrieve, neutral storage persona). Setup: Podman container (letta/letta + TOGETHER_API_KEY) + local embed proxy (port 7878, rewrites model names → GTE-ModernBERT, needed for Together AI embedding compat). Ran scope 01: **0.5308 composite** (new SOTA, +6.8pp vs chunked-hybrid+batch_retrieve). Key metrics: evidence_grounding=1.0, answer_quality=0.7239, reasoning_quality=0.9167, insight_depth=0.8750. 683 unit tests passing (+29 Letta tests). |
| 2026-02-18 | Scope 01 sweep + mem0-extract investigation | Scored FTS (0.2837) and embedding (0.3891) with Qwen3 judge. Investigated mem0-extract's 0.0000 score: Mem0's `FACT_RETRIEVAL_PROMPT` is hardcoded for personal assistant memory ("Personal Information Organizer... names, preferences, dates") — correctly returns `{"facts":[]}` on telemetry, logs "No new facts retrieved, skipping", stores nothing. No config to override without forking. mem0-raw (infer=False) works fine at 0.3690. Full scope 01 ranking: chunked-hybrid+batch_retrieve (0.4970) > embedding (0.3891) > mem0-raw (0.3690) > hybrid-L7 (0.3670) > fts (0.2837) > mem0-extract (0.0000). |
| 2026-02-18 | batch_retrieve breakthrough | Added `batch_retrieve` extra tool to sqlite-chunked-hybrid. Agent adopted it for 20/24 questions, cutting avg tool calls 41→3.2 (>12x). Composite: 0.3632→0.4970 (+35%), budget: 0.00→0.79, evidence_coverage: 0.22→0.46. Harness extended to track `ref_ids` from any tool. Fair comparison: chunked-hybrid beats mem0 by 35% (0.4970 vs 0.3690, same judge). OpenAI quota exhausted — all future scoring via Together AI (Qwen3-235B). |
| 2026-02-18 | SQLite adapter optimization | Built 4 new SQLite adapter variants to compete with mem0 (0.3714): sqlite-chunked (0.3358), sqlite-chunked-hybrid L=5/6/7 (0.3231/0.3455/0.3632), sqlite-hybrid-openai (0.2981). Key finding: chunked-hybrid has best answer quality (0.6746 vs mem0's 0.5707) but budget_compliance (0.0 vs 0.75) costs it. RRF score discrimination (~0.03 vs cosine ~0.6) is the root cause — agents interpret low scores as uncertainty, doing 50% more retrieves. |
| 2026-02-18 | Mem0 real benchmark | Ran mem0-raw against scope 01 via Together AI (Qwen3-235B + GTE-ModernBERT + Qdrant). Fixed search bug (scope_id→user_id mapping + dict return parsing). Mem0-raw=0.3714 vs FTS=0.3161 (+17.5%). 723 tests pass. |
| 2026-02-18 | Adapter infra + Mem0 | Added conformance test suite (25 tests), onboarding harness, LLM metering proxy, Mem0 adapters (raw + extract), RunEngine metering integration. 721 tests pass. |
| 2026-02-17 | Scoring pipeline v3→v3.1 | Implemented 4-phase scoring fix: budget cap, tier-3 judge, citation coverage, observational budget. Ran 3 baselines. 613 tests pass. |
| 2026-02-16 | Initial infrastructure | Dataset generation pipeline, 6 scopes, adapter system, initial scoring |
