# LENS Benchmark: Project Status Report

**Last Updated**: 2026-02-18
**Scoring Pipeline**: v3.1 (pairwise judge + citation coverage + observational budget)
**Agent LLM**: gpt-4o-mini
**Judge LLM**: gpt-4o-mini
**Token Cap**: 32,768 (standard preset)
**Dataset**: 6 scopes, 144 questions, 720 episodes
**Unit Tests**: 613 passing

---

## Executive Summary

LENS (Longitudinal Evidence-backed Narrative Signals) is a benchmark for evaluating whether AI agent memory systems can synthesize conclusions from evidence scattered across many sequential episodes, rather than finding answers in a single document.

**Current state**: Core infrastructure is feature-complete. We have 6 domain-diverse dataset scopes, a contamination-resistant two-stage data generation pipeline, a three-tier scoring system with pairwise LLM judging, and benchmark results across 3 SQLite-based retrieval variants. The scoring pipeline (v3.1) produces interpretable, non-zero composite scores that correctly rank retrieval strategies.

**Key finding**: Embedding-based retrieval (0.4778 composite) outperforms keyword-only FTS (0.4519) by 5.7%. The benchmark reveals that all naive RAG systems have *negative* longitudinal advantage (-0.39 to -0.42) — they score lower on synthesis questions than on simple controls. This is the core signal LENS is designed to measure: a memory system that truly helps should flip this to positive.

---

## Latest Benchmark Results (2026-02-17)

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
| `sqlite-fts` | BM25 keyword (FTS5) | **Benchmarked** |
| `sqlite-embedding-openai` | Semantic (OpenAI text-embedding-3-small) | **Benchmarked** |
| `sqlite-hybrid-openai` | BM25 + OpenAI RRF | **Benchmarked** |
| `sqlite-embedding` | Semantic (Ollama local) | Complete |
| `sqlite-hybrid` | BM25 + Ollama RRF | Complete |

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

### What's Working

1. **Interpretable composites**. Budget compliance is observational (not gated), using `1 - violations/total_questions`. Graduated scores (0.84–0.92) instead of binary 0/1.

2. **Correct system ordering**. Embedding consistently outperforms FTS across answer_quality (0.63 vs 0.61), insight_depth (0.74 vs 0.68), evidence_coverage (0.29 vs 0.25), and action_quality (0.50 vs 0.42).

3. **Pairwise judge functioning**. Position-debiased A/B judging produces scores in the 0.61–0.63 range — realistic for naive RAG baselines.

4. **Graduated action_quality**. Judge-based scoring gives 0.42–0.50 instead of old bimodal 0/1 from substring matching.

5. **Perfect evidence grounding (1.0)**. All cited refs exist in the vault — no hallucinated citations.

6. **Full resource logging**. Token usage, wall time, and violation rates captured per run in scorecard details.

### What Needs Attention

1. **longitudinal_advantage is negative (-0.39 to -0.42)**. Agents score *lower* on synthesis questions than on null_hypothesis controls. Current naive RAG systems are better at confirming "there is no anomaly" than synthesizing cross-episode patterns. A memory system that truly helps should flip this to positive.

2. **fact_recall is low (0.18)**. Substring matching finds only 18% of key_facts. Expected — key_facts use calibrated domain terms that agents paraphrase. The judge-based answer_quality (0.62) is the more accurate measure.

3. **citation_coverage equals evidence_coverage (0.25-0.29)**. Agents aren't yet producing `[ref_id]` inline citations consistently. These should diverge once agents adapt to the stronger citation prompt.

4. **Budget violations correlate with retrieval sophistication**. Hybrid: 16% violations, embedding: 12%, FTS: 8%. More context surfaced → more tokens used. Not necessarily bad.

### Question Type Difficulty Gradient

From earlier analysis (hybrid adapter):

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

1. **Run judge reliability analysis**: `scripts/judge_reliability.py` is ready. Validates pairwise judge stability (target: Cohen's kappa >= 0.6).

2. **Wire real memory systems**: Build adapters for architecturally distinct systems (managed summary, agent-managed, index framework, graph-based). The scoring pipeline is trustworthy enough.

### Target Systems

| System | Type | Notes |
|--------|------|-------|
| Mem0 | Managed memory | Python SDK, cloud API |
| Zep | Conversation memory | Python SDK, self-hosted or cloud |
| Letta (MemGPT) | Agentic memory | Agent manages own memory |
| LangChain Memory | Framework memory | ConversationBufferMemory + VectorStore |
| LlamaIndex | Index-based | VectorStoreIndex with temporal metadata |

### Publication Path

| Requirement | Status |
|-------------|--------|
| Working scoring pipeline | Done (v3.1) |
| 3 baseline systems benchmarked | Done |
| Budget compliance (non-zero composites) | Done |
| Pairwise judging (tier 2 + tier 3) | Done |
| Resource usage logging | Done |
| Multi-judge agreement (Cohen's kappa) | Script ready, not yet run |
| Results across ≥5 real memory systems | Not started |
| Human baseline | Harness built, not run |
| Statistical significance tests | Not started |

---

## Session Log

| Date | Session | Key Changes |
|------|---------|-------------|
| 2026-02-17 | Scoring pipeline v3→v3.1 | Implemented 4-phase scoring fix: budget cap, tier-3 judge, citation coverage, observational budget. Ran 3 baselines. 613 tests pass. |
| 2026-02-16 | Initial infrastructure | Dataset generation pipeline, 6 scopes, adapter system, initial scoring |
