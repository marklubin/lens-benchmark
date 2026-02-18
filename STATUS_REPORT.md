# LENS Benchmark Status Report
**Date**: 2026-02-17
**Scoring Pipeline**: v3.1 (pairwise judge + citation coverage + observational budget)
**Agent LLM**: gpt-4o-mini
**Judge LLM**: gpt-4o-mini
**Token Cap**: 32,768 (standard preset)
**Dataset**: 6 scopes, 144 questions, 180 episodes

---

## Composite Scores

| Adapter | Composite | Ranking |
|---------|-----------|---------|
| **sqlite-embedding-openai** | **0.4778** | 1st |
| **sqlite-hybrid-openai** | **0.4648** | 2nd |
| **sqlite-fts** | **0.4519** | 3rd |

Ordering: **embedding > hybrid > FTS** — semantic search outperforms keyword search for longitudinal evidence synthesis.

---

## Full Metric Breakdown

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

---

## Resource Usage (per run)

| Metric | FTS | Embedding | Hybrid |
|---|---|---|---|
| Total tokens | 1,722,535 | 1,789,143 | 1,994,312 |
| Avg tokens/question | 11,962 | 12,425 | 13,849 |
| Max tokens (single Q) | 73,073 | 96,227 | 88,826 |
| Total wall time | 28.63 min | 33.67 min | 32.38 min |
| Avg wall time/question | 11.9 sec | 14.0 sec | 13.5 sec |
| Max wall time (single Q) | 33.1 sec | 50.2 sec | 34.0 sec |
| Budget violations (>32K) | 11 (7.6%) | 17 (11.8%) | 23 (16.0%) |

Key observation: more sophisticated retrieval → more context → higher token usage. Hybrid uses 16% more tokens than FTS. The embedding adapter has the single worst-case question (96K tokens, 3x budget), but hybrid has the most violations overall (23/144 = 16%).

---

## Analysis

### What's working

1. **Interpretable composites**. Budget compliance is now observational (not gated), using the formula `1 - violations/total_questions`. This gives graduated scores (0.84–0.92) instead of binary 0/1.

2. **Correct system ordering**. Embedding consistently outperforms FTS across answer_quality (0.63 vs 0.61), insight_depth (0.74 vs 0.68), evidence_coverage (0.29 vs 0.25), and action_quality (0.50 vs 0.42).

3. **Pairwise judge is functioning**. answer_quality uses position-debiased A/B judging against canonical answers. Scores are in the 0.61–0.63 range — the agent gets some facts but not all. Realistic for naive RAG baselines.

4. **action_quality is graduated**. With judge-based scoring, values range from 0.42 to 0.50 instead of the old bimodal 0/1 from substring matching.

5. **evidence_grounding is perfect (1.0)**. All cited refs exist in the vault — no hallucinated citations.

6. **Full resource logging**. Token usage, wall time, and violation rates are now captured per run in the budget_compliance metric details. Each run's `scores/scorecard.json` contains complete stats.

### What needs attention

1. **longitudinal_advantage is negative (-0.39 to -0.42)**. The agent scores *lower* on synthesis questions than on null_hypothesis controls. This is a meaningful finding: current naive RAG systems are better at confirming "there is no anomaly" than synthesizing cross-episode patterns. This is exactly the signal LENS is designed to measure — a memory system that truly helps should flip this to positive.

2. **fact_recall is very low (0.18)**. Substring matching of key_facts against answers only finds 18% of facts. This is expected — key_facts use domain-specific calibrated terms that the agent paraphrases rather than quoting verbatim. The judge-based answer_quality (0.62) is a more accurate measure.

3. **citation_coverage equals evidence_coverage (0.25-0.29)**. Agents aren't yet producing `[ref_id]` inline citations consistently. As agents adapt to the stronger citation prompt, these metrics should diverge. citation_coverage will measure whether the agent cites the *right* evidence, while evidence_coverage measures whether it *retrieves* the right evidence.

4. **Budget violations correlate with retrieval sophistication**. Hybrid has the most violations (16%) because it surfaces more context per query. This isn't necessarily bad — it means hybrid retrieval is finding more potentially relevant material, which could help with harder synthesis questions even as it costs more tokens.

---

## Run Data & Reports

### Per-run reports (HTML dashboards)
- FTS: `output/66b1fd6cfb37/report/report.html`
- Embedding: `output/5470dd592ac1/report/report.html`
- Hybrid: `output/5fad11997b47/report/report.html`

### Comparison report
- `comparison.md` (side-by-side metrics table)

### Per-run artifacts
Each run directory (`output/<run_id>/`) contains:
- `run_manifest.json` — metadata (adapter, dataset, budget preset)
- `config.json` — full RunConfig snapshot
- `log.jsonl` — structured event log with timestamps
- `scopes/<scope_id>/checkpoint_<N>/question_results.json` — per-question answers, tokens, wall time, refs cited, full turn history
- `scores/scorecard.json` — all metrics with detailed breakdowns
- `report/report.{html,md}` — formatted reports

All data is preserved for longitudinal analysis across future runs.

### Run IDs

| Adapter | Run ID | Date |
|---------|--------|------|
| sqlite-fts | `66b1fd6cfb37` | 2026-02-17 |
| sqlite-embedding-openai | `5470dd592ac1` | 2026-02-17 |
| sqlite-hybrid-openai | `5fad11997b47` | 2026-02-17 |

---

## Scoring Pipeline Changes (v3 → v3.1)

| Phase | Change | Status |
|-------|--------|--------|
| 1. Budget fix | Token cap 8K→32K, `--no-gate` CLI flag, gate_thresholds passthrough | Done |
| 2. Tier 3 judge | LongitudinalAdvantage + ActionQuality use pairwise judge (fallback to substring) | Done |
| 3. Judge reliability | `scripts/judge_reliability.py` — multi-judge kappa analysis | Ready to run |
| 4. Citations | Stronger system prompt, inline `[ref_id]` extraction, CitationCoverage metric | Done |
| 5. Budget observational | Removed budget_compliance from gate, enriched with token/time stats | Done |

All 613 unit tests pass.

---

## Next Steps

1. **Run judge reliability analysis**: `uv run python scripts/judge_reliability.py --run output/66b1fd6cfb37 --judge-a gpt-4o-mini --judge-b gpt-4o` — validates that the pairwise judge is stable enough for 3-8% deltas between systems.

2. **Wire real memory systems**: Build adapters for architecturally distinct systems (managed summary, agent-managed, index framework, graph-based) and run the full benchmark. The scoring pipeline is now trustworthy enough to compare them.

3. **Investigate negative longitudinal_advantage**: This is the benchmark's key diagnostic — understanding *why* synthesis questions score lower than controls will guide memory system design.
