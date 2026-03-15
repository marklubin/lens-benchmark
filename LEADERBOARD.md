# LENS Leaderboard

```
┌──────────────────────────────────────────────┐
│  LENS // Benchmark Results                   │
│  Last updated: 2026-03-15                    │
└──────────────────────────────────────────────┘
```

---

## > V1: ADAPTER BENCHMARK (Legacy) //

**Setup**: Modal driver, Qwen3.5-35B-A3B agent, 6 scopes (S07-S12), budget-constrained agent querying.

This measures **end-to-end answer quality** when a real LLM agent interrogates each memory system through LENS's tool interface. The agent writes its own queries — no pre-authored search strings.

| Rank | Adapter | Mean AQ | Category | Notes |
|-----:|---------|--------:|----------|-------|
| 1 | graphrag-light | 0.462 | Graph | Entity-relationship preprocessing |
| 2 | sqlite-chunked-hybrid | 0.431 | Hybrid | BM25 + embedding |
| 3 | letta | 0.413 | Agent Memory | Letta/MemGPT core memory |
| 4 | hopping-hybrid | 0.408 | Hybrid | Multi-hop + embedding |
| 5 | hopping | 0.404 | Hop-based | Multi-hop retrieval |
| 6 | letta-sleepy | 0.404 | Agent Memory | Letta with sleep consolidation |
| 7 | hierarchical-hybrid | 0.388 | Hybrid | Hierarchical + embedding |
| 8 | triadv1-pairs | 0.377 | Triad | Entity/relation/event triples |
| 9 | hierarchical | 0.369 | Hierarchical | Multi-level summaries |
| 10 | letta-v4 | 0.338 | Agent Memory | Multi-agent Letta |
| 11 | null | 0.328 | Baseline | Returns empty results |

### Key Findings

1. **Agent query quality is the binding constraint.** AQ spread narrows from 0.340 (static driver) to 0.134 (modal driver). Memory architecture matters less when the agent writes its own queries.

2. **Hybrid retrieval advantage collapses under agent querying.** hopping-hybrid drops -0.281, hierarchical-hybrid -0.298 vs. their static driver scores. Their static dominance was an artifact of precise pre-authored queries.

3. **graphrag-light is the only adapter that gains from agent querying** (+0.058 vs. static). Entity-relationship preprocessing creates a query-robust index that survives imprecise agent queries.

4. **Null baseline is uncomfortably close** (0.328). The agent's LLM reasoning partially compensates for having no memory at all — it answers from parametric knowledge plus question context.

### Per-Scope Breakdown (V1)

| Adapter | S07 | S08 | S09 | S10 | S11 | S12 |
|---------|----:|----:|----:|----:|----:|----:|
| graphrag-light | 0.48 | 0.51 | 0.44 | 0.42 | 0.49 | 0.43 |
| sqlite-chunked-hybrid | 0.45 | 0.47 | 0.42 | 0.40 | 0.46 | 0.39 |
| letta | 0.43 | 0.44 | 0.40 | 0.39 | 0.44 | 0.38 |
| null | 0.35 | 0.36 | 0.31 | 0.43 | 0.30 | 0.22 |

Note: Null wins S10 outright (0.43 vs. graphrag-light 0.42), indicating that scope's signal may be partially recoverable from question context alone.

---

## > V2: MEMORY STRATEGY ABLATION //

**Setup**: 10 scopes, M=3 repetitions per cell, Fact F1 scoring (few-shot Qwen grader). 1,900 answers graded out of 2,100 generated (90.5%).

This isolates the **memory consolidation strategy** from the retrieval architecture. All policies use the same underlying storage — only the memory management policy varies.

| Rank | Policy | Fact F1 | n | Analogous System | Description |
|-----:|--------|--------:|----:|------------------|-------------|
| 1 | policy_core_faceted | 0.466 | 271 | Multi-agent faceted memory | 4 parallel folds (entity/relation/event/cause) + merge |
| 2 | policy_summary | 0.443 | 268 | Rolling summary | Progressive summarization |
| 3 | policy_core | 0.441 | 275 | Letta/MemGPT core memory | Single-fold core memory block |
| 4 | policy_core_structured | 0.432 | 271 | Mastra/ACE pattern | Structured schema-driven memory |
| 5 | policy_core_maintained | 0.398 | 265 | Fold + refinement | Core memory with periodic refinement |
| 6 | policy_base | 0.381 | 274 | Raw retrieval | Retrieval only, no consolidation |
| 7 | null | 0.055 | 276 | No memory | Score floor |

### Key Findings

1. **Faceted memory wins** (+0.025 over core). Decomposing into entity/relation/event/cause folds and merging captures more signal than a single consolidation pass.

2. **Refinement hurts** (-0.043 vs. core). The maintained policy's periodic refinement prunes useful signal that hasn't yet been reinforced. Over-consolidation destroys information.

3. **Summary rises with more scopes.** Adding S13 (implicit decision) and S14 (epoch classification) — where summary leads — pushed it to rank 2. Progressive summarization excels at temporal boundary and decision reconstruction tasks.

4. **No universal best strategy.** Kendall's W = 0.145 (weak concordance). Per-scope winners include faceted (S07, S15, S16), core (S09, S11), summary (S13, S14), structured (S08, S10), and base (S12).

### Per-Scope Breakdown (V2, Fact F1)

| Policy | S07 | S08 | S09 | S10 | S11 | S12 | S13 | S14 | S15 | S16 |
|--------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| policy_base | 0.339 | 0.351 | 0.349 | 0.114 | 0.356 | 0.277 | 0.260 | 0.513 | 0.753 | 0.442 |
| policy_core | 0.458 | 0.301 | 0.643 | 0.103 | 0.456 | 0.201 | 0.324 | 0.543 | 0.808 | 0.511 |
| policy_core_faceted | 0.547 | 0.373 | 0.598 | 0.091 | 0.343 | 0.233 | 0.375 | 0.590 | 0.828 | 0.679 |
| policy_core_maintained | 0.323 | 0.181 | 0.611 | 0.074 | 0.455 | 0.156 | 0.372 | 0.552 | 0.717 | 0.526 |
| policy_core_structured | 0.529 | 0.426 | 0.538 | 0.178 | 0.410 | 0.156 | 0.267 | 0.512 | 0.750 | 0.475 |
| policy_summary | 0.376 | 0.388 | 0.561 | 0.093 | 0.395 | 0.162 | 0.522 | 0.650 | 0.722 | 0.604 |

- S10 (clinical_trial) is the hardest scope — near-floor for all policies (max 0.178)
- S12 (therapy_chat) remains very difficult — raw retrieval (base) beats all synthesis
- S15 (value_inversion) is the easiest — clearer numeric signal patterns

---

## > METHODOLOGY //

### Scoring

**V1** uses `answer_quality` (AQ): pairwise LLM comparison of candidate answer vs. canonical ground truth, position-debiased, scored by Qwen3.5-35B-A3B.

**V2** uses `fact_f1`: per-fact binary grading by few-shot Qwen3.5-35B-A3B, then F1 across all key facts. More granular than AQ — measures individual fact recall rather than holistic answer quality.

### Dataset Generation

All scopes use the two-stage information isolation pipeline:
1. **PlanOutline** (gpt-5.2, sees full spec) — produces per-episode structured data sheets with concrete metrics
2. **RenderEpisodes** (gpt-4.1-nano, blind to storyline) — formats data sheets into terse log entries independently

This prevents contamination: the renderer can't editorialize because it doesn't know what's signal.

### Validation Gates

| Metric | Target | Purpose |
|--------|--------|---------|
| Contamination (max single-ep coverage) | <80% | No single episode answers any question |
| Naive baseline (longitudinal) | <50% | LLM with full context shouldn't trivially pass |
| Key fact hit rate | >90% | All signal actually present in generated episodes |
| Word count | >340/episode | Sufficient density for realistic retrieval |
| Forbidden words | 0 | No editorial commentary ("increasing", "concerning", etc.) |

### Budget Constraints

The agent operates under strict budget limits:
- Maximum tool calls per question
- Maximum tokens per question
- Maximum latency per question

Budget violations are penalized in Tier 1 scoring. This prevents brute-force approaches that simply retrieve everything.

---

## > SUBMITTING RESULTS //

See [SUBMISSION_GUIDE.md](docs/guides/SUBMISSION_GUIDE.md) for the full process.

**Requirements**:
- Run all 6 benchmark scopes (S07-S12) for V1 results
- Include run config, raw answers, and scored output
- No cherry-picking — all scopes required, all questions answered
- Budget compliance enforced

Open a PR modifying this file with your results and the required artifacts.
