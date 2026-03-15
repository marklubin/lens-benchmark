# LENS Leaderboard

```
┌──────────────────────────────────────────────┐
│  LENS // Benchmark Results                   │
│  Last updated: 2026-03-14                    │
└──────────────────────────────────────────────┘
```

---

## > V1: ADAPTER BENCHMARK //

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

**Setup**: 7 scopes, M=3 repetitions per cell, Fact F1 scoring (few-shot Qwen grader). 1407 answers graded out of 1470 generated (95.7%).

This isolates the **memory consolidation strategy** from the retrieval architecture. All policies use the same underlying storage — only the memory management policy varies.

| Rank | Policy | Mean F1 | Analogous System | Description |
|-----:|--------|--------:|------------------|-------------|
| 1 | core_faceted | 0.511 | Multi-agent faceted memory | 4 parallel folds (entity/relation/event/cause) + merge |
| 2 | core | 0.486 | Letta/MemGPT core memory | Single-fold core memory block |
| 3 | core_structured | 0.472 | Mastra/ACE pattern | Structured schema-driven memory |
| 4 | summary | 0.457 | Rolling summary | Progressive summarization |
| 5 | core_maintained | 0.427 | Fold + refinement | Core memory with periodic refinement |
| 6 | base | 0.412 | Raw retrieval | Retrieval only, no consolidation |
| 7 | null | 0.059 | No memory | Score floor |

### Key Findings

1. **Faceted memory wins** (+0.025 over core). Decomposing into entity/relation/event/cause folds and merging captures more signal than a single consolidation pass.

2. **Refinement hurts** (-0.059 vs. core). The maintained policy's periodic refinement prunes useful signal that hasn't yet been reinforced. Over-consolidation destroys information.

3. **Any consolidation beats raw retrieval.** Even rolling summary (+0.045 over base) provides meaningful lift. The gap to null (0.412 vs. 0.059) confirms LENS measures real memory capability.

4. **Structured memory is competitive** (0.472 vs. 0.486 core). Schema-driven approaches preserve signal well but impose overhead that slightly reduces coverage.

### Per-Scope Breakdown (V2, Mean F1)

| Policy | S01 | S07 | S08 | S10 | S12 | S14 | S15 |
|--------|----:|----:|----:|----:|----:|----:|----:|
| core_faceted | 0.52 | 0.54 | 0.50 | 0.48 | 0.42 | 0.53 | 0.59 |
| core | 0.49 | 0.51 | 0.48 | 0.46 | 0.40 | 0.50 | 0.56 |
| null | 0.06 | 0.07 | 0.05 | 0.06 | 0.04 | 0.06 | 0.08 |

- S12 (therapy_chat) is the hardest scope across all policies
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
