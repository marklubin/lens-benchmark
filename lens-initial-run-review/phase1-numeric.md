# LENS Benchmark: Phase 1 Numeric Scopes (S01-S06) Comparison Report

## Executive Summary

This report presents the results of the LENS benchmark's Phase 1 evaluation across six numeric scopes (S01-S06). LENS (Longitudinal Evidence-backed Narrative Signals) tests whether AI agent memory systems can synthesize conclusions from evidence scattered across many sequential episodes — a capability essential for real-world monitoring, investigation, and decision-support tasks.

**Top-line findings:**

- No memory system exceeds 50% composite score under controlled conditions (same agent LLM, same judge, same budget). With a frontier LLM (Letta + Sonnet), the 50% barrier is broken — see Section 7.
- **sqlite-chunked-hybrid** leads at 0.454 mean composite (8K budget), but the margin over second-place cognee (0.421) is not statistically significant at the pairwise level.
- **Model quality has large effects**: Letta with Claude Sonnet reaches 0.547 mean composite — higher than any adapter with GPT-OSS-120B. However, this comparison confounds the LLM with the adapter (see Section 7.2 for caveats).
- **Token budget universally helps**: 16K budget outperforms 8K for every adapter tested (p=0.016), with a mean gain of +0.041.
- **Distractors work as intended**: Compaction, which dominated under 30-episode signal-only conditions (NBA=0.73), collapsed to #7 of 8 at full scale with 90 distractors per scope.
- **Graph-based systems do not scale**: Graphiti failed on 3 of 6 scopes due to entity extraction timeouts at 120 episodes.

**Cross-phase note:** Phase 1 uses GPT-OSS-120B as the agent LLM and gated composites. Phase 2 (narrative) and Phase 3 (SRS) use Qwen3.5-35B-A3B with ungated composites. Scores are not directly comparable across phases. Rankings within each phase — where the agent LLM is held constant — are the valid unit of comparison.

---

## 1. Dataset Design

### 1.1 Scope Descriptions

Each scope models a different domain where longitudinal evidence synthesis is critical:

| Scope | Domain | Signal Pattern |
|-------|--------|----------------|
| S01 | Cascading infrastructure failure | Latency metrics climbing across services over time |
| S02 | Financial fraud detection | Suspicious transaction patterns emerging across accounts |
| S03 | Clinical deterioration | Subtle physiological markers shifting toward crisis |
| S04 | Environmental contamination | Chemical readings escalating across monitoring stations |
| S05 | Insider threat detection | Access patterns and data exfiltration indicators |
| S06 | Market regime shift | Trading volume and volatility signals preceding regime change |

### 1.2 Corpus Structure

Each scope contains 120 episodes:

- **30 signal episodes** — contain the actual evidence chain, spread across 5 arc phases (baseline, early_signal, red_herring, escalation, root_cause)
- **90 distractor episodes** — format-matched but topically orthogonal, designed to dilute signal without being obviously irrelevant
- **~700 words per episode** — terse operational metrics, logs, and configurations (no prose commentary)
- **~84K tokens total** per scope

### 1.3 Question Design

Each scope has 4 checkpoints with 6 questions each (24 total per scope). Question types:

- **Longitudinal**: Requires synthesizing a trend across multiple episodes (e.g., "What pattern do you see in geo-lookup latency over episodes 5-25?")
- **Null hypothesis**: Tests whether the agent can distinguish signal from noise (e.g., "Is there evidence of database corruption, or can the symptoms be explained by another mechanism?")
- **Action recommendation**: Requires the agent to propose interventions grounded in the evidence chain
- **Varies**: Additional question types specific to the domain

The benchmark is designed so that **no single episode can answer any question**. Signal emerges only from the progression across episodes. This is enforced by the two-stage generation pipeline: a planner encodes signal as numeric progressions, then a separate renderer formats numbers without knowing their significance.

---

## 2. Evaluation Setup

### 2.1 Primary Evaluation: Phase 5

The definitive numeric scope evaluation used the following configuration:

| Parameter | Value |
|-----------|-------|
| Agent LLM | GPT-OSS-120B on Cerebras (~3000 tok/s) |
| Judge LLM | Qwen3-235B on Cerebras (~1400 tok/s) |
| Token budgets | 8K and 16K |
| Adapters tested | 8 (sqlite-chunked-hybrid, cognee, graphiti, mem0-raw, letta, letta-sleepy, compaction, null) |
| Target configurations | 96 (8 adapters x 6 scopes x 2 budgets) |
| Runs scored | 90 of 96 (6 graphiti failures) |

### 2.2 Scoring Framework

The composite score combines four metrics:

- **Answer quality** (weight: dominant) — LLM judge performs pairwise comparisons of adapter answers against ground truth
- **Evidence grounding** — binary check: does the answer cite retrievable evidence?
- **Evidence coverage** — what fraction of key facts does the retrieved context cover?
- **Budget compliance** — did the adapter stay within the token budget?

A **hard gate** forces the composite to 0.0 if evidence_grounding < 0.5 OR budget_compliance < 0.5. This gate significantly shapes the rankings (see Section 5).

### 2.3 Adapter Descriptions

| Adapter | Architecture | Approach |
|---------|-------------|----------|
| **sqlite-chunked-hybrid** | SQLite FTS + vector search | Chunks episodes, retrieves via hybrid keyword + semantic search |
| **cognee** | Knowledge graph + vector store | Builds structured knowledge graph from episodes |
| **graphiti** | Temporal knowledge graph | Entity extraction and temporal relationship tracking |
| **mem0-raw** | Raw vector memory | Stores episode embeddings, retrieves by similarity |
| **letta** | Stateful agent memory | Letta agent with archival + recall memory |
| **letta-sleepy** | Letta with sleep-time compute | Letta agent that processes episodes during "sleep" phases |
| **compaction** | Progressive summarization | Compacts episodes into running summary |
| **null** | No memory (baseline) | Answers questions with no stored context |

---

## 3. Primary Results: Phase 5

### 3.1 Composite Score Rankings (8K Budget)

| Rank | Adapter | S01 | S02 | S03 | S04 | S05 | S06 | Mean | 95% CI |
|------|---------|-----|-----|-----|-----|-----|-----|------|--------|
| 1 | sqlite-chunked-hybrid | 0.424 | 0.482 | 0.381 | 0.535 | 0.520 | 0.386 | **0.454** | [0.406, 0.502] |
| 2 | cognee | 0.438 | 0.402 | 0.374 | 0.471 | 0.418 | 0.423 | **0.421** | [0.397, 0.446] |
| 3 | graphiti | 0.491 | 0.220 | --- | --- | --- | 0.467 | **0.393** | [0.220, 0.491] |
| 4 | mem0-raw | 0.345 | 0.320 | 0.186 | 0.419 | 0.407 | 0.299 | **0.329** | --- |
| 5 | letta | 0.288 | 0.338 | 0.215 | 0.455 | 0.424 | 0.239 | **0.327** | --- |
| 6 | letta-sleepy | 0.300 | 0.313 | 0.252 | 0.451 | 0.377 | 0.240 | **0.322** | --- |
| 7 | compaction | 0.339 | 0.000 | 0.198 | 0.364 | 0.309 | 0.259 | **0.245** | [0.137, 0.328] |
| 8 | null | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | --- |

The null adapter scores 0.000 across all scopes because it has no memory and therefore always fails the evidence_grounding gate.

### 3.2 Scope Difficulty

Averaging across all adapters reveals consistent difficulty ordering:

- **Easiest**: S04 (environmental contamination / multi-AZ cloud outage) — clear numeric metric progressions that are relatively straightforward to track across episodes
- **Hardest**: S03 (clinical deterioration) — subtle clinical markers that are harder to synthesize because the signal is less obviously numeric and requires domain-aware interpretation

Rankings are strongly concordant across scopes, with Kendall's W = 0.755 (see Section 4.1), meaning the relative ordering of adapters is consistent regardless of which domain is tested.

### 3.3 Graphiti Failures

Graphiti completed only 3 of 6 scopes (S01, S02, S06). Scopes S03, S04, and S05 all failed due to entity extraction timeouts. Each of the 120 episodes requires approximately 5 LLM calls for entity and relationship extraction, totaling ~600 LLM calls per scope just for graph construction. With remote LLM APIs, this does not complete within practical time limits. This is a fundamental scaling limitation of graph-based entity extraction approaches when applied to large sequential corpora.

---

## 4. Statistical Analysis

### 4.1 Cross-Scope Concordance

| Test | Statistic | Interpretation |
|------|-----------|----------------|
| Kendall's W | **0.755** | Strong agreement in adapter rankings across scopes |

A W of 0.755 indicates that the relative performance of adapters is highly consistent across all six domains. This is important: it means the benchmark is measuring a general capability (longitudinal synthesis), not domain-specific knowledge.

### 4.2 Pairwise Significance

Of 28 possible adapter pairs, **12 are statistically significant at p < 0.05** (all at p = 0.031, Wilcoxon signed-rank test):

| Pair | Direction |
|------|-----------|
| cognee vs compaction | cognee > compaction |
| cognee vs letta-sleepy | cognee > letta-sleepy |
| cognee vs mem0-raw | cognee > mem0-raw |
| cognee vs null | cognee > null |
| letta vs null | letta > null |
| letta-sleepy vs null | letta-sleepy > null |
| mem0-raw vs null | mem0-raw > null |
| sqlite-chunked-hybrid vs null | chunked-hybrid > null |
| sqlite-chunked-hybrid vs compaction | chunked-hybrid > compaction |
| sqlite-chunked-hybrid vs letta | chunked-hybrid > letta |
| sqlite-chunked-hybrid vs letta-sleepy | chunked-hybrid > letta-sleepy |
| sqlite-chunked-hybrid vs mem0-raw | chunked-hybrid > mem0-raw |

Notable non-significant pairs: sqlite-chunked-hybrid vs cognee, letta vs letta-sleepy, letta vs mem0-raw. The top two adapters are not statistically distinguishable from each other, and the middle tier (mem0-raw, letta, letta-sleepy) forms a cluster.

### 4.3 Budget Effect (8K vs 16K)

16K budget outperforms 8K for **every adapter tested** (p = 0.016, Wilcoxon signed-rank):

| Adapter | 8K Mean | 16K Mean | Delta |
|---------|---------|----------|-------|
| graphiti | 0.393 | 0.459 | +0.067 |
| compaction | 0.245 | 0.300 | +0.055 |
| letta | 0.327 | 0.366 | +0.039 |
| mem0-raw | 0.330 | 0.368 | +0.039 |
| sqlite-chunked-hybrid | 0.454 | 0.492 | +0.037 |
| letta-sleepy | 0.322 | 0.348 | +0.026 |
| cognee | 0.421 | 0.444 | +0.023 |

Mean improvement: **+0.041**. The effect is universal but modest, suggesting that while more context helps, the bottleneck is retrieval quality rather than retrieval quantity.

---

## 5. Hard Gate Impact Analysis

The hard gate (composite forced to 0.0 when evidence_grounding < 0.5 OR budget_compliance < 0.5) has an outsized influence on the final rankings. Analysis of 263 numeric runs:

| Adapter | Runs | Gate Fired (%) | Mean (gated) | Mean (ungated) | Delta |
|---------|------|----------------|-------------|----------------|-------|
| sqlite-chunked-hybrid | 50 | 55% | 0.144 | 0.437 | +0.293 |
| compaction | 43 | 44% | 0.337 | 0.465 | +0.128 |
| letta | 29 | --- | 0.241 | 0.441 | +0.200 |
| letta-sleepy | 29 | --- | 0.241 | 0.428 | +0.187 |
| cognee | 26 | 92% | 0.025 | 0.365 | +0.340 |
| mem0-raw | 28 | --- | 0.122 | 0.376 | +0.255 |
| null | 27 | 100% | 0.000 | 0.189 | +0.189 |

### 5.1 Key Observations

**sqlite-chunked-hybrid has the highest gate-fire rate among competitive adapters (55%)** yet still leads overall. This is because when the gate does not fire, its answer quality is the highest of any adapter. The gate fires primarily on **budget_compliance** — a single retrieval call returns more context than the 8K budget allows (typically 4-5x over budget). This is a structural issue with the adapter's retrieval granularity, not a quality problem.

**Cognee's #2 ranking is unreliable.** With a 92% gate-fire rate and 100% judge TIE rate (see Section 6), cognee's composite score is driven entirely by the 8% of runs that pass the gate, evaluated on mechanical metrics alone. The ranking should be treated with caution.

**Without gating, rankings shift significantly.** Compaction rises because its summaries always pass evidence_grounding and budget_compliance. sqlite-chunked-hybrid's true answer quality advantage becomes clearer when not penalized for budget overruns.

---

## 6. Judge Reliability

The LLM judge (Qwen3-235B) performs pairwise comparisons to evaluate answer quality.

| Metric | Value |
|--------|-------|
| Total judge calls | 7,652 |
| Position A wins | 2,503 (32.7%) |
| Position B wins | 3,052 (39.9%) |
| Ties | 2,097 (27.4%) |
| A/(A+B) ratio | 0.451 |

The A/(A+B) ratio of 0.451 indicates **minimal position bias** — the judge is not systematically favoring whichever answer appears first.

### 6.1 Per-Adapter TIE Rates

| Adapter | TIE Rate | Implication |
|---------|----------|-------------|
| cognee | **100%** | Judge never discriminates; answer_quality defaults to 0.5 |
| letta | ~38% | Moderate judge uncertainty |
| letta-sleepy | ~38% | Moderate judge uncertainty |
| graphiti | 20.5% | Judge can usually discriminate |
| mem0-raw | 4-6% | Judge reliably discriminates |
| null | 4-6% | Judge reliably discriminates |
| sqlite-chunked-hybrid | 4-6% | Judge reliably discriminates |
| compaction | 4-6% | Judge reliably discriminates |

**Cognee's 100% TIE rate is a critical measurement failure.** The judge is unable to distinguish cognee's answers from the reference in any pairwise comparison, which means cognee's composite score is entirely determined by mechanical metrics (evidence_grounding, evidence_coverage, budget_compliance). Cognee actually achieves the best evidence_coverage of any adapter (0.632 on S01), suggesting its graph-based retrieval produces high-quality context chunks — but this quality is invisible to the composite because the judge cannot evaluate it.

---

## 7. Model Quality vs. Memory Architecture

### 7.1 Letta with Claude Sonnet

When the Letta internal LLM is upgraded from GPT-OSS-120B to Claude Sonnet (a significantly more capable model), scores increase dramatically:

| Adapter | S01 | S02 | S03 | S04 | S05 | S06 | Mean |
|---------|-----|-----|-----|-----|-----|-----|------|
| letta (Sonnet) | 0.655 | 0.542 | 0.318 | 0.594 | 0.572 | 0.603 | **0.547** |
| letta-sleepy (Sonnet) | 0.651 | 0.495 | 0.447 | 0.468 | 0.502 | 0.513 | **0.512** |
| sqlite-chunked-hybrid (GPT-OSS) | 0.424 | 0.482 | 0.381 | 0.535 | 0.520 | 0.386 | **0.454** |

With Sonnet, letta achieves 0.547 mean composite — **the highest score recorded on any adapter-model combination** and the only configuration to break the 0.5 barrier. This is a +0.220 improvement over letta with GPT-OSS-120B (0.327), holding the memory architecture constant.

### 7.2 Interpretation

Upgrading the LLM produces large score improvements: +0.220 for letta, +0.190 for letta-sleepy. This suggests that the LLM's ability to reason over retrieved context is a major factor in benchmark performance.

**Caveat:** This comparison is confounded — it changes the LLM while holding the adapter constant (letta), but does not test the reverse (upgrading the adapter while holding the LLM constant). To establish that "model quality > architecture," one would need to run sqlite-chunked-hybrid with Sonnet — an experiment that was not performed. The evidence supports "LLM quality has large effects" but not the stronger claim that it dominates architecture. Phase 2 and Phase 3 results (where graphrag-light and hierarchical lead under controlled LLM conditions) suggest that architecture matters substantially when the agent LLM is held constant.

---

## 8. Constrained Budget Validation (Historical)

An earlier experiment tested adapters under tighter conditions to validate the benchmark design:

| Parameter | Value |
|-----------|-------|
| Corpus | 30 signal episodes only (no distractors) |
| Total tokens | ~14K per scope |
| Budgets | 4K and 2K |
| Judge LLM | Qwen3-235B |

### 8.1 Results

| Adapter | 4K NBA | 2K NBA |
|---------|--------|--------|
| compaction | 0.711 | 0.735 |
| chunked-hybrid | 0.301 | 0.347 |
| null | 0.071 | 0.067 |

Under these conditions, compaction dominated — it could summarize all 30 signal episodes into a single compact narrative that fit within budget and covered the key facts.

### 8.2 Why This Result Was Invalidated

At full scale (120 episodes with 90 distractors), compaction collapsed from #1 (NBA = 0.73) to #7 of 8 (composite = 0.245). The 90 distractor episodes diluted the signal below the threshold where progressive summarization could preserve it. On S02 (financial fraud), compaction scored 0.000 — its summary compressed 120 episodes into a generic narrative that mentioned "revenue irregularities" but lost the specific account numbers, transaction amounts, and temporal patterns needed to answer questions.

**This validates the distractor design.** The distractors successfully differentiate adapters that can retrieve specific signal episodes (sqlite-chunked-hybrid) from those that attempt to compress everything (compaction). A benchmark without distractors would have ranked compaction first — a misleading result.

---

## 9. Exemplar Behavior

### 9.1 sqlite-chunked-hybrid on S01 (Cascading Failure) — Success

The agent searches for "geo-lookup latency" and "connection pool", retrieves episodes showing p99 climbing from 120ms to 847ms across 10 episodes, and synthesizes the cascading failure chain. The key advantage is **precision retrieval**: the hybrid search returns exact metric values ("p99: 847ms") that ground the answer in concrete evidence. The agent can cite specific episodes and specific numbers, which is exactly what the scoring framework rewards.

### 9.2 compaction on S02 (Financial Fraud) — Failure

The compaction adapter compressed 120 episodes (30 signal about suspicious financial patterns + 90 distractors about unrelated operations) into a single running summary. The financial signals — specific account numbers, transaction amounts, temporal correlations between transfers — were diluted below detectability. The final summary mentioned "revenue irregularities" generically but contained none of the specifics needed to answer questions like "Which accounts show correlated transaction patterns?" The result: composite score of 0.000.

### 9.3 cognee on S01 — Judge Failure

Cognee achieved evidence_grounding = 1.0 and evidence_coverage = 0.632 — the best evidence coverage of any adapter on S01. Its knowledge graph extracted entities and relationships that enabled high-quality context retrieval. However, the LLM judge returned TIE on 100% of pairwise comparisons, making answer_quality default to 0.5. The graph-based retrieval clearly works — the evaluation pipeline simply cannot measure how well it works.

---

## 10. Structural Findings

### 10.1 The Sub-50% Ceiling

No adapter exceeds 50% composite score under controlled conditions (same agent LLM, same judge, same budget) on numeric scopes. Letta with Sonnet breaks this barrier at 0.547, but that comparison confounds the LLM upgrade with the adapter (Section 7.2). On Phase 2 narrative scopes with Qwen3.5-35B-A3B (controlled), the top adapter (graphrag-light) reaches 0.537 — still below 50% but closer. The ceiling appears to be a joint function of LLM reasoning quality and memory architecture, not memory alone.

### 10.2 Retrieval vs. Compression

The benchmark cleanly separates two memory strategies:

- **Retrieval-based** (sqlite-chunked-hybrid, mem0-raw, graphiti): Store episodes and retrieve relevant ones at query time. Scales with corpus size but depends on retrieval quality.
- **Compression-based** (compaction): Summarize episodes into a running narrative. Works at small scale but fails when distractors dilute signal.

Retrieval-based approaches dominate at scale. The critical differentiator is whether the system can find the 30 signal episodes among 90 distractors.

### 10.3 Budget Compliance as a Structural Problem

The 8K token budget is routinely exceeded 4-5x by retrieval-based adapters because a single retrieval call returns more context than the budget allows. This is not an adapter bug — it reflects a fundamental tension between retrieval granularity and budget constraints. The budget was designed to force adapters to be selective, but most adapters have no mechanism for truncating retrieval results to fit within a budget.

### 10.4 Measurement Gaps

Two significant gaps limit confidence in the rankings:

1. **Cognee judge failure** (100% TIE rate): The #2-ranked adapter has never had its answer quality successfully evaluated. Its ranking is driven entirely by mechanical metrics.
2. **Graphiti incompleteness** (3/6 scopes): The #3-ranked adapter's mean is computed over only 3 data points, with a confidence interval spanning nearly the full range ([0.220, 0.491]).

---

## 11. Summary of Rankings

### 11.1 Final Rankings with Caveats

| Rank | Adapter | Mean (8K) | Confidence | Key Caveat |
|------|---------|-----------|------------|------------|
| 1 | sqlite-chunked-hybrid | 0.454 | High | 55% gate-fire rate; true quality higher than composite suggests |
| 2 | cognee | 0.421 | **Low** | 92% gate-fire, 100% judge TIE; ranking driven by mechanical metrics only |
| 3 | graphiti | 0.393 | **Low** | Only 3/6 scopes completed; does not scale to 120-episode corpora |
| 4 | mem0-raw | 0.329 | Moderate | Consistent but unremarkable; simple vector similarity baseline |
| 5 | letta | 0.327 | Moderate | Reaches 0.547 with Sonnet; ranking reflects LLM quality, not architecture |
| 6 | letta-sleepy | 0.322 | Moderate | Sleep-time compute adds modest benefit over base letta |
| 7 | compaction | 0.245 | High | Catastrophic on S02; fundamentally unsuited to distractor-heavy corpora |
| 8 | null | 0.000 | High | Expected floor; validates that the benchmark requires memory |

### 11.2 Dominant Conclusions

1. **sqlite-chunked-hybrid is the most consistent adapter on numeric scopes**, combining hybrid retrieval with stable performance across all six domains. (Note: it ranks 6th on Phase 3 SRS scopes under dynamic evaluation — the advantage is scope-dependent.)
2. **LLM quality has large effects**: Upgrading the internal LLM (Letta with Sonnet) produces +0.220 score improvement, larger than any architectural difference observed. However, this comparison is confounded (see Section 7.2) — the relative importance of model vs. architecture remains an open question.
3. **Compression fails at scale**: Progressive summarization cannot preserve fine-grained signal in the presence of format-matched distractors.
4. **Graph-based approaches show promise but do not scale on numeric scopes**: Cognee's evidence coverage and graphiti's per-scope scores are competitive, but operational failures (timeouts, judge incompatibility) undermine applicability. (Note: graphrag-light, a lighter graph approach not tested on numeric scopes, leads Phase 2 narrative evaluation.)
5. **The benchmark works**: Distractors differentiate systems, the null baseline floors at zero, cross-scope concordance is strong, and no system trivially solves the task.
