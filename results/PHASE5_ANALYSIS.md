# LENS Benchmark Phase 5: Full Analysis

**90 runs | 8 adapters | 6 domains | 2 context budgets | 120 episodes with distractors**

---

## The Headline

**Only one memory system — basic text retrieval — statistically outperforms a naive LLM that simply reads all the evidence in context.** Every other architecture (knowledge graphs, agent memory, running summaries) either ties or *loses* to the naive baseline.

This is the central finding of LENS: the problem isn't retrieval. It's preservation.

---

## 1. Overall Rankings

### Reported (All 90 Runs)

| Rank | System | Composite | 95% CI | NBA | vs Naive |
|------|--------|-----------|--------|-----|----------|
| 1 | **sqlite-chunked-hybrid** | **0.473** | [0.432, 0.518] | 0.541 | **BEATS** |
| 2 | cognee (knowledge graph) | 0.432 | [0.416, 0.449] | 0.500 | Ties |
| 3 | graphiti (knowledge graph) | 0.426 | [0.325, 0.514] | 0.496 | Ties |
| 4 | mem0-raw (vector store) | 0.349 | [0.302, 0.391] | 0.441 | **Loses** |
| 5 | letta (agent memory) | 0.346 | [0.297, 0.394] | 0.400 | **Loses** |
| 6 | letta-sleepy (agent + consolidation) | 0.335 | [0.293, 0.379] | 0.390 | **Loses** |
| 7 | compaction (running summary) | 0.272 | [0.188, 0.346] | 0.391 | **Loses** |
| 8 | null (no memory) | 0.000 | [0.000, 0.000] | 0.231 | — |

### Corrected (Judge-Verified Runs Only)

Post-hoc audit discovered that the LLM judge failed silently on 21/90 runs (23%), affecting cognee (12/12), letta (4/12), letta-sleepy (4/12), and graphiti (1/6). Failed judgments defaulted to TIE (0.5), systematically inflating NBA and answer_quality for affected systems.

| Rank | System | Judged Runs | Corrected Composite | Corrected NBA | vs Naive |
|------|--------|-------------|--------------------|----|----------|
| 1 | **sqlite-chunked-hybrid** | 12/12 | **0.473** | **0.541** | **BEATS** |
| 2 | cognee | **0/12** | **unknown** | **unknown** | **Cannot determine** |
| 3 | graphiti | 5/6 | 0.418 | 0.495 | Ties |
| 4 | mem0-raw | 12/12 | 0.349 | 0.441 | **Loses** |
| 5 | compaction | 12/12 | 0.272 | 0.391 | **Loses** |
| 6 | letta | 8/12 | **0.296** | **0.350** | **Loses** |
| 7 | letta-sleepy | 8/12 | **0.289** | **0.335** | **Loses** |
| 8 | null | 12/12 | 0.000 | 0.231 | — |

**Key corrections:**
- **Cognee cannot be ranked** — zero judge data means its NBA and answer_quality are entirely fabricated (TIE defaults). Its mechanical metrics (evidence_grounding, evidence_coverage) are valid but the composite is unreliable.
- **Letta drops from 0.346 → 0.296 composite** (−14.5%) and **0.400 → 0.350 NBA** when unjudged runs removed
- **Letta-sleepy drops from 0.335 → 0.289** (−13.7%) and **0.390 → 0.335 NBA**
- Rankings for fully-judged systems (chunked-hybrid, mem0, compaction, null) are unchanged

**NBA** (Naive Baseline Advantage) = pairwise win rate against a context-stuffed LLM. 0.5 = tie. Only sqlite-chunked-hybrid's CI is entirely above 0.5.

---

## 2. Statistical Significance

Wilcoxon signed-rank tests, paired by (scope, budget), N=12 per pair:

| Comparison | Mean Δ | p-value | |
|-----------|--------|---------|--|
| chunked-hybrid > letta | +0.127 | 0.001 | *** |
| chunked-hybrid > letta-sleepy | +0.138 | 0.0005 | *** |
| chunked-hybrid > mem0-raw | +0.124 | 0.0005 | *** |
| chunked-hybrid > compaction | +0.201 | 0.001 | *** |
| chunked-hybrid > cognee | +0.041 | 0.204 | ns (but cognee judge-invalid) |
| cognee > letta | +0.086 | 0.012 | * (cognee inflated) |
| cognee > mem0-raw | +0.083 | 0.001 | *** (cognee inflated) |
| cognee > compaction | +0.160 | 0.002 | ** (cognee inflated) |
| letta > letta-sleepy | +0.012 | 0.266 | ns |

**Reliable tiers (fully-judged systems only):**
- **Tier 1:** chunked-hybrid (the only system that beats naive baseline)
- **Tier 2:** mem0-raw (significantly worse than chunked-hybrid, better than compaction)
- **Tier 3:** letta, letta-sleepy (corrected composites ~0.29, significantly worse)
- **Tier 4:** compaction (catastrophic failure on some scopes)

**Note:** Cognee's tier placement cannot be determined without re-scoring with a working judge. Its mechanical metrics (evidence_grounding = 1.0, evidence_coverage) are reliable and suggest it belongs in Tier 1-2, but confirmation requires re-running the judge.

---

## 3. The Lossy Abstraction Trap

Tracing through actual agent transcripts reveals the mechanism behind these rankings. Every complex architecture introduces a **lossy transformation** between raw evidence and what the agent sees at question time:

| System | Transformation | What's Lost | Evidence |
|--------|---------------|------------|----------|
| chunked-hybrid | **None** (raw text indexed) | Nothing | Retrieves ep_025: "Cr = 140 µg/L" verbatim |
| cognee | Entity extraction → graph | Temporal ordering, intermediate values | Graph has "WQ-03: Cr=60" but not the 14→30→42→55→60 progression |
| mem0-raw | Vector embedding | Early/subtle signals | Similarity search finds dramatic ep_025 (Cr=132) but misses ep_002 (Cr=4, first exceedance) |
| letta | Agent-managed memory | Search reliability | Q1 finds 5 episodes; Q2 (paraphrase of Q1) finds 0 in 9 attempts |
| letta-sleepy | Agent memory + consolidation | Same as letta + compression noise | Consolidation produces task-agnostic summaries that don't improve retrieval |
| compaction | Rolling LLM summary | Numeric precision at scale | "Revenue showed irregular patterns" replaces specific quarterly figures |

**The winning system preserves everything. Every other system throws something away before the agent can use it.**

### Concrete Side-by-Side: Q3 "What is the source of chromium contamination?"

All systems answering the same longitudinal question at checkpoint 99-120 (scope 04, 16k budget):

| System | Refs Retrieved | Answer Length | Correct? | Key Detail |
|--------|---------------|---------------|----------|------------|
| **chunked-hybrid** | 3 (ep_025, ep_018, ep_023) | 1,642 chars | **Yes** — identifies unpermitted pipe at RM 18.6 | Cites specific Cr gradient: 3→5→52 µg/L |
| **letta** | 5 (ep_018-024) | 2,567 chars | Partial — identifies gradient, misses pipe | More verbose, less precise |
| **letta-sleepy** | **0** | **0 chars** | **Failed** — empty answer | 9 search attempts, all 0 results |
| **compaction** | 3 (ep_001-003) | 2,607 chars | Partial — identifies gradient from Day 1 only | Cites only earliest episodes |
| **mem0-raw** | 2 (ep_018, ep_026) | 2,293 chars | **Wrong** — misidentifies station order | Claims WQ-02 downstream of source |
| **cognee** | **0** | **0 chars** | **Failed** — empty answer | 9 searches, 157K tokens burned |

**The pattern:** Chunked-hybrid retrieves the 3 right episodes on the first try. Every other system either fails to find the critical episode (ep_025 with the "discharge pipe" field note), retrieves wrong episodes, or fails search entirely.

### Empty Answer Rates (scope 04, all checkpoints)

| System | Empty/Total | Rate |
|--------|------------|------|
| chunked-hybrid | 4/24 | **17%** |
| compaction | 10/24 | 42% |
| mem0-raw | 8/24 | 33% |
| cognee | 9/24 | 38% |
| letta | 12/24 | **50%** |
| letta-sleepy | 10/24 | 42% |

Chunked-hybrid's 17% empty rate is structural (distractor_resistance and severity_assessment questions that all systems struggle with). Letta's 50% empty rate reflects systematic search thrashing.

---

## 4. The Compaction Collapse

The compaction adapter demonstrates a **catastrophic failure mode** at scale:

| Scope | 8k | 16k | Note |
|-------|-----|------|------|
| s01 | 0.339 | 0.366 | OK |
| s02 | **0.000** | **0.000** | **Complete collapse** |
| s03 | 0.198 | 0.227 | Degraded |
| s04 | 0.364 | 0.440 | OK |
| s05 | 0.309 | 0.445 | Marginal |
| s06 | 0.259 | 0.322 | Degraded |

120 episodes × ~500 tokens = ~60,000 tokens compressed into a ~4,000-token rolling summary (**15:1 ratio**). On scope 02 (financial irregularities), where signal is encoded in specific quarterly figures, the summary discards all numeric precision. **This is a death sentence for summarize-then-retrieve architectures** — they work on toy benchmarks but collapse on realistic episode counts.

---

## 5. The Agent Memory Paradox

Letta (agent-managed memory) and letta-sleepy (with consolidation) reveal a deep tension:

**The promise:** The agent "understands" and "organizes" information, building progressively richer memory structures.

**The reality:** Understanding requires lossy compression. Longitudinal synthesis requires lossless preservation. These are in direct tension.

### Concrete Evidence

**Letta Q3** (scope 04): "What is the source of chromium contamination?"
- Search 1: finds 5 relevant episodes → produces 2,567-char answer with spatial gradient ✓

**Letta-sleepy Q3** (scope 04, same question):
- Search 1-9: finds 0 episodes → 88,547 tokens burned → empty answer ✗

**The same system architecture, with consolidation added, performs WORSE on the same question.** The consolidated memory store is larger and noisier, making search less reliable.

### Consolidation Doesn't Help

Letta-sleepy's "sleep" consolidation runs during ingestion but produces memories that the question-answering agent doesn't preferentially retrieve. At question time, all tool calls are identical to base letta: generic `memory_search` queries. The consolidated summaries compete with raw memories in the same search space.

**The key insight**: consolidation "doesn't know what to optimize for — it's 'make it better, clean it up.'" Without a task-specific loss function, consolidation optimizes for generic coherence rather than evidence preservation. It's compression without an objective.

Result: letta vs. letta-sleepy is statistically indistinguishable (p = 0.266). Corrected composites: letta = 0.296, letta-sleepy = 0.289.

---

## 6. Cognee: Data Quality Caveat

Cognee (knowledge graph) showed the **lowest variance** across scopes (σ = 0.026) and appeared to be Tier 1 in the initial analysis. However, the judge reliability audit revealed that **all 12 cognee runs have empty judge caches** — the LLM judge never executed for any cognee run.

This means:
- Cognee's NBA of 0.500 is entirely fabricated (TIE defaults)
- Cognee's answer_quality of 0.500 is entirely fabricated
- Cognee's composite of 0.432 is inflated by these default values

**What IS reliable for cognee:**
- evidence_grounding = 1.0 (all 12 runs) — cognee always retrieves evidence
- evidence_coverage varies by scope (0.174–0.413)
- Budget compliance = 0.0 across all runs (same issue as letta/sleepy)
- Low variance (σ = 0.026) is real — the graph provides structural consistency

**Interpretation:** Cognee's structural metrics suggest it IS a consistent system that always retrieves something relevant. But without judge data, we cannot determine whether its answers are actually better or worse than the naive baseline. Re-scoring with a working judge is required before making any claims about cognee's ranking.

---

## 7. Budget Effect: More Context Uniformly Helps

Every adapter improves with 16k vs 8k context budget:

| Adapter | 8k | 16k | Δ | % |
|---------|-----|------|---|---|
| compaction | 0.245 | 0.300 | +0.055 | +22.5% |
| graphiti | 0.393 | 0.459 | +0.067 | +17.0% |
| letta | 0.327 | 0.366 | +0.039 | +12.1% |
| mem0-raw | 0.330 | 0.368 | +0.039 | +11.8% |
| chunked-hybrid | 0.454 | 0.492 | +0.037 | +8.2% |
| letta-sleepy | 0.322 | 0.348 | +0.026 | +7.9% |
| cognee | 0.421 | 0.444 | +0.023 | +5.4% |

**The systems that benefit least from more context are the ones with the best pre-filtering** (cognee's graph, chunked-hybrid's BM25+embedding). Systems that need more raw text (compaction, graphiti) benefit most.

---

## 8. Cross-Scope Consistency

| Adapter | s01 | s02 | s03 | s04 | s05 | s06 | σ |
|---------|-----|-----|-----|-----|-----|-----|---|
| chunked-hybrid | 0.445 | 0.530 | 0.389 | 0.582 | 0.470 | 0.422 | 0.072 |
| cognee† | 0.463 | 0.424 | 0.397 | 0.459 | 0.414 | 0.436 | 0.026 |
| letta | 0.320 | 0.365 | 0.253 | 0.460 | 0.436 | 0.245 | 0.091 |
| mem0-raw | 0.373 | 0.361 | 0.203 | 0.438 | 0.389 | 0.330 | 0.080 |
| compaction | 0.352 | 0.000 | 0.212 | 0.402 | 0.377 | 0.291 | 0.150 |

†cognee composites include inflated answer_quality/NBA (judge never ran)

**Scope 03 (clinical signals) and scope 06 (supply chain) are hardest** — all systems drop. Scope 04 (environmental drift) is easiest. Signal density and narrative structure affect difficulty more than domain.

---

## 9. Question Type Analysis

NBA win rates by question type (fully-judged systems only):

| Question Type | chunked-hybrid | mem0-raw | compaction | null |
|--------------|----------------|----------|------------|------|
| longitudinal | 0.711 | 0.605 | 0.541 | 0.226 |
| evidence_sufficiency | 0.702 | 0.459 | 0.378 | 0.147 |
| severity_assessment | 0.689 | 0.311 | 0.540 | 0.111 |
| action_recommendation | 0.679 | 0.577 | 0.310 | 0.158 |
| paraphrase | 0.653 | 0.467 | 0.429 | 0.190 |
| counterfactual | 0.477 | 0.362 | 0.361 | 0.060 |
| temporal | 0.435 | 0.354 | 0.244 | 0.138 |
| distractor_resistance | 0.430 | 0.340 | 0.239 | 0.169 |
| negative | 0.427 | 0.385 | 0.374 | 0.278 |

**Memory helps most for evidence-heavy questions** (severity, evidence sufficiency, action recommendation) and **least for temporal and distractor resistance**. Retrieving relevant chunks helps aggregate evidence, but temporal ordering and distractor filtering require *reasoning* that retrieval alone doesn't provide.

---

## 10. Judge Reliability

### The Problem

21 of 90 scored runs (23%) have empty judge caches, meaning the LLM judge never executed. All pairwise comparisons for these runs default to TIE (0.5), which:
- Inflates NBA for systems that would score below 0.5 (letta, letta-sleepy)
- Creates phantom data for cognee (100% of runs affected)
- Reduces statistical power for significance tests

### Affected Systems

| System | Runs Judged | Runs Unjudged | Impact |
|--------|------------|---------------|--------|
| cognee | 0/12 | 12/12 | **Complete** — no valid judge data |
| letta | 8/12 | 4/12 | NBA inflated 0.400 → 0.350 (corrected) |
| letta-sleepy | 8/12 | 4/12 | NBA inflated 0.390 → 0.335 (corrected) |
| graphiti | 5/6 | 1/6 | Minor impact |

### Root Cause

The judge failures correlate with runs scored during specific time windows when the Cerebras API was experiencing availability issues. The judge function silently caught connection errors and returned TIE defaults rather than failing loudly — a design decision intended to prevent scoring crashes, but which masks data quality issues.

### Recommendation

Re-score cognee, letta s04-s05, and letta-sleepy s04-s05 with a working judge (Together AI Qwen3-235B or local). Until then, treat cognee rankings as provisional and use corrected letta/sleepy numbers.

---

## 11. Implications for Memory System Design

1. **Store raw evidence; synthesize on demand.** The memory system should be a database, not an intelligence. Do understanding at question time, when the objective is clear.

2. **Index along multiple dimensions.** BM25 catches keyword matches that embeddings miss (early/subtle episodes). Embeddings catch semantic similarity that BM25 misses. Neither alone is sufficient.

3. **If you preprocess, make it additive, not substitutive.** Add extracted entities as supplementary indices alongside raw text. Don't replace raw text with extracted representations.

4. **Consolidation needs a task-specific loss function.** Generic "make it better" is not an optimization objective. Define what evidence must survive and verify that it does.

5. **Build graceful degradation.** When search fails, answer from available context rather than returning empty. Letta's worst failure mode is burning 80K tokens to produce nothing.

6. **More context > smarter retrieval.** The uniform 8k→16k improvement suggests expanding the context window is more cost-effective than sophisticated memory architectures.

---

## Experimental Setup

- **LLM:** Cerebras GPT-OSS-120B (agent), Cerebras GPT-OSS-120B (judge)
- **Embeddings:** Alibaba-NLP/gte-modernbert-base (768 dims, Together AI)
- **Episodes per scope:** 120 (100 signal + 20 topically-orthogonal distractors)
- **Scopes:** 6 diverse domains (cascading failure, financial irregularities, clinical signals, environmental drift, insider threat, supply chain)
- **Context budgets:** 8,192 and 16,384 cumulative result tokens
- **Questions:** 20 per scope across 9 question types
- **Scoring:** 3-tier (mechanical + LLM judge + differential), pairwise NBA against context-stuffed baseline
- **Judge reliability:** 69/90 runs (77%) fully judged; 21/90 (23%) have judge defaults (see Section 10)

**Companion documents:**
- [FAILURE_THEORY.md](FAILURE_THEORY.md) — qualitative analysis of agent transcripts explaining *why* complex systems underperform
- [ANALYSIS.md](ANALYSIS.md) — earlier version without judge reliability correction
