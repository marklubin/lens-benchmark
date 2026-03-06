# Phase 3: Semantic Retrieval Stress (S10-S12) — Comparison Report

## Overview

Phase 3 is the most adversarial dataset design in the LENS benchmark. Unlike numeric scopes (S01-S06) where signal and distractor episodes occupy distinct topical domains, or narrative scopes (S07-S09) where document formats vary, SRS episodes are **structurally and semantically identical** between signal and distractor tracks. An embedding search for "clinical trial hepatotoxicity" returns both signal episodes about Patient X's deteriorating liver enzymes AND distractor episodes about Patient Y's routine labs — because both are clinical trial monitoring data using the same vocabulary, document structure, and terminology.

All results use **Modal Qwen3.5-35B-A3B** as the universal LENS agent LLM. Composites are **ungated**. Static driver results are presented separately as a retrieval quality sanity check, not for direct comparison with dynamic driver results.

Total: **77 Modal-era scored runs** across 12 adapters and 3 scopes.

---

## Dataset Design

### Scope Descriptions

| Scope | Domain | Signal Pattern | Adversarial Element |
|-------|--------|----------------|---------------------|
| S10 — Clinical Deterioration | Phase IIb clinical trial records | Delayed hepatotoxicity; ALT/AST elevations appear 4-6 weeks post-dose with dose-dependent pattern | Distractor episodes are clinical trial data from other arms/timepoints with normal labs — same format, same acronyms, same tables |
| S11 — Zoning Variance | Municipal government records | Developer influence through intermediary LLCs; campaign contributions, favorable votes contradicting staff recommendations | Distractor episodes are routine zoning proceedings — same minutes format, same property records, same filings |
| S12 — Therapy Progression | AI wellness companion chat logs | Gradual psychological trajectory from avoidance to potential clinical depression; sleep deterioration, social withdrawal | Distractor episodes are other users' therapy sessions — same conversational format, same wellness topics |

### Corpus Structure

- **40 episodes per scope** (20 signal + 20 distractor)
- **~5,000 words per episode**
- **~200K tokens total** per scope
- **1:1 signal-to-distractor ratio** — semantic similarity already provides strong adversarial pressure

### Why SRS Exists

Standard memory benchmarks use topically distinct corpora where semantic search trivially separates relevant from irrelevant content. SRS eliminates this convenience: signal and distractor episodes use the same domain vocabulary, follow the same document format, reference similar entity types, and contain comparable data density. The only difference is which specific entities, values, and progressions constitute the benchmark signal.

This mirrors real operational environments. A clinical data monitor reviews hundreds of trial reports — every report is a clinical trial, and the question is which data points form a safety signal. A compliance officer reviews government records — every document is a zoning filing, and the question is which filings reveal a pattern of influence.

---

## Results: Dynamic Driver Rankings

| Rank | Adapter | N | Composite | AnsQ | NBA | EvGnd | EvCov | BudC | InsDp | ReasQ |
|------|---------|---|-----------|------|-----|-------|-------|------|-------|-------|
| 1 | **hierarchical** | 6 | **0.554** | 0.345 | 0.656 | 1.000 | 0.346 | 0.000 | 0.967 | 1.000 |
| 2 | hopping | 6 | 0.548 | 0.379 | 0.627 | 1.000 | 0.320 | 0.000 | 0.933 | 0.983 |
| 3 | graphrag-light | 6 | 0.544 | 0.397 | 0.657 | 1.000 | 0.269 | 0.000 | 0.900 | 1.000 |
| 4 | hopping-hybrid | 3 | 0.544 | 0.421 | 0.667 | 1.000 | 0.228 | 0.000 | 0.867 | 1.000 |
| 5 | letta | 6 | 0.541 | 0.345 | 0.628 | 1.000 | 0.304 | 0.000 | 0.950 | 1.000 |
| 6 | hierarchical-hybrid | 3 | 0.531 | 0.406 | 0.645 | 1.000 | 0.246 | 0.000 | 0.800 | 1.000 |
| 7 | sqlite-chunked-hybrid | 6 | 0.527 | 0.435 | 0.629 | 0.993 | 0.227 | 0.000 | 0.800 | 0.983 |
| 8 | letta-sleepy | 6 | 0.520 | 0.322 | 0.621 | 1.000 | 0.282 | 0.000 | 0.900 | 0.967 |
| 9 | letta-entity | 9 | 0.376 | 0.378 | 0.151 | 0.505 | 0.110 | 1.000 | 0.700 | 0.000 |
| 10 | triadv1-pairs | 6 | 0.328 | 0.333 | 0.581 | 0.000 | 0.000 | 0.667 | 0.000 | 1.000 |
| 11 | letta-v4 | 14 | 0.269 | 0.286 | 0.367 | 0.269 | 0.011 | 1.000 | 0.157 | 0.000 |
| 12 | null | 6 | 0.248 | 0.329 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

### Key Observations

1. **The top 8 adapters are tightly clustered** — only 0.034 points separate #1 (hierarchical, 0.554) from #7 (sqlite-chunked-hybrid, 0.527). SRS compresses the performance spread when all adapters use the same agent LLM.

2. **Summarization-based adapters lead.** hierarchical (#1), hopping (#2), and hopping-hybrid (#4) all use multi-level or rolling summarization. On SRS, where raw retrieval cannot distinguish signal from distractor, summarization acts as implicit signal filtering — the LLM compression step prioritizes notable patterns over routine data.

3. **Evidence grounding is near-perfect** for the top 8 adapters (0.993-1.000). On SRS, if an adapter retrieves episodes at all, they are correctly cited. The challenge is retrieval discrimination, not citation formatting.

4. **Budget compliance is universally zero** for retrieval-based adapters. Same structural issue as narrative scopes — 5,000-word episodes exceed the 8K token budget.

5. **letta-v4 collapses** (0.269, rank 11). Its internal Q&A agent, which bypasses the LENS agent loop, produces low evidence grounding (0.269) and zero reasoning quality. The Qwen3.5-35B-A3B model running through Letta's agent framework does not reliably cite episode IDs or expose chain-of-thought reasoning.

6. **graphrag-light is no longer dominant** (#3, 0.544 vs #1 hierarchical at 0.554). On narrative scopes, graphrag-light led by 0.10+ points. On SRS, its entity graph advantage is diluted because signal and distractor episodes share entity types — the graph links both signal and distractor entities equally.

---

## Per-Scope Breakdown

### S10 — Clinical Deterioration (Hardest)

Clinical trial data with maximum adversarial pressure — identical terminology, lab panels, and document structure between signal and distractor tracks.

| Rank | Adapter | N | Composite |
|------|---------|---|-----------|
| 1 | hierarchical | 2 | 0.528 |
| 2 | letta | 2 | 0.519 |
| 3 | letta-sleepy | 2 | 0.519 |
| 4 | hopping | 2 | 0.516 |
| 5 | hopping-hybrid | 1 | 0.503 |
| 6 | graphrag-light | 2 | 0.501 |
| 7 | sqlite-chunked-hybrid | 2 | 0.476 |
| 8 | hierarchical-hybrid | 1 | 0.466 |
| 9 | letta-entity | 2 | 0.407 |
| 10 | triadv1-pairs | 2 | 0.343 |
| 11 | letta-v4 | 5 | 0.262 |
| 12 | null | 2 | 0.249 |

S10 is the hardest SRS scope. Mean composite across all adapters: 0.441. The clinical data uniformity creates maximum adversarial pressure — lab panels in signal and distractor episodes use identical headers, units, and reference ranges.

### S11 — Zoning Variance (Middle)

Government records with distinctive proper nouns (LLC names, parcel IDs) that provide some keyword anchors.

| Rank | Adapter | N | Composite |
|------|---------|---|-----------|
| 1 | hopping-hybrid | 1 | 0.575 |
| 2 | hopping | 2 | 0.561 |
| 3 | graphrag-light | 2 | 0.555 |
| 4 | hierarchical | 2 | 0.553 |
| 5 | sqlite-chunked-hybrid | 2 | 0.549 |
| 6 | letta-sleepy | 2 | 0.542 |
| 7 | letta | 2 | 0.531 |
| 8 | hierarchical-hybrid | 1 | 0.526 |
| 9 | letta-entity | 5 | 0.329 |
| 10 | triadv1-pairs | 2 | 0.325 |
| 11 | letta-v4 | 3 | 0.297 |
| 12 | null | 2 | 0.244 |

S11 produces the tightest clustering among the top 8 — only 0.049 points from #1 to #8. Distinctive proper nouns in zoning records (LLC names, board member names, parcel IDs) give all retrieval approaches reasonable keyword anchors.

### S12 — Therapy Progression (Easiest)

Conversational chat logs with emotional markers — some natural language hooks for retrieval.

| Rank | Adapter | N | Composite |
|------|---------|---|-----------|
| 1 | hierarchical-hybrid | 1 | 0.601 |
| 2 | hierarchical | 2 | 0.582 |
| 3 | graphrag-light | 2 | 0.578 |
| 4 | letta | 2 | 0.574 |
| 5 | hopping | 2 | 0.567 |
| 6 | sqlite-chunked-hybrid | 2 | 0.557 |
| 7 | hopping-hybrid | 1 | 0.554 |
| 8 | letta-sleepy | 2 | 0.500 |
| 9 | letta-entity | 2 | 0.462 |
| 10 | triadv1-pairs | 2 | 0.317 |
| 11 | letta-v4 | 6 | 0.260 |
| 12 | null | 2 | 0.251 |

S12 is the easiest SRS scope. The therapy chat format provides emotional markers and narrative hooks (sleep deterioration mentions, relationship updates) that help both summarization and retrieval approaches. hierarchical-hybrid leads — its combination of multi-level summarization and raw retrieval handles the conversational format well.

---

## Architecture Analysis

### Why Summarization Leads on SRS

The top 4 includes three summarization-based adapters (hierarchical, hopping, hopping-hybrid). On SRS, where raw retrieval returns both signal and distractor episodes equally, summarization provides a critical advantage: **implicit signal filtering**.

When the LLM compresses 40 episodes into a multi-level summary, it makes relevance judgments — emphasizing notable patterns (dose-dependent toxicity, suspicious voting patterns, psychological deterioration) and de-emphasizing routine data (normal lab panels, standard zoning approvals, stable mood reports). The summary has already done the discrimination work that keyword and embedding search cannot.

This explains why summarization adapters improve relative to their Phase 2 (narrative) rankings. On narrative scopes, graphrag-light's entity graph provided structural signals that pure text search lacked. On SRS, the entity graph links both signal and distractor entities equally (both are clinical trial data with patients, labs, and dosing), so the graph advantage is reduced. Summarization's implicit filtering becomes relatively more valuable.

### sqlite-chunked-hybrid: Consistent but No Longer Dominant

sqlite-chunked-hybrid ranks 7th (0.527) — competitive but mid-pack. Its BM25 + embedding hybrid provides reasonable retrieval on all three scopes, but lacks the signal filtering that summarization provides or the structural linking that graphrag-light provides. On S11 (zoning), where distinctive proper nouns give BM25 keyword anchors, it rises to 5th. On S10 (clinical), where keywords can't discriminate, it drops to 7th.

The adapter's defining strength — preserving raw text verbatim with no lossy transformation — is less valuable on SRS because the raw text of signal and distractor episodes is nearly interchangeable.

### graphrag-light: Strong but Not Dominant

graphrag-light ranks 3rd overall (0.544) — strong but not the clear leader it was on narrative scopes. The entity graph still provides value (graph traversal can connect specific patients to specific lab trajectories to specific dosing arms), but the advantage is smaller because signal and distractor episodes share entity types. A graph node for "ALT elevation" connects to both signal and distractor episodes.

### The letta-v4 / letta-entity Problem

Both delegated adapters (letta-v4 rank 11, letta-entity rank 9) significantly underperform compared to the standard letta adapter (rank 5). The issue is their internal Q&A agents:

- **letta-v4**: Evidence grounding = 0.269 (vs 1.000 for standard letta), reasoning quality = 0.000. The internal agent doesn't reliably cite episode IDs.
- **letta-entity**: NBA = 0.151 (vs 0.628 for standard letta). Entity-focused memory organization loses longitudinal signal.

The standard letta adapter, which uses the LENS agent loop for Q&A (not its own internal agent), performs at the same level as other top-8 adapters.

---

## SRS Difficulty Gradient

| Scope | Mean Composite (top-8) | Character |
|-------|----------------------|-----------|
| S10 — Clinical Deterioration | 0.506 | Hardest — maximum terminology overlap |
| S11 — Zoning Variance | 0.549 | Middle — some keyword anchors (proper nouns) |
| S12 — Therapy Progression | 0.564 | Easiest — emotional/narrative hooks |

The difficulty gradient follows keyword discriminability: S10 (clinical) has the most uniform vocabulary between signal and distractor; S12 (therapy) has the most natural language variation that retrieval can exploit.

---

## Cross-Phase Comparison

| Metric | Phase 2 (S07-S09) | Phase 3 (S10-S12) |
|--------|-------------------|-------------------|
| Top adapter | graphrag-light (0.537) | hierarchical (0.554) |
| Top-8 spread | 0.139 | 0.034 |
| Mean composite (top-8) | 0.405 | 0.539 |
| Best evidence grounding | 0.996 | 1.000 |
| Budget compliance (typical) | 0.000-0.322 | 0.000 |
| Dominant architecture | Graph (entity linking) | Summarization (implicit filtering) |

**Key differences:**

1. **SRS compresses the spread.** On narrative scopes, graphrag-light led by 0.10+ points. On SRS, the top 8 are within 0.034. When signal and noise are semantically identical, no single architecture has a decisive advantage.

2. **Different architectures lead.** Graph-based retrieval dominates narrative scopes (where entity relationships span document types). Summarization dominates SRS (where implicit filtering separates signal from noise during compression).

3. **Higher absolute scores on SRS.** Mean top-8 composite is 0.539 (SRS) vs 0.405 (narrative). This reflects the smaller adversarial challenge from fewer, more focused episodes — not easier synthesis. SRS episodes are adversarial in a different way (semantic overlap) vs narrative (vocabulary mismatch, episode size).

---

## Key Findings

### 1. The Graph Advantage Is Proportional to Topical Distinctiveness

graphrag-light's entity graph dominated narrative scopes (#1, 0.537, +0.10 lead) because signal and distractor entities occupied different topical domains — the graph naturally separated them. On SRS, where signal and distractor episodes share entity types (both are clinical trials with patients and lab panels, both are zoning filings with LLCs and votes), the graph links both tracks equally. graphrag-light drops to #3 (0.544) with no meaningful lead. The structural signal that makes graphs powerful — entity relationships — only discriminates when the entities themselves are distinguishable. This is a fundamental limitation: graph-based memory encodes *what is connected to what*, but cannot judge *which connections matter* without additional context.

### 2. Summarization Acts as an Implicit Relevance Judgment

Summarization adapters (hierarchical #1, hopping #2, hopping-hybrid #4) lead SRS because LLM compression inherently makes relevance judgments. When the model compresses 40 episodes into a summary, it emphasizes notable patterns (dose-dependent toxicity, suspicious voting patterns) and de-emphasizes routine data (normal lab panels, standard approvals). This is a different *kind* of memory operation than retrieval — it is closer to consolidation in biological memory, where the act of compression separates signal from noise. The implication is that effective agent memory may need a consolidation stage that operates on different principles than the retrieval stage.

### 3. No Single Architecture Wins Across Content Types

The cross-phase comparison is definitive: graph-based memory leads on narrative, summarization leads on SRS, and raw hybrid retrieval leads on numeric (Phase 1). Each architecture excels on the content type that matches its structural assumptions. Real-world agent workloads contain all three content types simultaneously — structured metrics, narrative documents, and semantically overlapping records. A memory system built around any single mechanism will fail on at least one class of content. This argues against the "pick the best architecture" framing and toward systems that integrate multiple memory mechanisms — but the letta-v4/entity results (finding 4) show that naive multi-store approaches introduce their own failures.

### 4. Multi-Agent Memory Coordination Remains Unsolved

letta-v4 (rank 11) and letta-entity (rank 9) attempt to solve the multi-mechanism problem by delegating memory management to specialized internal agents. On SRS, both significantly underperform standard letta (rank 5), which uses the same storage but routes Q&A through the LENS agent loop instead of an internal agent. The failure is not in storage or retrieval — it is in the coordination layer. Internal agents introduce citation format errors, non-deterministic consolidation, and reasoning quality breakdowns. The architectural complexity intended to handle diverse memory needs creates failure modes that simpler systems avoid. A structural solution to multi-mechanism memory — one that doesn't require independent agents to coordinate through message-passing — appears necessary.

---

## Experimental Controls

| Parameter | Value |
|-----------|-------|
| LENS agent LLM | Qwen3.5-35B-A3B (Modal vLLM) |
| Letta internal agents | Qwen3.5-35B-A3B (via openai-proxy → Modal) |
| Embedding model | GTE-ModernBERT-base (Modal) |
| graphrag-light entity extraction LLM | Qwen3.5-35B-A3B (Modal) |
| hierarchical/hopping summarization LLM | Qwen3.5-35B-A3B (Modal) |
| Driver | Dynamic (agent formulates own queries) |
| Budget preset | constrained-8k |
| Scoring | Ungated weighted composite |
| Runs excluded | RunPod (Llama 3.3 70B), static driver |
| Total runs | 77 |
| Data source | salinas machine (all S10-S12 runs) + obispo (S07-S09 overlap, deduplicated) |
