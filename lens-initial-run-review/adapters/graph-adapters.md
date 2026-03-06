# Graph-Based Adapters: cognee, graphiti, graphrag-light

All three adapters extract entities and relationships into graph structures. They differ in infrastructure requirements, graph backend, and how they combine graph signals with text search. Graph approaches show promise — especially graphrag-light on semantic retrieval stress — but are limited by entity extraction scalability and infrastructure complexity.

---

## Scope Categories

| Phase | Scopes | Episodes | Signal / Distractor | Words/Episode |
|-------|--------|----------|---------------------|---------------|
| **Phase 1: Numeric** | S01--S06 | 120 | 30 + 90 | ~700 |
| **Phase 2: Narrative** | S07--S09 | 40 | 20 + 20 | ~5,000 |
| **Phase 3: Semantic Retrieval Stress (SRS)** | S10--S12 | 40 | 20 + 20 | ~5,000 |

Phase 3 (SRS) scopes are structurally identical to Phase 2 but designed to defeat embedding similarity, forcing retrieval systems to rely on deeper semantic understanding.

---

## Architectures

### cognee

Embedded GraphRAG stack: LanceDB (vectors) + Kuzu (graph DB) + SQLite.

- **Ingest**: `cognee.add(episode_text)` buffers raw text; no processing at ingest time.
- **Prepare**: `cognee.cognify()` runs LLM entity extraction, graph construction, and summarization. Takes 30--60 minutes on remote LLM APIs.
- **Search**: LanceDB vector search over graph-aware chunks.
- **Infrastructure**: No container required — all embedded databases. Requires 6 monkey-patches and ACL disabling.
- **Entity extraction**: Together AI LLM.

### graphiti (Zep)

FalkorDB-backed temporal knowledge graph with bi-temporal edge invalidation.

- **Ingest**: `add_episode()` triggers LLM entity extraction at ingest time (5+ LLM calls per episode).
- **Prepare**: Embed entity descriptions for semantic search.
- **Search**: `EDGE_HYBRID_SEARCH` — graph edges + episode mentions mapped back to source episodes.
- **Infrastructure**: Requires FalkorDB container + Neo4j-compatible driver.
- **Entity extraction**: Runs at ingest; failed on remote LLM APIs with 120-episode corpora due to timeouts.

### graphrag-light

Lightweight graph over SQLite FTS5 with multi-signal fusion.

- **Ingest**: Store text in SQLite FTS5.
- **Prepare**: LLM extracts entities and relationships; embeds entity descriptions.
- **Search**: 3-signal RRF fusion (BM25 text match + entity embedding similarity + one-hop graph neighbors).
- **Infrastructure**: SQLite only — no external databases.
- **Entity extraction**: LLM-based with dedup via normalization. Had a KeyError bug where returned display-form names did not match the normalized lowercase keys in the graph (fixed).

---

## Phase 1: Numeric Scopes (S01--S06)

### cognee

**Phase 5 evaluation** (GPT-OSS-120B agent, 12/12 runs scored):

| Metric | Value |
|--------|-------|
| Overall Mean | **0.432** [0.397, 0.446] |
| Rank | **#2 of 8 adapters** |
| 8K Mean | 0.421 |
| 16K Mean | 0.444 |
| Budget compliance | 1.0 |
| Evidence grounding | 1.0 |
| TIE rate from judge | **100%** |

Per-scope at 8K budget: S01=0.438, S02=0.402, S03=0.374, S04=0.471, S05=0.418, S06=0.423.

The 100% TIE rate means the judge returned "tie" for every answer quality comparison. The composite score is driven entirely by mechanical metrics (evidence_grounding, evidence_coverage) — answer quality is unknown. This ranking cannot be trusted without re-scoring.

**Constrained validation** (S01 only, Qwen3-235B judge):

- 2K NBA: 0.855 (anomalous — budget_compliance=0.167; only 2/12 questions answered within budget)
- 4K NBA: 0.477; AnsQ: 0.402; EvCov: 0.260

The 2K result is a methodological artifact and should be disregarded.

**7-Adapter sweep** (S01 only, Qwen3-32B): Scored 0.000 (hard-gated, evidence_grounding=0). Entity extraction was too slow with the 32B model, producing incomplete indices.

### graphiti

**Phase 5 evaluation** (GPT-OSS-120B agent, 6/12 runs completed):

| Metric | Value |
|--------|-------|
| Overall Mean | **0.426** [0.220, 0.491] |
| Rank | **#3 of 8 adapters** |
| 8K Mean | 0.393 |
| 16K Mean | 0.459 (+0.067, largest budget effect of any adapter) |
| Budget compliance | 1.0 |
| Evidence grounding | 1.0 |
| Reasoning quality | 0.917 |
| Insight depth | 0.833 |

Per-scope at 8K budget: S01=0.491, S02=0.220, S06=0.467.

Three scopes failed entirely (S03, S04, S05) due to entity extraction timeouts. The wide confidence interval reflects this missing data. Graph-based entity extraction does not scale to 120-episode corpora with remote LLM APIs.

**Constrained validation** (S01, Qwen3-235B judge):

- 2K NBA: 0.270; 4K NBA: 0.517; 4K AnsQ: 0.559; Budget compliance: 0.708
- Sharp budget degradation: 0.517 at 4K drops to 0.270 at 2K. Graph traversal requires minimum budget to produce useful context.

**7-Adapter sweep** (Qwen3-32B): Only 2/6 scopes completed (S01=0.000 hard-gated, S02=0.321). Timed out on S03--S06.

### graphrag-light

Not evaluated on numeric scopes. Added to the benchmark later for SRS evaluation.

---

## Phase 2: Narrative Scopes (S07--S09)

### cognee

**3-rep run** (Llama era):

| Metric | Value |
|--------|-------|
| Mean | **0.327** +/- 0.029 |
| S07 | 0.298 |
| S08 | 0.364 |
| S09 | 0.319 |
| Ingest time | 0s (deferred to prepare) |
| Question time | 530.7s |
| Mean tokens | 222,312 |
| Tool calls | 20 |

Zero ingest overhead: all processing happens during `cognify()` in the prepare phase.

### graphiti

**Failed on narrative scopes.** Context window overflow at checkpoint 16. Accumulated entity/relationship data (~106K tokens) plus generation budget exceeded the 113K context window. This is a fundamental architectural limitation: the graph grows superlinearly with episode count.

### graphrag-light

Not evaluated on narrative scopes.

---

## Phase 3: Semantic Retrieval Stress (S10--S12)

### cognee

Not evaluated on SRS scopes.

### graphiti

Not evaluated on SRS scopes due to the scaling failure observed in earlier phases.

### graphrag-light

**Agent LLM:** Qwen3.5-35B-A3B (Modal vLLM) for all runs below.

**Dynamic driver evaluation:**

| Scope | Score | Run ID |
|-------|-------|--------|
| S10 | 0.496 | `b46ae86050fe` |
| S11 | 0.585 | `abc3476a4797` |
| S12 | 0.583 | `6ca518eb7963` |
| **Mean** | **0.555** | **see note** |

*Note: Phase 3 report computes a slightly different aggregate (0.544, N=6) due to per-run weighting across all dynamic-driver runs. The per-scope means above are from the 3 specific run IDs listed.*

**Static driver evaluation:**

| Scope | Score | Run ID |
|-------|-------|--------|
| S10 | 0.514 | `11dc9cc02017` |
| S11 | 0.658 | `4c7b08a51153` |
| S12 | 0.563 | `570997c3e031` |
| **Mean** | **0.578** | **#3 of 7 adapters** |

Most resilient adapter to driver change: only -0.023 drop from static to modal (compared to -0.101 for sqlite-chunked-hybrid). The entity dedup KeyError was fixed by normalizing returned names and verifying existence before use. Frequent entity extraction parse failures occur but are handled gracefully without crashing.

---

## Cross-Phase Summary

**Comparability caveat:** Numeric column uses GPT-OSS-120B (gated composites). Narrative column uses Qwen3.5-35B-A3B (ungated). SRS columns use Qwen3.5-35B-A3B (ungated). Cross-column comparisons are not controlled.

| Adapter | Numeric (GPT-OSS, gated) | Narrative (Qwen, ungated) | SRS Static (Qwen) | SRS Dynamic (Qwen) |
|---------|---------------------|----------------------|-----------------------|----------------------|
| cognee | 0.432 (judge-invalid) | 0.436 | -- | -- |
| graphiti | 0.426 (3 scopes missing) | FAILED | -- | -- |
| graphrag-light | -- | **0.537** | **0.578** | **0.544** |

---

## Key Findings

**1. graphrag-light is the standout on semantic retrieval stress.** It ranks #1 with the modal driver (0.555) and #3 with static (0.578). Its 3-signal RRF fusion — BM25 text, entity embeddings, and one-hop graph neighbors — compensates for imprecise agent queries. The -0.023 static-to-modal drop is the smallest of any adapter, indicating that graph-based retrieval produces semantically coherent results even when queries degrade.

**2. Cognee's #2 numeric ranking is unreliable.** The 100% TIE rate from the judge means answer quality was never actually measured. The composite is carried by mechanical metrics (evidence_grounding=1.0, strong evidence_coverage). These metrics are valid, but the overall ranking cannot be trusted without re-scoring with a judge that produces non-tie verdicts.

**3. Graphiti does not scale.** It failed on 3/6 numeric scopes (entity extraction timeout), failed entirely on narrative scopes (context window overflow at checkpoint 16), and was never attempted on SRS. The temporal knowledge graph grows superlinearly with episode count. 120-episode corpora with remote LLM APIs are beyond its operational envelope.

**4. Graph traversal acts as a semantic bridge.** graphrag-light's minimal performance drop under driver change suggests that one-hop graph expansion retrieves contextually relevant episodes that pure vector or keyword search would miss. This property is particularly valuable when agent query formulation is noisy.

**5. Infrastructure burden varies by orders of magnitude.** graphrag-light needs only SQLite. Cognee runs embedded but requires 6 monkey-patches and ACL disabling. Graphiti requires a FalkorDB container and careful timeout configuration.

**6. Entity extraction is the universal bottleneck.** All three adapters depend on LLM entity extraction, which is slow (30--60+ minutes for cognee's `cognify()`), fragile (parse failures in graphrag-light), and scales poorly (graphiti timeouts on large corpora). Any production deployment of graph-based memory must solve entity extraction throughput.
