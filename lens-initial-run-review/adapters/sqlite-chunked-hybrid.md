# sqlite-chunked-hybrid

## Architecture

- **Pattern**: Store-and-Retrieve (passive store, LENS agent drives queries)
- **Ingest**: Chunk episodes into sections, embed with GTE-ModernBERT-base, insert into SQLite FTS5
- **Search**: RRF-fused hybrid (BM25 keyword + cosine semantic), k=60
- **No LLM at ingest or prepare()** — pure indexing
- **Batch retrieval**: Exposes a `batch_retrieve` tool (fetch multiple episodes by ID in one call), which collapsed tool calls from ~41 to 3.2
- **Dependencies**: SQLite (stdlib), embedding endpoint. No containers, no external databases.

---

## Phase 1: Numeric Scopes (S01-S06)

120 episodes (30 signal + 90 distractor), ~700 words/episode, terse operational metrics and logs.

### Main Evaluation (Phase 5) — GPT-OSS-120B (Cerebras)

- 12/12 runs scored (6 scopes x 2 budgets: 8K/16K)
- Agent/Judge: GPT-OSS-120B (Cerebras). Embeddings: GTE-ModernBERT-base (Modal)
- **Overall Mean (8K+16K combined): 0.473 [0.406, 0.502] (95% CI) — #1 of 8 adapters**
- 8K Mean: 0.454, 16K Mean: 0.492 (+0.037)
- Note: Phase 1 report uses 8K-only means (0.454) for cross-adapter comparison
- Per-scope (8K): S01=0.424, S02=0.482, S03=0.381, S04=0.535, S05=0.520, S06=0.386
- Wins 4/6 scopes. Only adapter with CI entirely above 0.40
- Statistically significantly better than letta (p=0.031), letta-sleepy (p=0.031), mem0-raw (p=0.031), compaction (p=0.031)
- NOT significantly better than cognee (p=0.204) — but cognee had 100% judge TIE rate (unreliable)

### Constrained Budget Validation — Qwen3-235B (Together AI)

- 12 runs: 6 scopes x 2 budgets (4K/2K tokens), 30 signal episodes only (no distractors)
- 4K NBA: 0.301 [0.250, 0.354], 2K NBA: 0.347 [0.295, 0.401]
- Per-scope (4K): S01=0.479, S02=0.274, S03=0.263, S04=0.345, S05=0.294, S06=0.352
- Budget degradation not significant (p=0.072)
- Ranked #2 behind compaction (which dominated the small-corpus regime)

### Phase 3 (S01 Only, With Distractors) — GPT-OSS-120B

- 8K: AnsQ=0.477, NBA=0.568, EvCov=0.194 (run `d2ba166bb282`)
- 16K: AnsQ=0.369, NBA=0.516, EvCov=0.177 (run `9003aa5a81f8`)
- **#1 overall** — simple FTS+embedding beat all complex memory architectures
- Budget compliance=0.000 (avg 39K tokens vs 8K budget — 4.8x over)

### 7-Adapter Sweep — Qwen3-32B (RunPod)

- 30 episodes, no distractors. Mean composite: 0.365 (#2 behind compaction 0.399)
- Per-scope: S01=0.370, S02=0.386, S03=0.296, S04=0.443, S05=0.370, S06=0.325

---

## Phase 2: Narrative Scopes (S07-S09)

40 episodes (20 signal + 20 distractor), ~5000 words/episode, chat logs/memos/emails.

### Static Driver — Qwen3.5-35B-A3B (Modal)

- S07=0.302, S08=0.404, S09=0.425, **Mean=0.377 — #1 of 3 tested**
- Per-metric: EvGnd=1.000 (all), BudC=0.00 (all — single episode blows 8K budget)
- Run IDs: S07=`c13f9ed236d3`, S08=`e19a3a980938`, S09=`3354d5623072`

### 3-Rep Determinism (Llama 3.3 70B Era)

- Composite: 0.348 +/- 0.049
- Ingest: 76.5s, Question time: 623.9s, Total: 700.5s
- Mean tokens: 237,509, Tool calls: 20
- 82 token budget violations across 90 questions

---

## Phase 3: Semantic Retrieval Stress (S10-S12)

40 episodes (20 signal + 20 distractor), ~5000 words/episode, structurally identical episodes designed to defeat pure embedding similarity search.

### Static Driver

- S10=0.519, S11=0.691, S12=0.589, **Mean=0.600 — #1 of 7 adapters**
- Run IDs: S10=`70c67b943fa3`, S11=`97b913da439f`, S12=`44a460070f36`

### Modal/Dynamic Driver (Agent Formulates Own Queries)

- S10=0.395, S11=0.555, S12=0.547, **Mean=0.499 — #5 of 8**
- Delta vs static: **-0.101** (biggest drop of any adapter)
- Run IDs: S10=`9129751df7ba`, S11=`18d19b12a8ea`, S12=`232f396f9d29`
- This is the key finding: retrieval is excellent when queries are well-formed, but the agent doesn't formulate optimal queries. The bottleneck is agent query quality, not memory architecture.

---

## Cross-Phase Summary

| Phase | Scopes | Best Score | Rank | N Runs |
|-------|--------|-----------|------|--------|
| Numeric | S01-06 | 0.473 | **1st** | 12 |
| Narrative | S07-09 | 0.377 | **1st** | 3 |
| SRS (static) | S10-12 | 0.600 | **1st** | 3 |
| SRS (modal) | S10-12 | 0.499 | 5th | 3 |

---

## Key Findings

1. **Most consistent adapter overall** — #1 across all three scope categories (with static driver).
2. **Preserves raw text verbatim** — no lossy transformation. Retrieves exact quotes. Every other architecture introduces summarization, graph extraction, or compression that discards evidence.
3. **Static-to-modal driver drop is largest (-0.101)** — retrieval quality is high but agent query formulation is the bottleneck.
4. **Budget compliance is structurally zero** on narrative/SRS episodes (~5000 words each).
5. **Zero infrastructure** — no containers, no external databases, no API keys beyond embedding endpoint.
6. **Ingest speed**: ~1.9s/episode (embed + index).
