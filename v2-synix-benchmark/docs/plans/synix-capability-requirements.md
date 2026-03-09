# Synix Capability Requirements for LENS V2

Requirements derived from memory strategy analysis. Each section lists what the Synix platform must provide.

## V2 First Pass Policy Set (graph deferred)

| Policy | Artifact Families | Synix Primitives |
|--------|------------------|-----------------|
| `null` | none | — |
| `policy_base` | chunks + search indexes | MapSynthesis + SearchSurface |
| `policy_core` | base + core memory | + FoldSynthesis |
| `policy_summary` | base + summaries | + GroupSynthesis + ReduceSynthesis |

All four policies compose from existing Synix transform primitives. `policy_graph` deferred — see section 4.

---

## 1. Chunks + Search Indexes (policy_base)

**Platform chunker:**
- Configurable chunk size and overlap
- Stable chunk IDs with lineage: chunk → source episode + offset
- Chunk ordinal inherits from parent episode (for checkpoint filtering)

**Search surface (default):**
- FTS + cosine + RRF fusion exposed as a built-in query surface
- RRF k parameter exposed and configurable
- Retrieval caps configurable (max results, max context tokens)

**Embedding:**
- Platform handles embed calls through broker
- Batch embedding at build time

---

## 2. Core Memory (policy_core)

**Sequential fold transform:**
- `fold(llm_update, episodes[:N], initial_state)` — one LLM call per episode, sequential
- Configurable block schema: block names, max size per block
- Output is an opaque text blob per checkpoint (not a search surface)
- Deterministic with caching — same episodes + same prompt = same result

**Lineage:**
- Fold output → list of source episode IDs that contributed

**Runtime exposure:**
- Not a search surface — just prepended context the agent always sees
- Synix provides the blob, LENS policy decides how to inject it

## 3. Summaries (policy_summary)

**Batch reduce transform:**
- Group episodes into ordered batches, LLM summarize each batch, then summarize summaries
- Hierarchical: partials → final synthesis
- Output is a text blob per checkpoint (day one: injected context, not a search surface)

**Platform grouping primitive:**
- `group(episodes, size=N, order_by="ordinal", direction="asc")` → list of batches
- Generic — reusable for summaries, core memory batching, any ordered windowing

**Partial retention (ablation target, not day one):**
- Store partial summaries with IDs and lineage (partial → source episode batch)
- If retained, partials can be indexed into the existing search surface for RAG-over-summaries
- Same applies to core memory: store historical fold states as searchable partials

**Lineage:**
- Final summary → list of partial IDs → list of source episode IDs

**Runtime exposure (day one):**
- Not a search surface — just prepended context
- Synix provides the blob, LENS policy decides injection

## 2b. Background Maintenance (Letta Sleepy-Time Equivalence)

**V2 simplification: post-fold refinement transform.**
- Runs once per checkpoint on the fold output (not between episodes)
- Single LLM pass: consolidate, prune, resolve contradictions
- Input: fold output + source episodes
- Output: refined core memory blob (replaces fold output)
- Same sealed checkpoint model — no runtime mutation

**Known simplification:**
- Real Letta runs maintenance *between* episodes during the fold
- We run it as a batch cleanup after the fold completes
- For LENS purposes (measuring retrieval quality), this may not matter
- Ablation target: compare fold-only vs fold+refinement

**Deferred platform question (not v2):**
- Editable draft vs sealed release lifecycle for artifacts
- Maintenance as incremental mutation between checkpoints
- Reconciliation of draft state back into checkpoint lifecycle
- This is a Synix platform versioning concern, not a LENS benchmark concern

## 4. Graph (policy_graph) — NEEDS MORE DESIGN WORK

### What we know

**Target capabilities:** entity extraction, entity dedup/merge, relationship tracking, N-hop traversal, community detection (Leiden), community summaries, graph-boosted retrieval fused with chunk search. Parity with mem0 (triple store), graphiti (temporal), cognee/GraphRAG (communities).

**Extraction is straightforward:**
- MapSynthesis per episode → JSON artifacts with entities + relationships
- Parallel, cached, fingerprinted — standard Synix transform

**Merge/dedup strategy must be pluggable:**
- FuzzyNameMatch, ExactMatch, LLM-assisted — user chooses
- Merge processes extractions in ordinal order (fold semantics)
- Graph DB is stateful with a cursor for incremental application

### The open design question

Graph construction is really a multi-stage pipeline:
1. Extract (Map, LLM) — episode → entities + relationships
2. Merge/Dedup (stateful, possibly LLM) — raw extractions → canonical entity model
3. Community Detection (algorithmic) — entity graph → clusters
4. Community Summaries (Map, LLM) — cluster → summary text

Each stage produces artifacts that should be cached and fingerprinted independently. But the intermediate representations (canonical entity model, community assignments) are structured data, not text — they don't fit cleanly into the current artifact model.

**Attempted approaches and why they're awkward:**
- **Graph as a projection** (like SynixSearch): breaks down because stages 2-4 need LLM calls and caching, which are build-time concerns, not release-time
- **Graph as Map → Release materializer**: works for extraction + simple merge, but community detection and summaries can't live in release without making release expensive and uncacheable
- **Graph as Map → Fold → Release**: fold handles merge, but community stages still don't fit
- **Graph stages as separate artifacts**: conceptually clean (entity artifacts, edge artifacts, community artifacts, summary artifacts) but stage 2 (merge/dedup) still needs stateful cross-artifact awareness that doesn't map to existing transform patterns

### Parking this — what we need to resolve

1. How does Synix model structured (non-text) intermediate artifacts?
2. Can the merge/dedup stage be a transform that takes all extraction artifacts and outputs canonical entity + edge artifacts?
3. Where do community detection and summarization live in the DAG?
4. How does the graph query surface compose with existing FTS + semantic search?

### What we can build now without resolving this

- MapSynthesis extraction works today — we can produce and cache extraction artifacts
- The graph DB materialization pattern (ordered log + pluggable merge + cursor) is clear
- Basic graph query (entity search + N-hop + episode ID boosting) is well-understood
- Community features are additive — can layer on after the core graph works

### For LENS v2 specifically

We may not need full platform graph support for the benchmark. A LENS-specific graph builder that orchestrates the stages externally (calling Synix transforms + custom materialization) could work as a stopgap while the platform design matures.
