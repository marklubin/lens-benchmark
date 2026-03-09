# Memory Strategy Ablation Plan — V2 First Pass

> Created: 2026-03-09
> Status: Planning
> Depends on: Synix PRs #92, #82, #83 → T005 → T007, T008

## Motivation

The V1 headline finding was "no memory system exceeds 50% composite score and simple retrieval beats all complex architectures." The V2 ablation tests whether that's because:

1. **Retrieval is sufficient** — finding the right chunks is all you need, synthesis is wasted compute
2. **V1 implementations were bad** — the *strategy* is sound but the adapter implementations (Letta, compaction, etc.) were limited
3. **Budget matters more than strategy** — more chunks beats distilled context
4. **Domain-dependent** — different strategies win on different scope types (numeric vs. narrative)

V2 eliminates implementation noise by building all strategies on the same Synix substrate. Same chunker, same embeddings, same search infrastructure. Policies only differ in what *additional* context they inject.

## First-Pass Policies

### 1. null

**Strategy class**: Stateless / no memory

**What the agent sees**: Nothing. Question only.

**Synix primitives**: None.

**Real-world analogs**:
- Base ChatGPT / Claude / Gemini with no memory features enabled
- LangChain `ConversationBufferMemory` (stuffs raw history into context, no persistence)
- Any stateless "prompt-and-forget" agent deployment

**Tests**: The floor. How well can the agent answer from the question alone?

---

### 2. policy_base

**Strategy class**: Chunked RAG with hybrid retrieval (BM25 + embeddings + RRF fusion)

**What the agent sees**: Search tool returning ranked chunks via hybrid retrieval.

**Synix primitives**: MapSynthesis (chunker) + SearchSurface (FTS5 + cosine + RRF)

**Real-world analogs**:
- **Zep** — hybrid vector + BM25 retrieval over conversation chunks
- **LangChain / LlamaIndex** vanilla RAG pipelines — chunk, embed, retrieve
- **Graphlit** — semantic chunking + hybrid search infrastructure
- **Mem0** with `infer=False` — bypasses extraction, pure vector store over raw text
- **Pinecone / Weaviate / Qdrant** as memory backends — chunk-and-embed is the default pattern
- **OpenAI Assistants API** `file_search` tool — chunked vector retrieval over uploaded files
- **ODEI** — vector + keyword hybrid retrieval layer

**Tests**: Does finding the right chunks suffice, or does the agent need synthesized context?

---

### 3. policy_core

**Strategy class**: Incremental distillation / editable persistent state

**What the agent sees**: Search tool (same as base) + core memory block in system prompt (fold output).

**Synix primitives**: policy_base + FoldSynthesis (sequential, one LLM call per episode, ordered by timestamp)

**Real-world analogs**:
- **Letta (MemGPT)** — the canonical example. Core memory blocks always in system prompt, agent self-edits via `core_memory_replace` / `core_memory_append`
- **ChatGPT Memory** — extracts persistent user facts, injected into every conversation. Effectively a fold: each conversation may update stored facts
- **Gemini Memory** — persistent fact extraction across sessions
- **Mem0** with extraction enabled — extracts user preferences/facts, stores as managed memories
- **LangMem** (LangChain team) — `create_memory_manager` that extracts and updates semantic memories
- **A-Mem** (2025) — Zettelkasten-inspired, each memory unit enriched with keywords/tags/links, dynamically updated
- **SimpleMem** (2026) — lightweight persistent memory with incremental updates

**Tests**: Does maintaining a distilled "what I know" block help vs. just searching raw chunks?

---

### 4. policy_core_maintained

**Strategy class**: Incremental distillation + background refinement / sleep-time compute

**What the agent sees**: Search tool (same as base) + refined core memory block in system prompt (fold output after consolidation pass).

**Synix primitives**: policy_base + FoldSynthesis + MapSynthesis (single refinement pass: consolidate, prune, resolve contradictions)

**Real-world analogs**:
- **Letta sleep-time agents** — background consolidation between interactions, reorganizes archival memory into core
- **Letta V4 multi-agent** — separate ingest/sleep/QA agents; sleep agent runs consolidation pass
- **Google "sleep-time compute"** (2025) — pre-compute abstractions between queries, amortize reasoning
- **Cofounder** (General Intelligence Company, 2026) — event-based decision log with background memory processing
- **EverMemOS** (2026) — self-organizing memory OS with structured consolidation for long-horizon reasoning
- **Zep temporal knowledge graph** — background process tracking how facts change over time, resolves contradictions

**Known simplification**: Real Letta runs maintenance *between* episodes during the fold. We run it as a batch cleanup after the fold completes. For LENS purposes (measuring retrieval quality at checkpoints), this may not matter. Ablation target: compare fold-only (policy_core) vs. fold+refinement (this policy).

**Tests**: Does a cleanup/consolidation pass over the fold output improve quality, or is raw accumulation good enough? Directly measures the value of "sleep-time" processing.

---

### 5. policy_summary

**Strategy class**: Multi-level recursive summarization / compression hierarchy

**What the agent sees**: Search tool (same as base) + hierarchical summary blob in system prompt (group → reduce output).

**Synix primitives**: policy_base + GroupSynthesis (ordered batches) + ReduceSynthesis (hierarchical reduce)

**Real-world analogs**:
- **Claude Code** context compaction — compresses history into summary when context fills
- **Cursor / Windsurf / Aider** — similar context compaction for coding agents
- **HiAgent** (2025) — chunks working memory by subgoals, summarizes action-observation pairs hierarchically
- **NEXUSSUM** (ACL 2025) — hierarchical LLM agents for long-form summarization
- **Synapse** (2025) — hierarchical consolidation from episodic to semantic memory, +23% multi-hop accuracy at 95% fewer tokens
- **Mem0** chat history compression — 80% prompt token reduction via optimized memory representations
- **LangChain `ConversationSummaryBufferMemory`** — summarizes older messages, keeps recent ones verbatim

**Tests**: Does multi-level compression (episodes → group summaries → global summary) beat single-level fold or raw retrieval?

---

## Ablation Dimensions

### Isolated Comparisons

| Comparison | What it isolates |
|---|---|
| `base - null` | Value of retrieval (finding relevant chunks) |
| `core - base` | Marginal value of fold-based core memory on top of retrieval |
| `core_maintained - core` | Marginal value of consolidation/refinement pass |
| `summary - base` | Marginal value of hierarchical summarization on top of retrieval |
| `core vs summary` | Head-to-head: incremental distillation vs. batch compression |
| `*_16k - *_8k` | Whether more budget (more chunks) outweighs better synthesis |

### Budget Dimension

Each policy at two token budgets:

| Policy | 8K | 16K |
|---|---|---|
| null | null_8k | null_16k |
| policy_base | base_8k | base_16k |
| policy_core | core_8k | core_16k |
| policy_core_maintained | core_maintained_8k | core_maintained_16k |
| policy_summary | summary_8k | summary_16k |

### Scope Selection

**Screening study (32-40 configs)**: 4 scopes (2 numeric + 2 narrative) × 5 policies × 2 budgets.

**Full study (72-90 configs)**: 9 scopes (6 numeric + 3 narrative) × 5 policies × 2 budgets.

Numeric scopes test strategies on structured telemetry data (where V1 sqlite-chunked-hybrid dominated). Narrative scopes test on long-form text (where V1 Letta family dominated). If strategies are domain-dependent, this split will reveal it.

## Strategy Coverage Assessment

The 5 first-pass policies cover 3 of the ~7 major strategy classes in production today:

| Class | Covered | Policies |
|---|---|---|
| Stateless | Yes | null |
| Chunked RAG | Yes | policy_base |
| Incremental distillation | Yes | policy_core, policy_core_maintained |
| Hierarchical compression | Yes | policy_summary |
| Knowledge graphs | No | kill list |
| Temporal knowledge graphs | No | kill list |
| Multi-agent shared memory | No | kill list |
| Procedural / skill memory | No | kill list |

## Kill List — Deferred Strategies

Strategies that can't be faithfully modeled with Synix v1 primitives. Tracked here for future phases.

### Knowledge Graph + Entity Resolution

**Products**: Graphiti (temporal KG), Cognee (GraphRAG), Neo4j GraphRAG, Microsoft GraphRAG, Mem0 graph memory

**Strategy**: Extract entities/relationships from text, deduplicate/merge, build traversable graph, use N-hop neighborhood + community summaries to augment retrieval.

**Synix blocker**: Stateful merge/dedup across artifacts, community detection, graph projection adapter. Extraction (MapSynthesis) is straightforward; everything after extraction needs platform design work.

**Priority**: High — most products are adopting graph memory. Biggest gap in our coverage.

### Temporal Knowledge Graph

**Products**: Zep (temporal facts with validity tracking), Graphiti (bi-temporal edges)

**Strategy**: Track how facts *change* over time, not just accumulate. Invalidate stale facts, surface contradictions, query "what was true at time T?"

**Synix blocker**: Temporal-aware artifact versioning. Current artifacts are immutable snapshots; temporal queries need a version history per entity.

**Priority**: Medium — subset of graph, naturally extends once graph lands.

### Multi-Agent Shared Memory

**Products**: Letta V4 (ingest/sleep/QA agents), CrewAI shared memory, MAGMA (2026 paper)

**Strategy**: Multiple specialized agents reading/writing to a shared memory bank concurrently. Different agents responsible for ingestion, consolidation, and answering.

**Synix blocker**: SDK v2 buffer for concurrent instant writes. Current sealed checkpoint model doesn't support concurrent mutation.

**Priority**: Medium — important for production agent architectures, but LENS's benchmark model (single agent per policy) doesn't require it.

### Procedural / Skill Memory

**Products**: Voyager (Minecraft skill library), JARVIS, hierarchical procedural memory (ICLR 2026)

**Strategy**: Learn and reuse action sequences, not just declarative facts. Store successful tool-use patterns, code snippets, decision procedures.

**Synix blocker**: Completely different artifact semantics — these aren't text blobs, they're executable patterns. Out of scope for LENS's current focus on longitudinal evidence synthesis.

**Priority**: Low for LENS — different benchmark domain entirely.

### Associative / Spreading Activation Memory

**Products**: A-Mem (Zettelkasten links), OpenClaw associative hierarchical memory

**Strategy**: Dynamic link traversal at query time — retrieve a memory, follow links to related memories, spread activation through the network.

**Synix blocker**: Graph query at runtime, not build-time. Requires dynamic link resolution in the search/retrieval path.

**Priority**: Low — academic, few production implementations.

### Observation / Event Memory

**Products**: Observational memory (VentureBeat 2025), Cofounder event-based decision log

**Strategy**: Real-time event stream with structured extraction. Each observation is a dated, prioritized, structured record (not prose). Query by event type, time range, priority.

**Synix blocker**: SDK v2 buffer for instant structured writes. Partially modelable via FoldSynthesis if we accept batch-only processing.

**Priority**: Medium — emerging pattern, partially covered by policy_core_maintained.

### Memory-as-Filesystem

**Products**: Letta "Context Repositories" (2026) — git-based versioning of memory with branching/merging

**Strategy**: Programmatic context management with git semantics. Branch memory per conversation, merge insights back to main. Version control for knowledge.

**Synix blocker**: Synix has refs and snapshots but not branch-per-agent semantics or merge operations across memory branches.

**Priority**: Low — very new (Feb 2026), unclear if the pattern generalizes.

## References

- [Mem0 vs Zep vs LangMem vs MemoClaw Comparison 2026](https://dev.to/anajuliabit/mem0-vs-zep-vs-langmem-vs-memoclaw-ai-agent-memory-comparison-2026-1l1k)
- [Top 10 AI Memory Products 2026](https://medium.com/@bumurzaqov2/top-10-ai-memory-products-2026-09d7900b5ab1)
- [Memory in the Age of AI Agents Survey (Dec 2025)](https://arxiv.org/abs/2512.13564)
- [A-Mem: Agentic Memory for LLM Agents (2025)](https://arxiv.org/html/2502.12110v11)
- [Letta Docs — MemGPT Concepts](https://docs.letta.com/concepts/memgpt/)
- [Cofounder Memory System (2026)](https://www.generalintelligencecompany.com/writing/introducing-cofounder-our-state-of-the-art-memory-system-in-an-agent)
- [Survey of AI Agent Memory Frameworks — Graphlit](https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks)
- [ODEI vs Mem0 vs Zep Comparison 2026](https://dev.to/zer0h1ro/odei-vs-mem0-vs-zep-choosing-agent-memory-architecture-in-2026-15c0)
- [NEXUSSUM: Hierarchical LLM Agents (ACL 2025)](https://aclanthology.org/2025.acl-long.500.pdf)
- [Observational Memory Cuts AI Agent Costs 10x — VentureBeat](https://venturebeat.com/data/observational-memory-cuts-ai-agent-costs-10x-and-outscores-rag-on-long)
