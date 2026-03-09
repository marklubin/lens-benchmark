# Deferred Memory Strategies — Deep Dive

> Created: 2026-03-09
> Status: Research / Kill List
> Purpose: Document the memory strategy classes we cannot yet model in Synix v1, their implementations, theoretical basis, likely value, and Synix design unknowns.

These strategies are excluded from the V2 first pass because they require Synix platform capabilities that don't exist yet. This document captures enough detail to inform when and how to build them.

---

## 1. Knowledge Graph + Entity Resolution (GraphRAG)

### How It Works

A multi-stage pipeline that builds a queryable property graph from unstructured text:

1. **Entity extraction**: LLM processes each text chunk independently, emitting `(entity, type, description)` tuples and `(source, relation, target)` triples. Typically parallelized across chunks.
2. **Element summarization**: Multiple extractions of the same entity across chunks are merged into a single descriptive summary per entity node.
3. **Entity resolution / deduplication**: Match extracted names to canonical entities. Approaches range from exact string matching (Microsoft GraphRAG default) to fuzzy matching (Jaccard, edit distance) to LLM-assisted resolution ("are these the same entity?").
4. **Graph construction**: Entities become nodes, relationships become edges. Edge weights may encode co-occurrence frequency, extraction confidence, or semantic similarity.
5. **Community detection**: Leiden algorithm partitions the graph into hierarchical communities of closely-related entities. Recursive — detects sub-communities until reaching leaf nodes.
6. **Community summarization**: LLM generates a summary of each community, capturing the collective meaning of clustered entities and their relationships.
7. **Retrieval**: Query hits both the chunk index (local search) and community summaries (global search). Results fused via RRF or similar.

### Products Implementing This

| Product | Variant | Key Differentiator |
|---|---|---|
| **Microsoft GraphRAG** | Full pipeline (extract → Leiden → community summaries → local+global search) | Open-source reference implementation, hierarchical community detection |
| **Cognee** | GraphRAG + LanceDB vectors + Kuzu graph DB | Adds vector search alongside graph traversal |
| **Neo4j GraphRAG** | Property graph with native graph algorithms | Enterprise graph DB, native Cypher queries |
| **Mem0 graph memory** | Triple store (subject-predicate-object) + vector retrieval | Simpler than full GraphRAG — no community detection |
| **Our graphrag-light adapter** (V1) | NetworkX DiGraph + SQLite FTS + entity embeddings + 1-hop neighborhood + RRF | Self-contained, no external DB. Phase 7b: 0.555 composite |
| **LightRAG** (2025) | Lightweight alternative to Microsoft GraphRAG | Faster indexing, simpler community structure |

### Theoretical Basis

Knowledge graphs encode *relational structure* that flat text and embeddings cannot capture. When a question requires understanding how entities relate to each other ("which teams were affected by the cascading failure?"), graph traversal can surface connections that no amount of keyword or embedding search over raw text will find.

Community summaries address *global queries* — questions about the overall corpus rather than specific passages. "What are the main themes?" cannot be answered by retrieving individual chunks; it requires a pre-computed abstraction over the entire graph.

The theoretical advantage over policy_base (chunked RAG): structured relationships survive the lossy compression of embedding. Two chunks may have similar embeddings but describe completely different relationships between the same entities.

### Likelihood of Genuine Value

**Mixed signal from V1 results:**
- graphrag-light scored 0.555 composite in Phase 7b (modal driver) — competitive but not dominant
- graphrag-light was the *most resilient* to agent query formulation (-0.023 drop vs. -0.05 average from static to modal)
- Cognee scored 0.432 in Phase 5 but had 100% TIE rate on judge (scoring failure, not strategy failure)
- Graphiti scored 0.426 in Phase 5 but only completed 6/12 runs (entity extraction timeout on larger scopes)

**The LENS-specific challenge**: LENS tests longitudinal evidence synthesis — "what changed over time?" — not relational reasoning. Graph structure captures *what entities exist and how they relate* but doesn't natively encode that the *nature of interactions changed across episodes*. This is the fundamental limitation noted in graphrag-light's comments.

**Assessment**: Moderate value. Likely helps on questions requiring entity-centric reasoning ("which component was the root cause?") but adds little for temporal progression questions ("how did latency evolve?"). Worth testing but unlikely to dominate chunked retrieval + fold on LENS's question distribution.

### Synix Design Unknowns

1. **Stateful merge across artifacts**: Entity dedup requires comparing the *current* extraction against *all previous* extractions. This is a fold with global state — every new extraction must be resolved against the accumulated canonical entity set. FoldSynthesis processes one artifact at a time sequentially, but the accumulated state is a text blob, not a queryable entity index. The fold would need to maintain structured JSON state (entity registry) across iterations, not just text.

2. **Graph as a projection type**: Currently Synix has `synix_search` (FTS5 + embeddings) and `flat_file` (markdown). A graph needs a new projection adapter — materializing a property graph (Neo4j, FalkorDB, NetworkX pickle, or a custom SQLite schema) at release time. What's the adapter contract? Does it write to an external graph DB or bundle a portable graph format?

3. **Community detection as a transform**: Leiden runs on the *whole graph*, not on individual artifacts. It's inherently a ReduceSynthesis over the full entity set. But the input isn't text — it's a graph structure. Synix transforms operate on `list[Artifact]` → `list[Artifact]`. The graph would need to be serialized as artifact content (JSON adjacency list?) and the transform would deserialize, run Leiden, serialize back. Awkward but workable.

4. **Entity resolution quality**: Microsoft's own research notes that entity resolution accuracy below 85% makes the entire system unreliable — every graph traversal compounds resolution errors. This means the extraction + resolution pipeline needs to be high-quality before the downstream graph has value. If resolution is bad, the graph is worse than no graph.

5. **Incremental updates**: Microsoft GraphRAG v0.5+ supports incremental updates with consistent entity IDs. Synix's cache model (hash comparison → rebuild or skip) would need to support *appending* new extractions to an existing graph rather than rebuilding from scratch at each checkpoint.

---

## 2. Temporal Knowledge Graph

### How It Works

Extends a knowledge graph with time-awareness. Every edge carries validity intervals:

- **Event time (T)**: When the fact actually occurred or was true
- **Ingestion time (T')**: When the system first learned about the fact

This bi-temporal model enables:
- Querying "what was true at time T?" (point-in-time snapshots)
- Detecting fact invalidation ("X was the lead engineer" → "X transferred to another team")
- Tracking how relationships *evolve* — the same edge (A → B) may exist across multiple validity windows with different properties
- Retroactive corrections — ingesting a fact at T' that refers to event time T < T'

### Products Implementing This

| Product | Implementation |
|---|---|
| **Graphiti** (Zep) | Neo4j/FalkorDB backend. Bi-temporal edges with `(t_valid, t_invalid)`. Real-time incremental updates without batch recomputation. Three-tier graph: episode subgraph, semantic entity subgraph, community subgraph. |
| **Zep** (commercial) | Temporal knowledge graph as the core memory layer. Tracks how facts change over time. Combines graph memory with vector search. Outperforms MemGPT on Deep Memory Retrieval benchmark. |

### Theoretical Basis

Standard knowledge graphs are *static snapshots* — they represent the current state of knowledge but lose the history of how that knowledge evolved. For longitudinal tasks (exactly what LENS measures), temporal graphs are theoretically ideal: they natively answer "what changed?" by comparing graph states across time.

The bi-temporal model specifically handles the asymmetry between *when something happened* and *when the agent learned about it*. In LENS's benchmark model, episodes arrive in chronological order but may contain retroactive information ("last week's deployment actually caused..."). A temporal graph can correctly position this information on the event timeline, not just the ingestion timeline.

### Likelihood of Genuine Value

**Theoretically high for LENS specifically.** LENS's core challenge is longitudinal synthesis — detecting signal that only emerges from the *progression* across episodes. Temporal graphs are the only strategy class that natively represents this progression as a first-class query dimension.

**V1 evidence is inconclusive:**
- Graphiti (temporal graph) scored 0.426 in Phase 5 but only completed 6/12 runs due to entity extraction timeouts
- The failures were operational (extraction too slow at 120 episodes), not strategic
- We never got a clean signal on whether temporal awareness actually helps

**Assessment**: High theoretical value, insufficient empirical evidence. If graph strategy lands, temporal should be the first extension — it directly addresses LENS's core measurement.

### Synix Design Unknowns

1. **Artifact versioning**: Current Synix artifacts are immutable. A temporal entity needs a *version history* — the same entity label with different content at different validity intervals. This is fundamentally different from "rebuild the artifact." It's "this artifact has multiple valid states indexed by time." Does Synix need a `TemporalArtifact` type, or can this be modeled as metadata on regular artifacts?

2. **Temporal queries in the search layer**: `release.search("latency", time_range=(T1, T2))` doesn't exist. The search retriever would need to filter by validity intervals, not just by content relevance. This requires either extending the FTS5/embedding search with temporal filters, or building a separate temporal query path.

3. **Edge invalidation during fold**: When a new episode contradicts an earlier fact, the temporal graph doesn't delete the old edge — it sets `t_invalid` on the old edge and creates a new one. In Synix's immutable artifact model, this would mean producing a *new* artifact that references the old one with invalidation metadata. The fold transform would need to emit both "create new" and "invalidate old" operations.

4. **Dependency on graph strategy**: Temporal is an extension of graph, not independent. All the graph unknowns (entity resolution, graph projection adapter, community detection) must be resolved first.

---

## 3. Multi-Agent Shared Memory

### How It Works

Multiple specialized agents share a common memory bank with different read/write patterns:

- **Ingest agent**: Processes incoming data, writes extracted facts/observations to shared memory
- **Consolidation agent**: Runs periodically (or on threshold), reorganizes and compresses memory
- **QA agent**: Reads memory at query time, answers questions
- **Planning agent**: Reads memory for context, writes plans/decisions back

Each agent may have its own private working memory in addition to access to the shared bank. Consistency guarantees vary: some systems use locks, others use eventual consistency, others use git-style branching and merging.

### Products Implementing This

| Product | Architecture |
|---|---|
| **Letta V4** | Three-agent loop: ingest agent (extracts + stores), sleep agent (consolidation), QA agent (answers). Core memory blocks shared across agents. Our V1 adapter: 0.562 composite. |
| **CrewAI** | Role-based multi-agent framework. ChromaDB-backed shared short-term + long-term memory. Each crew member reads/writes to shared context. Interagent misalignment accounts for 36.9% of failures. |
| **MAGMA** (Jan 2026) | Multi-graph memory — each memory item represented across orthogonal semantic, temporal, causal, and entity graphs. Retrieval is policy-guided traversal across these views. Designed for agent swarms. |
| **Letta Context Repositories** (Feb 2026) | Git-backed memory filesystem. Multiple subagents get isolated worktrees, process memory concurrently, merge changes through git conflict resolution. Progressive disclosure via file hierarchy. |
| **AutoGen** | Microsoft's multi-agent framework. Shared conversation history as memory. No structured long-term memory by default. |
| **MCP Memory Service** | Open-source persistent memory for multi-agent pipelines (LangGraph, CrewAI, AutoGen). Knowledge graph + autonomous consolidation. |

### Theoretical Basis

The motivation is *separation of concerns*: the agent that's best at extracting information from raw text isn't necessarily the same agent that's best at answering questions from extracted facts. Different memory operations (write, consolidate, query) benefit from different system prompts, different models, and different context windows.

The deeper theoretical claim: human memory consolidation happens *offline* (during sleep), not during the experience itself. The dual-process theory (System 1 fast encoding, System 2 slow consolidation) maps to fast ingest agents + slow background consolidation agents.

### Likelihood of Genuine Value

**Our V1 evidence says: limited value for LENS.** Letta V4 (three-agent) scored 0.562 vs standard Letta 0.606 on narrative scopes — the multi-agent version was *worse*. The core memory blocks (patterns, hypotheses, entities, events at 5K each) lost fine-grained evidence compared to the simpler archival search.

**But the V1 implementation was constrained**: the three agents shared a single Letta server with fixed block schemas. A properly designed multi-agent system with flexible shared memory might do better.

**The real value likely emerges at scale**: when the memory bank is large enough that a single agent can't effectively manage both ingestion and retrieval, specialization starts to pay off. At LENS's current scale (120 episodes, ~84K tokens), a single agent can handle everything. At 1000+ episodes, multi-agent may become necessary.

**Assessment**: Low value for LENS at current scale. Medium value as a future extension if we increase corpus size. The interesting ablation (single vs. multi-agent on the same memory bank) doesn't need new Synix primitives — it needs a multi-agent orchestrator.

### Synix Design Unknowns

1. **Concurrent writes to shared state**: Synix's model is build → release → read. There's no concept of multiple writers mutating the same artifact bank concurrently. SDK v2's buffer (deferred) sketches a fast write path, but it's append-only JSONL — not concurrent structured updates.

2. **Agent-scoped views**: Different agents need different views of the same memory. The ingest agent sees raw episodes + its extraction buffer. The QA agent sees the search index + core memory. Synix releases are a single consistent view. Would need either multiple named projections per release (already supported) or agent-scoped filtering at query time.

3. **Consistency model**: What happens when the consolidation agent is rewriting core memory while the QA agent is reading it? Synix's sealed checkpoint model prevents this by design (read from released snapshot, never from in-progress build). But that means the QA agent always reads *stale* memory — it can never see the consolidation agent's latest work until a new release is cut.

4. **Orchestration is LENS-side, not Synix-side**: Synix doesn't need to know about multi-agent. It just needs to support fast writes (buffer) and concurrent reads (already works — Release objects are independent). The multi-agent orchestration is a LENS runner concern.

---

## 4. Procedural / Skill Memory

### How It Works

Instead of storing declarative knowledge (facts, observations, summaries), the agent stores *executable procedures* — code snippets, tool-use patterns, decision trees, or action templates that can be retrieved and reused in similar situations.

Voyager's architecture (the canonical example):

1. **Skill library**: A vector-indexed collection of JavaScript functions (in Minecraft). Each skill has a natural language description for retrieval and executable code for reuse.
2. **Skill composition**: Complex behaviors are built by composing simpler skills. "Build a house" retrieves and chains "gather wood" → "craft planks" → "place blocks."
3. **Self-verification**: After executing a skill, the agent checks whether the goal was achieved. Failed skills are refined through iterative prompting with environment feedback.
4. **Curriculum-driven acquisition**: An automatic curriculum proposes progressively harder tasks, driving the agent to acquire new skills.

### Products Implementing This

| Product | Domain | Approach |
|---|---|---|
| **Voyager** (2023) | Minecraft | JavaScript skill library, vector-indexed by description, compositional |
| **JARVIS** (2023) | Multi-tool | Task planner + tool selector + executor with skill memory |
| **Hierarchical Procedural Memory** (ICLR 2026) | General agents | Bayesian selection + contrastive refinement of procedure library |
| **Agent Skills survey** (Feb 2026) | Survey paper | Categorizes skill acquisition, security, architecture patterns across the field |
| **Anthropic Tool Use** (Nov 2025) | API | Tool Search, Programmatic Tool Calling, Tool Learning — industry-level skill integration |
| **OpenAI function calling** | API | Structured tool definitions — the substrate procedural memory would index over |

### Theoretical Basis

Declarative memory answers "what do I know?" Procedural memory answers "what can I do?" In cognitive science, these are fundamentally different memory systems (hippocampus vs. basal ganglia/cerebellum). An agent that only has declarative memory must re-derive procedures from scratch every time; an agent with procedural memory can retrieve and adapt proven approaches.

The theoretical win is *amortized reasoning*: solving a problem once and storing the solution is cheaper than solving it from scratch every time a similar problem appears. This is especially valuable for tool-use-heavy agents where the correct tool invocation pattern is complex but reusable.

### Likelihood of Genuine Value

**Not applicable to LENS.** LENS measures longitudinal evidence synthesis — the agent's job is to reason about what happened across episodes, not to execute procedures or use tools in novel ways. There's no "skill" to learn and reuse in the LENS task format.

**Assessment**: Zero value for LENS. This is a fundamentally different benchmark domain (embodied agents, tool-use agents). Including it would be testing a capability LENS isn't designed to measure.

### Synix Design Unknowns

1. **Artifact content type**: Synix artifacts are text strings. Procedural memory needs *executable* content — code, tool schemas, action templates. Could be stored as text (JSON tool definitions, code strings), but the semantics are completely different from episodes or summaries.

2. **Retrieval by capability, not content**: Skill retrieval needs to match *what the skill does* (its capability description) to *what the agent needs* (the current task). This is a different retrieval problem than content-similarity search. Would need a capability-indexed projection rather than a content-indexed one.

3. **No current use case in LENS**: This strategy class is entirely outside LENS's evaluation scope. No Synix design work is warranted until there's a concrete use case.

---

## 5. Associative / Spreading Activation Memory

### How It Works

Memories are organized as a *network* of interconnected nodes. Retrieval works by *spreading activation*: accessing one memory activates it, and that activation propagates along links to related memories, surfacing contextually relevant information that wouldn't be found by direct search.

A-Mem's architecture (NeurIPS 2025):

1. **Zettelkasten structure**: Each memory unit ("note") is enriched with LLM-generated keywords, tags, contextual descriptions, and dynamically constructed links to semantically related memories.
2. **Dynamic linking**: When a new memory is added, the LLM identifies connections to existing memories based on embedding similarity *and* semantic reasoning. Links are bidirectional and weighted.
3. **Retroactive updates**: Adding a new memory can trigger updates to *existing* memories — their contextual descriptions and links are revised in light of the new information. The network continuously self-organizes.
4. **Retrieval**: Query activates initial matches (embedding similarity), then traverses links to find related memories that embedding search alone would miss.

### Products Implementing This

| Product | Approach |
|---|---|
| **A-Mem** (NeurIPS 2025) | Zettelkasten-inspired, ChromaDB backend, dynamic inter-memory links, retroactive updates |
| **OpenClaw proposal** (2026) | Associative hierarchical memory with human-like recall patterns |
| **MAGMA** (Jan 2026) | Multi-graph traversal (semantic + temporal + causal + entity) — spreading activation across graph views |

### Theoretical Basis

Human memory is associative — recalling one memory triggers related memories through a network of associations. This is why a smell can trigger a vivid childhood memory: the activation spreads from the olfactory association to the episodic memory through learned links.

Standard embedding retrieval is *point-wise* — each memory is scored independently against the query. Associative memory adds *relational* retrieval — memories that are low-relevance individually but highly connected to high-relevance memories get surfaced. This helps with multi-hop reasoning: if the query matches memory A, and A links to B, and B links to C, the system surfaces C even though the query has no direct semantic overlap with C.

A-Mem's retroactive update mechanism is particularly interesting: it means the memory network *improves* as more memories are added, because existing memories get richer context. This is the opposite of most systems where adding more memories just increases noise.

### Likelihood of Genuine Value

**Moderate theoretical value for LENS.** LENS questions specifically require multi-hop reasoning across episodes. A memory that links "p99 latency: 600ms" in episode 12 to "deployment rollback in region-east" in episode 8 could surface the connection that pure embedding search misses.

**But empirically unproven at scale.** A-Mem was evaluated on standard benchmarks (LongBench, InfiniteBench), not on longitudinal synthesis tasks. The retroactive update cost grows quadratically with memory size — each new memory potentially updates all existing memories.

**Assessment**: Medium value. The link-traversal retrieval is a strict superset of embedding retrieval — it can only help, not hurt. But the construction cost may be prohibitive at 120+ episodes, and the benefit over well-tuned hybrid search is unclear.

### Synix Design Unknowns

1. **Dynamic inter-artifact links**: Synix artifacts have `input_ids` (provenance), which is a DAG relationship. Associative links are different — they're *lateral* connections between artifacts at the same level, not parent-child relationships. Would need a new link type or a separate link index.

2. **Retroactive mutation**: When a new artifact is added, existing artifacts' metadata/descriptions may need updating. This violates Synix's immutability model. Options: (a) treat link updates as a new artifact version (version history), (b) store links in a separate mutable index outside the artifact store, (c) rebuild links as a batch transform after all artifacts are ingested.

3. **Activation-based retrieval**: The search retriever needs to support graph traversal, not just vector/FTS scoring. After initial matches, follow links N hops, accumulate activation scores, return top-K by total activation. This is a new search mode, not expressible through existing FTS5 + cosine infrastructure.

4. **Construction cost**: A-Mem uses LLM calls for link generation and retroactive updates. At 120 episodes, this could mean 120 × 119 = 14,280 potential link evaluations (though in practice, only top-K by embedding similarity are considered). The cost model needs to be understood before committing to platform support.

---

## 6. Observation / Event Memory

### How It Works

Instead of storing raw text or summaries, the system maintains a *structured event log* — each entry is a dated, prioritized, categorized observation capturing a specific fact or event.

Mastra's Observational Memory architecture:

1. **Observer agent**: Watches the conversation and produces structured observations. Each observation has: date, priority (high/medium/low), category, and a terse factual description. Not prose — structured records.
2. **Append-only log**: Observations accumulate in an ordered log that replaces raw message history. The log is always in the system prompt.
3. **Prompt caching**: Because the log is append-only (new observations always go at the end), the entire prefix is cache-stable. Anthropic/OpenAI prompt caching gives 4-10x cost reduction on the cached prefix.
4. **Reflection / garbage collection**: When the log exceeds a threshold (default 40K tokens), a Reflector agent runs second-level compaction: merges related observations, drops superseded ones, preserves the structured format.
5. **Two-level structure**: Top-level bullets are tasks/events, sub-bullets are details. Grouped by date with inline timestamps.

### Products Implementing This

| Product | Approach |
|---|---|
| **Mastra Observational Memory** (Feb 2026) | Observer + Reflector agents, structured event log, prompt caching. 94.87% on LongMemEval with gpt-5-mini. Claims 10x cost reduction vs. RAG. |
| **Cofounder** (General Intelligence Company, 2026) | Event-based decision log. Processes events more frequently on smaller chunks. Operational audit trail rather than summary. |
| **HINDSIGHT / TEMPR** (Dec 2025) | Structured observation extraction with temporal awareness. Our V1 adapter scored 0.213 — but failed operationally (20-100s/episode ingest, 17.3GB image). |

### Theoretical Basis

The insight: most memory systems either store too much (raw text) or lose too much (summaries). Structured observations hit a middle ground — they're denser than raw text (only facts, no filler) but more precise than summaries (individual dated events, not abstractions).

The prompt caching economics are compelling: by keeping the observation log append-only, every API call benefits from cached-prefix pricing. A 50-turn conversation with 30K tokens of observations pays full price once and gets 4-10x discounts on every subsequent turn. For long-running agents, this dominates the cost model.

The structured format also enables filtering that neither raw text nor summaries support: "show me all high-priority observations from the last week" is trivial with structured events but impossible with a compressed summary.

### Likelihood of Genuine Value

**High, and partially testable now.** Mastra's 94.87% on LongMemEval is the highest reported score on that benchmark. The structured event log is essentially what our FoldSynthesis already produces if prompted correctly — "extract dated observations from this episode and append to the log."

**The key insight for LENS**: Observation memory is a *variant of policy_core* where the fold prompt produces structured observations instead of free-form core memory. We could test this as a policy_core prompt variant without any new Synix primitives.

**Assessment**: High value. The strategy is largely implementable today as a prompt engineering variant of FoldSynthesis. The Reflector (garbage collection) maps to our policy_core_maintained refinement pass. Consider adding as a prompt variant ablation within policy_core rather than a separate deferred strategy.

### Synix Design Unknowns

1. **Mostly solved**: FoldSynthesis + MapSynthesis refinement already models this pattern. The "structured observation" format is a prompt concern, not a platform concern.

2. **Real-time append**: Mastra's system is designed for real-time conversation (append observation after each turn). Synix processes in batch (all episodes at once). This difference doesn't matter for LENS (we process checkpoint prefixes as batches anyway).

3. **Prompt caching optimization**: The cost advantage of observation memory comes from prompt caching infrastructure. This is an inference-provider concern (Anthropic/OpenAI APIs), not a Synix platform concern. The Modal broker already handles caching.

4. **Possible promotion to first pass**: Given that this maps cleanly to existing primitives, consider promoting "structured observation fold" from the kill list to a policy_core prompt variant in the first pass ablation.

---

## 7. Memory-as-Filesystem

### How It Works

Memory is stored as a *versioned filesystem* with git semantics. Agents interact with memory by reading and writing files, and all changes are automatically version-controlled.

Letta's Context Repositories (Feb 2026):

1. **File-based memory**: Agent context is stored as files in a local directory. Different types of memory (core facts, conversation history, learned patterns) are different files or directories.
2. **Git versioning**: Every change is automatically committed with informative commit messages. Full history is preserved and browsable.
3. **Agent-controlled organization**: The agent itself can reorganize files, update frontmatter descriptions, move files between directories. Progressive disclosure: the agent controls what's pinned to context vs. what's filed away.
4. **Multi-agent via worktrees**: Each subagent gets an isolated git worktree. Concurrent memory processing with standard git merge for conflict resolution.
5. **Terminal access**: Because memory is files on disk, the agent can use all standard tools (grep, sed, scripts) to manage memory — not limited to memory-specific APIs.

### Products Implementing This

| Product | Approach |
|---|---|
| **Letta Context Repositories** (Feb 2026) | Git-backed filesystem, agents write scripts to restructure memory, isolated worktrees for subagents |
| **Letta Filesystem** (2026) | Earlier iteration — structured file storage for agent state |
| **Claude Code** CLAUDE.md / memory files | Simple variant — markdown files the agent reads/writes, no git versioning |
| **Cursor Rules** / `.cursorrules` | Static memory files — project context injected into every prompt |
| **Synix itself** | `.synix/` directory with content-addressed objects and refs — git-inspired architecture |

### Theoretical Basis

The claim: existing memory APIs (search, insert, update) are too restrictive. Agents are general-purpose reasoners that can write code — if you give them a filesystem, they can build their own memory organization adapted to the task. The filesystem is a maximally flexible interface.

Git versioning adds safety (can always revert), auditability (full history of memory changes), and concurrency (worktree isolation + merge). These are properties that custom memory APIs must build from scratch.

The deeper insight: *memory organization should evolve with the agent's understanding*. Early in a task, flat files suffice. As the agent learns more, it may reorganize into hierarchies, create indices, split files. The filesystem allows this organic evolution without API changes.

### Likelihood of Genuine Value

**Low for LENS, high for real-world agents.** LENS's benchmark format (stream episodes → answer questions at checkpoints) doesn't benefit from agent-controlled memory organization. The episodes are fixed, the questions are predetermined, and the agent doesn't iterate on its memory structure over multiple sessions.

**For real-world long-running agents**, filesystem memory is compelling. Letta Code's early results show agents that restructure their own context over time, developing task-specific memory layouts that no pre-designed API would have produced.

**Assessment**: Zero value for LENS. High value for Synix as a platform feature (unified-memory use case). Different evaluation needed.

### Synix Design Unknowns

1. **Synix already has the building blocks**: `.synix/` is a content-addressed object store with refs (git-inspired). Releases are named snapshots. The distance from "git-like build system" to "git-backed memory filesystem" is smaller than it looks.

2. **Agent-writable artifacts**: Current Synix artifacts are produced by transforms (LLM pipeline steps). Filesystem memory means the agent writes artifacts directly. This overlaps with SDK v2's buffer concept but goes further — the agent can also *reorganize* existing artifacts, not just append new ones.

3. **Progressive disclosure / pinning**: The agent needs to control which artifacts are always in context vs. filed away. Synix's FlatFile projection is a simple version of this (renders selected artifacts as markdown). A more sophisticated version would let the agent dynamically select what's pinned.

4. **Not a Synix v1 priority**: The memory-as-filesystem pattern is most valuable for real-world agents, not benchmarks. Defer until the unified-memory use case drives requirements.

---

## Priority Summary

| # | Strategy | Value for LENS | Synix Difficulty | Recommendation |
|---|---|---|---|---|
| 1 | Knowledge Graph (GraphRAG) | Moderate | High — entity resolution, graph projection, community detection | Build after graph artifact model is designed |
| 2 | Temporal Knowledge Graph | High (theoretical) | Very High — depends on #1 + temporal versioning + temporal queries | Build as extension of #1 |
| 3 | Multi-Agent Shared Memory | Low at current scale | Medium — SDK v2 buffer + orchestration is LENS-side | Defer; not a Synix concern |
| 4 | Procedural / Skill Memory | None | N/A | Out of scope for LENS entirely |
| 5 | Associative / Spreading Activation | Medium | High — lateral links, activation retrieval, retroactive mutation | Research; may partially model as graph variant |
| 6 | Observation / Event Memory | High | **Low — already buildable as FoldSynthesis prompt variant** | **Consider promoting to first-pass ablation** |
| 7 | Memory-as-Filesystem | None for LENS | Medium | Defer to unified-memory use case |

### Immediate Action Item

**Observation memory (#6) should be re-evaluated for first-pass inclusion.** It maps cleanly to existing Synix primitives (FoldSynthesis with a structured-observation prompt + MapSynthesis refinement). The only question is whether the prompt engineering to produce structured observations vs. free-form core memory is different enough to warrant a separate policy, or whether it's a within-policy hyperparameter.

## References

- [Microsoft GraphRAG — GitHub](https://github.com/microsoft/graphrag)
- [Microsoft GraphRAG — From Local to Global (2024)](https://arxiv.org/html/2404.16130v2)
- [Zep: Temporal Knowledge Graph Architecture (Jan 2025)](https://arxiv.org/abs/2501.13956)
- [Graphiti: Knowledge Graph Memory — Neo4j Blog](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [MAGMA: Multi-Graph Agentic Memory (Jan 2026)](https://arxiv.org/abs/2601.03236)
- [CrewAI Memory Docs](https://docs.crewai.com/en/concepts/memory)
- [Letta Context Repositories (Feb 2026)](https://www.letta.com/blog/context-repositories)
- [Voyager: Open-Ended Embodied Agent (2023)](https://arxiv.org/abs/2305.16291)
- [Agent Skills Survey (Feb 2026)](https://arxiv.org/html/2602.12430)
- [A-MEM: Agentic Memory for LLM Agents (NeurIPS 2025)](https://arxiv.org/abs/2502.12110)
- [Mastra Observational Memory (Feb 2026)](https://mastra.ai/blog/observational-memory)
- [Mastra Observational Memory — Research](https://mastra.ai/research/observational-memory)
- [Memory in the Age of AI Agents Survey (Dec 2025)](https://arxiv.org/abs/2512.13564)
- [Observational Memory Cuts AI Agent Costs 10x — VentureBeat](https://venturebeat.com/data/observational-memory-cuts-ai-agent-costs-10x-and-outscores-rag-on-long)
- [GraphRAG Dataflow — Microsoft](https://microsoft.github.io/graphrag/index/default_dataflow/)
