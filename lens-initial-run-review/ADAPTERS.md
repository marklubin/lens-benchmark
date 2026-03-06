## Adapters Evaluated

LENS evaluates memory systems through a standardized adapter interface. Each adapter implements three core operations: `ingest()` (store an episode), `search()` (find relevant information), and `prepare()` (optional pre-query processing at checkpoints). The runner streams episodes into the adapter and tests the agent at checkpoints. All adapters expose the same tools to the agent — `memory_search`, `memory_retrieve`, and `memory_capabilities` — so differences in score reflect differences in how information is stored, indexed, and retrieved.

### Null Adapter

**Registry name:** `null`
**Type:** Baseline (no memory)
**Source:** `src/lens/adapters/null.py`

The null adapter stores nothing and returns nothing. It serves as the floor — any memory system that fails to beat null is providing negative value. The agent receives the question but has no tools to search or retrieve; it can only reason from the question text itself.

- **Ingest:** No-op. The method accepts all parameters (`episode_id`, `scope_id`, `timestamp`, `text`, `meta`) and discards them.
- **Search:** Returns an empty list unconditionally.
- **Retrieve:** Returns `None` for any `ref_id`.
- **Dependencies:** None. The adapter imports only `lens.adapters.base` types and the registry decorator. No external packages, no network calls, no filesystem access.
- **Scopes evaluated:** All 12 (S01 through S12), both 8k and 16k token budgets. Scores 0.000 composite everywhere because the evidence grounding gate requires citations — with no memory to search, the agent cannot produce any.
- **Issues:** None. It does exactly what it is designed to do.

The null adapter's role in the benchmark is structural, not aspirational. Any adapter that scores at or below null is actively harmful — the agent would be better off with no memory system at all. In practice, several early adapter configurations failed to clear this bar on specific question types, which validated null as a meaningful floor rather than a trivial one.

---

### SQLite Chunked-Hybrid

**Registry name:** `sqlite-chunked-hybrid`
**Type:** Lightweight retrieval (no containers, no external services)
**Source:** `src/lens/adapters/sqlite_variants.py` (class `SQLiteChunkedHybridAdapter`, line 1084)

The top-performing adapter across nearly every benchmark phase. Episodes are split into section-based chunks of approximately 150 words, each independently embedded via an OpenAI-compatible embedding API (GTE-ModernBERT-base, served on Modal). Chunks and full episodes are stored in an in-process SQLite database. At search time, two retrieval signals are computed independently and fused via Reciprocal Rank Fusion (RRF): BM25 full-text search over the FTS5 index (operating at episode level) and cosine similarity over chunk embeddings (operating at chunk level).

#### Ingest

1. The full episode text is inserted into the `episodes` table. An FTS5 trigger automatically indexes it for BM25 search.
2. The episode is split into chunks using section-based chunking: the text is divided on markdown `###` headers, the date header from the first line is prepended to every chunk for temporal context, and sections exceeding 150 words are further split at paragraph boundaries. Chunks shorter than 20 words are discarded as noise.
3. Each chunk is embedded via a single batched API call.
4. Chunks with their embeddings are stored in the `chunks` table, keyed by `{episode_id}_c{index}`.

#### Search

The adapter runs two independent searches against the same query, then merges the results:

1. **BM25 (keyword):** The query is escaped for FTS5 syntax (each token quoted and joined with OR) and matched against the full-text index. This excels at finding exact terms like service names, metric labels, and error codes.
2. **Embedding (semantic):** The query is embedded and compared via cosine similarity against all stored chunk embeddings. This captures conceptual similarity even when the exact terms differ.
3. **RRF fusion:** Results from both lists are merged using Reciprocal Rank Fusion with `k=60`. Each document's RRF score is `1/(k + rank_a) + 1/(k + rank_b)`, where documents absent from a list receive a rank of `len(list) + 1`. The merged list is deduplicated by episode ID and truncated to the requested limit (default 7).

The agent receives chunk-level excerpts with `source_episode_id` in the metadata, allowing it to retrieve the full episode when a chunk looks relevant.

#### Retrieve

Retrieval checks the `chunks` table first (by `chunk_id`), then falls back to the `episodes` table (by `episode_id`). This dual-lookup means both search-result ref_ids (chunk-level) and direct episode references (from FTS or from citations) resolve correctly.

#### Extra Tool: `batch_retrieve`

The adapter exposes a `batch_retrieve` tool that fetches multiple documents by ref_id in a single call. This was the single biggest efficiency win in the project. Without it, the agent made roughly 41 individual `memory_retrieve` tool calls per question, burning most of its token budget on tool-call overhead. With `batch_retrieve`, the agent typically makes 3 to 4 calls per question: one search, one batch retrieve, and one or two follow-up searches if needed.

#### Dependencies

No containers, no external databases, no background processes. The adapter runs entirely in-process using Python's stdlib `sqlite3` module. The only external dependency is an embedding API endpoint, configured via `LENS_EMBED_BASE_URL` and `LENS_EMBED_MODEL` environment variables.

#### Scopes Evaluated

All 12 scopes (S01 through S12), both 8k and 16k token budgets, across static and Modal evaluation drivers. Over 200 runs completed without a single adapter-level crash.

#### Issues

Embedding API rate limits under concurrent load (when running multiple evaluation threads against the same Modal endpoint) caused transient failures early in Phase 5. The fix was micro-batching: chunks from a single episode are embedded in one API call rather than individually. After that fix, the adapter has been completely stable.

#### Performance

Leads or ties for first place in every benchmark phase where it was evaluated:

- Phase 5 (numeric scopes, Cerebras inference): 0.473 mean composite (8K+16K combined; 0.454 at 8K only), highest of 8 adapters.
- Phase 6 narrative scopes (static driver): 0.805 mean composite across S07 through S09.
- SRS scopes (static driver): 0.600 mean composite across S10 through S12.

The consistent finding is that simple hybrid retrieval — BM25 for lexical precision, embeddings for semantic recall, RRF to merge — outperforms every dedicated memory system tested. No graph database, no entity extraction pipeline, and no LLM-in-the-loop processing required.

#### Example I/O

**Search request:**
```
memory_search(query="geo-lookup latency trend")
```

**Internal pipeline:**
```
BM25 returns:   [ep_012 (rank 1), ep_015 (rank 2), ep_009 (rank 3), ...]
Embedding returns: [ep_015 (rank 1), ep_012 (rank 2), ep_018 (rank 3), ...]
RRF fusion:     [ep_015 (score 0.033), ep_012 (score 0.033), ep_009 (score 0.020), ...]
```

**Agent receives:** A list of chunk excerpts (up to 500 characters each) with `ref_id`, `score`, and `metadata` containing `source_episode_id`. The agent then calls `batch_retrieve` with the ref_ids it wants to read in full.

---

### Compaction

**Registry name:** `compaction`
**Type:** Summarization baseline
**Source:** `src/lens/adapters/compaction.py`

The simplest possible "smart" baseline. All episodes are buffered in memory during ingest (instant, no processing). At each checkpoint, `prepare()` calls an LLM to compress everything into a single summary document. The agent then searches and retrieves from that summary rather than individual episodes.

#### Ingest

No-op in terms of processing. The episode dictionary (id, scope, timestamp, text, metadata) is appended to an in-memory list. There is no embedding, no indexing, and no I/O. This makes compaction the fastest adapter at ingest time — it defers all computation to the `prepare()` step.

#### Prepare

At each checkpoint, the adapter builds a single prompt containing all buffered episodes in chronological order, formatted as `[episode_id] timestamp: text`. A system prompt instructs the LLM to compress the episodes into a summary that preserves numeric values exactly, cites episode IDs for specific data points, and focuses on cross-episode patterns rather than repeating each entry.

If the total episode text fits within the context window (configurable via `COMPACTION_MAX_INPUT_CHARS`, default ~200K tokens), summarization happens in a single LLM call. If it exceeds the window, the adapter uses incremental batching: episodes are split into batches that fit, each batch is summarized independently, and a final merge call combines the batch summaries into one coherent document. The merge prompt instructs the LLM to preserve all citations, focus on cross-batch patterns, and remove redundancy.

The LLM model is controlled by `LENS_LLM_MODEL` (default: `Qwen/Qwen3.5-35B-A3B`), and output length is controlled by `COMPACTION_MAX_TOKENS` (default: 2000).

#### Search

Returns the summary as a single `SearchResult` with `ref_id="compaction_summary"` and `score=1.0`. The search query is ignored — the adapter always returns the same summary regardless of what was asked. If no summary exists but episodes are buffered (e.g., `prepare()` has not been called), it falls back to returning individual episode snippets.

#### Retrieve

Supports three kinds of ref_ids:
- `"compaction_summary"` — returns the full summary text.
- `"compaction_fallback"` — returns all episodes concatenated (emergency fallback).
- Any original episode ID (e.g., `scope_01_ep_005`) — returns that episode's raw text, enabling the agent to verify citations from the summary against source material.

#### Dependencies

Requires the `openai` Python package for LLM API calls. No containers, no databases. API credentials are resolved from `LENS_LLM_API_KEY` / `OPENAI_API_KEY` and `LENS_LLM_API_BASE` / `OPENAI_BASE_URL` environment variables.

#### Scopes Evaluated

Scopes S01 through S06 (numeric/structured data scopes), both 8k and 16k budgets. The adapter was not run on narrative scopes (S07 through S09) because those use long-form episodes (~5,500 words each). With 40 narrative episodes, the total input exceeds 220,000 words — well beyond the 65K token context window available to the compaction LLM, even with incremental batching producing summaries too long to merge effectively.

#### Issues

**Phase 1-2 dominance was an artifact of corpus size.** In early benchmark phases with only 30 episodes (~14K tokens) and no distractors, compaction achieved NBA (Normalized Benchmark Accuracy) of 0.790 — it simply summarized everything in one pass and the agent had all the information it needed. This made it appear to be the best strategy.

When the benchmark expanded to 120 episodes (~84K tokens) with 90 distractor episodes in Phase 3, compaction collapsed to 0.294 answer quality. The distractor episodes — topically plausible but containing no signal — diluted the summary. The LLM could not reliably distinguish signal from noise at compression time, so key facts were lost or buried. This was actually a validation of the experimental design: the distractors successfully separated real memory systems from those that simply fit everything in context.

**Cannot scale to narrative-length episodes.** The fundamental constraint is that compaction requires fitting all episodes (or their intermediate summaries) into a single LLM context window at `prepare()` time. As corpus size grows, the summarization degrades — either from information loss during compression or from cascading errors across multiple batching rounds.

#### Notable

Compaction is embarrassingly simple and embarrassingly effective on small corpora. It represents the strategy that every complex memory system must beat: "just summarize everything and search the summary." Most dedicated memory systems do not clear this bar until the corpus grows large enough that the summary strategy breaks down. The point at which compaction fails is, in effect, the point at which the benchmark starts measuring something real.

#### Example I/O

**Prepare step (single-pass, 30 episodes):**
```
System: You are a memory compaction agent...
User: EPISODES (30 total, 2024-01-01 to 2024-01-30):
  [scope_01_ep_001] 2024-01-01: Service metrics dashboard...
  [scope_01_ep_002] 2024-01-02: ...
  ...
  COMPRESSION OBJECTIVE: Compress these episodes into a summary...

LLM output (2000 tokens):
  Infrastructure monitoring summary spanning 2024-01-01 to 2024-01-30.
  DNS resolution latency increased from 12ms [scope_01_ep_003] to 340ms
  [scope_01_ep_018]. Service-B error rate: 0.02% [scope_01_ep_005] →
  4.7% [scope_01_ep_022]...
```

**Search request (any query):**
```
memory_search(query="root cause of service degradation")
→ Returns: [SearchResult(ref_id="compaction_summary", text="Infrastructure monitoring summary...", score=1.0)]
```

The agent then calls `memory_retrieve(ref_id="compaction_summary")` to read the full summary, and optionally retrieves cited episode IDs to verify specific claims.

---

## Letta-Family Adapters

The LENS benchmark evaluates four Letta-based memory adapters (plus one ablation variant), all built on the Letta platform (formerly MemGPT). Letta provides containerized agent infrastructure with archival memory (vector-searchable passage storage) and core memory (small, always-in-context blocks the agent can read and update). Each adapter explores a different strategy for organizing and consolidating information across episodes.

All Letta adapters share common infrastructure requirements: a Letta server container (port 8283), a custom embedding proxy (`letta_embed_proxy.py`, port 7878) that routes embedding calls to the configured backend, and the `letta-client` Python library. The Letta server runs PostgreSQL internally for metadata and uses a vector index for archival passage retrieval.

**Cross-phase comparability caveat:** Letta scores below mix different agent LLMs across phases — GPT-OSS-120B (numeric/Phase 1), Claude Sonnet (static-driver narrative and SRS), and Qwen3.5-35B-A3B (dynamic-driver narrative and SRS). Cross-phase means should not be interpreted as controlled comparisons. Within-phase rankings (where the agent LLM is held constant) are the valid comparison unit.

Across the full evaluation, standard Letta achieved the highest mean composite score in the family (0.606 across 12 scopes, mixed LLMs), followed by Letta-Sleepy (0.572), Letta-V4 (0.375 across 6 scopes), and Letta-Entity (0.349 across 6 scopes). A recurring finding across all four adapters is that the quality of the LLM powering the Letta agent matters more than the memory architecture: Sonnet-era runs consistently scored 2-3x higher than Qwen-era runs on the same adapter and scope.


### Letta (Standard)

**Registry name:** `letta`
**Adapter type:** Vector-search archival memory

Standard Letta stores episodes as archival passages in a vector database and retrieves them via semantic search. One Letta agent is created per scope, serving as the isolation and namespace unit. The agent is given a neutral persona that explicitly avoids editorializing.

**Ingest.** Each episode is stored deterministically via `passages.create()`, which writes the full episode text (prefixed with the episode ID and timestamp) directly into Letta's archival vector store. No LLM is involved during ingest.

**Search.** The LENS agent harness calls `passages.search()` with the query, which performs semantic vector search over the stored archival passages. Results are parsed to extract episode IDs from the `[ep_id]` prefix format. The adapter also exposes a `batch_retrieve` extended tool that lets the agent fetch multiple full episodes in a single call rather than one at a time.

**Prepare.** No-op. There is no consolidation or preprocessing step between ingestion and question-answering.

**Evaluated on:** All 12 scopes (S01-S12), using both static and dynamic drivers. On numeric scopes (Phase 1), the Letta internal agent LLM was Sonnet (for the "Sonnet comparison" runs) or used the same GPT-OSS-120B agent LLM. On narrative and SRS scopes (Phase 2/3, dynamic driver), the Letta internal LLM was Qwen3.5-35B-A3B via openai-proxy → Modal (the default; `LETTA_LLM_MODEL` was never set in sweep scripts).

**Results.** Standard Letta achieved 0.606 mean composite across all 12 scopes, the highest in the Letta family and competitive with the overall benchmark leaders. On scope 01 alone it scored 0.5308 (formerly SOTA before multi-scope evaluation). It achieved perfect evidence grounding (1.0) and the highest insight depth (0.8750) of any adapter on scope 01. Its strength lies in simplicity: raw passage storage preserves all fine-grained evidence, and semantic search over those passages provides reliable retrieval without the information loss that comes from summarization or compression.

**Known issues.** The embedding proxy routing required a custom `letta_embed_proxy.py` to redirect Letta's embedding calls to the correct backend, since Letta's built-in embedding configuration does not natively support all providers used in the benchmark. The default httpx timeout of 60 seconds was too short for Together AI on large contexts (120-180s latency); this was increased to 300 seconds. The Letta server occasionally returns passages as objects versus lists, so the adapter defensively handles both formats.


### Letta-Sleepy

**Registry name:** `letta-sleepy`
**Adapter type:** Vector-search archival memory with sleep-time consolidation

Letta-Sleepy extends standard Letta with Letta's native sleep-time compute feature (`enable_sleeptime=True`). A background sleep agent automatically consolidates the primary agent's core memory blocks at a configurable frequency. The hypothesis: pre-organizing cross-episode patterns during a consolidation phase should help the Q&A agent locate relevant evidence more efficiently.

**Ingest.** Two-pronged approach. First, `passages.create()` stores the full episode text in archival memory, guaranteeing retrieval regardless of agent behavior. Second, a shortened summary of the episode is sent to the agent via `messages.create()`, which triggers core memory updates and activates the sleep-time agent for background consolidation. After each episode, the conversation buffer is reset to its initial state (preserving core memory and archival passages) to prevent context overflow across episodes.

**Search.** Returns two types of results. First, the sleep-consolidated core memory blocks (persona and any sleep-written blocks) are included as the top search result with score 1.0, providing the agent with the synthesized navigational context. Second, standard semantic search over archival passages provides the raw episode evidence.

**Prepare.** At each checkpoint, sends a consolidation prompt asking the agent to review archival memory and update core memory blocks with key patterns, trends, and developments. The conversation buffer is then reset.

**Evaluated on:** All 12 scopes (S01-S12).

**Results.** Letta-Sleepy scored 0.572 mean composite across 12 scopes, consistently second in the Letta family. Sleep-time consolidation adds roughly 5 percentage points on narrative scopes but slightly hurts on numeric scopes. Four consolidation prompt variants were tested on scope 01: V1 (comprehensive summary, 0.4290) and V2 (actionable filter, 0.4596) both performed worse than standard Letta (0.5308), while V3 (delta/causal synthesis, 0.5776) was the only variant that helped. V3's advantage correlates inversely with standard Letta's base score, meaning sleep consolidation is compensatory for weak retrieval rather than universally additive.

**Known issues.** The original implementation relied on the LLM agent calling `insert_archival_memory` during ingest, which proved unreliable — Sonnet simply would not call the tool when prompted. The fix was the two-pronged approach: deterministic `passages.create()` for guaranteed storage plus a shortened agent message for consolidation context. This change dropped run time from approximately 3 hours to approximately 20 minutes per scope and brought scores from 0.0 (gated by evidence grounding failures) to 0.44-0.61. The sleep-time frequency is set via direct PostgreSQL manipulation inside the Letta container (`podman exec` with `psql`), since the Letta REST API does not expose this configuration.


### Letta-V4 (Core Memory)

**Registry name:** `letta-v4`
**Adapter type:** Three-agent architecture with structured core memory blocks

Letta-V4 takes a fundamentally different approach from the passage-storage adapters. Instead of relying on raw episode retrieval, it uses an Ingest agent to process each episode and maintain four structured core memory blocks: patterns (5K chars), hypotheses (5K), entities (5K), and events (5K). A standalone Sleep agent consolidates these blocks at checkpoints. A separate Q&A agent answers questions using the shared core memory blocks plus archival search for supporting detail.

**Ingest.** Episode text is stored deterministically via `passages.create()` in both the Ingest and Q&A agents' archival stores (Letta archival memory is per-agent, so the Q&A agent needs its own copy). No agent message is sent during ingest; core memory updates happen exclusively at checkpoints via `prepare()` to avoid overloading the Letta server.

**Search.** The adapter's `search()` method returns an empty list. All question-answering goes through the dedicated Q&A agent via `answer_question()`, which sends the question as a message. The Q&A agent uses its core memory blocks (containing synthesized patterns and entity histories) as starting context and calls `archival_memory_search` internally to find supporting evidence.

**Prepare.** Two-step consolidation. First, the Ingest agent searches its archival memory and updates the shared core memory blocks with key findings. Second, the Sleep agent reconciles information across blocks: forming hypotheses, promoting recurring events to patterns, pruning stale entities, and condensing overfull blocks.

**Evaluated on:** Scopes S07-S12 (narrative and SRS scopes). S07-S09 used Anthropic Sonnet as Letta's internal LLM; S10-S12 used Qwen3.5-35B-A3B.

**Results.** Letta-V4 scored 0.375 mean composite across 6 scopes, substantially below standard Letta (0.606). The Sonnet-era scopes (S07-S09) scored 0.586, 0.489, and 0.612, while the Qwen-era scopes (S10-S12) scored 0.282, 0.281, and 0.000. The S12 score of 0.0 was caused by a complete evidence grounding gate failure: Qwen, operating through Letta's tool interface, did not produce episode citations in its answers.

**Known issues.** Core memory compression loses the fine-grained evidence that the scorer requires. Evidence coverage averaged only 0.169, because the Q&A agent cites episodes but not the specific evidence references needed for scoring. The four core memory blocks (20K chars total) cannot hold the detail present in 120 episodes of raw text. Late-episode ingest latency also increased significantly: episodes 29-35 of S07 required 100-130 seconds each (versus 30-50 seconds early) as the growing Letta conversation context slowed agent processing.

An ablation variant, **Letta-V4-NoSleep** (registry name `letta-v4-nosleep`), skips the sleep consolidation phase entirely. It subclasses Letta-V4 and overrides `prepare()` to be a no-op, testing whether the value of V4 comes from the three-agent compression architecture or from the active consolidation step.


### Letta-Entity

**Registry name:** `letta-entity`
**Adapter type:** Two-agent architecture with dynamic entity-focused core memory

Letta-Entity is a variation on the V4 approach that focuses on entity tracking rather than general pattern detection. An Ingest agent maintains a single large "entities" core memory block (20K chars) that tracks key entities, their evolving states, relationships, and evidence citations. A Q&A agent shares this block and uses archival search for detailed evidence retrieval.

**Ingest.** Two-pronged, following the lesson learned from Letta-Sleepy. First, `passages.create()` stores the full episode text in both the Ingest and Q&A agents' archival stores. Second, a truncated summary (first 2000 characters) is sent to the Ingest agent, which updates the entities block using `core_memory_replace` to maintain structured entity entries with type, status, history, relationships, and evidence citations. The conversation buffer is reset after each episode.

**Search.** Returns an empty list. All question-answering goes through the Q&A agent via `answer_question()`, which uses the shared entities block for entity-aware context and `archival_memory_search` for episode retrieval.

**Prepare.** No-op. Unlike Letta-V4, there is no separate consolidation step; the entity tracker is updated incrementally during each ingest call.

**Evaluated on:** Scopes S07-S12 (narrative and SRS scopes).

**Results.** Letta-Entity scored 0.349 mean composite across 6 scopes, the lowest in the Letta family. Individual scope scores varied widely: S08 (0.550) and S12 (0.559) scored well (these were Sonnet-era runs with correct citations), while S09 (0.228) and S11 (0.000) were significantly degraded.

**Known issues.** A systematic citation prefix doubling bug caused evidence grounding gate failures on multiple scopes. The adapter produced citations like `ep_corporate_acquisition_08_ep_001` instead of `corporate_acquisition_08_ep_001`, where an `ep_` prefix was being doubled. The `_extract_inline_refs()` function (shared with Letta-V4) includes cleanup logic to strip spurious leading `ep_` prefixes, but the entity adapter's specific citation patterns triggered the bug in ways the cleanup did not fully handle. The entity tracking approach also adds architectural complexity (two agents, a 20K-char entity block, per-episode LLM calls during ingest) without improving retrieval quality over standard Letta's simple passage storage.


### Cross-Adapter Analysis

The Letta family evaluation produced a clear hierarchy and several findings relevant to memory system design:

1. **Simplicity wins.** Standard Letta's raw passage storage with vector search (0.606) outperformed every structured memory variant. Direct search over unmodified episode text preserves the fine-grained evidence that the LENS scorer requires.

2. **Consolidation is compensatory, not additive.** Letta-Sleepy's sleep-time consolidation helps most when base retrieval is weak (low-signal scopes) and slightly hurts when retrieval is already strong. The delta/causal synthesis framing (V3) is the only variant that consistently helps; comprehensive summaries and actionable filters both degrade performance.

3. **Core memory compression is lossy in the wrong way.** Both V4 and Entity adapters attempt to build evolving internal representations during ingest. These representations abstract away from source episodes, which causes evidence coverage and evidence grounding failures during scoring. The 20K-char budget for core memory blocks cannot substitute for searchable access to 120 full episodes.

4. **LLM quality dominates architecture choice.** Across all four adapters, the quality of the LLM powering the Letta agent (Sonnet versus Qwen) had a larger effect on scores than the choice of memory architecture. This suggests that investing in agent LLM capability yields higher returns than investing in memory system complexity.

5. **Deterministic storage is non-negotiable.** The lesson from the Letta-Sleepy archival fix applies broadly: never rely on an LLM agent to call a specific tool for critical data operations. All four adapters now use `passages.create()` for guaranteed episode storage, with agent messages reserved for optional processing and consolidation.

---

# Heavy Adapters and Novel Architectures

This section covers the external memory systems and novel architectures evaluated in LENS. These adapters require either containerized infrastructure (Qdrant, FalkorDB, PostgreSQL), LLM-powered ingestion pipelines, or both. They represent the state of the art in agent memory — and collectively demonstrate how much operational complexity these systems add relative to their retrieval quality.

---

## Mem0-Raw

**Registry name:** `mem0-raw` | **Type:** Vector search via Mem0 platform (containerized)

Mem0 is a memory platform designed for chatbot personalization. The "raw" variant bypasses Mem0's extraction LLM — which is hardcoded for personal facts like names and preferences — and stores episodes directly as vectors with `infer=False`.

**Architecture.** Episodes are stored via `Memory.add(text, infer=False)`, which writes vectors directly to a Qdrant backend. The `infer=False` flag is critical: without it, Mem0's extraction prompt begins "You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences" and finds zero useful facts in operational metrics data, storing nothing. Search is pure cosine similarity over Qdrant vectors.

**Dependencies.** Qdrant container (port 6333), OpenAI-compatible embedding API. Mem0 also requires an LLM client configuration even when `infer=False` — it crashes on startup without one.

**Results.** 0.349 mean composite across scopes S01-S06. Mid-pack but functional. The vector search itself works fine; the extraction layer is the problem.

**Issues.** The companion `mem0-extract` adapter scored 0.000 — Mem0's extraction LLM correctly identifies zero personal facts in "p99: 320ms, pool_util: 87%" and stores nothing. There is no configuration to change the extraction prompt without forking the library. Additionally, Together AI rejects the `dimensions` parameter that Mem0's embedder sends, requiring a monkey-patch (`MEM0_EMBED_NO_DIMS=1`). The `mem0.search()` return shape is unstable across versions, alternating between `{"results": [...]}` and a bare list. Running a full Qdrant container for cosine similarity over a few hundred embeddings is architectural overkill.

---

## Cognee (GraphRAG)

**Registry name:** `cognee` | **Type:** Graph-aware knowledge pipeline with embedded databases

Cognee builds a knowledge graph from ingested episodes using LLM-based entity extraction, then retrieves via vector search over graph-aware chunks. It uses LanceDB (vectors), Kuzu (graph), and SQLite internally — no external containers needed.

**Architecture.** Ingestion stores raw episode text via `cognee.add()`. The expensive step is `cognee.cognify()`, which runs LLM entity extraction, relationship identification, graph construction, and summarization. Search queries the CHUNKS collection in LanceDB; graph structure informs chunk boundaries and relationships.

**Dependencies.** Cognee library (async-only, requiring a threading bridge), OpenAI-compatible LLM and embedding endpoints. All databases are embedded — no containers.

**Results.** When it works (scope 01 in Phase 5), Cognee achieves 0.432 mean composite with the best evidence_coverage (0.6319) of any system. The GraphRAG chunking and knowledge graph indexing produce particularly retrievable chunk representations. However, scopes 02-06 all timed out in practice, making Cognee scope-01-only for realistic use.

**Issues (six separate monkey-patches required).**

1. `cognify()` takes 30-60+ minutes per scope on Together AI serverless. Entity extraction via Llama-3.3-70B is extremely slow.
2. Cognee 0.5.2+ wraps search results in `{"dataset_id": UUID, "search_result": [...]}` instead of a flat list. The adapter silently returned empty results, scoring 0.000 until debugged.
3. Concurrent `cognify()` and search deadlock the embedded Kuzu graph database. ACL had to be disabled entirely.
4. Together AI rejects the `dimensions` parameter from LiteLLM's `aembedding()`. Required monkey-patching.
5. Cognee's tokenizer does not recognize Together AI model names, causing a `tiktoken` KeyError. Required patching `tiktoken.encoding_for_model()`.
6. LLM token limits during extraction produce partial entity graphs with no error raised.

---

## Graphiti

**Registry name:** `graphiti` | **Type:** Bi-temporal knowledge graph with FalkorDB backend

Graphiti builds a temporal knowledge graph with bi-temporal edge invalidation — edges carry both "valid time" (when the fact was true in the world) and "transaction time" (when it was recorded). Entity extraction via LLM populates FalkorDB, a Redis-compatible graph database.

**Architecture.** Episodes are stored and processed via LLM entity extraction, which builds a knowledge graph in FalkorDB with timestamped entities and relationships. Search uses EDGE_HYBRID_SEARCH — BM25 plus cosine similarity on graph edges, with episode mentions mapped back to source episodes.

**Dependencies.** FalkorDB container (port 6379), OpenAI-compatible LLM and embedding APIs. Async-only library requiring a threading bridge with a dedicated daemon thread hosting its own event loop.

**Results.** 0.426 mean composite on 3 completed scopes (S01, S02, S06). Perfect budget_compliance (1.0), evidence_grounding (1.0), and strong reasoning_quality (0.917). However, answer_quality (0.675) and longitudinal_advantage (-0.337) are weaker than Letta, suggesting the graph structure adds overhead without improving temporal synthesis relative to simpler passage retrieval.

**Issues.** Together AI rejects embedding batches larger than 1MB (HTTP 413), fixed with 20-item batch chunking via a monkey-patched `_ChunkedEmbedder`. Entity extraction is slow enough that scopes 03-05 timed out entirely, leaving only 6 of 12 planned Phase 5 runs completed. RediSearch underscore escaping also required patching.

---

## GraphRAG-Light

**Registry name:** `graphrag-light` | **Type:** Lightweight entity graph with RRF fusion (no external containers)

A lightweight alternative to Cognee and Graphiti that builds an entity graph using NetworkX (in-process) with LLM-based entity extraction. Search fuses three signals via Reciprocal Rank Fusion: BM25 (FTS5), entity embedding similarity, and graph 1-hop neighborhood expansion.

**Architecture.** Episodes are stored in SQLite during ingestion — fast, no LLM calls required. The prepare step runs LLM entity extraction, stores entities in a NetworkX graph with embeddings, and performs LLM-based entity deduplication. Search combines three signals through RRF: FTS5 BM25 keyword matching, entity embedding cosine similarity, and graph 1-hop neighborhood expansion that pulls in related entities and their source episodes.

**Dependencies.** NetworkX (in-process), OpenAI-compatible LLM and embedding APIs. No containers.

**Results.** Most resilient adapter to driver change — only -0.023 drop from static to modal driver, compared to -0.101 for sqlite-chunked-hybrid. Became the top-ranked adapter with the modal driver at 0.555 mean composite. Graph-based retrieval may produce more semantically coherent results that compensate for imprecise agent queries.

**Issues.** Entity deduplication had a KeyError (fixed in session 31): `_llm_dedup_check()` returned display-form names (e.g., "Elm Street") but the graph stores normalized lowercase keys. Fix was to normalize returned names and verify existence before use. Entity extraction parse failures are frequent but handled gracefully.

---

## Hierarchical

**Registry name:** `hierarchical` / `hierarchical-hybrid` | **Type:** Multi-level summary index

Builds a hierarchical summary structure across four levels: L0 (raw episodes), L1 (per-episode summaries), L2 (group summaries over clusters of episodes), and L3 (global summary). The hybrid variant adds FTS5 and embedding search across all levels.

**Architecture.** Episodes are buffered during ingestion. The prepare step runs LLM summarization at each level, processing incrementally so only new episodes trigger work. The hybrid variant searches across all summary levels using FTS5 keyword matching and embedding similarity. The non-hybrid variant simply returns the global summary.

**Dependencies.** OpenAI-compatible LLM and (for the hybrid variant) embedding APIs. No containers.

**Results.** Competitive at 0.574 mean composite (static driver, SRS scopes) and 0.519 with the modal driver. The multi-level approach provides good coverage without the fragility of graph-based systems.

**Issues.** None significant. Clean implementation.

---

## Hopping

**Registry name:** `hopping` / `hopping-rag` / `hopping-hybrid` | **Type:** Incremental rolling summary

Maintains a rolling summary that gets updated as new episodes arrive. When the buffer exceeds a token threshold, new episodes are merged into the existing summary via LLM. The hybrid variant adds RAG-style search over raw episodes alongside the rolling summary.

**Architecture.** Episodes are buffered during ingestion. When the buffer limit is reached, a compaction step merges new episodes into the existing rolling summary via LLM. Search returns the rolling summary, or in the hybrid variant, combines semantic search across raw episodes with the summary.

**Dependencies.** OpenAI-compatible LLM and (for the hybrid variant) embedding APIs. No containers.

**Results.** The only adapter that improved with the modal/dynamic driver (+0.064). The rolling-summary approach may mesh better with an agent's iterative query-refine pattern than with pre-computed single-shot queries. Generally underperforms at 0.462 mean composite (static SRS).

**Issues.** The rolling summary approach loses fine-grained evidence across longer episode sequences. As new information is merged, specific metrics and timestamps get compressed away, making it difficult to cite precise evidence.

---

## Triad V1

**Registry name:** `triadv1-panel` / `triadv1-pairs` | **Type:** Multi-agent object-store architecture

A novel approach using four specialized agents — Entity, Relation, Value, and Event — each maintaining a structured "notebook" conforming to a 5-field meta-schema (identity, schema, interface, state, lifecycle). On search, all agents are consulted and their outputs synthesized.

**Architecture.** During ingestion, each episode is sent to all four agents in parallel. Each agent updates its domain-specific notebook — a structured text store tracking its area of concern. The panel variant queries all four agents in parallel during search. The pairs variant runs 6-way cross-references between agent pairs, then synthesizes the results.

**Dependencies.** OpenAI-compatible LLM (Modal vLLM in practice). Threading for parallel agent calls.

**Results.** Scores 0.000 due to a fundamental citation format mismatch: agents return synthetic `notebook-*` references instead of episode IDs, failing the evidence_grounding hard gate. Underlying answer quality was actually non-trivial (0.23-0.35 range) but gated to zero by the missing citations.

**Issues.** Originally hung indefinitely. Fixed with timeouts (120s per agent call), notebook caps (8K characters), and episode truncation (2K characters). Runs now complete in under 15 minutes. The citation format is a structural mismatch with LENS's evidence_grounding requirement and would need architectural changes — not just configuration — to produce valid episode citations.

---

## Hindsight (Removed)

**Registry name:** `hindsight` | **Type:** TEMPR retrieval engine (semantic + BM25 + graph + temporal, RRF-fused)

Hindsight uses TEMPR (Temporal Enhanced Memory Pattern Retrieval) — a four-signal retrieval engine fusing semantic similarity, BM25, graph-based entity relationships, and temporal proximity via Reciprocal Rank Fusion. It also exposes a `memory_reflect` tool for native longitudinal synthesis.

**Architecture.** Episodes are stored one at a time via `client.retain()`, which triggers entity extraction on every call (20-100 seconds per episode). Search runs TEMPR RRF-fused retrieval across all four signals.

**Dependencies.** Hindsight container (17.3 GB image containing PostgreSQL and the TEMPR engine). Eight environment variables with non-obvious naming — `HINDSIGHT_API_EMBEDDINGS_OPENAI_*` requires the `_OPENAI_` infix even when using Together AI as the backend.

**Results.** NBA 0.168, statistically indistinguishable from the null adapter (0.150). Evidence_coverage = 0.000 across all constrained runs.

**Issues.**

- `retain()` takes 20-100 seconds per episode due to entity extraction. Thirty episodes at 50 seconds each means 25 minutes of overhead per checkpoint before the agent can even begin answering.
- `retain_batch()` sends all episodes to the embedding API at once, triggering HTTP 413. Had to revert to sequential calls.
- Budget compliance collapse: 19/24 violations (0.208). Consolidation overhead consumes the agent's entire token budget before it can answer questions.
- Hindsight reformats ingested text internally, destroying the original episode content and making it impossible to cite evidence. Evidence_coverage = 0.167.
- TEMPR's RRF-fused scores are unusably low (~0.03), causing the agent to interpret results as low confidence and search repeatedly, burning more tokens in a futile loop.
- The native `memory_reflect` tool for longitudinal synthesis exists but is rarely invoked by the agent.
- Environment variable naming cost hours of debugging: `HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai` requires keys under `HINDSIGHT_API_EMBEDDINGS_OPENAI_*` with the `_OPENAI_` infix.

**Status: Removed from evaluation (session 19).** A 17.3 GB container image delivering zero demonstrated value over doing nothing.
