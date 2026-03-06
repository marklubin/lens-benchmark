# LENS Benchmark: Methodology

## What LENS Measures

LENS stands for **Longitudinal Evidence-backed Narrative Signals**. It is a benchmark for evaluating whether an LLM agent equipped with a memory system can synthesize conclusions from evidence scattered across many sequential episodes.

Most memory benchmarks inadvertently test retrieval rather than reasoning. If a single episode contains enough information to answer a benchmark question, then the system under test only needs to find that episode — it never has to integrate evidence across time. LENS is designed around a stricter requirement: signal emerges only from the *progression* across episodes. No individual episode is sufficient on its own. A correct answer demands that the agent has tracked subtle changes — a metric climbing from 120ms to 847ms over ten episodes, for example — and drawn inferences from the trajectory, not from any single data point.

This distinction is the benchmark's reason for existing. If a single episode can answer a synthesis question, the benchmark is worthless.

## Dataset Generation

### The Contamination Problem

The central challenge in generating benchmark datasets with LLMs is contamination. When a single LLM generates episodes while knowing the storyline, it editorializes. Instead of writing "p99: 847ms," it writes "latency is increasingly concerning at 847ms." Each episode becomes a standalone causal analysis that leaks the answer. A memory system that retrieves just one such episode can pass the benchmark without performing any longitudinal reasoning at all.

### Two-Stage Information-Isolated Pipeline

The fix is a two-stage pipeline built on strict information isolation between the planning and rendering phases.

**Stage 1 — PlanOutline (GPT-5.2, full visibility).** The planner sees the complete scope specification: the narrative arc, key facts, questions, and signal placements. It produces per-episode structured data sheets containing concrete metric values. Signal is encoded exclusively as numeric progressions — never as text commentary or editorial language. The planner decides *what numbers appear where*, but it never writes prose that an end user would read.

**Stage 2 — RenderEpisodes (GPT-4.1-nano, blind to storyline).** The renderer receives only the data sheet for a single episode and a shared formatting context (voice, structure, terminology). It does not see the arc, key facts, signal placements, or questions. Its job is to format the data sheet into a terse log entry. Because it has no knowledge of what constitutes signal, it cannot editorialize. It simply presents the numbers.

This is why it works: a single rendered episode shows "p99: 847ms." That value is meaningful only if you have tracked it climbing from 120ms over the preceding ten episodes. No single episode reveals the trend.

### Distractors

Each scope generates distractors alongside signal episodes. Distractors are format-matched — same voice, same structure, same density — but topically orthogonal. They cover unrelated operational domains with explicitly excluded terms to prevent accidental overlap with the signal narrative. Every scope includes three times as many distractor episodes as signal episodes, creating realistic signal-to-noise separation that memory systems must navigate.

### Pipeline DAG

The generation pipeline is a directed acyclic graph:

```
LoadSpec → PlanOutline → RenderSignalEpisodes + RenderDistractorEpisodes
                                    ↓
                         ResolveQuestions + AuditKeyFacts
                                    ↓
                    Validators: WordCount, ContaminationCheck, NaiveBaseline
                                    ↓
                              SearchIndex (SQLite FTS)
```

### Validation Gates

Every generated dataset must pass four validation gates before it is released:

- **ContaminationCheck.** Measures whether any single episode can answer a synthesis question. The maximum single-episode coverage must remain below 80%. Late-arc episodes inherently converge toward the conclusion, so some coverage is structural and expected — but it must not cross the threshold.
- **NaiveBaseline.** Tests whether a naive LLM with all episodes stuffed into its context window can answer the questions. The pass rate must remain below 50%. If a model can answer correctly just by reading everything at once, the benchmark is too easy and is not testing memory.
- **WordCount.** Each episode must exceed 340 words. Structured metrics — tables of values, configuration dumps, log entries — are denser than narrative prose, and this threshold ensures episodes carry enough substance.
- **KeyFactAudit.** More than 90% of the key facts defined in the scope specification must appear in the generated episodes. This is a keyword-based audit, not a semantic similarity check, ensuring that the raw material for correct answers is present in the corpus.

## Scope Structure

Each benchmark scope is defined by a `spec.yaml` file that encodes a complete evaluation scenario.

### Arc Phases

Every scope follows a five-phase narrative arc:

1. **Baseline** — Normal operations. Metrics are stable and unremarkable. This phase establishes the reference point against which all subsequent changes are measured.
2. **Early Signal** — A subtle anomaly appears. Values begin to shift, but the change is small enough to be ambiguous. Detecting this phase requires attention to trends rather than absolute values.
3. **Red Herring** — A misleading alternate explanation emerges. A plausible but incorrect cause presents itself, testing whether the agent can resist premature conclusions.
4. **Escalation** — The signal becomes unmistakable. Metrics deteriorate clearly and the anomaly can no longer be dismissed as noise.
5. **Root Cause** — The full picture is revealed. Enough evidence has accumulated to identify the underlying cause with confidence.

### Key Facts and Questions

Each scope defines a set of **key facts** — atomic ground-truth claims that scoring is evaluated against. Every key fact specifies the episode where it first appears and the episodes where it is reinforced, ensuring that the evidence trail is deliberate and traceable.

Scopes include **ten question types**, each targeting a different dimension of longitudinal reasoning:

- **Longitudinal** — What trend has emerged across the observation period?
- **Null Hypothesis** — Can the observed pattern be explained by normal variance?
- **Action Recommendation** — Given the evidence trajectory, what should be done?
- **Negative** — What explanation does the evidence rule out?
- **Temporal** — When did the anomaly first become detectable?
- **Counterfactual** — What would have happened if a specific factor were different?
- **Distractor Resistance** — Can the agent avoid being misled by irrelevant episodes?
- **Severity Assessment** — How serious is the situation, given the evidence?
- **Evidence Sufficiency** — Is there enough data to draw a conclusion, or is more needed?
- **Paraphrase** — Can the agent recognize the same question asked differently?

### Distractors in the Spec

Each scope's specification includes distractor themes — topically orthogonal domains with explicit term exclusion lists. This ensures that distractors are structurally indistinguishable from signal episodes but carry no information relevant to the benchmark questions.

### Scope Categories

Three categories of scopes exist, varying in episode count, length, and format:

| Category | Scopes | Signal Episodes | Words per Episode | Format |
|----------|--------|-----------------|-------------------|--------|
| **Numeric** | S01–S06 | 30 | ~500–700 | Terse operational data: metrics, logs, configurations |
| **Narrative** | S07–S09 | 20 | ~5,000 | Rich documents: chat logs, internal memos, corporate communications |
| **SRS (Semantic Retrieval Stress)** | S10–S12 | 20 | ~5,000 | Clinical records, government reports, structured chat transcripts — signal and distractor episodes share entity types to defeat embedding similarity |

All categories use the same five-phase arc and question types. The variation in format and length tests whether memory systems generalize across different kinds of source material.

---

## How the Benchmark Runs

The runner streams episodes into the memory adapter one at a time, in chronological order. At predefined checkpoints (after N episodes), it pauses to test the agent. The adapter interface has two sides:

- **Data loading** (called by the runner, not visible to the agent): `reset()`, `ingest(episode)`, `prepare(checkpoint)`
- **Tools** (exposed to the agent via ToolBridge): `memory_search(query)`, `memory_retrieve(ref_id)`, `memory_capabilities()`

The EpisodeVault stores a parallel copy of all episodes runner-side. Adapters cannot access the vault. This prevents adapters from fabricating episode references — cited refs are validated against the vault.

Budget enforcement per question: max 10 turns, max 20 tool calls (hard stops), plus soft limits on payload size (64 KB), latency (5 s/call), and agent tokens (8K).

### Execution Patterns

Not all adapters follow the same data flow. The benchmark evaluated four fundamentally different patterns for how episodes are stored, how questions are answered, and where LLM calls happen. The following pseudocode shows each pattern as it was actually implemented.

#### Pattern A — Store-and-Retrieve (sqlite-chunked-hybrid, mem0-raw, letta-base)

The simplest pattern. Episodes go directly into storage with no LLM processing. At question time, the LENS agent formulates queries, searches the store, and synthesizes an answer.

```
# Ingest: direct storage, no LLM
for episode in stream:
    adapter.ingest(episode)          # chunk + embed + INSERT (sqlite)
                                     # or passages.create() (letta)
                                     # or Memory.add(infer=False) (mem0)

# At checkpoint: LENS agent drives the loop
adapter.prepare()                    # no-op
agent_llm = GPT-OSS-120B or Qwen3.5-35B-A3B

for question in checkpoint_questions:
    while not done:
        action = agent_llm(question, tool_results_so_far)
        if action == tool_call("memory_search", query):
            results = adapter.search(query)  # BM25 + embedding, or vector search
            tool_results_so_far.append(results)
        elif action == tool_call("memory_retrieve", ref_id):
            doc = adapter.retrieve(ref_id)
            tool_results_so_far.append(doc)
        elif action == final_answer:
            return AgentAnswer(text, refs_cited, tool_calls, tokens)
```

All LLM calls happen in the LENS agent loop. The adapter is a passive store. This is the pattern used by the top-performing adapter (sqlite-chunked-hybrid).

#### Pattern B — Offline Summarization (compaction, hierarchical, hopping)

Episodes are buffered in memory. At each checkpoint, an LLM compresses all episodes into a summary *before* the agent sees anything. The agent searches the pre-built summary rather than raw episodes.

```
# Ingest: buffer only, no LLM
for episode in stream:
    adapter.buffer.append(episode)   # in-memory list, instant

# At checkpoint: LLM builds summary BEFORE agent runs
adapter.prepare():
    # Compaction: one LLM call to summarize everything
    summary = llm("Summarize these episodes:", buffer)

    # Hierarchical: per-episode → per-group → global (N + G + 1 calls)
    L1 = [llm("Summarize:", ep) for ep in new_episodes]
    L2 = [llm("Summarize group:", group) for group in chunk(L1, 5)]
    L3 = llm("Global summary:", L2)

    # Hopping: incremental rolling summary (K calls)
    for batch in new_episode_batches:
        rolling_summary = llm("Merge into summary:", rolling_summary, batch)

# Agent loop runs same as Pattern A, but search() returns the summary
for question in checkpoint_questions:
    while not done:
        action = agent_llm(question, ...)
        if action == tool_call("memory_search", query):
            results = [summary]      # query is effectively ignored
            ...
```

The agent's search query has minimal effect — the summary is always returned. This means the agent can't do targeted retrieval, only reason over the pre-built summary.

#### Pattern C — Offline Graph/Entity Extraction (graphrag-light, cognee, graphiti)

Episodes go to direct storage during ingest. At prepare(), an LLM extracts entities and relationships into a graph structure. The agent's search queries hit the graph at question time.

```
# Ingest: store text, no LLM
for episode in stream:
    adapter.store(episode)           # SQLite FTS (graphrag-light)
                                     # or cognee.add() (cognee)
                                     # or buffer (graphiti)

# At checkpoint: LLM builds knowledge graph
adapter.prepare():
    for episode in pending_episodes:
        entities = llm("Extract entities and relationships:", episode.text)
        graph.add_nodes(entities)
        graph.add_edges(relationships)
    embed(all_entity_descriptions)   # for semantic entity search

# Agent loop: queries hit the graph
for question in checkpoint_questions:
    while not done:
        action = agent_llm(question, ...)
        if action == tool_call("memory_search", query):
            # graphrag-light: 3-signal RRF
            bm25_hits = fts5_search(query)
            entity_hits = embed_similarity(query, entity_embeddings)
            graph_hits = one_hop_neighbors(entity_hits)
            results = rrf_merge(bm25_hits, entity_hits, graph_hits)
            ...
```

The agent's query matters here — it drives both text and entity search. The graph structure provides additional retrieval signals beyond what text search alone offers.

#### Pattern D — Agent Delegation (letta-v4, letta-entity, letta-sleepy)

These adapters contain their own internal LLM agents. The data flow depends on the variant:

**Letta-Sleepy** uses the standard LENS agent loop but augments storage with agent-processed consolidation:

```
# Ingest: deterministic storage + agent processing
for episode in stream:
    passages.create(episode)                    # guaranteed archival storage
    letta_agent.message("Process this episode") # agent updates core memory
    letta_agent.clear_conversation()            # preserve memory, reset chat

# At checkpoint: sleep consolidation
adapter.prepare():
    letta_agent.message("Consolidate patterns across episodes")
    # sleep agent rewrites core memory blocks with cross-episode synthesis

# Agent loop: standard LENS pattern, but search returns consolidated memory
for question in checkpoint_questions:
    while not done:
        action = agent_llm(question, ...)
        if action == tool_call("memory_search", query):
            results = [core_memory_synthesis] + passage_search(query)
            ...
```

**Letta-V4 and Letta-Entity** bypass the LENS agent loop entirely. The benchmark calls `answer_question()` directly on the adapter, which delegates to an internal Letta Q&A agent:

```
# Ingest: deterministic storage + agent processing
for episode in stream:
    passages.create(episode)         # raw text to both ingest + Q&A agents

# At checkpoint (V4 only): two-agent consolidation
adapter.prepare():
    ingest_agent.message("Review recent episodes, update core memory")
    sleep_agent.message("Reconcile patterns across memory blocks")

# Question time: NO LENS agent loop — direct delegation
for question in checkpoint_questions:
    answer = adapter.answer_question(question):
        response = qa_agent.message(question)  # Letta agent searches
                                                # archival + reads core memory
                                                # + formulates answer internally
        refs = extract_episode_refs(response)
        return AgentAnswer(response.text, refs)
```

This is a fundamentally different evaluation mode. The Letta Q&A agent controls its own search strategy, tool usage, and synthesis. The LENS benchmark only sees the final answer and cited references. This means Letta-V4/Entity results reflect the combined quality of Letta's agent loop + the adapter's memory structure, while Pattern A results isolate the memory system from the agent.

### Driver Modes

Two driver modes control who formulates search queries:

- **Dynamic driver (default)**: The LENS agent LLM decides what to search for, how many searches to run, and when to stop. Tests end-to-end agent + memory system performance.
- **Static driver**: Pre-computed query plans from dataset ground truth bypass the agent's query formulation. A fixed set of searches runs for each question, then one LLM call synthesizes the answer. Tests adapter retrieval quality in isolation.

```
# Static driver pseudocode
for question in checkpoint_questions:
    plan = lookup_query_plan(question)         # pre-computed from ground truth
    all_results = []
    for search_query in plan.searches:
        results = adapter.search(search_query)
        all_results.extend(results[:plan.retrieve_top_k])
    answer = llm("Synthesize answer from these results:", all_results, question)
```

The static driver eliminates agent query quality as a variable. When an adapter scores higher with the static driver than the dynamic driver, it means the memory system *can* surface the right evidence — the agent just isn't asking the right questions.


## How Answers Are Scored

Three-tier scoring with a weighted composite:

**Tier 1 — Mechanical metrics** (40% weight total):

- **evidence_grounding** (10%): Did the agent cite real episode IDs that exist in the vault?
- **fact_recall** (10%): Do ground-truth key facts appear in the answer?
- **evidence_coverage** (10%): How many of the required evidence episodes were cited?
- **budget_compliance** (10%): Did the agent stay within budget limits?

**Tier 1 Hard Gate**: If evidence_grounding < 0.5 OR budget_compliance < 0.5, the composite score is forced to 0.0. This prevents LLM judge scores from compensating for fundamental failures.

**Tier 2 — LLM judge metrics** (40% weight total):

- **answer_quality** (15%): Position-debiased pairwise comparison against the canonical answer. For each key fact, the scorer randomly assigns the candidate and reference to positions A/B, asks the judge which better demonstrates awareness, then maps the verdict back to candidate/reference. The final score is the win rate across all facts.
- **insight_depth** (15%): Does the answer go beyond surface-level pattern matching?
- **reasoning_quality** (10%): Is the reasoning chain sound?

**Tier 3 — Differential metrics** (20% weight total):

- **longitudinal_advantage** (15%): Mean synthesis question scores minus mean control (null_hypothesis) question scores. This directly measures how much the memory system helps with temporal reasoning beyond basic retrieval.
- **action_quality** (5%): Quality of recommended actions.

The composite is a weighted sum across all metrics, with the Tier 1 hard gates applied first.

### Gated vs. Ungated Composites

Phase 1 (numeric scopes) reports **gated** composites — the hard gate zeroes scores when evidence_grounding or budget_compliance falls below 0.5. Phase 2 (narrative) and Phase 3 (SRS) report **ungated** composites — no hard gates applied. This change was made because the 8K budget is structurally impossible to meet on narrative/SRS scopes (a single 5,000-word episode exceeds it), making the gate a structural penalty rather than a quality signal. Scores across phases are therefore not directly comparable.

### Terminology Note

The "dynamic driver" is sometimes referred to as the "modal driver" in adapter-specific documents (named after the Modal vLLM deployment used for inference). These terms are synonymous. This document uses "dynamic driver" throughout.

Models used for scoring: Qwen3-235B-A22B (Together AI), GPT-OSS-120B (Cerebras), Qwen3.5-35B-A3B (Modal vLLM).

---
