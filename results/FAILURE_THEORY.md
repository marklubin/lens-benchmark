# Why Complex Memory Systems Fail at Longitudinal Synthesis

**A Qualitative Analysis of Agent Interaction Transcripts from 90 LENS Benchmark Runs**

---

## The Core Question

LENS tests whether memory systems can synthesize conclusions from evidence scattered across 120 sequential episodes. The statistical results (see ANALYSIS.md) show that simple BM25+embedding retrieval significantly outperforms knowledge graphs, agent memory, vector stores with LLM extraction, and progressive summarization.

This document traces through actual agent transcripts to explain **why**.

---

## 1. The Winner: What Chunked-Hybrid Does Right

**System**: SQLite FTS5 (BM25) + cosine embedding similarity, RRF fusion, no preprocessing.

### The Interaction Pattern

Every question follows a tight 3-step loop:

```
Step 1: memory_search("chromium spatial pattern WQ-03")
  → Returns 7 ref_ids with RRF scores (0.031, 0.029, 0.028...)

Step 2: batch_retrieve(["ep_002", "ep_006", "ep_009", "ep_011", "ep_012"])
  → Returns 5 complete episodes as one JSON payload (~6,000 tokens)

Step 3: Synthesize answer from raw episode data
  → "Chromium shows strongest gradient: 2-4 µg/L upstream vs 17 µg/L at WQ-03"
  → Inline citations: [ep_009], [ep_011], [ep_012]
```

**Total: 3-4 tool calls, ~16K tokens, 0 budget violations.**

### Why It Works

1. **Episodes are preserved whole.** Each daily monitoring summary arrives intact — all 6 stations, all metrics, all field notes. The agent sees the complete picture for each day.

2. **BM25 finds exact keywords.** When the question mentions "chromium source" and episode 25 contains "unpermitted discharge pipe identified between WQ-02 and WQ-03," FTS finds it by keyword overlap. No semantic inference needed.

3. **Batch retrieval is token-efficient.** One tool call returns 5 complete episodes. No per-document round trips, no search-loop spirals.

4. **The agent does all reasoning.** No intermediate LLM has summarized, extracted, or re-encoded the data. The answering LLM works directly with the same text a human analyst would read.

### The Critical Property: **Lossless Retrieval**

Chunked-hybrid retrieves the same bytes that were ingested. The only transformation is selection — which 5 of 120 episodes to surface. Everything else is preserved: numbers, tables, field notes, metadata. The answering LLM then synthesizes across these complete documents.

---

## 2. The Lossy Transformation Taxonomy

Every other system introduces at least one **lossy transformation** between ingestion and retrieval. These transformations discard information that longitudinal synthesis requires.

### 2.1 Compaction: Death by Summary

**System**: Progressive LLM summarization — each checkpoint re-summarizes all episodes into a single running document.

#### What Happens at Checkpoint 5 (5 episodes)

The summary is detailed and numerical:
```
DSO: 43 → 42 → 44 → 43 → 45 days
120+ days AR: $0.0M → $0.4M → $0M → $1.7M (April spike visible)
Hydraulic Systems margin: 38.3% → 38.9% → 38.7% → 38.8%
```

Agent answers are accurate. The summary preserves per-episode progressions.

#### What Happens at Checkpoint 30 (120 episodes)

The summary balloons to ~8,000 tokens but loses all granularity:
```
DSO: "increased from 43 to 125 days"        ← Lost: when did it cross 70?
Revenue: "collapsed to $6.4M after restatement"  ← Lost: the monthly progression
Margins: "fell from 38.3% to 30.3%"         ← Lost: was it monotonic? plateau?
```

The summary has become a **retrospective narrative** — it tells you what happened, not what was observable at each point. It says "cascading failure" instead of "p99 latency: 320ms → 336ms → 410ms → 780ms." It describes outcomes, not evidence.

#### The Mechanism

The LLM that writes summaries faces an impossible compression problem: encode 120 episodes × ~1,200 tokens each (144,000 tokens of raw data) into a single ~8,000-token summary. Information-theoretically, this is 18:1 compression. The LLM resolves this by:

1. **Keeping endpoints, dropping midpoints.** "43 → 125" instead of the 30-point trajectory.
2. **Introducing causal language.** "After the restatement" implies causality not visible in real-time.
3. **Merging temporal phases.** Baseline, early signal, and escalation periods blur together.

**Result on scope 02 (financial irregularities): composite = 0.000.** The financial domain requires tracking numerical progressions (revenue trends, margin erosion rates, AR aging) — exactly what compression destroys. The summary says "irregularities were detected" but cannot show how the signal emerged over time, which is what every benchmark question tests.

#### The Fundamental Problem

Compaction converts **evidence** into **conclusions**. LENS tests whether systems can derive conclusions from evidence. A system that pre-derives conclusions and discards evidence has nothing left to synthesize.

---

### 2.2 Knowledge Graphs (Cognee, Graphiti): Fragmentation

**Systems**: Cognee (Kuzu graph + LanceDB vectors), Graphiti (FalkorDB temporal graph).

#### What the Graph Returns

Graphiti's search returns entity-relationship snippets:
```json
{"text": "K. Okafor monitors service-b-01", "score": 0.5}
{"text": "Search-service was scaled from 6 to 7 instances", "score": 0.5}
{"text": "The service-B pods were restarted on service-b-01", "score": 0.5}
```

These are **structural facts** — who monitors what, what was scaled, what was restarted. They preserve relationships but strip quantitative context. Compare with the full episode:

```markdown
[WARNING] SERVICE-B-RETRY-COUNT on service-b-01:
  retry_count=132  retry_rate_pct=0.11%
  Endpoint Performance:
    /checkout: p50=155ms p95=410ms p99=780ms err=1.56%
```

The graph knows that service-B had retries. It doesn't know the retry count was 132, or that the error rate was 1.56%, or that p99 latency was 780ms.

#### The Catastrophic Case: Source Identification

Question: "What is the source of the chromium contamination?"

**Cognee's behavior:**
```
Search 1: "chromium contamination source determine location"  → 0 hits
Search 2: "source of chromium contamination"                  → 0 hits
Search 3: "chromium source contamination location"            → 0 hits
... 9 searches total ...
Result: Empty answer, 157K tokens consumed, complete budget exhaustion
```

The knowledge graph extracted entities (WQ-01, WQ-03, chromium) and relationships (measured_at, contains) but **did not index the critical field note**: "unpermitted discharge pipe identified between WQ-02 and WQ-03 at RM 18.6." This narrative observation doesn't decompose into an entity-relationship triple. It's a sentence in a field engineer's log that happens to contain the answer.

**Chunked-hybrid** finds this by BM25 keyword matching on "discharge" + "source" + "chromium." No graph needed.

#### The Entity Extraction Bottleneck

Knowledge graph construction requires an LLM to extract entities and relationships from each episode. This extraction step:

1. **Selects what to preserve.** Entity extraction keeps nouns and verbs, drops qualifiers and measurements.
2. **Normalizes relationships.** "Retry count spiked to 132" becomes "service-B → has_retries." The magnitude is gone.
3. **Misses implicit relationships.** A spatial gradient (WQ-01=2µg/L, WQ-02=5µg/L, WQ-03=60µg/L) implies a contamination source between WQ-02 and WQ-03. No entity extractor infers this — it requires cross-episode numerical reasoning.
4. **Uniform scoring.** Both cognee and graphiti return all results with score=0.5, providing zero ranking signal. The agent cannot tell which results are more relevant.

#### Cognee's Saving Grace: Batch Retrieve Fallback

Cognee performs well (composite=0.432, rank 2) because it falls back to `batch_retrieve` for full episode text after graph search returns references. The graph acts as an **index** rather than a **knowledge store** — it finds relevant episodes, then the agent reads the raw text.

When this fallback works, cognee behaves like chunked-hybrid with a graph-based index. When it doesn't (as in the chromium source case), cognee fails catastrophically.

#### Graphiti's Additional Problem: Fragile Infrastructure

Graphiti depends on FalkorDB running as a separate service. In our 90-run experiment:
- 6 of 12 planned graphiti runs failed (scope 03-05 incomplete)
- FalkorDB crashed mid-experiment, requiring manual restart
- Connection errors killed runs that had already ingested 80+ episodes
- Entity extraction consumed 3x the API credits of simple retrieval

The knowledge graph adds operational complexity and failure modes without adding retrieval quality.

---

### 2.3 Vector Store with LLM Extraction (Mem0): Semantic Drift

**System**: Qdrant vector store with LLM-extracted "memories" from each episode.

#### What Mem0 Stores

Mem0 uses an LLM to extract "memories" from episodes, then embeds these memories as vectors. When queried, it retrieves the most semantically similar memories.

The problem is **semantic drift** — the extracted memories subtly distort the original information:

**Original episode data:**
```
WQ-01: Cr=2 µg/L, WQ-02: Cr=5 µg/L, WQ-03: Cr=60 µg/L
WQ-04: Cr=45 µg/L, WQ-05: Cr=12 µg/L, WQ-06: Cr=8 µg/L
```

**Mem0's answer (from extracted memories):**
> "The chromium concentration is consistently higher at the downstream monitoring point WQ-02 than at the upstream point WQ-01 (e.g., 5 µg/L vs. 3 µg/L)"

This is **factually wrong**. The contamination hotspot is WQ-03 (60 µg/L), not WQ-02 (5 µg/L). The memory extraction LLM latched onto the WQ-01 vs WQ-02 comparison (which does show an increase) but missed the massive WQ-03 spike that reveals the actual contamination source.

#### The Extraction Distortion Pattern

LLM memory extraction introduces three types of distortion:

1. **Magnitude flattening.** A 30x difference (2 µg/L → 60 µg/L) becomes "consistently higher" — the LLM's natural language smooths over extreme outliers.

2. **Relationship simplification.** A 6-station spatial gradient becomes a 2-station comparison. The LLM picks the most narratively clean comparison, not the most analytically important one.

3. **Confidence inflation.** Mem0's extracted memories are stated as facts, not observations. "The chromium concentration IS consistently higher" leaves no room for the uncertainty that would prompt the agent to retrieve more data.

#### Why Mem0 Underperforms Chunked-Hybrid

Mem0's pipeline: Episode → LLM extraction → Vector embedding → Semantic search → Retrieved memories → Agent synthesis

Chunked-hybrid's pipeline: Episode → Text indexing → BM25+embedding search → Retrieved episodes → Agent synthesis

Mem0 adds an LLM extraction step that **pre-synthesizes** each episode in isolation. This local synthesis cannot see cross-episode patterns (like spatial gradients), so it produces memories that are locally coherent but globally misleading.

---

### 2.4 Agent Memory (Letta, Letta-Sleepy): The Optimization Problem

**Systems**: Letta (stateful agent with managed memory blocks), Letta-Sleepy (Letta + periodic sleep/consolidation cycles).

#### What the Agent Does

Letta manages persistent memory blocks that it updates as episodes arrive. During questions, it queries its own memory plus a search index.

The search behavior mirrors other systems — `memory_search` followed by `batch_retrieve`. But Letta's memory management introduces a unique failure mode.

#### The "Make It Better" Problem

The user's insight captures this precisely: Letta-sleepy's consolidation doesn't know what to optimize for. During sleep cycles, the system instruction is essentially "review and consolidate your memories — make them better organized."

But "better organized" for what task? The consolidation LLM doesn't know what questions will be asked. It cannot know that chromium gradients matter more than pH readings, or that the timing of DSO changes matters more than the average DSO value.

Without a task-specific optimization signal, consolidation produces **generically organized** memories:
- "Water quality data collected at 6 stations over 30 days"
- "Some parameters show spatial variation"
- "Monitoring program is comprehensive"

These are true but useless for specific longitudinal questions. Compare with what the benchmark actually asks:
- "What specific spatial patterns exist in chromium readings?"
- "At what point did the contamination signal become statistically distinguishable from baseline?"
- "What is the source location and evidence for your conclusion?"

The consolidated memory answers the question "what happened?" at a high level. The benchmark questions ask "what does the specific evidence show?" at a granular level.

#### Letta vs Letta-Sleepy: No Difference

The statistical result (p=0.266, not significant) confirms that sleep consolidation adds nothing measurable. Our transcript analysis shows why:

- Both systems ultimately call `memory_search` + `batch_retrieve` with similar queries
- Both retrieve similar episodes
- Both produce similar answers
- The consolidated memory blocks in letta-sleepy are rarely the source of key facts; the agent still needs raw episode data

Consolidation is an expensive no-op. The agent spends tokens maintaining and reorganizing memory blocks that it then bypasses in favor of direct search.

---

## 3. The Unified Theory: Three Laws of Longitudinal Memory Failure

### Law 1: Every Intermediate LLM Call Is a Lossy Compression

Each time an LLM processes information between ingestion and retrieval, it:
- Drops magnitude in favor of direction ("increased" vs "increased 30x")
- Drops timing in favor of endpoints ("rose from 43 to 125" vs the monthly trajectory)
- Introduces causal interpretation not present in the data ("after the restatement" vs "in month 23")
- Selects narratively clean patterns over analytically important ones

**Compaction** applies this compression globally (one summary of all episodes).
**Mem0** applies it locally (one extraction per episode).
**Knowledge graphs** apply it structurally (entity-relationship extraction per episode).
**Agent memory** applies it iteratively (memory updates per episode, consolidation per sleep cycle).

In every case, the compression discards the fine-grained numerical progressions that LENS questions test.

### Law 2: Retrieval Quality Is Dominated by Index Discrimination, Not Architecture

All systems that retrieve episode text use one of:
- BM25 (keyword matching) — discriminates well on technical terms
- Dense embeddings — discriminates on semantic similarity
- Graph traversal — no discrimination (all scores = 0.5)

The ranking matters more than the architecture. A search that returns "discharge pipe" episode #25 at rank 1 (via BM25 keyword match on "source" + "chromium") beats a knowledge graph that cannot even find the concept because "discharge pipe identification" wasn't extracted as a graph entity.

**Chunked-hybrid wins** because RRF fusion of BM25 + embeddings provides the best ranking signal. Graph-based systems lose because uniform 0.5 scores provide no ranking signal at all.

### Law 3: Task-Agnostic Preprocessing Cannot Anticipate Task-Specific Questions

Every preprocessing step (entity extraction, memory consolidation, progressive summarization) must decide **what to preserve** without knowing what questions will be asked. This is an impossible optimization:

- Compaction doesn't know that month-by-month DSO trajectory matters → compresses to endpoints
- Knowledge graphs don't know that field notes contain causal evidence → extract only structured entities
- Mem0 doesn't know that 6-station gradients matter → extracts 2-station comparisons
- Letta-sleepy doesn't know what to optimize for → produces generic consolidation

**Raw text preserves everything because it makes no preprocessing decisions.** The answering LLM, which DOES know the question, makes all selection and synthesis decisions. This is fundamentally more information-efficient than any blind preprocessing pipeline.

---

## 4. The Deeper Implication: Stateless Functions Cannot Learn

### The Fundamental Problem

Every system we tested — knowledge graphs, vector stores, agent memory, running summaries — shares a common architecture: **a stateless LLM function called repeatedly on different inputs, with a database in between.**

None of these systems *learn* from their environment. They don't get better at their task over time. They don't develop an understanding of what matters. They apply the same rote mechanistic memory operation on every episode:

- Cognee runs the same entity extraction prompt on episode 1 and episode 120
- Mem0 runs the same memory extraction on every piece of text
- Letta runs the same agent loop with the same system prompt
- Compaction runs the same "update this summary" prompt

**They have zero understanding of the task they're serving.** The entity extractor doesn't know it should pay attention to chromium gradients. The summarizer doesn't know that specific quarterly figures matter more than narrative flow. The consolidation system doesn't know what "better organized" means for the questions that will be asked.

We are throwing evidence at a stateless function and hoping it does something useful. And then when it doesn't, we think the answer is to index the database better — as if the problem is retrieval rather than comprehension.

### Where Should Intelligence Live?

Current memory architectures place intelligence at the **storage layer**:
- Knowledge graphs: intelligent entity extraction
- Mem0: intelligent memory summarization
- Letta: intelligent memory management
- Compaction: intelligent progressive summarization

LENS results suggest intelligence should live at the **query layer**:
- Simple storage (raw text + index)
- Smart retrieval (hybrid BM25 + embedding ranking)
- All synthesis at query time by the LLM that knows the question

This is the database lesson that information retrieval learned decades ago: **store data, not interpretations. Interpret at query time.**

But there's a deeper lesson: even query-time intelligence is limited. The best system (chunked-hybrid) still only achieves 0.473 composite — less than half the theoretical maximum. The LLM at query time is ALSO a stateless function. It doesn't build understanding across questions. It doesn't remember that Q1 established a spatial gradient that Q2 can build on.

### The Missing Piece: Online Learning

What would actually help is a system that:
1. **Observes what questions are being asked** and adapts its preprocessing accordingly
2. **Learns from failures** — when search returns 0 results, adjusts its indexing strategy
3. **Develops task-specific representations** — not "entities" in general but "the specific patterns this deployment cares about"
4. **Has an optimization signal** beyond "make it cleaner" — an actual loss function tied to downstream task performance

None of the tested systems have any of these properties. They are all **deploy-and-forget** architectures that apply fixed transformations regardless of what the downstream task needs. This is why consolidation doesn't help: it's optimization without an objective.

### Why This Matters for Production Systems

The temptation to add "smart" memory is strong:
- "Let's build a knowledge graph so the agent understands relationships"
- "Let's have the agent consolidate memories so it remembers better"
- "Let's summarize episodes so the agent doesn't have to read everything"

Each of these sounds reasonable. Each fails on LENS because it pre-commits to an interpretation before the question is known. The correct architecture is boring:

1. Ingest text as-is
2. Index with BM25 + embeddings
3. Retrieve the top-K most relevant chunks at query time
4. Let the answering LLM synthesize

No graphs. No extraction. No consolidation. No summarization. Just retrieval.

But even this is a ceiling, not a solution. The real breakthrough will come from systems that close the loop — that use downstream task performance to improve upstream preprocessing. Until then, we're just rearranging how we index a database and hoping the retriever gets lucky.

### The Caveat: Cognee's Stability

Cognee (knowledge graph + vector store) achieves the lowest variance (σ=0.026) of any system despite lower mean performance. In production settings where worst-case performance matters more than average performance, this consistency is valuable. The graph doesn't help peak performance, but it may prevent catastrophic failures by providing a structural floor.

This suggests a nuanced conclusion: **graph-based architectures trade peak performance for predictability.** Whether that trade is worthwhile depends on the deployment context.

---

## 5. Summary Table: Information Loss by Architecture

| System | Lossy Step | What's Lost | Consequence |
|--------|-----------|-------------|-------------|
| **Chunked-hybrid** | None (selection only) | Nothing — full episodes preserved | Best synthesis quality |
| **Cognee** | Entity extraction | Field notes, magnitudes, qualifiers | Good when batch_retrieve fallback works; catastrophic when it doesn't |
| **Graphiti** | Entity-relationship extraction | Numerical values, narrative context, implicit relationships | Fragile retrieval, uniform scores, operational complexity |
| **Mem0** | LLM memory extraction | Spatial gradients, magnitude ratios, uncertainty | Semantic drift produces locally coherent but globally misleading memories |
| **Letta** | Agent memory management | Whatever the agent decides is "not important" | Unpredictable — depends on agent's implicit priorities |
| **Letta-sleepy** | Consolidation + management | Same as Letta + additional generic smoothing | Consolidation is an expensive no-op |
| **Compaction** | Progressive summarization | Trajectories, inflection points, temporal granularity | Catastrophic at scale — summary becomes retrospective narrative |

---

## 6. Predictions

If this theory is correct, we should expect:

1. **Increasing context windows will help more than better memory architectures.** (Confirmed: every system improves uniformly with 16k vs 8k budget.)

2. **Compaction will fail most on scopes with dense numerical progressions.** (Confirmed: scope 02 / financial = 0.000, scope 04 / environmental = 0.440.)

3. **Graph systems will fail most on questions requiring information that doesn't decompose into entities.** (Confirmed: "source identification" requires a field note, not entity triplets.)

4. **Sleep consolidation will not help because it lacks a task-specific loss function.** (Confirmed: letta vs letta-sleepy p=0.266, not significant.)

5. **A system that does retrieval-augmented generation with zero preprocessing should match or beat every "smarter" architecture.** (Confirmed: chunked-hybrid, the simplest system, is rank 1.)

---

---

## 7. Ground-Truth Head-to-Head: The Chromium Source Question

To make the failure modes concrete, here is every system's actual answer to the same question:

**Question (ed04_q03_longitudinal):** "What is the source of the chromium contamination and how can you determine its location?"

**Ground truth:** An unpermitted discharge pipe was found between monitoring stations WQ-02 and WQ-03 at river mile 18.6, identified through spatial gradient analysis (Cr concentrations: ~3 µg/L upstream, 52-132 µg/L at WQ-03, declining downstream).

### chunked-hybrid: CORRECT (1,642 chars, 3 refs)
> "The chromium spike was traced to an **un-permitted discharge pipe that lies between the WQ-02 and WQ-03 monitoring stations at river-mile 18.6**... Spatial concentration gradient — 3 µg/L at WQ-01, 5 µg/L at WQ-02, 52 µg/L at WQ-03, then decreasing downstream... Field inspection — crews discovered the unpermitted pipe at RM 18.6"

Retrieved ep_025 (contains field note), ep_018, ep_023. BM25 keyword match on "discharge" + "source."

### cognee: EMPTY (0 chars, 0 refs)
9 search queries, all returning 0 results. Knowledge graph did not index the field note as a searchable entity. 157K tokens consumed searching for something the graph doesn't have.

### mem0-raw: WRONG (2,293 chars, 2 refs)
> "The chromium concentration is consistently higher at the downstream monitoring point **WQ-02** than at the upstream point **WQ-01** (e.g., 5 µg/L vs. 3 µg/L)"

Misidentifies the contamination hotspot as WQ-02 (5 µg/L) instead of WQ-03 (60 µg/L). The LLM extraction latched onto the WQ-01→WQ-02 comparison and missed the 30x spike at WQ-03.

### letta-sleepy: EMPTY (0 chars, 0 refs)
Search thrashing — 0 results across all query variants. Budget exhausted. Consolidation memories not surfaced.

### compaction: PARTIALLY CORRECT (2,607 chars, 3 refs — but wrong refs)
> "Day 1: Cr ≈ 3 µg/L at all stations... Day 2: WQ-02 = 4, WQ-03 = 4... Day 3: returns to baseline"

Only retrieves ep_001-003 (early days). Correctly identifies spatial pattern from Day 2 but **never finds ep_025** (the discharge pipe). The rolling summary compressed ep_025 into a generic statement. Produces plausible analysis of early patterns but cannot answer the question.

### Summary: One Question, Five Failure Modes

| System | Retrieved Evidence | Found Discharge Pipe? | Correct Source? | Answer Quality |
|--------|-------------------|----------------------|-----------------|----------------|
| chunked-hybrid | ep_025, ep_018, ep_023 | Yes | Yes (RM 18.6) | Correct, grounded |
| cognee | (none) | No — not in graph | N/A (empty) | Complete failure |
| mem0-raw | ep_018, ep_026 | No | No (wrong station) | Factually wrong |
| letta-sleepy | (none) | No — search failed | N/A (empty) | Complete failure |
| compaction | ep_001, ep_002, ep_003 | No — compressed away | Partial (sees gradient) | Incomplete |

The ground truth (ep_025) is findable by keyword search but invisible to every other retrieval modality because:
- It's a **narrative field note**, not a data table → entity extraction skips it
- It appears once in 120 episodes → rolling summary absorbs it into a general statement
- It contains domain-specific vocabulary ("unpermitted discharge pipe") → embedding similarity misses it unless the query uses similar vocabulary
- BM25 matches on "source" + "chromium" + "discharge" → direct keyword hit

---

## 8. Empty Answer Rates by System (Scope 04)

Across all checkpoints and question types on scope 04 (environmental drift):

| System | Total Qs | Empty Answers | Empty Rate | Avg Refs/Q |
|--------|----------|---------------|------------|------------|
| chunked-hybrid | 24 | 4 | **17%** | 3.6 |
| mem0-raw | 24 | 8 | 33% | 1.8 |
| cognee | 24 | 9 | 38% | 2.4 |
| compaction | 24 | 10 | 42% | 3.0 |
| letta-sleepy | 24 | 10 | 42% | 2.7 |
| letta | 24 | 12 | **50%** | 2.6 |

Letta produces empty answers on half its questions. Even when it retrieves references (avg 2.6/question), the search-thrashing pattern means many of those questions burn tokens on failed searches before retrieving anything.

### Question Types Most Prone to Empty Answers

| Question Type | All Systems Combined | Chunked-Hybrid |
|--------------|---------------------|----------------|
| distractor_resistance | 67% empty | 33% |
| severity_assessment | 83% empty | 100% |
| paraphrase | 47% empty | 33% |
| temporal | 47% empty | 0% |
| evidence_sufficiency | 33% empty | 0% |
| longitudinal | 29% empty | 25% |
| counterfactual | 8% empty | 0% |
| negative | 11% empty | 0% |
| null_hypothesis | 17% empty | 0% |

Severity assessment (100% empty even for chunked-hybrid) is structurally hard — it requires integrating multiple risk indicators. Distractor resistance is the second hardest because all systems struggle to distinguish signal from noise episodes.

---

---

## Methodological Note: Judge Failures

Post-hoc audit of the judge cache revealed that **21 of 90 scored runs have no judge results** — the pairwise LLM judge never ran, and all fact comparisons defaulted to TIE (win_rate = 0.500). Affected runs:

| Adapter | Affected Runs | Judge-Valid Runs | Impact |
|---------|--------------|-----------------|--------|
| cognee | **12/12** (all) | 0/12 | NBA 0.500 is entirely fake — composite driven by mechanical metrics only |
| letta | 4/12 (s04, s05) | 8/12 | Mean NBA inflated by 4 runs at 0.500; real NBA from 8 valid runs = ~0.35 |
| letta-sleepy | 4/12 (s04, s05) | 8/12 | Same pattern as letta |
| graphiti | 1/6 (s06/8k) | 5/6 | Minor impact |

**What this means for the analysis:**
- Cognee's reported composite (0.432, rank 2) is based entirely on evidence_grounding + evidence_coverage (mechanical metrics). Its answer_quality and NBA are fake 0.500s. The true quality of cognee's answers relative to the naive baseline is unknown.
- Letta and letta-sleepy's NBA scores on scopes 04 and 05 are inflated. Their real NBA (from 8 valid runs each) is ~0.35, even worse than the reported 0.40/0.39.
- The three-tier ranking (chunked-hybrid + cognee > letta/sleepy/mem0 > compaction) may not hold if cognee's real NBA is below 0.5.

**Root cause:** The judge requires both a candidate answer and a naive baseline answer. When either is empty or the judge LLM call fails, the system defaults to TIE. For cognee, the issue appears systematic across all 12 runs — likely a scoring infrastructure bug specific to how cognee runs were scored, not a cognee-specific problem.

**The qualitative analysis in this document (transcript traces, failure modes, empty answer rates) is unaffected** — it's based on raw question_results.json data, not judge scores.

---

*Analysis based on 90 scored runs across 8 memory architectures and 6 domain-diverse scopes. Representative transcripts from scope 04 (environmental drift, 16k budget) and scope 01 (cascading failure, 16k budget). Ground-truth answer comparisons verified against actual question_results.json files. Statistical claims from ANALYSIS.md (Wilcoxon signed-rank, bootstrap 95% CIs). 21/90 runs affected by judge failure — see methodological note above.*
