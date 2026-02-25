# Scenario Walkthrough: Why Simple Retrieval Wins

**A single question, seven memory architectures, one answer.**

This document traces a canonical LENS benchmark question through every memory system, connecting scaling behavior (Figures A-E) to a concrete case study (the swimlane trace and comparison table). The goal: show *why* simple chunked retrieval outperforms knowledge graphs, agent memory, and progressive summarization at longitudinal evidence synthesis.

---

## 1. The Scenario

**Scope 04: Environmental Drift** — a water quality monitoring domain.

Six monitoring stations (WQ-01 through WQ-06) line a river. Over 30 simulated days, chromium concentrations rise at station WQ-03 while upstream stations remain at baseline. Mixed in are 90 distractor episodes covering unrelated infrastructure work: DNS migration, storage capacity planning, authentication audits. Total: 120 episodes.

The signal pattern:
- **Days 1-5** (baseline): All stations report chromium < 5 µg/L
- **Days 6-15** (early signal): WQ-03 begins climbing — 8, 12, 17 µg/L — while WQ-01/02 hold steady
- **Days 16-22** (escalation): WQ-03 hits 52 µg/L, then 105 µg/L (exceeding EPA MCL)
- **Days 23-25** (root cause): A field inspection at RM 18.6 discovers an unpermitted discharge pipe between WQ-02 and WQ-03

The red herring: agricultural runoff causes turbidity spikes at multiple stations, but the turbidity pattern doesn't correlate with the chromium gradient. A careful analyst sees that nutrient loading explains turbidity but not hexavalent chromium.

---

## 2. How Systems Scale (Figures A-E)

Before looking at the specific question, the scaling graphs reveal how each architecture behaves as the episode count grows from 0 to 120.

### Ingest Latency (Figure A)

Two tiers emerge:
- **Letta and chunked-hybrid** process each episode during ingest (~300-700ms per episode) — letta via API calls to its server, chunked-hybrid via SQLite FTS indexing + embedding computation
- **Graphiti and compaction** show near-zero ingest times because they buffer episodes and do real work in `prepare()` — entity extraction and summarization happen in bulk before the question phase

Cognee and mem0-raw don't appear because their ingest timing isn't logged to `log.jsonl` (they use internal async pipelines).

### Token Consumption — The Real Scaling Signal (Figure C)

This is the most revealing graph. As more episodes are ingested:
- **Chunked-hybrid** rises steadily from ~6K to ~76K tokens per question — it retrieves more episodes as the corpus grows, but the increase is linear and bounded by the retrieval limit
- **Letta, letta-sleepy, and compaction** climb to 50-65K tokens — the agent makes more search attempts as early queries fail to surface relevant episodes from a larger haystack
- **Null** stays flat at ~5K tokens — it has no memory to search, so token consumption doesn't scale with data
- **Graphiti** stays relatively flat at ~20K — its knowledge graph returns entity-level facts rather than full episodes

The key insight: **token consumption is a proxy for search thrashing.** Systems that can't find relevant evidence in 1-2 queries burn tokens reformulating queries. This is visible in Figure D (tool calls) where letta-sleepy climbs from 3 calls at checkpoint 1 to 8+ calls at later checkpoints.

### Empty Answer Rate (Figure E)

The most directly interpretable metric: what fraction of questions get no answer at all?

- **Chunked-hybrid** maintains the lowest empty rate across all checkpoints (14-33%), demonstrating consistent retrieval success
- **Null** has high empty rates (33-70%) because it has no memory — it can only answer from the question itself
- **Letta** and **compaction** degrade sharply from checkpoint 2 onward, rising to 60-80% empty answers as the corpus grows and retrieval becomes unreliable
- **Letta-sleepy** follows the same degradation pattern — more data means more failed searches means more empty answers

---

## 3. The Critical Question

**Question**: "What is the source of the chromium contamination and how can you determine its location?"

**Checkpoint**: After 99 episodes (or 120 for adapters that only recorded the final checkpoint)

**Ground truth**: An unpermitted industrial discharge between monitoring stations WQ-02 and WQ-03 at river-mile 18.6. The spatial gradient proves this — WQ-01/02 remain at baseline while WQ-03 shows peak concentrations and downstream stations show progressive dilution. Agricultural runoff is a red herring: it explains turbidity but not the chromium gradient.

**Key evidence**: Episode 025 — a field inspection note recording the discovery of the discharge pipe.

### Comparison Table

| System | Tool Calls | Tokens | Time | Answer | Found Pipe? | Found RM 18.6? |
|--------|-----------|--------|------|--------|-------------|-----------------|
| **chunked-hybrid** | **3** | **22,308** | **2.7s** | **1,642 chars** | **Yes** | **Yes** |
| compaction | 4 | 35,774 | 62s | 2,607 chars | No | No |
| null | 6 | 4,537 | 1.4s | 212 chars | No | No |
| letta | 8 | 74,722 | 127s | 2,567 chars | No | No |
| letta-sleepy | 9 | 88,547 | 69s | Empty | No | No |
| cognee | 9 | 157,187 | 20s | Empty | No | No |
| mem0-raw | 9 | 180,102 | 7s | 2,293 chars | No | No |

*(See `results/figures/scenario_trace_swimlane.png` for the visual tool-call sequence.)*

### System-by-System Walkthrough

#### Chunked-Hybrid: 3 calls, correct answer

```
Search 1: "chromium contamination source determine location" → 10 results (RRF scores 0.033-0.027)
Search 2: "source of chromium contamination ... source identification chromium leak" → 10 results
  → ep_025 at rank 5 (score 0.027) — the field inspection note
Batch Retrieve: [ep_025, ep_018, ep_023] → 3 complete episodes in one call
```

The agent reads the raw episode text:
> "unpermitted discharge pipe identified between WQ-02 and WQ-03 at RM 18.6"

It synthesizes a grounded answer citing the spatial gradient (WQ-01: 3, WQ-02: 5, WQ-03: 132 µg/L) and the field inspection confirmation. Three references, all valid.

**Why it works**: BM25 matches on "source" + "chromium" + "discharge" — direct keyword overlap with episode 025. The episode is stored verbatim, so all details (RM 18.6, station readings) are preserved.

#### Letta-Sleepy: 9 calls, empty answer

```
Search 1: "chromium contamination source determine location" → (no useful results)
Search 2: "source of chromium contamination location determination" → (no useful results)
Search 3: "chromium source contamination analysis" → (no useful results)
...
Search 9: "chromium source contamination source identification site" → (no useful results)
Search 10: budget exhausted at 88,547 tokens
```

Ten progressively longer queries, each returning nothing actionable. The agent reformulates but never finds evidence. It exhausts the 16K token budget (actual: 88K with violations) and the 10-turn limit without producing any answer.

**Why it fails**: Letta's memory architecture stores extracted facts and entity relationships, not raw episode text. The field inspection note (episode 025) is a narrative — "crews inspected the reach where the gradient changed most sharply" — not a structured entity. Letta's retrieval index doesn't surface it because it was never extracted as a discrete fact.

#### Cognee: 9 calls, empty answer

```
9 memory_search calls across 157,187 tokens → 0 useful retrievals → empty answer
```

Similar search-thrashing pattern. Cognee's knowledge graph indexed entities and relationships from the monitoring data but didn't capture the narrative field note that contains the actual answer. The discharge pipe discovery is encoded in natural language prose, not as a graph node.

**Why it fails**: Entity extraction works well for structured data (station readings, concentrations) but misses narrative details that don't fit the entity-relationship model. The answer exists in the corpus but is invisible to the graph.

#### Mem0-Raw: 9 calls, wrong answer

Mem0 retrieves episodes ep_018 and ep_026 — escalation-phase data showing elevated chromium. But it never finds ep_025 (the field inspection). Its answer identifies the wrong station and mislocates the source.

**Why it fails**: Mem0's vector search returns semantically similar episodes but misses the specific keyword match. Episode 025 uses domain vocabulary ("unpermitted discharge pipe") that doesn't embed close to "chromium contamination source."

#### Compaction: 4 calls, partial answer

Compaction retrieves ep_001, ep_002, ep_003 — early baseline episodes. Its running summary absorbed episode 025 into a generalized statement about the investigation, losing the specific RM 18.6 location and the discharge pipe detail.

**Why it fails**: Progressive summarization is lossy. As the summary grows from 5 episodes to 120, per-episode details are compressed away. The single most important detail — a one-sentence field note about a pipe at a specific river mile — gets absorbed into "investigation revealed the contamination source."

#### Null: 6 calls, generic answer

With no memory at all, null can only hallucinate a generic response about contamination investigation methodology. It gets 212 characters of plausible-sounding but ungrounded text.

---

## 4. Connecting to the General Theory

This single question illustrates three recurring patterns from the full 90-run analysis:

### Pattern 1: Search Thrashing Wastes Budgets

When retrieval fails, agents don't stop — they reformulate and retry. Each retry consumes tokens for the query, the empty response, and the agent's internal reasoning about what to try next. Letta-sleepy burned 88K tokens (5.5x the 16K budget) on 9 failed searches. This pattern explains why token consumption in Figure C diverges so dramatically between adapters.

### Pattern 2: Lossy Transformations Destroy Critical Details

The answer to this question is one sentence in one episode out of 120. Any transformation that summarizes, extracts entities, or re-encodes information has a chance of discarding that sentence. The probability that a specific detail survives grows with its redundancy — if the discharge pipe were mentioned in 10 episodes, knowledge graphs might capture it. But longitudinal benchmarks test exactly the case where signal is sparse and distributed.

### Pattern 3: Keyword Match Beats Semantic Inference for Specific Facts

Episode 025 contains "unpermitted discharge pipe" and the question asks about "the source of chromium contamination." BM25 matches these directly through shared terms ("source," "contamination"). Embedding similarity requires the model to have learned that "discharge pipe" is semantically close to "contamination source" — a reasonable inference, but one that competes with 119 other episodes also discussing contamination.

---

## Artifacts

| File | Description |
|------|-------------|
| `results/figures/scaling_ingest_latency.png` | Figure A: Per-episode ingest latency |
| `results/figures/scaling_wall_time.png` | Figure B: Question wall time per checkpoint |
| `results/figures/scaling_tokens.png` | Figure C: Token consumption per checkpoint |
| `results/figures/scaling_tool_calls.png` | Figure D: Tool calls per checkpoint |
| `results/figures/scaling_empty_rate.png` | Figure E: Empty answer rate per checkpoint |
| `results/figures/scenario_trace_swimlane.png` | Swimlane: tool call sequence per adapter |
| `results/tables/scenario_comparison.tex` | LaTeX comparison table |
| `results/scaling_analysis.json` | Raw scaling data (all adapters, all checkpoints) |
| `results/scenario_trace_data.json` | Full structured trace data |
