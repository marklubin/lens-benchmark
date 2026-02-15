# LENS: Longitudinal Evaluation of Networked Systems

## What it is

LENS is a benchmark for evaluating **memory systems** — any system that stores, indexes, and retrieves information over time (think AI memory layers, RAG systems, personal knowledge bases, CRM-like longitudinal stores). It answers the question: *how well does your memory system help an agent reason about information that accumulated over time?*

## The core insight

Most retrieval benchmarks dump a static corpus and test search quality. Real memory systems receive information **incrementally** — a therapy session each week, a support ticket each day — and need to surface patterns that only become visible after enough data has accumulated. LENS tests this temporal dimension directly.

## How it works

### 1. The adapter wraps your memory system

You implement a `MemoryAdapter` — a thin wrapper that exposes your system as three tools:

- `search(query, filters, limit)` — semantic or keyword search over stored memories
- `retrieve(ref_id)` — fetch a full document by ID
- `get_capabilities()` — declare what your system supports (search modes, filter fields, extra tools)

The adapter also has `ingest()` and `reset()` for data loading, but those are called by the runner, not the agent.

### 2. Episodes stream in chronologically

The runner feeds episodes (timestamped text records belonging to a persona) one at a time, in temporal order. Each episode is ingested into the adapter. This simulates real-world usage — the system doesn't get the full dataset up front.

### 3. At checkpoints, an LLM agent interrogates the memory

After N episodes have been ingested, the benchmark pauses and asks questions. A budget-constrained LLM agent receives each question and can only answer by calling the adapter's tools — `search`, `retrieve`, `get_capabilities`. The agent has no direct access to the raw episodes; it must discover information through the adapter's interface.

The agent gets a fixed budget: max turns, max tool calls, max payload bytes, max tokens. This prevents brute-force strategies and rewards systems that surface relevant information efficiently.

### 4. Three question types test different capabilities

- **Longitudinal** — requires synthesizing across many episodes to identify patterns (e.g., *"What pattern of anxiety triggers has emerged over the course of therapy?"*). This is the hard test. It rewards memory systems that can surface cross-episode patterns, not just retrieve individual records.

- **Null hypothesis** — answerable from a single specific episode (e.g., *"What happened on January 15th?"*). This is the baseline. Any functioning search system should handle these. The delta between longitudinal and null-hypothesis performance measures how much the memory system helps with *temporal reasoning* beyond basic retrieval.

- **Action recommendation** — requires both longitudinal insight and judgment (e.g., *"Based on the full history, what should the therapist focus on next?"*). Tests whether the memory system provides enough context for downstream decision-making.

### 5. Ground truth enables mechanical scoring

Each question has a `GroundTruth` with:
- `canonical_answer` — the reference answer
- `key_facts` — factual claims that must appear in the agent's answer
- `required_evidence_refs` — specific episode IDs the answer should draw from

This allows scoring without an LLM judge for the core metrics.

## Scoring tiers

### Tier 1 — Mechanical (no LLM judge)

| Metric | What it measures |
|--------|-----------------|
| Evidence grounding | Are the agent's retrieved refs real episodes? (anti-hallucination) |
| Fact recall | What fraction of ground-truth key facts appear in the answer? |
| Evidence coverage | Did the agent find the *right* episodes? |
| Budget compliance | Did the agent stay within its resource budget? |

### Tier 2 — LLM Judge

| Metric | What it measures |
|--------|-----------------|
| Answer quality | Overall correctness vs. canonical answer |
| Insight depth | Does the answer show cross-episode synthesis? |
| Reasoning quality | Is the reasoning chain coherent? |

### Tier 3 — Differential

| Metric | What it measures |
|--------|-----------------|
| Longitudinal advantage | How much better does the system do on longitudinal vs. null-hypothesis questions? This is the headline metric — it isolates the value of temporal memory. |
| Action quality | How good are the action recommendations? |

## What makes it different

- **Temporal streaming** — Episodes arrive over time, not all at once. The same question asked at checkpoint 5 vs. checkpoint 40 should yield different answers because the memory has accumulated more signal.
- **Agent-mediated** — The benchmark doesn't call your API directly. An LLM agent uses your system as a tool, which tests the full retrieval-to-synthesis pipeline and rewards systems that expose useful capabilities.
- **Dynamic capability discovery** — The agent calls `get_capabilities()` to learn what filters, search modes, and extra tools the adapter supports, then adapts its strategy. Systems that expose richer interfaces get a natural advantage.
- **Budget constraints** — Fixed resource budgets prevent brute-force. A system that surfaces the right information in 3 tool calls scores the same as one that takes 15, but the 15-call system is closer to its budget limit.
- **Differential scoring** — The longitudinal advantage metric directly measures *how much memory helps* by comparing performance on questions that require temporal reasoning vs. questions that don't.
