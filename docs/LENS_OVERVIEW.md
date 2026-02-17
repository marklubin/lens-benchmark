# LENS: Longitudinal Evidence-backed Narrative Signals

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

### 4. Ten question types test different capabilities

**Synthesis types** (require cross-episode reasoning):

- **Longitudinal** — requires synthesizing across many episodes to identify patterns. This is the core test — it rewards memory systems that surface cross-episode patterns, not just retrieve individual records.
- **Negative** — correctly identifying that a suspected cause is NOT supported by evidence. Tests whether the system can distinguish real signal from noise.
- **Temporal** — when did a pattern start? What was the progression? Requires tracking changes over time.
- **Counterfactual** — what would happen if X were different? Tests causal understanding from accumulated evidence.
- **Paraphrase** — same underlying question rephrased differently. Tests robustness of retrieval and reasoning.
- **Distractor resistance** — correctly ignoring topically similar but irrelevant episodes. Tests precision under noise.
- **Severity assessment** — how severe is the issue? What's the impact? Requires weighing evidence across episodes.
- **Evidence sufficiency** — is there enough evidence to support a conclusion? Tests epistemic calibration.

**Control type** (baseline):

- **Null hypothesis** — answerable from a single specific episode. Any functioning search system should handle these. The delta between synthesis and null-hypothesis performance is the `longitudinal_advantage` metric — the headline number that isolates the value of temporal memory.

**Decision type**:

- **Action recommendation** — requires both longitudinal insight and judgment. Tests whether the memory system provides enough context for downstream decision-making.

### 5. Ground truth enables mechanical scoring

Each question has a `GroundTruth` with:
- `canonical_answer` — the reference answer
- `key_facts` — factual claims that must appear in the agent's answer
- `required_evidence_refs` — specific episode IDs the answer should draw from

This allows scoring without an LLM judge for the core metrics.

## Scoring tiers

### Tier 1 — Mechanical (no LLM judge)

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| Evidence grounding | 10% | Are the agent's retrieved refs real episodes? (anti-hallucination) |
| Fact recall | 10% | What fraction of ground-truth key facts appear in the answer? |
| Evidence coverage | 10% | Did the agent find the *right* episodes? |
| Budget compliance | 10% | Did the agent stay within its resource budget? |

**Hard gate**: If `evidence_grounding` or `budget_compliance` < 0.5, the composite score is zeroed out. This prevents higher-tier scores from compensating for fundamental mechanical failures like hallucinated references or budget violations.

### Tier 2 — LLM Judge

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| Answer quality | 15% | Pairwise comparison: candidate answer vs. canonical ground truth, position-debiased. For each key fact, the judge picks which answer better demonstrates the finding. |
| Insight depth | 15% | Does the answer draw from 2+ distinct episodes? (cross-episode synthesis) |
| Reasoning quality | 10% | Is the answer substantive (>50 chars) with active tool use? |

### Tier 3 — Differential

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| Longitudinal advantage | 15% | Mean fact-recall for synthesis questions minus control questions. **The headline metric** — directly measures how much temporal memory helps beyond basic retrieval. |
| Action quality | 5% | How good are the action recommendations? |

## What makes it different

- **Temporal streaming** — Episodes arrive over time, not all at once. The same question asked at checkpoint 5 vs. checkpoint 40 should yield different answers because the memory has accumulated more signal.
- **Agent-mediated** — The benchmark doesn't call your API directly. An LLM agent uses your system as a tool, which tests the full retrieval-to-synthesis pipeline and rewards systems that expose useful capabilities.
- **Dynamic capability discovery** — The agent calls `get_capabilities()` to learn what filters, search modes, and extra tools the adapter supports, then adapts its strategy. Systems that expose richer interfaces get a natural advantage.
- **Budget constraints** — Fixed resource budgets prevent brute-force. A system that surfaces the right information in 3 tool calls scores the same as one that takes 15, but the 15-call system is closer to its budget limit.
- **Differential scoring** — The longitudinal advantage metric directly measures *how much memory helps* by comparing performance on questions that require temporal reasoning vs. questions that don't.
- **Contamination-resistant dataset generation** — Datasets are generated via a two-stage pipeline: a full-context planner encodes signal as numeric progressions, then a blind renderer formats each episode independently without knowing the storyline. This prevents the LLM from editorializing and ensures signal only emerges from the progression across episodes, not from any single episode. See [methodology.md](methodology.md) for details.
