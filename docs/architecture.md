# Architecture

Technical deep-dive into the LENS benchmark internals.

## Core Data Flow

```
Episode stream ──► Adapter.ingest() ──► [adapter storage]
                                              │
                   at checkpoints...          │
                                              ▼
Question ──► AgentHarness ──► ToolBridge ──► Adapter.search()
                  │                          Adapter.retrieve()
                  │                          Adapter.get_capabilities()
                  ▼
            AgentAnswer ──► ScorerEngine ──► ScoreCard
                                │
                          Tier 1: mechanical
                          Tier 2: LLM judge
                          Tier 3: differential
```

The runner streams episodes into the adapter, pauses at checkpoints to run the agent, then feeds results to the scorer. The `EpisodeVault` stores a parallel copy of all episodes runner-side for evidence validation — adapters cannot access the vault.

## Core Models

Defined in `src/lens/core/models.py`:

### Input

- **`Episode`** — A single timestamped text record. Fields: `episode_id`, `scope_id`, `timestamp`, `text`, `meta`. Frozen dataclass.

### Dataset

- **`Question`** — A benchmark question. Fields: `question_id`, `scope_id`, `checkpoint_after`, `question_type`, `prompt`, `ground_truth`.
- **`GroundTruth`** — Ground truth for a question. Fields: `canonical_answer`, `required_evidence_refs` (episode IDs the answer should cite), `key_facts` (factual claims that must appear).

### Agent Output

- **`AgentAnswer`** — The agent's answer to a question. Fields: `question_id`, `answer_text`, `turns` (full conversation), `tool_calls_made`, `total_tokens`, `wall_time_ms`, `budget_violations`, `refs_cited`.
- **`QuestionResult`** — Pairs a `Question` with an `AgentAnswer` plus `retrieved_ref_ids` and `valid_ref_ids` (subset verified against the vault).

### Runner Output

- **`CheckpointResult`** — Results at a single checkpoint: `question_results`, `validation_errors`, `budget_used`, `timing`.
- **`ScopeResult`** — All checkpoints for a scope.
- **`RunResult`** — Complete results for a benchmark run across all scopes.

### Scoring

- **`MetricResult`** — A single metric: `name`, `tier` (1/2/3), `value` (0.0–1.0), `details`.
- **`ScoreCard`** — Aggregate scoring: `run_id`, `adapter`, `dataset_version`, `budget_preset`, list of `MetricResult`, `composite_score`.

## Adapter System

### MemoryAdapter ABC (`src/lens/adapters/base.py`)

The abstract base class defines two categories of methods:

**Data loading** (called by the runner, not exposed to agent):
- `reset(scope_id)` — Clear all state for a scope. Called once before episode stream begins.
- `ingest(episode_id, scope_id, timestamp, text, meta)` — Ingest a single episode. Must complete within 200ms, no LLM calls allowed.
- `prepare(scope_id, checkpoint)` — Optional hook before questions at a checkpoint. Adapters may build indices or consolidate memories.

**Mandatory tools** (exposed to the agent via `ToolBridge`):
- `search(query, filters, limit) -> list[SearchResult]` — Search memory.
- `retrieve(ref_id) -> Document | None` — Fetch a full document by reference ID.
- `get_capabilities() -> CapabilityManifest` — Declare what the adapter supports.

**Optional**: `call_extended_tool(tool_name, arguments)` for adapter-defined extra tools.

### CapabilityManifest

Declares adapter capabilities for dynamic tool discovery:
- `search_modes` — e.g., `["semantic"]`, `["semantic", "keyword"]`
- `filter_fields` — List of `FilterField` (name, type, description, enum values)
- `max_results_per_search` — Default 10
- `supports_date_range` — Boolean
- `extra_tools` — List of `ExtraTool` (name, description, JSON Schema parameters)

The agent calls `memory_capabilities` at the start of each question to learn what's available and adapts its search strategy accordingly.

### AdapterRegistry (`src/lens/adapters/registry.py`)

Adapters are registered in two ways:

1. **Built-in**: `@register_adapter("name")` decorator (e.g., `NullAdapter`, `SQLiteAdapter`).
2. **External plugins**: `lens.adapters` entry point group. Discovered lazily on first miss.

`get_adapter(name)` returns the adapter class. `list_adapters()` returns all registered adapters.

## Agent Harness

### AgentHarness (`src/lens/agent/harness.py`)

Orchestrates the agent loop for a single question:

1. Builds tool definitions from the adapter's capability manifest via `ToolBridge`.
2. Creates a `BudgetEnforcement` tracker.
3. Runs `llm_client.run_agent_loop()` with a system prompt, the question, and tool definitions.
4. For each tool call: checks budget → dispatches via `ToolBridge` → records metrics.
5. Extracts the final answer text from the last assistant turn.
6. Returns an `AgentAnswer` with full conversation, tool usage stats, and budget violations.

The system prompt instructs the agent to use `memory_search`, `memory_retrieve`, and `memory_capabilities` to find information, synthesize findings, and cite evidence by ref_id.

### ToolBridge (`src/lens/agent/tool_bridge.py`)

Translates between the LLM tool-call interface and adapter methods:

- `build_tool_definitions(adapter)` — Reads the `CapabilityManifest` and produces `ToolDefinition` objects for: `memory_search`, `memory_retrieve`, `memory_capabilities`, plus any `extra_tools`.
- `dispatch_tool_call(adapter, tool_call, max_payload_bytes)` — Routes tool calls to adapter methods. Truncates payloads exceeding the byte limit.

### BudgetEnforcement (`src/lens/agent/budget_enforcer.py`)

Per-question budget limits via `QuestionBudget`:

| Limit | Default | Enforcement |
|-------|---------|-------------|
| `max_turns` | 10 | Hard stop (raises `BudgetViolation`) |
| `max_total_tool_calls` | 20 | Hard stop |
| `max_payload_bytes` | 65,536 | Warning only |
| `max_latency_per_call_ms` | 5,000 | Records violation |
| `max_agent_tokens` | 8,192 | Records violation |

Hard stops (`max_turns`, `max_total_tool_calls`) immediately terminate the agent loop. Other violations are recorded for scoring but don't halt execution.

### LLM Clients (`src/lens/agent/llm_client.py`)

- `BaseLLMClient` — ABC with `run_agent_loop(system_prompt, user_message, tools, tool_executor, max_turns, turn_callback)`.
- `OpenAIClient` — Production client using OpenAI's API with tool-use support.
- `MockLLMClient` — Returns a fixed answer without tool calls. Used in smoke tests.

## Runner

### RunEngine (`src/lens/runner/runner.py`)

Orchestrates the full benchmark run:

1. Resolves the adapter class from the registry.
2. Groups questions by scope and checkpoint.
3. For each scope:
   - Calls `adapter.reset(scope_id)`.
   - Streams episodes in timestamp order, calling `adapter.ingest()` for each.
   - Stores each episode in the `EpisodeVault`.
   - At checkpoints: calls `adapter.prepare()`, then runs the `AgentHarness` for each question.
   - Validates cited refs against the vault (`valid_ref_ids` = refs that actually exist).
4. Saves artifacts (run manifest, config, per-checkpoint question results, log).

### EpisodeVault (`src/lens/runner/anticheat.py`)

Runner-side storage for anticheat:

- Stores episode text as ingested. Adapters cannot access the vault.
- `has(episode_id)` — Validates that a cited ref actually exists.
- `verify_quote(episode_id, quote)` — Verifies that a quote is an exact substring.
- Used by the runner to compute `valid_ref_ids` for each `QuestionResult`.

This prevents adapters from fabricating episode references or accessing raw text at query time.

## Scoring

### Overview

Scoring uses three tiers with a weighted composite. Defined across `src/lens/scorer/`:

- `tier1.py` — Mechanical metrics (evidence_grounding, fact_recall, evidence_coverage, budget_compliance)
- `tier2.py` — LLM judge metrics (answer_quality, insight_depth, reasoning_quality)
- `tier3.py` — Differential metrics (longitudinal_advantage, action_quality)
- `aggregate.py` — Composite weighting and tier 1 hard gate
- `judge.py` — Pairwise LLM judge implementation
- `engine.py` — `ScorerEngine` that runs all metrics
- `registry.py` — `@register_metric` decorator and metric discovery
- `base.py` — `BaseMetric` ABC

### ScorerEngine (`src/lens/scorer/engine.py`)

Runs all registered metrics against a `RunResult`:

```python
scorer = ScorerEngine(judge_fn=my_llm_judge)
scorecard = scorer.score(run_result)
```

The optional `judge_fn` callable is injected into metrics that support it (currently `answer_quality`). Metrics implement `configure(judge_fn=...)` to receive the judge.

### Tier 1 Hard Gate

From `aggregate.py`:

```python
TIER1_GATE_THRESHOLDS = {
    "evidence_grounding": 0.5,
    "budget_compliance": 0.5,
}
```

If either gated metric falls below its threshold, the composite score is 0.0. This prevents LLM judge scores from compensating for fundamental failures like hallucinated references or budget violations.

### Composite Weights

```python
DEFAULT_WEIGHTS = {
    "evidence_grounding": 0.10,
    "fact_recall": 0.10,
    "evidence_coverage": 0.10,
    "budget_compliance": 0.10,
    "answer_quality": 0.15,
    "insight_depth": 0.15,
    "reasoning_quality": 0.10,
    "longitudinal_advantage": 0.15,
    "action_quality": 0.05,
}
```

Weights sum to 1.0. If some metrics are missing (e.g., no judge_fn), the composite normalizes over present metrics.

### Pairwise Answer Quality (`src/lens/scorer/judge.py`)

The `answer_quality` metric uses position-debiased pairwise comparison:

1. For each key fact, randomly assign candidate and reference answers to positions A and B.
2. Ask the judge: "Which response better demonstrates awareness of this finding?"
3. Map the positional verdict (A/B/TIE) back to candidate/reference.
4. Score: candidate wins = 1.0, tie = 0.5, reference wins = 0.0.
5. Win rate = mean score across all facts across all questions.

This approach is more robust than absolute scoring because relative comparisons are easier for LLM judges and position bias is controlled via random assignment.

### Longitudinal Advantage (`src/lens/scorer/tier3.py`)

The headline differential metric:

```
longitudinal_advantage = mean(synthesis_scores) - mean(control_scores)
```

- **Synthesis question types**: longitudinal, negative, temporal, counterfactual, paraphrase, distractor_resistance, severity_assessment, evidence_sufficiency
- **Control question types**: null_hypothesis

This directly measures how much the memory system helps with temporal reasoning beyond basic retrieval.
