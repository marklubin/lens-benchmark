# LENS Benchmark

**Longitudinal Evidence-backed Narrative Signals** — a benchmark for evaluating memory systems for AI agents.

Most retrieval benchmarks dump a static corpus and test search quality. Real memory systems receive information **incrementally** — a support ticket each day, a sensor reading each hour, a clinical note each week — and need to surface patterns that only become visible after enough data has accumulated. LENS tests this temporal dimension directly.

LENS streams timestamped episodes into your memory system chronologically, then pauses at checkpoints to ask questions that require synthesizing evidence scattered across many episodes. A budget-constrained LLM agent interrogates your system through its search/retrieve interface, and three tiers of scoring measure everything from basic evidence grounding to longitudinal reasoning advantage.

## How It Works

```
                    ┌─────────────┐
                    │  spec.yaml  │    Dataset definition
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Episodes  │    Timestamped text records
                    └──────┬──────┘
                           │ stream chronologically
                    ┌──────▼──────┐
                    │   Adapter   │    Your memory system wrapper
                    │  (ingest)   │
                    └──────┬──────┘
                           │ at checkpoints...
                    ┌──────▼──────┐
                    │    Agent    │    Budget-constrained LLM
                    │  (search,   │    interrogates memory via
                    │  retrieve)  │    adapter's tool interface
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Scorer    │    3-tier scoring
                    │  (mechanical│    mechanical → LLM judge
                    │   → judge   │    → differential
                    │   → diff)   │
                    └─────────────┘
```

1. **Adapter wraps your memory system** — You implement `MemoryAdapter` with `search`, `retrieve`, and `get_capabilities`. The runner calls `ingest` to feed episodes.
2. **Episodes stream in chronologically** — The runner feeds timestamped episodes one at a time. Your adapter stores them however it wants.
3. **At checkpoints, an LLM agent interrogates memory** — A budget-constrained agent receives each question and can only answer by calling `memory_search`, `memory_retrieve`, and `memory_capabilities`. It has no direct access to raw episodes.
4. **Three-tier scoring** — Mechanical metrics (fact recall, evidence grounding), LLM judge (pairwise answer quality), and differential metrics (longitudinal advantage over baseline retrieval).

## Quick Start

```bash
# Install
pip install lens-bench
# or with uv
uv pip install lens-bench

# Run smoke test (null adapter + mock LLM)
uv run lens smoke
```

## Writing an Adapter

Subclass `MemoryAdapter` from `src/lens/adapters/base.py`:

```python
from lens.adapters.base import (
    MemoryAdapter,
    CapabilityManifest,
    SearchResult,
    Document,
)

class MyMemoryAdapter(MemoryAdapter):
    """Wrap your memory system for LENS evaluation."""

    def reset(self, scope_id: str) -> None:
        """Clear all state for a scope."""
        ...

    def ingest(self, episode_id: str, scope_id: str,
               timestamp: str, text: str, meta: dict | None = None) -> None:
        """Ingest a single episode. Must complete within 200ms, no LLM calls."""
        ...

    def search(self, query: str, filters: dict | None = None,
               limit: int | None = None) -> list[SearchResult]:
        """Search memory for relevant information."""
        ...

    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve a full document by reference ID."""
        ...

    def get_capabilities(self) -> CapabilityManifest:
        """Declare search modes, filter fields, and extra tools."""
        return CapabilityManifest(
            search_modes=["semantic"],
            max_results_per_search=10,
        )
```

The agent discovers your adapter's capabilities at runtime via `get_capabilities()` and adapts its strategy. Systems that expose richer interfaces (filter fields, date ranges, extra tools) get a natural advantage.

Built-in adapters: `null` (returns empty results, used as baseline), `sqlite` (SQLite FTS + semantic search). External adapters can be registered via `lens.adapters` entry points.

## Running a Benchmark

```bash
# Run benchmark with an adapter
lens run --dataset data.json --adapter sqlite --out output/

# Score the results
lens score --run output/

# Generate report
lens report --run output/

# Compare two runs
lens compare output1/ output2/

# List available adapters and metrics
lens adapters
lens metrics
```

## Scoring

Nine metrics across three tiers, with a weighted composite score:

### Tier 1 — Mechanical (no LLM judge)

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| `evidence_grounding` | 10% | Fraction of cited ref_ids that exist in the episode vault (anti-hallucination) |
| `fact_recall` | 10% | Fraction of ground-truth key facts found in the answer text |
| `evidence_coverage` | 10% | Fraction of required evidence episodes actually retrieved |
| `budget_compliance` | 10% | 1.0 minus 0.1 per budget violation (turns, tool calls, tokens, latency) |

**Hard gate**: If `evidence_grounding` or `budget_compliance` falls below 0.5, the composite score is zeroed out. This prevents higher-tier scores from compensating for fundamental mechanical failures.

### Tier 2 — LLM Judge

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| `answer_quality` | 15% | Pairwise comparison: candidate answer vs. canonical ground truth per key fact, position-debiased |
| `insight_depth` | 15% | Fraction of questions where the agent cited refs from 2+ distinct episodes |
| `reasoning_quality` | 10% | Fraction of questions with substantive answers (>50 chars) and tool use |

### Tier 3 — Differential

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| `longitudinal_advantage` | 15% | Mean fact-recall for synthesis questions minus control questions. **The headline metric** — isolates the value of temporal memory. |
| `action_quality` | 5% | Mean fact-recall for action recommendation questions |

## Dataset Scopes

Six scopes across different domains, each with 30 signal episodes + 90 distractor episodes and 24 questions:

| Scope | Domain | Signal |
|-------|--------|--------|
| 01 Cascading Failure | System logs | API gateway cascading dependency failure |
| 02 Financial Irregularity | Financial reports | Progressive revenue recognition manipulation |
| 03 Clinical Signal | Clinical notes | Drug-drug interaction hepatotoxicity signal |
| 04 Environmental Drift | Environmental monitoring | Upstream chromium contamination from unpermitted discharge |
| 05 Insider Threat | Security logs | Systematic IP exfiltration by departing employee |
| 06 Market Regime | Market analysis | Hidden equity-bond correlation breakdown from policy shift |

Each scope follows a five-phase narrative arc: baseline → early signal → red herring → escalation → root cause. Signal is distributed across episodes so that no single episode answers any question — patterns only emerge from the progression.

## Project Structure

```
src/lens/
  adapters/       # Memory system adapters (base ABC, null, sqlite, registry)
  agent/          # Agent harness, tool bridge, budget enforcement, LLM clients
  cli/            # Click CLI (run, score, report, compare, smoke, etc.)
  core/           # Data models (Episode, Question, GroundTruth, AgentAnswer, ScoreCard)
  datagen/synix/  # Two-stage dataset generation pipeline
  datasets/       # Dataset loading
  matcher/        # Answer matching
  report/         # Report generation
  runner/         # Benchmark runner with EpisodeVault anticheat
  scorer/         # 3-tier scoring (tier1, tier2, tier3, aggregate, judge)
datasets/scopes/  # Dataset specifications and generated artifacts
tests/unit/       # Unit tests
docs/             # Detailed documentation
```

## Documentation

- [Architecture deep-dive](docs/architecture.md) — Core data flow, adapter system, agent harness, runner, scoring internals
- [Dataset methodology](docs/methodology.md) — Two-stage pipeline, contamination prevention, validation gates
- [Calibration learnings](docs/calibration.md) — Naive baseline calibration process and key fact design rules
- [Conceptual overview](docs/LENS_OVERVIEW.md) — What makes LENS different from static retrieval benchmarks

## Development

```bash
# Run tests
uv run pytest tests/unit/ -v

# Build a dataset scope
uv run synix build src/lens/datagen/synix/pipeline.py \
  --source-dir datasets/scopes/01_cascading_failure \
  --build-dir datasets/scopes/01_cascading_failure/generated \
  -j 8 -vv

# Validate a build
uv run synix validate src/lens/datagen/synix/pipeline.py \
  --build-dir datasets/scopes/01_cascading_failure/generated --json
```
