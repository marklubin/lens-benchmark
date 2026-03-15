# Quick Start

```
┌──────────────────────────────────────────────┐
│  LENS // Getting Started                     │
└──────────────────────────────────────────────┘
```

## 1. Install

```bash
git clone https://github.com/synix-dev/lens-benchmark.git
cd lens-benchmark
uv sync --all-extras
```

Requires Python 3.11+. Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Smoke Test

Verify the installation with the built-in smoke test. This uses the null adapter (returns empty results) and doesn't require an LLM:

```bash
uv run lens smoke
```

Expected: exits cleanly with a score summary showing the null baseline.

## 3. List Available Adapters

```bash
uv run lens adapters
```

Built-in adapters:
- `null` — returns empty results (score floor baseline)
- `sqlite` — SQLite FTS5 full-text search + optional embedding similarity

## 4. Run Against a Scope

Run the SQLite adapter against the cascading failure scope:

```bash
# Compile the scope dataset
uv run lens compile --scope-dir datasets/scopes/01_cascading_failure \
  --output data_s01.json

# Run the benchmark
uv run lens run --dataset data_s01.json --adapter sqlite --out output/s01_sqlite/
```

This streams 120 episodes (30 signal + 90 distractors) into the adapter chronologically, then asks 24 questions at checkpoints.

## 5. Score Results

Scoring requires an LLM for Tier 2 (judge) metrics. Set up the LLM endpoint:

```bash
# Using a local or remote OpenAI-compatible endpoint
export LENS_LLM_API_BASE=http://localhost:8000/v1
export LENS_LLM_API_KEY=dummy

# Score the run
uv run lens score --run output/s01_sqlite/ --judge-model your-model-name
```

The scorer produces:
- `scores.json` — per-question scores across all 9 metrics
- `scorecard.json` — aggregate scores by tier and composite

## 6. Generate Report

```bash
uv run lens report --run output/s01_sqlite/
```

## 7. Compare Against Null Baseline

```bash
# Run null adapter on the same scope
uv run lens run --dataset data_s01.json --adapter null --out output/s01_null/
uv run lens score --run output/s01_null/ --judge-model your-model-name

# Compare
uv run lens compare output/s01_sqlite/ output/s01_null/
```

Your adapter should beat null. If it doesn't, your memory system is providing negative value — the agent does better with no memory at all.

## 8. Next Steps

- **Write your own adapter**: [ADAPTER_GUIDE.md](ADAPTER_GUIDE.md)
- **Run the full benchmark**: [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)
- **Understand the scoring**: See the Scoring section in [README.md](../../README.md)
- **Design a new scope**: [SCOPE_GUIDE.md](SCOPE_GUIDE.md)

## CLI Reference

```bash
uv run lens --help           # All commands
uv run lens run --help       # Run options
uv run lens score --help     # Scoring options
uv run lens report --help    # Report options
uv run lens adapters         # List adapters
uv run lens metrics          # List scoring metrics
uv run lens smoke            # Smoke test
uv run lens verify           # Verify generated scope
```
