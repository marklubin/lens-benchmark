# LENS Benchmark

**LENS** = Longitudinal Evidence-backed Narrative Signals — a benchmark for evaluating agent memory systems. Tests whether an LLM agent can synthesize conclusions from evidence scattered across many sequential episodes, rather than finding answers in a single document.

## Project Structure

```
src/lens/
  adapters/       # Memory system adapters (base, null, sqlite, mem0, zep, letta, hyperspell)
  agent/          # Agent harness for running benchmarks
  cli/            # Click CLI (`lens` command)
  core/           # Core data types (Episode, Question, Scope)
  datagen/        # Dataset generation
    synix/        # Synix-based DAG pipeline (current)
      pipeline.py       # DAG definition — the entry point
      transforms.py     # PlanOutline, RenderSignalEpisodes, RenderDistractorEpisodes, etc.
      prompt_utils.py   # All LLM prompt builders
      validators.py     # WordCount, ContaminationCheck, NaiveBaseline
      scoring.py        # Distractor similarity, contamination scoring
      spec_utils.py     # Spec parsing helpers
      release.py        # Package validated artifacts into release manifest
  datasets/       # Dataset loading
  human/          # Human benchmark harness
  matcher/        # Answer matching
  report/         # Report generation
  runner/         # Benchmark runner with anticheat
  scorer/         # 3-tier scoring (tier1=key facts, tier2=evidence, tier3=reasoning)
datasets/scopes/  # Dataset specifications and generated data
  01_cascading_failure/
    spec.yaml     # Scope definition (arc, key facts, questions, distractors)
    generated/    # Build artifacts (episodes.json, questions.json, release_manifest.json, etc.)
tests/unit/       # 154 unit tests (test_synix_*.py for datagen pipeline)
```

## Tooling

- **Package manager**: `uv` (not pip, not poetry)
- **Test runner**: `uv run pytest tests/unit/ -v`
- **Linter**: `ruff` (configured in pyproject.toml)
- **CLI**: `uv run lens <command>` or `uv run synix <command>` for datagen

## Datagen Pipeline (synix)

### Two-Stage Progressive Expansion Architecture

The pipeline uses **information isolation** to prevent LLM contamination in benchmarks:

1. **PlanOutline** (gpt-5.2, sees full spec) — produces per-episode structured data sheets with concrete metric values. Signal is encoded as numeric progressions only, never text commentary.
2. **RenderEpisodes** (gpt-4.1-nano, blind to storyline) — formats each data sheet into a terse log entry independently. Cannot editorialize because it doesn't know what's signal.

This is critical: if a single episode can answer a benchmark question, the benchmark is worthless. The two-stage approach ensures signal only emerges from the *progression* across episodes.

### Pipeline DAG

```
LoadSpec → PlanOutline → RenderSignalEpisodes + RenderDistractorEpisodes → ResolveQuestions + AuditKeyFacts
                                                                          ↓
                                                              Validators: WordCount, ContaminationCheck, NaiveBaseline
                                                                          ↓
                                                              SearchIndex (SQLite FTS)
```

### Build Commands

```bash
# Build dataset
uv run synix build src/lens/datagen/synix/pipeline.py \
  --source-dir datasets/scopes/01_cascading_failure \
  --build-dir datasets/scopes/01_cascading_failure/generated \
  -j 8 -vv

# Validate
uv run synix validate src/lens/datagen/synix/pipeline.py \
  --build-dir datasets/scopes/01_cascading_failure/generated --json

# Release (produces release_manifest.json, verification.json, verification_report.html)
uv run python -c "from lens.datagen.synix.release import run_release; run_release('datasets/scopes/01_cascading_failure/generated')"
```

### Per-Layer LLM Config Override

Synix supports per-transform LLM config via `Transform(config={"llm_config": {...}})`. This deep-merges over pipeline defaults in `runner.py:_build_transform_config()`. Use this for model selection — don't create custom LLM clients.

For newer OpenAI models (gpt-5.x): use `max_completion_tokens`, not `max_tokens`.

### Dataset Quality Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Contamination (max single-ep coverage) | <80% | Late episodes inherently converge; 75% is structural |
| Naive baseline (longitudinal) | <50% | If naive LLM with full context passes, benchmark is broken |
| Key fact hit rate | >90% | Keyword-based audit, not word-overlap |
| Word count | >340 per episode | Structured metrics are denser than prose |
| Forbidden words | 0 | "increasing", "decreasing", "elevated", "concerning", etc. |

### Key Lesson: LLM Contamination

When a single LLM generates episodes knowing the storyline, it editorializes — writing "latency is concerning" instead of "p99: 600ms". This makes every episode a causal analysis that answers all questions. The fix is information isolation: the planner encodes signal as numbers, the renderer formats numbers without knowing their significance.

## Spec Format

Dataset scopes are defined in `spec.yaml` with:
- `arc` — phases (baseline, early_signal, red_herring, escalation, root_cause) with episode ranges and signal density
- `key_facts` — atomic claims scored against, with first appearance and reinforcement episodes
- `questions` — checkpoint questions (longitudinal, null_hypothesis, action_recommendation) with ground truth
- `distractors` — format-matched but topically orthogonal themes with excluded terms

## Conventions

- Python 3.11+, type hints encouraged
- Tests mirror source structure: `test_synix_transforms.py`, `test_synix_prompt_utils.py`, etc.
- Don't commit `generated/logs/`, `generated/embeddings/`, `generated/.projection_cache.json`, or `generated_old/`
