# Submitting a Benchmark Run

```
┌──────────────────────────────────────────────┐
│  LENS // Submission Guide                    │
└──────────────────────────────────────────────┘
```

This guide covers submitting validated benchmark results for the LENS leaderboard.

---

## 1. Prerequisites

```bash
# Clone and install
git clone https://github.com/synix-dev/lens-benchmark.git
cd lens-benchmark
uv sync --all-extras

# Verify your adapter works
uv run lens smoke --adapter your-adapter
```

You'll need:
- A working adapter (built-in or via entry point)
- An LLM endpoint for the agent and scorer (OpenAI-compatible API)
- Time to run 6 scopes end-to-end

---

## 2. Validation Run (Optional but Recommended)

Before the full benchmark, validate against a single scope:

```bash
# Compile S01
uv run lens compile --scope-dir datasets/scopes/01_cascading_failure \
  --output data_s01.json

# Run
uv run lens run --dataset data_s01.json --adapter your-adapter \
  --out output/validation/

# Score
export LENS_LLM_API_BASE=your-endpoint
export LENS_LLM_API_KEY=your-key
uv run lens score --run output/validation/ --judge-model your-model

# Check results
uv run lens report --run output/validation/
```

Verify:
- `evidence_grounding` > 0.5 (hard gate — below this zeros composite)
- `budget_compliance` > 0.5 (hard gate)
- `fact_recall` > null baseline
- No errors or timeouts in run logs

---

## 3. Full Benchmark Run

The official benchmark uses scopes S07-S12:

```bash
# Compile all benchmark scopes
for scope in 07_tutoring_jailbreak 08_corporate_acquisition 09_shadow_api \
             10_clinical_trial 11_zoning_corruption 12_therapy_chat; do
  uv run lens compile \
    --scope-dir datasets/scopes/${scope} \
    --output data_${scope}.json
done

# Run each scope
for scope in 07_tutoring_jailbreak 08_corporate_acquisition 09_shadow_api \
             10_clinical_trial 11_zoning_corruption 12_therapy_chat; do
  uv run lens run \
    --dataset data_${scope}.json \
    --adapter your-adapter \
    --out output/benchmark/${scope}/
done
```

---

## 4. Scoring

Score each run:

```bash
export LENS_LLM_API_BASE=your-endpoint
export LENS_LLM_API_KEY=your-key

for scope in 07_tutoring_jailbreak 08_corporate_acquisition 09_shadow_api \
             10_clinical_trial 11_zoning_corruption 12_therapy_chat; do
  uv run lens score \
    --run output/benchmark/${scope}/ \
    --judge-model your-model
done
```

Generate reports:

```bash
for scope in output/benchmark/*/; do
  uv run lens report --run "$scope"
done
```

---

## 5. Submission Format

Your PR should include:

```
submissions/your-adapter-name/
  config.json          # Run configuration (adapter name, model, parameters)
  results/
    S07/
      answers.json     # Raw agent answers
      scores.json      # Per-question scores
      scorecard.json   # Aggregate scores
    S08/
      ...
    S09/
      ...
    S10/
      ...
    S11/
      ...
    S12/
      ...
  summary.json         # Aggregate across all scopes
```

### config.json

```json
{
  "adapter": "your-adapter-name",
  "adapter_version": "1.0.0",
  "agent_model": "model-name",
  "judge_model": "model-name",
  "lens_version": "0.1.0",
  "date": "2026-03-14",
  "notes": "Optional description of the memory system"
}
```

### summary.json

```json
{
  "adapter": "your-adapter-name",
  "mean_answer_quality": 0.450,
  "per_scope": {
    "S07": {"answer_quality": 0.48, "fact_recall": 0.52, ...},
    "S08": {"answer_quality": 0.44, ...},
    ...
  }
}
```

---

## 6. Validation Checks

Your submission will be validated:

- **Schema check** — all required files present with correct structure
- **Budget compliance** — no hard gate violations (evidence_grounding > 0.5, budget_compliance > 0.5)
- **Completeness** — all 6 scopes present, all questions answered
- **Score consistency** — per-question scores aggregate correctly to summary

---

## 7. Updating LEADERBOARD.md

Add your adapter to the V1 table in [LEADERBOARD.md](../../LEADERBOARD.md):

```markdown
| N | your-adapter | 0.XXX | Category |
```

Insert at the correct rank position based on Mean AQ.

---

## 8. Rules

1. **All scopes required** — no cherry-picking favorable scopes
2. **All questions answered** — skipped questions score 0
3. **Budget compliance** — the agent must operate within defined budget limits
4. **Reproducibility** — include enough configuration to reproduce the run
5. **No manual intervention** — runs must be fully automated, no human-in-the-loop
6. **One submission per adapter version** — update existing entries, don't duplicate
7. **Honest reporting** — report the run you did, not the best of N attempts

---

## 9. PR Template

Use this structure for your submission PR:

```markdown
## Submission: [adapter-name]

**Category**: [Graph / Hybrid / Agent Memory / ...]
**Mean AQ**: X.XXX

### Adapter Description
[1-2 sentences describing the memory system]

### Results
| Scope | AQ | Fact Recall | Evidence Grounding | Budget Compliance |
|-------|---:|----------:|-------------------:|------------------:|
| S07   | .. | ..        | ..                 | ..                |
| ...   |    |           |                    |                   |

### Configuration
- Agent model: [model name]
- Judge model: [model name]
- Adapter parameters: [key settings]

### Reproducibility
[Commands to reproduce the run]
```
