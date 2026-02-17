# Dataset Generation Methodology

How LENS datasets are generated, why the pipeline is structured the way it is, and what validation gates ensure quality.

## The Contamination Problem

The central challenge in generating benchmark datasets with LLMs is **contamination**: when a single LLM generates episodes knowing the storyline, it editorializes.

Instead of writing:
```
p99_latency: 847ms
connection_pool_waiting: 14 threads
retry_rate: 0.23
```

It writes:
```
Latency is increasingly concerning at 847ms, suggesting a cascading failure pattern.
Connection pool exhaustion appears to be worsening, pointing to upstream dependency issues.
```

When this happens, every individual episode becomes a causal analysis that answers all the benchmark questions. A benchmark where any single episode answers the question is worthless — it tests search, not longitudinal reasoning.

## Two-Stage Pipeline Architecture

The fix is **information isolation**. The pipeline uses two LLMs with different context:

### Stage 1: PlanOutline (sees full spec)

- **Model**: gpt-5.2 (high-capability planner)
- **Input**: Complete spec including arc phases, key facts, signal placements, questions
- **Output**: Per-episode structured data sheets with concrete metric values

The planner encodes signal as **numeric progressions only** — never text commentary. It decides what numbers go in each episode to create the pattern, but the output is a structured data sheet, not prose.

### Stage 2: RenderEpisodes (blind to storyline)

- **Model**: gpt-4.1-nano (cheap, fast renderer)
- **Input**: Shared context (voice, format, entity names) + one data sheet
- **Does NOT receive**: Arc structure, key facts, signal placements, questions

The renderer formats each data sheet into a terse log entry independently. It cannot editorialize because it doesn't know what's signal and what's noise.

### Why This Works

Signal only emerges from the **progression** across episodes, not from any single episode. A single episode shows "p99: 847ms" — meaningful only if you've tracked it climbing from 120ms over the previous 10 episodes. The renderer can't insert editorial commentary because it has no idea that 847ms is abnormal.

## Pipeline DAG

Implemented in `src/lens/datagen/synix/pipeline.py` using the Synix build system:

```
LoadSpec (reads spec.yaml)
    │
    ▼
PlanOutline (gpt-5.2, full context)
    │
    ├──► RenderSignalEpisodes (gpt-4.1-nano, blind, parallel per episode)
    │
    ├──► RenderDistractorEpisodes (gpt-4.1-nano, blind, parallel per episode)
    │
    ├──► ResolveQuestions (maps phase-relative refs to episode IDs)
    │
    └──► AuditKeyFacts (keyword-based coverage check)
            │
            ▼
        Validators: WordCount, ContaminationCheck, NaiveBaseline
            │
            ▼
        SearchIndex (SQLite FTS + semantic embeddings)
```

### Transform Details

**LoadSpec** (`Source`): Reads `spec.yaml` from source directory, validates structure, computes content hash.

**PlanOutline** (`Transform`): Produces one signal outline + one distractor outline per theme. Supports pre-generated outlines at `source_dir/signal_outline.json` and `source_dir/distractor_outlines/{theme_id}.json` to avoid repeated LLM calls during iteration.

**RenderSignalEpisodes** (`Transform`): `split()` returns one group per episode for parallel execution. Each episode is rendered independently with only the data sheet and shared formatting context.

**RenderDistractorEpisodes** (`Transform`): Same isolation as signal rendering, but for distractor themes. Distractors are format-matched (same voice, same structure) but topically orthogonal.

**ResolveQuestions** (`Transform`): Maps phase-relative evidence references (e.g., "early_signal:3") to concrete episode IDs. Maps key fact IDs to fact text.

**AuditKeyFacts** (`Transform`): Keyword-based check that key facts are actually represented in the generated episodes. Uses domain-specific indicator terms.

## Spec Format

Dataset scopes are defined in `spec.yaml`:

### Arc

Five narrative phases with episode ranges and signal density:

```yaml
arc:
  - id: baseline
    episodes: 1-8
    signal_density: none
    description: Normal operations. Establish baseline metrics.

  - id: early_signal
    episodes: 9-15
    signal_density: low
    description: Subtle anomaly emerging, buried in noise.

  - id: red_herring
    episodes: 16-20
    signal_density: medium
    description: Misleading alternate explanation introduced.

  - id: escalation
    episodes: 21-26
    signal_density: high
    description: Signal becomes unmistakable.

  - id: root_cause
    episodes: 27-30
    signal_density: high
    description: Root cause identified or confirmable.
```

### Key Facts

Atomic ground-truth claims that scoring checks against:

```yaml
key_facts:
  - id: kf_latency_progression
    fact: "geo-lookup API latency degraded progressively over successive reporting periods"
    first_appears: early_signal:2
    reinforced_in:
      - escalation:1
      - root_cause:2
```

Each fact has a `first_appears` phase reference and `reinforced_in` list. These map to concrete episode IDs at build time.

### Questions

Ten question types with checkpoints and ground truth:

```yaml
questions:
  - id: cf01_q01_longitudinal
    type: longitudinal
    checkpoint_after: 30
    prompt: "What pattern of system degradation has emerged?"
    ground_truth:
      canonical_answer: "..."
      evidence:
        - early_signal:2
        - escalation:3
      key_facts:
        - kf_latency_progression
        - kf_pool_exhaustion
        - kf_not_dns
```

### Distractors

Format-matched but topically orthogonal themes:

```yaml
distractors:
  count: 90
  themes:
    - id: auth_audit
      topic: "Authentication system audit findings"
      excluded_terms: ["latency", "geo-lookup", "connection pool"]
    - id: dns_migration
      topic: "DNS provider migration project updates"
      excluded_terms: ["timeout", "retry", "cascade"]
```

## Validation Gates

### WordCount

Checks that signal and distractor episodes meet a minimum word count (default: 340 words). Short episodes lack enough metric data to be realistic.

### ContaminationCheck

LLM-based test: can synthesis questions be answered from a single episode?

For each synthesis question type (longitudinal, negative, temporal, counterfactual, paraphrase, distractor_resistance, severity_assessment, evidence_sufficiency), tests each signal episode individually. Computes key fact coverage from the single-episode answer.

- **Threshold**: Max single-episode coverage < 80%
- **Output**: `contamination_results.json` with per-question, per-episode scores

Late episodes inherently converge (they contain more signal), so ~75% coverage for the last few episodes is structural, not a bug.

### NaiveBaseline

LLM-based test: can a naive LLM with full episode context answer the question?

Feeds ALL episodes (signal + sampled distractors) into a single LLM prompt and asks the question. Uses LLM-as-judge for per-fact semantic matching (not word overlap).

Three-tier thresholds per question-type average:
- **Floor** (< 5%): Warning — signal may be missing or key facts poorly calibrated
- **Sweet spot** (5–49%): Benchmark is working as intended
- **Fail** (>= 50%): Error — benchmark is too easy; a naive LLM can answer without memory

Output: `baseline_results.json` with per-question scores and per-fact match details.

### KeyFactAudit

Keyword-based check that key facts are represented in generated episodes. Uses domain-specific indicator terms extracted from the fact text. Target hit rate: >90%.

## Question Types

| Type | What it tests | Calibration notes |
|------|--------------|-------------------|
| `longitudinal` | Synthesizing across many episodes to identify patterns | 3-4 mixed facts, at least 1 eliminative |
| `null_hypothesis` | Answerable from a single specific episode (control) | Skipped in baseline (no key_facts) |
| `action_recommendation` | Longitudinal insight + judgment for decision-making | Facts about reasoning process, not just entities |
| `negative` | Correctly identifying that a suspected cause is NOT supported | Always scores 0% on naive baseline (structural) |
| `paraphrase` | Same question rephrased — tests robustness | Mirrors parent question's difficulty |
| `temporal` | When did X start / what was the progression? | Multi-phase timing, not single inflection point |
| `counterfactual` | What would happen if X? / What if Y were different? | 3+ facts with temporal progression qualifiers |
| `distractor_resistance` | Correctly ignoring topically similar but irrelevant episodes | Causal mechanism facts, not negation facts |
| `severity_assessment` | How severe is the issue? What's the impact? | Add temporal progression fact |
| `evidence_sufficiency` | Is there enough evidence to support the conclusion? | 3+ facts minimum |

## Output Artifacts

A successful build produces:

| Artifact | Location | Description |
|----------|----------|-------------|
| `spec.json` | `layer0-spec/` | Parsed and validated spec |
| Signal outline | `layer1-outline/signal_outline.json` | Per-episode data sheets from planner |
| Distractor outlines | `layer1-outline/distractor_outline_*.json` | Per-theme distractor data sheets |
| Signal episodes | `layer2-signal_episodes/` | Rendered signal episodes (one JSON per episode) |
| Distractor episodes | `layer2-distractor_episodes/` | Rendered distractor episodes |
| Questions | `layer3-questions/` | Resolved questions with concrete episode refs |
| Key fact audit | `layer3-key_fact_audit/` | Fact coverage report |
| Baseline results | `baseline_results.json` | Naive baseline scores per question |
| Contamination results | `contamination_results.json` | Per-episode contamination scores |
| Search index | `search.db` | SQLite FTS + semantic embeddings |
| Manifest | `manifest.json` | Build provenance and artifact checksums |
