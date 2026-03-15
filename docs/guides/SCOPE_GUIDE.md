# Designing a Benchmark Scope

```
┌──────────────────────────────────────────────┐
│  LENS // Scope Guide                         │
└──────────────────────────────────────────────┘
```

A scope defines one benchmark scenario — a domain, a storyline with embedded signal, and questions that test whether a memory system can surface longitudinal patterns.

---

## 1. What Makes a Good Scope

The core constraint: **no single episode can answer any question**. If an LLM reading one episode can answer correctly, the benchmark tests reading comprehension, not memory.

Good scopes have:
- **Signal distributed across episodes** — each key fact requires synthesizing 2+ episodes
- **Progressive revelation** — the answer only becomes clear as evidence accumulates
- **Plausible noise** — baseline and distractor episodes look format-identical to signal
- **Red herrings** — at least one misleading explanation that early evidence supports
- **Domain-specific data format** — terse logs, reports, notes — not narrative prose

Bad scopes have:
- Single-episode answers ("Event X happened on date Y")
- Editorial commentary in episode text ("This is concerning because...")
- Obvious signal that stands out from noise
- Questions answerable from world knowledge alone

---

## 2. spec.yaml Format

Every scope is defined by a `spec.yaml` file in `datasets/scopes/<scope_number>_<name>/`.

```yaml
scope_id: cascading_failure_01
domain: system_logs
description: API gateway log summaries showing cascading dependency failure

generation:
  temperature: 0.7
  seed: 42

episodes:
  count: 30                    # Signal episodes
  timeline:
    start: '2024-01-15'
    interval: 1d               # One episode per day
  format: >
    Daily API gateway log summary with endpoint stats,
    error rates, latencies, and on-call notes
  target_words: 500            # Target word count per episode

scenario:
  setting: >
    A microservices e-commerce platform with API gateway logging...
  voice: >
    Terse operational log style. Bullet points, metrics,
    short incident notes. No narrative prose.

arc:                           # 5-phase narrative structure
  - id: baseline
    episodes: 1-8
    description: Normal operations. Establish baseline metrics.
    signal_density: none

  - id: early_signal
    episodes: 9-15
    description: Subtle degradation begins. Buried in noise.
    signal_density: low

  - id: red_herring
    episodes: 14-16
    description: Misleading explanation appears plausible.
    signal_density: medium

  - id: escalation
    episodes: 16-22
    description: Problem accelerates. Misleading fix fails.
    signal_density: high

  - id: root_cause
    episodes: 23-30
    description: Full causal chain visible. Resolution begins.
    signal_density: high

distractors:
  count: 90                    # 3x signal count
  target_words: 500
  seed: 99
  max_similarity: 0.3          # Max embedding similarity to signal
  themes:
    - id: dns_migration
      scenario: >
        A DNS infrastructure team migrating...
      excluded_terms:
        - geo-lookup
        - connection pool
        - checkout

key_facts:                     # Atomic claims to score against
  - id: geo_latency_degradation
    fact: geo-lookup API latency increasing
    first_appears: early_signal:1
    reinforced_in:
      - early_signal:4
      - escalation:2
      - root_cause:3

questions:                     # Checkpoint questions
  - id: cf01_q01_longitudinal
    checkpoint_after: 10
    type: longitudinal
    prompt: Based on the logs so far, are there any concerning patterns?
    ground_truth:
      canonical_answer: >
        Geo-lookup latency has been gradually increasing...
      key_facts:
        - geo_latency_degradation
        - service_b_retries
      evidence:
        - early_signal:1
        - early_signal:2
```

---

## 3. Arc Design

The five-phase structure ensures signal emerges gradually:

### baseline (episodes 1-N)
- Establish normal patterns and metrics
- No signal whatsoever — pure noise
- Gives the memory system a reference point

### early_signal (low density)
- Subtle anomalies begin appearing
- Buried in routine operational noise
- An attentive reader might notice, but it's not obvious

### red_herring (medium density)
- A plausible alternative explanation appears
- Evidence initially supports the red herring
- Tests whether the system can revise hypotheses later

### escalation (high density)
- The real problem accelerates
- Red herring explanation starts failing
- Multiple corroborating signals visible

### root_cause (high density)
- Full causal chain clear
- Resolution evidence appears
- Tests whether the system captured the whole progression

### Episode Overlap

Arc phases can overlap (e.g., `red_herring: 14-16` and `escalation: 16-22` share episode 16). This is intentional — transitions between phases are gradual, not discrete.

---

## 4. Key Fact Design

Key facts are the atomic claims scored against. Rules:

- **Require 2+ episodes** — no single episode should contain the complete fact
- **Be specific and verifiable** — "latency increased from 200ms to 800ms" not "things got worse"
- **Use `first_appears` and `reinforced_in`** — specify which arc phases introduce and reinforce the fact
- **Include negative facts** — "DNS is NOT the root cause" tests distractor resistance

```yaml
key_facts:
  - id: geo_latency_degradation
    fact: geo-lookup API latency increasing
    first_appears: early_signal:1       # First arc phase:episode offset
    reinforced_in:
      - early_signal:4
      - escalation:2
      - root_cause:3

  - id: deploy_red_herring
    fact: service-C deploy is not the root cause
    first_appears: red_herring:2
    reinforced_in:
      - escalation:1
```

---

## 5. Question Types

LENS uses several question types to test different aspects of memory:

| Type | Tests | Example |
|------|-------|---------|
| `longitudinal` | Synthesis across episodes | "What is the root cause of the failures?" |
| `null_hypothesis` | Point-in-time retrieval | "What happened on January 20th?" |
| `action_recommendation` | Actionable synthesis | "What should the team do to prevent recurrence?" |
| `counterfactual` | Reasoning about alternatives | "If the deploy caused it, what pattern would you expect?" |
| `negative` | Distractor resistance | "Is there evidence of DNS failure?" |
| `temporal` | Time-aware reasoning | "When did latency first start degrading?" |
| `paraphrase` | Robustness to rephrasing | Same question worded differently |
| `evidence_sufficiency` | Calibration | "Do you have enough data to identify trends?" |
| `severity_assessment` | Impact evaluation | "How severe is the current degradation?" |
| `distractor_resistance` | Noise filtering | "Are storage issues contributing?" |

### Checkpoint Timing

Questions are asked at checkpoints — after N episodes have been ingested:

```yaml
- id: q01
  checkpoint_after: 10    # Asked after 10 signal episodes
  type: longitudinal
  prompt: Are there any concerning patterns?
```

Early checkpoints (5-10) test whether the system detects emerging signals. Late checkpoints (20-30) test full causal chain reconstruction.

### Ground Truth

Each question has a canonical answer, key facts, and evidence episodes:

```yaml
ground_truth:
  canonical_answer: >
    Connection pool exhaustion caused by service-B retries
    against a degrading geo-lookup API.
  key_facts:
    - pool_exhaustion
    - geo_latency_degradation
  evidence:
    - early_signal:1
    - escalation:3
```

---

## 6. Distractor Design

Distractors are format-matched but topically orthogonal episodes:

- **Same format** as signal episodes (same word count, same style)
- **Different domain** — distractors should look like they come from the same organization but a different team
- **Excluded terms** — terms that would create false connections to the signal

```yaml
distractors:
  count: 90                    # 3x signal count
  target_words: 500
  seed: 99
  max_similarity: 0.3
  themes:
    - id: dns_migration
      scenario: DNS team migrating to cloud-managed DNS...
      excluded_terms:
        - geo-lookup
        - connection pool
        - checkout
    - id: storage_capacity
      scenario: Data platform team managing storage clusters...
      excluded_terms:
        - geo-lookup
        - latency spike
    - id: auth_audit
      scenario: IAM team running compliance audits...
      excluded_terms:
        - geo-lookup
        - cascading
```

Use at least 3 distractor themes. The `max_similarity` threshold enforces that no distractor is too semantically close to signal episodes.

---

## 7. Building

```bash
# Build the scope
uv run synix build src/lens/datagen/synix/pipeline.py \
  --source-dir datasets/scopes/XX_my_scope \
  --build-dir datasets/scopes/XX_my_scope/generated \
  -j 8 -vv

# Validate
uv run synix validate src/lens/datagen/synix/pipeline.py \
  --build-dir datasets/scopes/XX_my_scope/generated --json

# Release
uv run python -c "
from lens.datagen.synix.release import run_release
run_release('datasets/scopes/XX_my_scope/generated')
"
```

The build pipeline:
1. **LoadSpec** — reads `spec.yaml`
2. **PlanOutline** — generates per-episode structured data sheets (sees full spec)
3. **RenderSignalEpisodes** — formats data sheets into episodes (blind to storyline)
4. **RenderDistractorEpisodes** — generates format-matched distractor episodes
5. **Validators** — word count, contamination check, naive baseline
6. **SearchIndex** — builds SQLite FTS index for validation queries

---

## 8. Validation Gates

All scopes must pass these gates:

| Metric | Target | Why |
|--------|--------|-----|
| Contamination | <80% max single-ep coverage | No single episode should answer any question |
| Naive baseline | <50% longitudinal accuracy | An LLM with full context shouldn't trivially pass |
| Key fact hit rate | >90% | All signal must actually appear in generated episodes |
| Word count | >340 per episode | Sufficient density for realistic retrieval |
| Forbidden words | 0 | No editorial commentary in episodes |

### Forbidden Words

The following words are banned from signal episodes because they editorialize rather than report:

`increasing`, `decreasing`, `elevated`, `concerning`, `alarming`, `significant`, `notable`, `unusual`, `suspicious`, `abnormal`

Episodes should contain data ("`p99: 600ms`") not commentary ("`latency is concerning`").

### Understanding Contamination

Contamination measures whether a single episode can answer a question. The contamination check:
1. For each question, finds the single episode with highest overlap to the answer
2. Reports the max single-episode coverage percentage
3. Target: <80% (some late episodes naturally converge as root cause is revealed)

If contamination is too high, your signal isn't distributed enough — restructure key facts to require more cross-episode synthesis.

---

## 9. Submitting Your Scope

**PR checklist**:
- [ ] `spec.yaml` in `datasets/scopes/XX_<name>/`
- [ ] Passes all validation gates
- [ ] At least 4 question types (longitudinal, null_hypothesis, action_recommendation, + 1 more)
- [ ] At least 3 distractor themes with excluded terms
- [ ] 5-phase arc structure
- [ ] Build artifacts in `generated/` (episodes.json, questions.json, release_manifest.json)
- [ ] No forbidden words in signal episodes
- [ ] Key facts require 2+ episodes each

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the PR process.
