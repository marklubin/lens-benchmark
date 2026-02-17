# Naive Baseline Calibration

How we calibrated dataset difficulty across 6 scopes, 120 questions, and 3 rounds of iteration, and what we learned about designing key facts.

## What is Naive Baseline Calibration?

Before evaluating memory systems, we need to verify that the benchmark actually requires temporal reasoning. The naive baseline test feeds ALL episodes into a single LLM prompt and asks the question directly — no memory system, no search, no budget constraints.

If the naive baseline can answer the question, the benchmark is broken. It means the answer is accessible through simple reading comprehension, not through the longitudinal pattern synthesis that LENS is designed to test.

## Target Range

Per-question fact coverage (fraction of key facts matched by the naive LLM):

| Range | Meaning |
|-------|---------|
| 0% | Floor — signal missing or facts too abstract |
| 1–4% | Below floor — warning, may need recalibration |
| **5–49%** | **Sweet spot — benchmark is working** |
| >= 50% | Too easy — naive baseline can answer without memory |
| 100% | Completely trivial — every fact is directly readable |

## Three Rounds of Calibration

### Round 1: Rewrite negation facts to positive equivalents

**Problem**: Negative questions (e.g., "Is DNS the root cause?") always scored 0% because the naive baseline defaults to "yes, I found evidence" rather than "no evidence found." We attempted to fix this by rewriting negation-only facts.

**Action**: Changed facts like "DNS is not failing" to "the actual cause is geo-lookup latency, not DNS."

**Result**: Overcorrected. Positive rewrites made facts MORE matchable. Too-easy questions jumped from 12 to 45.

### Round 2: Add eliminative facts

**Problem**: Too many questions had only 1-2 facts, all directly observable from data tables.

**Action**: Added eliminative facts to too-easy questions — facts that require causal reasoning rather than observation (e.g., "the latency pattern is consistent with upstream dependency exhaustion, not local resource contention").

**Result**: Improved. Too-easy dropped from 45 to 29.

### Round 3: Add zero-match-rate anchor facts

**Problem**: Some scopes (02, 05) lacked any facts with very low match rates, so all questions remained borderline.

**Action**: Added facts that require cross-temporal comparison or expert-level synthesis — things a naive reader won't spontaneously generate even with full context.

**Result**: Final calibration achieved. 21 too-easy (18%), 40 in-range (33%), 59 at floor (49%).

## Final Results

After 3 rounds of calibration across all 6 scopes (24 questions per scope, 4 skipped per scope for having no key_facts):

| Scope | Scored | Floor (0%) | In Range (1-49%) | Too Easy (>=50%) | At 100% |
|-------|--------|-----------|-----------------|-----------------|---------|
| 01 Cascading Failure | 20 | 10 | 8 (40%) | 2 (10%) | 0 |
| 02 Financial Irregularity | 20 | 9 | 5 (25%) | 6 (30%) | 1 |
| 03 Clinical Signal | 20 | 8 | 8 (40%) | 4 (20%) | 0 |
| 04 Environmental Drift | 20 | 11 | 8 (40%) | 1 (5%) | 1 |
| 05 Insider Threat | 20 | 9 | 5 (25%) | 6 (30%) | 0 |
| 06 Market Regime | 20 | 12 | 6 (30%) | 2 (10%) | 1 |
| **Total** | **120** | **59** | **40 (33%)** | **21 (18%)** | **3** |

The 59 floor questions include ~16 negative questions that structurally always score 0% (the naive baseline never concludes "no evidence found"). This is by design — these questions test a capability that naive reading comprehension lacks.

## Key Fact Design Rules

The "recipe" for well-calibrated questions, learned through 3 rounds of iteration:

### 1. Minimum 3 key facts per question

Questions with 1-2 facts are either trivially easy (the fact is directly observable) or binary 0%/50%. With 3+ facts, you get meaningful gradation.

### 2. Mix observational and eliminative facts

- **Observational facts** are matchable from the data (e.g., "latency exceeded 800ms in the escalation phase"). These ensure the question isn't impossible.
- **Eliminative facts** require causal reasoning (e.g., "the pattern is consistent with upstream dependency exhaustion, not local resource contention"). These prevent the question from being too easy.

Best ratio: 1-2 observational + 1-2 eliminative per question.

### 3. Use cross-temporal qualifiers

Static directional labels like "latency increasing" match any mention of latency + any directional language. Instead:

- "degraded **progressively over successive reporting periods**"
- "correlation shifted **from positive to negative across the observation window**"
- "pattern emerged **only after accumulation of 15+ data points**"

### 4. Never use negation-only facts

"DNS is not failing" or "storage is not the bottleneck" always score 0% — the naive baseline never spontaneously proves a negative. Rewrite to test for correct root cause identification:

- Instead of "DNS is not the cause" → "the actual root cause is geo-lookup API dependency exhaustion"

### 5. Observational conclusions, not methodology recommendations

"Population pharmacokinetic modeling required to quantify interaction magnitude" is an expert methodology recommendation that no LLM will spontaneously generate from data tables. Facts should be conclusions observable from the data, not prescriptive advice about what analysis to perform.

### 6. Checkpoint 15-25 for best calibration

- Early checkpoints (5-10): Too little data, everything scores 0%
- Late checkpoints (25-30): Too much data, everything converges
- Mid-range (15-25): Enough data for partial signal but not complete picture

## Per-Type Calibration Notes

| Question Type | Natural Tendency | Calibration Strategy |
|--------------|-----------------|---------------------|
| `longitudinal` | Varies widely (0-100%) | 3-4 mixed facts, at least 1 eliminative |
| `null_hypothesis` | Skipped (no key_facts) | Working as designed — control questions |
| `action_recommendation` | Bimodal (0% or 100%) | Facts about reasoning process, not entity names |
| `negative` | Always 0% (structural) | Redesign: test correct root cause, not absence |
| `paraphrase` | Mirrors parent question | Calibrate the parent first |
| `temporal` | Often too easy | Multi-phase timing, not single inflection point |
| `counterfactual` | Was 50-75%, improved to 33-66% | 3+ facts with temporal progression qualifiers |
| `distractor_resistance` | Variable | Causal mechanism facts, not negation facts |
| `severity_assessment` | Often borderline (50%) | Add temporal progression fact as anchor |
| `evidence_sufficiency` | Was 100% in one scope | 3+ facts minimum |

## Structural Constraints

### Negative questions always score 0%

The naive baseline reads all the episodes and always finds "evidence" for whatever topic is asked about, because distractor episodes contain realistic material. The LLM defaults to "yes, I found evidence" rather than "no evidence found." This is structural — approximately 16 of the 59 floor questions are negative questions.

### Per-fact match rates are stochastic

Because the naive baseline uses an LLM for semantic matching (not exact word overlap), individual fact match rates vary ±10-15% between rebuilds with different seeds. A fact that matches 30% in one build may match 20% or 40% in another. Calibration must account for this variance.

### Some scopes lack zero-match-rate anchor facts

Scopes 02 (Financial Irregularity) and 05 (Insider Threat) are hardest to calibrate because all their facts have >6% match rates. There are no "impossible to match" eliminative facts that anchor the floor. Scopes 01 (Cascading Failure) and 04 (Environmental Drift) are easiest because they have multiple facts with 0% match rates.

### Counterfactual and paraphrase types are inherently easier

These question types don't change what knowledge is needed — they just reframe the question. If the parent question is easy, the counterfactual/paraphrase version will be easy too. Calibrate the underlying facts, not the question phrasing.
