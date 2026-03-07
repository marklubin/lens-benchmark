# Scoring V2

## Why The Scoring Changes

The prior scoring stack had too many overlapping or weakly grounded metrics and used hard gating in a way that distorted rankings and made results hard to reproduce and explain.

V2 scoring is designed to be:

- simpler
- easier to audit
- easier to rerun
- less sensitive to arbitrary policy choices

## Platform Boundary

Scoring stays in the LENS benchmark layer.

Synix should expose the artifact and tool payloads needed for scoring, but scoring logic, weighting, audit policy, and report generation should not move into Synix.

## Removed From The Primary Score

These are removed from the primary composite:

- hard gating
- budget compliance
- reasoning quality
- insight depth
- longitudinal advantage
- naive baseline advantage
- action quality

These may still be logged as diagnostics if useful.

## Primary Metrics

### 1. `fact_f1`

Measures whether the answer captures the canonical key facts for the question.

This is the main correctness signal.

### 2. `evidence_support`

Measures whether the cited evidence actually supports the answer's claims.

This is the main evidence-grounding signal.

### 3. `citation_validity`

Mechanical metric that checks whether cited refs resolve to canonical source evidence.

## Primary Composite

```text
primary_score = 0.5 * fact_f1 + 0.3 * evidence_support + 0.2 * citation_validity
```

No hard gates.

## Secondary Diagnostics

Track but do not include in the primary composite:

- latency
- prompt tokens
- completion tokens
- estimated cost
- retrieval count
- tool count
- budget overrun

## Scoring Pipeline

### Step 1: Fact Matching

Inputs:

- question
- canonical answer
- canonical key facts
- candidate answer

Output:

- `fact_precision`
- `fact_recall`
- `fact_f1`

### Step 2: Citation Validation

Inputs:

- cited refs
- run artifact state
- raw evidence store

Output:

- `citation_validity`
- list of invalid refs

### Step 3: Evidence Support

Inputs:

- question
- candidate answer
- cited evidence contents
- canonical evidence clusters

Output:

- `evidence_support`
- optional structured explanation for audit

## Human Audit

Audit at least `10%` of answers, stratified by:

- scope family
- runtime policy
- score band

Audit should confirm:

- fact matching seems reasonable
- evidence support judgments are sane
- invalid citation handling is correct

## Statistical Reporting

Report:

- per-policy mean and confidence interval
- pairwise comparisons with correction
- effect sizes
- cost and latency alongside score, but not inside it

## Policy Notes

### Budget

Budget is not part of the primary score.

Reason:

Budget is an operational constraint, not a correctness metric. It should be reported separately and may be used as a study filter or cost analysis dimension, but it should not zero or suppress otherwise valid answers.

### Determinism

Because primary runs should be near-deterministic and replayable, scorer outputs must be stored as first-class artifacts and rerunnable without new answer generation.
