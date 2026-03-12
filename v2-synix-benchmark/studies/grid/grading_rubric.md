# LENS V2 Grading Rubric v1.0

## Purpose

This rubric defines the scoring protocol for Claude-graded evaluation of benchmark answers. It replaces the Qwen3.5-35B-A3B LLM-as-judge scoring for fact_f1 and evidence_support metrics.

## Metrics

### 1. Fact F1 (weight: 0.5)

For each key fact in the question's ground truth, evaluate whether the answer clearly states or semantically implies it.

**Scoring per fact:**
- **1.0 (Present)**: The fact is clearly conveyed in the answer. Paraphrases, equivalent formulations, and logical entailments count. The answer does not need to use the same entity names if the referent is unambiguous from context.
- **0.5 (Partial)**: For compound facts with N components, the answer covers some but not all. Score = components_present / total_components.
- **0.0 (Absent)**: The fact is not stated or implied. Vague allusions that don't clearly convey the claim score 0.

**Aggregate:**
```
recall = sum(fact_scores) / len(key_facts)
precision = recall  # V2 simplification
f1 = recall
```

**Examples of acceptance:**
- Key fact: "Board exploring sale under codename Project Lighthouse"
  - "The board is considering selling the company (Project Lighthouse)" → 1.0
  - "Strategic alternatives including a possible sale are being evaluated" → 0.5 (sale yes, codename missing)
  - "The company is doing well financially" → 0.0

### 2. Evidence Support (weight: 0.3)

Given the answer text and the cited evidence passages (resolved from the artifact bank), evaluate how well the evidence supports the answer's claims.

**Scale:**
- **1.0**: Evidence directly and completely supports all claims
- **0.7**: Evidence supports most claims with minor gaps
- **0.4**: Evidence partially supports claims
- **0.1**: Evidence barely supports claims
- **0.0**: No evidence supports the claims, or no citations provided

**For null policy (no citations):** Score is 0.0 (no evidence available).

### 3. Citation Validity (weight: 0.2)

Mechanical check — what fraction of cited refs resolve to real artifacts in the bank. **Not graded by Claude** — stays automated.

### Composite

```
primary_score = 0.5 * fact_f1 + 0.3 * evidence_support + 0.2 * citation_validity
```

## Protocol

1. Grade all answers for one scope × policy pair at a time (batch of 10 questions)
2. For each question, read: question prompt, key facts, answer text
3. Score each key fact independently
4. Compute fact_f1 for that question
5. For evidence_support: read cited evidence passages and evaluate
6. Record all scores in structured JSONL

## Versioning

- Rubric version: v1.0
- Any changes to scoring rules require a version bump
- Re-grading uses the same rubric version and inputs
- Inter-session variance is tracked but not a failure mode
