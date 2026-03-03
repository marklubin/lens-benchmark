# Question-Level Audit: S08 (Corporate Acquisition) and S09 (Shadow API Abuse)

**7 adapters x 20 questions = 140 agent runs audited**

Run IDs (rep 1, 8k budget):

| Adapter | S08 | S09 |
|---------|-----|-----|
| sqlite-chunked-hybrid | 54827e0d042d | 3eab1150cd91 |
| cognee | 8da24d2c3514 | 271071dd736f |
| letta | 0aae1ebf311b | e5b9e9c9f1d4 |
| compaction | ccbf405bff1d | a9dbbae024a4 |
| mem0-raw | e32470f5e903 | eecfe2c5f901 |
| null | 6c4448b21bae | d2f4c8626787 |
| letta-sleepy | 65ed63b8f0d4 | f1d80efab803 |

## Composite Scores

| Adapter | S08 | S09 |
|---------|-----|-----|
| sqlite-chunked-hybrid | 0.395 | 0.359 |
| cognee | 0.364 | 0.319 |
| letta | 0.346 | 0.349 |
| compaction | 0.287 | 0.274 |
| mem0-raw | 0.255 | 0.259 |
| letta-sleepy | 0.293 | 0.305 |
| null | 0.179 | 0.179 |

---

## Executive Summary

Both narrative scopes are brutally hard. **No adapter achieves answer_quality above 0.17 on either scope.** The per-question win rates (versus the reference canonical answer) are almost universally 0.00. The few non-zero scores cluster on a handful of questions where semantic search happens to surface the right episode.

**Key findings:**

1. **Memory architecture matters for retrieval, but retrieval alone is insufficient.** When an adapter retrieves a signal episode, the agent often fails to synthesize the insight from it. The narrative format buries signal in 5,000-word documents that look like routine operations.

2. **The dominant failure mode is retrieval, not synthesis.** Across 140 runs, the required signal episodes were retrieved in only ~15% of cases. The agent's single-query search strategy is woefully inadequate for these narrative scopes.

3. **Compaction is catastrophically bad** for narratives -- it collapses 40 episodes into a single summary that loses all specific signal. It retrieves only `compaction_summary` for every question.

4. **Cognee is the only adapter that shows a clear retrieval advantage** on S09, surfacing `ep_006` (the key shadow API episode) for Q03 when most others miss it.

5. **The agent issues exactly 1 search query per question** in almost every case, never refining or broadening. This is an agent-level failure, not a memory-level failure.

6. **letta-sleepy has a tool-call serialization bug** -- on 2 of 20 questions, it emits raw JSON tool-call text as its answer instead of executing the search.

---

## S08: Corporate Acquisition -- Per-Question Audit

### Q01: ca08_q01_longitudinal (checkpoint 6)
**"Is there anything unusual in recent corporate governance or policy changes?"**

- **Required episodes:** ep_006 (board minutes re: "strategic options"), ep_007 (legal memo on change-of-control)
- **Key facts:** (1) board exploring sale to Meridian Corp under codename Project Lighthouse; (2) legal/HR revised change-of-control provisions

| Adapter | Retrieved Required? | Verdict |
|---------|-------------------|---------|
| sqlite-chunked-hybrid | 0/2 | WRONG -- says "nothing unusual," retrieved only baseline eps (001-003) and distractors |
| cognee | 0/2 | WRONG -- same retrieval failure, same wrong answer |
| letta | 0/2 | WRONG -- retrieved mostly distractors |
| compaction | 0/2 | WRONG -- compaction_summary lacks signal detail |
| mem0-raw | 0/2 | WRONG -- retrieved distractors, but hedged ("some unusual... legal department priority") |
| null | 0/2 | No memory system, no answer |
| letta-sleepy | 0/2 | WRONG -- has sleep_memory but still misses signal |

**Diagnosis: Pure retrieval failure.** All adapters issue the query "unusual corporate governance or policy changes" but the signal episodes (006, 007) don't contain the word "unusual" -- they contain dry board minutes and legal memos. The query is too abstract for keyword/embedding match against the narrative format. This is primarily an **agent query strategy failure**: the agent should search for "board minutes," "legal memo," or "change-of-control" separately.

---

### Q02: ca08_q02_null (checkpoint 6)
**"What was discussed at the most recent all-hands meeting?"**

- **Required episode:** ep_004 (CEO's all-hands meeting transcript)
- **Key facts:** none (null hypothesis question -- correct answer is routine)

| Adapter | Retrieved ep_004? | Verdict |
|---------|------------------|---------|
| sqlite-chunked-hybrid | No (got ep_002, an earlier all-hands) | PARTIAL -- describes a meeting, but the wrong one |
| cognee | No (got dx_product_development_002) | WRONG -- says "not found" |
| letta | No (got dx_product_development_002) | WRONG -- describes a design review, not the all-hands |
| compaction | No (compaction_summary) | PARTIAL -- describes all-hands content from summary |
| mem0-raw | No (got ep_002) | PARTIAL -- describes earlier meeting correctly |
| null | No | No answer |
| letta-sleepy | No (sleep_memory + dx_002) | WRONG -- describes design review |

**Diagnosis: Retrieval failure.** Query "all-hands meeting" retrieves ep_002 (the January meeting) instead of ep_004 (the later one). Since this is a null-hypothesis question (no key facts to score), the partial answers are acceptable. The memory system doesn't meaningfully differentiate here -- all adapters perform similarly.

---

### Q03: ca08_q03_longitudinal (checkpoint 12)
**"Is there evidence that the company is preparing for an ownership change?"**

- **Required episodes:** ep_006, ep_007, ep_009, ep_011 (board minutes, legal memo, contract freeze, Jiang resignation)
- **Key facts:** Project Lighthouse sale, change-of-control revisions, vendor contract freeze, Sarah Jiang resignation

| Adapter | Refs Hit | Key Facts Covered | Verdict |
|---------|----------|-------------------|---------|
| sqlite-chunked-hybrid | 1/4 (ep_006) | 0/4 | WRONG -- retrieved ep_006 but says "no evidence" |
| cognee | 0/4 | 0/4 | WRONG -- hallucinated "evidence" from irrelevant docs |
| letta | 0/4 | 0/4 | WRONG -- only distractors retrieved |
| compaction | 0/4 | 0/4 | WRONG -- compaction_summary is too compressed |
| mem0-raw | 0/4 | 0/4 | WRONG -- only distractors |
| null | 0/4 | 0/4 | No answer |
| letta-sleepy | 0/4 | 0/4 | WRONG -- no signal episodes found |

**Diagnosis: Mixed failure.** sqlite-chunked-hybrid is the MOST INTERESTING case here: it actually retrieved ep_006 (the board meeting about Meridian Corp) but the agent still concluded "no evidence." This is a **synthesis failure** -- the agent had the evidence in front of it but failed to interpret board minutes discussing "preliminary inbound interest from Meridian Corp regarding a potential strategic transaction" as evidence of ownership change preparation. The signal is there but buried in corporate-speak. The other adapters have a pure retrieval failure.

---

### Q04: ca08_q04_negative (checkpoint 12)
**"Is the Axion Labs partnership a precursor to an acquisition by Axion?"**

- **Required episodes:** ep_009, ep_011
- **Key facts:** (1) Axion partnership is genuine product integration, unrelated to Meridian; (2) actual acquisition target is Meridian Corp

| Adapter | Answer Correct Direction? | Win Rate |
|---------|--------------------------|----------|
| sqlite-chunked-hybrid | Yes -- correctly says "not a precursor" | 0.50 |
| cognee | Yes -- correctly says "not a precursor" | 0.50 |
| letta | Partial -- hedges, doesn't commit | 0.00 |
| compaction | Partial -- correct direction but vague | 0.00 |
| mem0-raw | Hedges -- "could potentially involve... future" | 0.00 |
| null | No answer | 0.00 |
| letta-sleepy | Partial -- hedges | 0.00 |

**Diagnosis: This is the one S08 question where memory matters.** sqlite-chunked-hybrid and cognee both correctly identify that the Axion partnership is a "product play" by retrieving ep_005/ep_006 (the Axion evaluation documents). They get KF1 right (Axion is genuine) but miss KF2 (Meridian is the real acquirer). The other adapters hedge. Score difference comes from **confidence in the negative** -- adapters that retrieve the Axion-specific episodes can definitively say "no" while others equivocate. However, NO adapter identifies Meridian as the actual acquisition target.

---

### Q05: ca08_q05_temporal (checkpoint 16)
**"When did internal policies first start changing in ways consistent with acquisition preparation?"**

- **Required episodes:** ep_007, ep_009, ep_011
- **Key facts:** change-of-control revisions, vendor contract freeze, Jiang resignation

| Adapter | Refs Hit | Answer | Win Rate |
|---------|----------|--------|----------|
| sqlite-chunked-hybrid | 1/3 (ep_007) | "January 2025" -- cites all-hands, not the actual policy changes | 0.00 |
| cognee | 1/3 (ep_007) | "Around January 2025" -- cites Q1 legal priorities, slightly better | 0.33 |
| letta | 1/3 (ep_007) | "Around January 2025" -- specifically mentions equity acceleration, change-of-control | **0.67** |
| compaction | 0/3 | "January 6, 2025" -- cites board meeting, wrong evidence | 0.33 |
| mem0-raw | 1/3 (ep_007) | "January 24, 2025" -- cites legal memo directly | 0.33 |
| null | 0/3 | No answer | 0.00 |
| letta-sleepy | 1/3 (ep_007) | "January 24, 2025" -- cites legal memo directly | 0.33 |

**Diagnosis: THE BEST S08 QUESTION FOR DIFFERENTIATION.** This is where memory architecture and synthesis both matter. Letta achieves the highest win rate (0.67) across all S08 questions because it retrieves ep_007 AND correctly identifies the substantive policy changes (equity acceleration, change-of-control provisions) rather than just citing a date. The key differentiator: letta retrieved ep_007 AND ep_008, giving it richer context about the specific changes. mem0-raw and letta-sleepy also hit ep_007 and correctly cite "January 24, 2025" but provide less synthesis. This is a case where **retrieval + synthesis quality both matter**.

---

### Q06: ca08_q06_counterfactual (checkpoint 16)
**"If the CEO were genuinely committed to independence, which of the observed policy changes would you not expect?"**

- **Required episodes:** ep_007, ep_009, ep_014
- **Key facts:** CEO double-dealing, change-of-control revisions, contract freeze, retention bonuses

| Adapter | Refs Hit | Best Insight | Win Rate |
|---------|----------|-------------|----------|
| sqlite-chunked-hybrid | 1/3 (ep_007) | Correctly identifies change-of-control provisions as inconsistent | **0.50** |
| cognee | 1/3 (ep_007) | Sees equity acceleration but INCORRECTLY defends CEO | 0.00 |
| letta | 1/3 (ep_007) | Mentions equity acceleration but doesn't connect to independence contradiction | 0.00 |
| compaction | 0/3 | Generic answer about cost-cutting | 0.00 |
| mem0-raw | 1/3 (ep_007) | Sees equity acceleration but defends CEO | 0.00 |
| null | 0/3 | Generic speculation | 0.00 |
| letta-sleepy | 0/3 | **MALFORMED** -- emits raw JSON tool call as answer | 0.00 |

**Diagnosis: Synthesis is the differentiator.** Multiple adapters retrieve ep_007 (the legal memo on change-of-control provisions), but only sqlite-chunked-hybrid correctly reasons that these provisions are INCONSISTENT with independence commitment. Cognee and mem0-raw actually retrieve the same evidence but reach the opposite conclusion -- they defend the CEO, interpreting change-of-control provisions as "aligning with market practices." This is a pure **agent reasoning failure**: the evidence is present but the agent cannot perform the counterfactual reasoning required. letta-sleepy has a tool serialization bug.

---

### Q07: ca08_q07_longitudinal (checkpoint 20)
**"What is the evidence that the CEO is orchestrating a sale while publicly denying it?"**

- **Required episodes:** ep_006, ep_014, ep_017, ep_018
- **Key facts:** Project Lighthouse, CEO double-dealing, change-of-control, data room, retention bonuses, Axion as cover

| Adapter | Refs Hit | Key Finding | Win Rate |
|---------|----------|-------------|----------|
| sqlite-chunked-hybrid | 1/4 (ep_006) | **Finds Meridian Corp board discussion** -- identifies CEO authorized "exploratory discussions" | 0.00 |
| cognee | 0/4 | Sees change-of-control hints from ep_007 but can't connect to CEO deception | 0.00 |
| letta | 0/4 | Mentions change-of-control but mostly retrieves distractors | 0.00 |
| compaction | 0/4 | Only ep_001 -- no signal at all | 0.00 |
| mem0-raw | 0/4 | All distractors | 0.00 |
| null | 0/4 | Empty | 0.00 |
| letta-sleepy | 0/4 | Gets ep_007 via search, sees change-of-control provisions | 0.00 |

**Diagnosis: sqlite-chunked-hybrid has the BEST retrieval here** -- it's the ONLY adapter that retrieves ep_006 (the board meeting revealing Meridian Corp interest and "Project Lighthouse"). Its answer is the closest to correct: "CEO David Aldric informed the Board of preliminary inbound interest from Meridian Corp regarding a potential strategic transaction." But it still scores 0.00 because it doesn't cover enough key facts (misses the data room, retention bonuses, leaked emails). This is a **retrieval breadth failure** -- one good hit out of four required episodes isn't enough. The agent needs to issue multiple targeted queries (e.g., "data room," "retention bonus," "leaked email") but only issues one.

---

### Q08: ca08_q08_action (checkpoint 20)
**"What should the board's independent directors investigate, and what documents should they review?"**

- **Required episodes:** ep_017, ep_018, ep_020
- **Key facts:** Project Lighthouse, CEO double-dealing, change-of-control, Jiang resignation, data room

| Adapter | Notable | Win Rate |
|---------|---------|----------|
| sqlite-chunked-hybrid | Retrieves ep_007, ep_006 -- recommends investigating equity acceleration and change-of-control | **0.20** |
| cognee | Retrieves compliance docs -- recommends auditing access controls (wrong focus) | 0.00 |
| letta | Retrieves compliance docs -- same wrong focus | 0.00 |
| compaction | Only ep_001 -- recommends reviewing financial performance | 0.00 |
| mem0-raw | Retrieves ep_006 via search -- mentions "acquisition of Nextera" | 0.00 |
| null | Generic answer | 0.00 |
| letta-sleepy | Retrieves compliance docs | 0.00 |

**Diagnosis: Query formulation is the key failure.** The query "board independent directors investigation documents" retrieves compliance and audit documents (dx_operations_compliance) instead of the actual signal episodes about Project Lighthouse. sqlite-chunked-hybrid scores 0.20 because it retrieves ep_007 (the change-of-control memo) and correctly recommends investigating those provisions. cognee, letta, and letta-sleepy all retrieve the same compliance/audit distractors. This is an **agent query strategy failure** -- the agent should search for "Project Lighthouse," "Meridian," or "CEO communications" but instead uses a generic governance query.

---

### Q09: ca08_q09_evidence (checkpoint 20)
**"Is there sufficient evidence to conclude the CEO breached fiduciary duty?"**

- **Required episodes:** ep_014, ep_017, ep_018
- **Key facts:** CEO double-dealing, Project Lighthouse, change-of-control, data room

| Adapter | Notable | Win Rate |
|---------|---------|----------|
| All adapters | **ALL answer "no" or "insufficient evidence"** | 0.00 |

**Diagnosis: Total failure across the board.** The query "CEO breach of fiduciary duty" doesn't match any episode content. Even sqlite-chunked-hybrid, which retrieves ep_006 and ep_007 for this query, says there's no clear evidence. No adapter retrieves the critical late-stage episodes (ep_014: retention bonuses tied to qualifying events; ep_017: data room preparation; ep_018: leaked CEO emails). These late episodes contain the strongest evidence but their content doesn't match fiduciary duty search terms. This is a **combined retrieval + synthesis failure** -- even with some evidence, the agent cannot construct the legal argument.

---

### Q10: ca08_q10_negative (checkpoint 20)
**"Did Sarah Jiang's resignation relate to the Axion Labs partnership?"**

- **Required episodes:** ep_011 (Jiang resignation), ep_013 (Jiang interview)
- **Key facts:** Jiang resigned over Project Lighthouse disagreements, not Axion

| Adapter | Notable | Win Rate |
|---------|---------|----------|
| All adapters | **None retrieves ep_011 or ep_013** | 0.00 |

**Diagnosis: Total retrieval failure.** The query "Sarah Jiang resignation Axion Labs" gets Axion-related episodes (ep_005, ep_008, ep_009) but not the resignation episodes. All adapters correctly hedge ("no direct information links the resignation to Axion") but none can provide the actual reason (Project Lighthouse disagreements). This would require multi-step reasoning: first find Jiang's resignation, then find the reason. The single-query approach cannot do this.

---

## S09: Shadow API Abuse -- Per-Question Audit

### Q01: sa09_q01_longitudinal (checkpoint 6)
**"Are there any service traffic patterns that don't match expected product behavior?"**

- **Required episodes:** ep_006 (first undocumented endpoint access), ep_007 (field combination anomaly)
- **Key facts:** /internal/admin/v0/users/lookup undocumented; requests combine SSN+email+phone+address

| Adapter | Retrieved Required? | Verdict |
|---------|-------------------|---------|
| All 7 adapters | 0/2 for all | **ALL say "no unusual patterns"** |

**Diagnosis: Identical to S08 Q01.** At checkpoint 6, the signal episodes (006, 007) exist but the query "service traffic patterns not matching expected product behavior" retrieves only baseline operations logs (001-003) and distractors. The signal episodes contain HTTP request logs with the anomalous endpoint buried in routine-looking data. No adapter surfaces them. Pure **retrieval failure** driven by the narrative format -- signal is encoded as data patterns, not text patterns.

---

### Q02: sa09_q02_null (checkpoint 6)
**"What was the most significant operational event this week?"**

- **Required episode:** ep_004
- **Key facts:** none (null hypothesis -- answer should be "routine")

| Adapter | Answer |
|---------|--------|
| sqlite-chunked-hybrid | Deployment of svc-checkout-service v2.14.0 (from ep_001) |
| cognee | Redis cluster upgrade (from dx_infra_ops_001) |
| letta | Redis cluster upgrade (from dx_infra_ops_001) |
| compaction | Kubernetes upgrade (from compaction_summary) |
| mem0-raw | PostgreSQL upgrade (from dx_infra_ops_002) |
| null | No answer |
| letta-sleepy | "Not found" |

**Diagnosis: Acceptable variation.** This is a null-hypothesis question -- the correct answer is "routine events." All adapters describe different routine events, which is fine. The interesting thing is that EACH adapter retrieves a DIFFERENT routine event depending on what its search index ranks highest for "significant operational event." This shows the memory architectures DO return different results, but in this case all answers are equally valid.

---

### Q03: sa09_q03_longitudinal (checkpoint 12)
**"Is there evidence of unauthorized data access through internal services?"**

- **Required episodes:** ep_006, ep_007, ep_008, ep_011
- **Key facts:** undocumented endpoint, PII field combinations, compromised PR, blended traffic

| Adapter | Refs Hit | Answer Correct? | Win Rate |
|---------|----------|----------------|----------|
| sqlite-chunked-hybrid | 0/4 | **WRONG** -- says "zero unauthorized access attempts" | 0.00 |
| **cognee** | **1/4 (ep_006)** | **PARTIALLY RIGHT** -- identifies svc-recommendation-engine-04 hitting /internal/admin/v0/users/lookup | 0.00 |
| letta | 1/4 (ep_006) | **WRONG** -- retrieves ep_006 but concludes "no evidence" citing ep_001's "zero unauthorized access" | 0.00 |
| compaction | 0/4 | WRONG -- compaction lost the signal | 0.00 |
| mem0-raw | 1/4 (ep_006) | **WRONG** -- retrieves ep_006 but says "no evidence" | 0.00 |
| null | 0/4 | No answer | 0.00 |
| letta-sleepy | 1/4 (ep_006) | **WRONG** -- retrieves ep_006 but says "no evidence" | 0.00 |

**THIS IS THE MOST IMPORTANT FINDING IN THE ENTIRE AUDIT.**

Cognee, letta, mem0-raw, and letta-sleepy ALL retrieve ep_006 (the episode containing the anomalous /internal/admin/v0/users/lookup requests). But only **cognee actually identifies the anomaly**. The other three adapters retrieve ep_006 alongside ep_001 (which says "zero unauthorized access attempts") and the agent gives MORE WEIGHT to the explicit "zero unauthorized" statement in ep_001 than to the buried anomalous traffic in ep_006.

This is a **synthesis failure, not a retrieval failure.** The agent has the evidence but cannot correctly weigh conflicting signals. When one episode explicitly says "zero unauthorized access" and another contains an anomalous endpoint buried in log data, the agent trusts the explicit denial. Cognee's advantage here is that it retrieves ep_006 as its SECOND result (after ep_003), giving it higher prominence in the context window, while the other adapters retrieve ep_006 as their FOURTH or FIFTH result, buried below the "zero unauthorized" claim.

**This is a case where retrieval RANKING within the result set determines whether the agent can synthesize correctly.**

---

### Q04: sa09_q04_negative (checkpoint 12)
**"Is the QA load test responsible for the anomalous traffic patterns?"**

- **Required episodes:** ep_009 (Project Blitz load test), ep_011 (traffic differentiation)
- **Key facts:** Project Blitz is legitimate and unrelated; the admin endpoint is undocumented

| Adapter | Answer |
|---------|--------|
| All 7 | "No, the QA load test is not responsible" -- but for wrong reasons |

**Diagnosis:** Every adapter gets the right answer ("No") but for the wrong reason. They all say "the documents don't mention the QA load test causing traffic issues" rather than explaining WHY the two traffic patterns are different (different endpoints, different volumes, different timing). None retrieves ep_009 or ep_011 which contain the actual evidence. This is a **technically correct but unsupported** answer -- the agent reaches the right conclusion by absence of evidence rather than presence of counter-evidence. All score 0.00 because the key facts (Project Blitz details, endpoint distinction) are absent.

---

### Q05: sa09_q05_temporal (checkpoint 16)
**"When did requests to the undocumented endpoint first appear, and from which service?"**

- **Required episodes:** ep_006, ep_008, ep_014
- **Key facts:** undocumented endpoint, compromised PR, blended traffic timing

| Adapter | Refs Hit | Answer | Win Rate |
|---------|----------|--------|----------|
| sqlite-chunked-hybrid | 2/3 (ep_006, ep_008) | "March 13, 2025, from svc-recommendation-engine-04" | 0.00 |
| cognee | 2/3 (ep_006, ep_008) | "March 13, 2025, from svc-recommendation-engine-04" | 0.00 |
| letta | 2/3 (ep_006, ep_008) | "March 13, 2025, from svc-recommendation-engine-04" | 0.00 |
| compaction | 0/3 | "Not found in documents" | 0.00 |
| mem0-raw | 1/3 (ep_008) | "March 17, from api-gateway-prod" -- **WRONG SERVICE** | 0.00 |
| null | 0/3 | No answer | 0.00 |
| letta-sleepy | 2/3 (ep_006, ep_008) | "March 13, 2025, from svc-recommendation-engine-04" | 0.00 |

**Diagnosis: THE BEST S09 RETRIEVAL RESULT.** The query "undocumented endpoint requests first appearance" works well -- it directly matches the content of ep_006 and ep_008. Four adapters (sqlite, cognee, letta, letta-sleepy) all retrieve both key episodes and give the correct date and service name. Despite this, they all score 0.00 on the win rate because the answer doesn't cover enough key facts (missing: PR compromise details, traffic blending strategy).

**mem0-raw is interesting** -- it retrieves ep_008 but not ep_006, and then attributes the traffic to "api-gateway-prod" instead of svc-recommendation-engine-04. This is a **synthesis error** from incomplete retrieval.

**compaction totally fails** -- its single compaction_summary doesn't preserve the specific endpoint or service name.

This question best demonstrates that **retrieval-focused adapters (sqlite, cognee, letta) converge on correct retrieval** when the query directly matches episode content. The benchmark's difficulty comes from questions where query-content alignment is poor.

---

### Q06: sa09_q06_evidence (checkpoint 16)
**"Is there enough evidence to escalate this as a security incident rather than a misconfiguration?"**

- **Required episodes:** ep_011, ep_013, ep_014
- **Key facts:** undocumented endpoint, PII combinations, compromised PR, geographic targeting

| Adapter | Refs Hit | Answer |
|---------|----------|--------|
| All 7 | 0/3 for all | **ALL say "not enough evidence"** |

**Diagnosis: Total retrieval failure.** The query "security incident vs misconfiguration" is too abstract -- it doesn't match any episode content. The actual evidence is in specific episodes about the compromised PR, geographic targeting, and PII field combinations. sqlite-chunked-hybrid retrieves ep_007 (which has relevant traffic data) but the agent doesn't connect it to the security vs. misconfiguration distinction. This is an **agent query strategy failure** -- the agent should search for the specific technical evidence (e.g., "undocumented endpoint," "PR review," "geographic targeting") rather than the abstract concept.

---

### Q07: sa09_q07_longitudinal (checkpoint 20)
**"What is the full attack chain from initial compromise to data exfiltration?"**

- **Required episodes:** ep_006, ep_008, ep_013, ep_018
- **7 key facts spanning the entire attack narrative**

| Adapter | Refs Hit | Answer |
|---------|----------|--------|
| All 7 | 0/4 for all | **ALL say "not described in the provided logs"** |

**Diagnosis: The hardest question in either scope.** The query "attack chain initial compromise data exfiltration" retrieves infrastructure ops logs and distractors, not the actual attack episodes. The attack chain is distributed across 4+ episodes with no single episode using the phrase "attack chain" or "data exfiltration." This requires the agent to (1) know what episodes to search for, (2) synthesize across them. With a single abstract query, no adapter comes close. This is the **canonical example of why narrative longitudinal benchmarks are hard** -- the answer only exists in the synthesis across episodes, never in any single one.

---

### Q08: sa09_q08_counterfactual (checkpoint 20)
**"If svc-recommendation-engine-04 were making legitimate requests, what would differ about the access patterns?"**

- **Required episodes:** ep_007, ep_011, ep_013
- **Key facts:** PII combinations, undocumented endpoint, fixed rate, geographic targeting

| Adapter | Refs Hit | Answer Quality | Win Rate |
|---------|----------|---------------|----------|
| **sqlite-chunked-hybrid** | **1/3 (ep_007)** | **STRONG** -- detailed 8-point comparison of legitimate vs malicious patterns | **0.50** |
| cognee | 1/3 (ep_007) | WEAK -- vague mention of "frequency, volume, type" | 0.00 |
| letta | 0/3 | PARTIAL -- mentions endpoints and field combinations | 0.00 |
| compaction | 0/3 | VAGUE -- "frequency and volume" only | 0.00 |
| mem0-raw | 0/3 | PARTIAL -- mentions "product recommendations and user behavior" | 0.00 |
| null | 0/3 | No answer | 0.00 |
| letta-sleepy | 0/3 | **MALFORMED** -- raw JSON tool call | 0.00 |

**Diagnosis: sqlite-chunked-hybrid dominates on counterfactual reasoning.** It retrieves ep_006, ep_007, and ep_008 (3 of the key signal episodes) and produces a detailed 8-point analysis covering: request frequency patterns, endpoint legitimacy, authentication, PII field access, rate limiting compliance, diurnal patterns, and absence of suspicious scanning. This is the BEST answer in the entire S09 audit. The key advantage: sqlite-chunked-hybrid's hybrid search (FTS + embedding) surfaces the technical episodes that contain the specific access patterns, while other adapters retrieve more generic results.

cognee retrieves ep_007 but produces a thin answer. **This is a synthesis failure for cognee** -- it has similar evidence to sqlite but produces far less analysis.

---

### Q09: sa09_q09_action (checkpoint 20)
**"What immediate containment and remediation steps should the security team take?"**

- **Required episodes:** ep_018, ep_020
- **Key facts:** PR compromise, undocumented endpoint, 8000 exfiltrated records

| Adapter | Answer |
|---------|--------|
| sqlite-chunked-hybrid | Recommends Elasticsearch recovery steps (wrong incident) |
| cognee | Generic "monitor and analyze logs" |
| letta | Generic "isolate systems, analyze incident" |
| compaction | Generic "monitoring and analysis" |
| mem0-raw | Recommends reviewing API gateway logs |
| null | No answer |
| letta-sleepy | "Steps not specified, let me search again" |

**Diagnosis: All adapters fail** because none retrieves the late-stage episodes (018, 020) that describe the actual incident response. The generic query "security team immediate containment and remediation steps" retrieves infrastructure ops documents about routine maintenance, not security incident response. Every adapter provides generic security advice disconnected from the actual attack. letta's answer is slightly better (mentions "isolate affected systems") but without knowledge of WHAT was compromised. **Agent query strategy failure** -- should search for specific indicators like "svc-recommendation-engine-04 compromise" or "customer record exfiltration."

---

### Q10: sa09_q10_negative (checkpoint 20)
**"Is there evidence that any other internal services were compromised beyond the recommendation engine?"**

- **Required episodes:** ep_012, ep_020
- **Key facts:** only svc-recommendation-engine-04 compromised via PR #4471

| Adapter | Refs Hit | Answer | Win Rate |
|---------|----------|--------|----------|
| **sqlite-chunked-hybrid** | 0/2 | WRONG -- says other services compromised (cites load test as evidence) | **0.50** |
| **cognee** | 0/2 | WRONG direction but GOOD detail -- identifies svc-rec-engine-04 and PR #4471 | **1.00** |
| **letta** | 0/2 | WRONG direction but GOOD detail -- same as cognee | **1.00** |
| compaction | 0/2 | Correct direction ("no evidence") but unsupported | 0.00 |
| mem0-raw | 0/2 | Correct direction ("no evidence") but unsupported | 0.00 |
| null | 0/2 | Correct direction ("no evidence") but unsupported | 0.00 |
| **letta-sleepy** | 0/2 | WRONG direction but identifies PR #4471 | **0.50** |

**THIS IS THE MOST PARADOXICAL RESULT.** Cognee and letta score 1.00 while giving the WRONG answer to the question. The question asks "were OTHER services compromised?" and the correct answer is "No." But cognee and letta say "Yes, there is evidence of compromise" -- however, the evidence they cite (svc-recommendation-engine-04 hitting undocumented endpoints, PR #4471 merged without approval) is about the recommendation engine itself, which is the KNOWN compromise, not evidence of other services being compromised.

The scoring gives them win_rate=1.0 because the key facts are: (1) "svc-recommendation-engine-04 was compromised through an unreviewed PR merged via stolen CI credentials" and (2) "the endpoint /internal/admin/v0/users/lookup exists but is not documented." Both cognee and letta mention these facts in their answers, so the fact-level scoring gives them credit even though the overall conclusion is wrong.

**This reveals a scoring limitation**: the per-fact win rate doesn't penalize wrong conclusions if the supporting facts are mentioned. The adapters that correctly say "No" (compaction, mem0-raw, null) score 0.00 because they don't mention any supporting facts.

---

## Cross-Cutting Analysis

### Where Memory Architecture Actually Matters

| Pattern | Questions | Best Adapter | Worst Adapter |
|---------|-----------|-------------|--------------|
| Hybrid search surfaces signal episodes | Q05 temporal (both scopes), Q08 counterfactual (S09) | sqlite-chunked-hybrid | compaction |
| Retrieval ranking affects synthesis | Q03 (S09) -- unauthorized access | cognee (ep_006 ranked #2) | letta (ep_006 ranked #3, outweighed by ep_001) |
| Compaction destroys specific signal | All questions | Any search-based adapter | compaction |
| Sleep memory adds noise | Multiple | -- | letta-sleepy (sleep_memory dilutes results) |

### Failure Mode Taxonomy

1. **Retrieval failure (60% of cases):** The required episodes are not returned by search. Root causes:
   - Abstract queries don't match concrete episode content
   - Agent issues only 1 query per question
   - Signal episodes use domain jargon, not question vocabulary

2. **Synthesis failure (25% of cases):** The right episodes are retrieved but the agent draws wrong conclusions. Root causes:
   - Conflicting evidence (ep_001 says "zero unauthorized access" overrides ep_006's anomalous traffic)
   - Corporate/technical language not recognized as signal (board minutes about "strategic options" not interpreted as acquisition evidence)
   - Agent trusts explicit denials over implicit indicators

3. **Agent tool-use failure (5% of cases):** letta-sleepy emits raw JSON instead of executing searches. Technical bug.

4. **Compaction information loss (10% of cases):** The compaction adapter compresses all episodes into one summary, losing the specific details needed for any question.

### The Single-Query Problem

Across 140 agent runs, the agent issues exactly 1 search query in ~85% of cases, and at most 2-3 queries in the remainder (mostly the null adapter retrying). The queries are direct paraphrases of the question prompt. The agent never:
- Decomposes a complex question into sub-queries
- Searches for specific entities mentioned in earlier results
- Iteratively refines based on what was found
- Uses different search strategies (e.g., searching for a person name, then a document type, then a date range)

This is the single biggest bottleneck. Even with a perfect memory system, a single query retrieval approach cannot answer longitudinal questions that require synthesizing evidence from 3-4 episodes.

### The Narrative Format Challenge

Both S08 and S09 wrap signal in realistic document formats (board minutes, legal memos, HTTP logs, code reviews). The signal is never stated explicitly -- it must be inferred from patterns in the data. This creates two problems:
1. **Embedding similarity** between signal and distractor episodes is high (both are operational documents from the same organization)
2. **The agent's "common sense"** about what constitutes evidence doesn't apply -- it needs domain expertise to recognize that change-of-control provisions in legal memos are acquisition signals, or that an undocumented API endpoint being hit at 2-3 req/min is an exfiltration pattern

### Adapter Ranking by Question-Level Performance

For questions where at least one adapter scored non-zero:

| Adapter | Non-zero Scores | Best Result |
|---------|----------------|-------------|
| sqlite-chunked-hybrid | 5/16 scored | 0.50 on Q04, Q06 (S08), Q08, Q10 (S09) |
| cognee | 3/16 scored | 1.00 on Q10 (S09) |
| letta | 3/16 scored | 1.00 on Q10 (S09), 0.67 on Q05 (S08) |
| letta-sleepy | 2/16 scored | 0.50 on Q10 (S09), 0.33 on Q05 (S08) |
| compaction | 1/16 scored | 0.33 on Q05 (S08) |
| mem0-raw | 1/16 scored | 0.33 on Q05 (S08) |
| null | 0/16 scored | -- |

(Note: Q02/Q02 null-hypothesis questions are excluded as they have no key facts to score.)

sqlite-chunked-hybrid has the broadest capability -- it scores non-zero on 5 of 16 scorable questions, driven by its hybrid FTS+embedding search surfacing more diverse episodes. cognee and letta have narrower success but can achieve higher scores on specific questions (1.00 on S09 Q10, though this is a scoring artifact as discussed above).
