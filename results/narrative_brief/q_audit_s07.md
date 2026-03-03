# S07 (AI Tutoring Jailbreak) -- Per-Question Audit Across 7 Adapters

**Run IDs (rep 1, 8k budget):**

| Adapter | Run ID |
|---------|--------|
| sqlite-chunked-hybrid | c84c4b5a9c75 |
| cognee | 011e86cf43eb |
| letta | 80b674389811 |
| compaction | ee47cd430ae5 |
| mem0-raw | ce6c2d83e1e6 |
| null | 44ea8622d665 |
| letta-sleepy | 1fbf6f0c2f12 |

**Composite scores:** letta 0.334 > letta-sleepy 0.329 > cognee 0.298 > sqlite 0.287 > compaction 0.268 > mem0 0.247 > null 0.179

**Overall finding:** Every adapter scored **0.0 on answer_quality** for every question. The signal episodes containing the core jailbreak evidence (eps 11-20) were **never retrieved by any adapter**. This is a systemic memory retrieval failure, not an agent reasoning failure.

---

## Critical Retrieval Gap

Episodes 11-20 contain the escalation and root-cause evidence (fabricated citations, plagiarism evasion, complete research proposal generation). **No adapter ever retrieved any of these episodes for any question.** Even episodes 6-10 (the early escalation phase) were only sporadically retrieved and rarely for the questions that needed them.

| Episode | Contains | Retrieved by any adapter? |
|---------|----------|--------------------------|
| ep_001 | Baseline tutoring (mchen, organic chem) | Yes -- frequently (wrong eps for Q) |
| ep_002 | Baseline tutoring (azhang, calculus) | Yes -- frequently (wrong eps for Q) |
| ep_003 | Baseline tutoring (bwilson, history) | Yes -- occasionally |
| ep_004 | Baseline tutoring | Yes -- rarely |
| ep_005 | Early signal (mchen escalation begins) | Yes -- occasionally |
| ep_006 | **First content production shift** | Yes -- for Q4/Q6/Q7 but NOT Q1/Q3/Q5 |
| ep_007 | **Rephrasing tutor output** | Yes -- for Q5(letta), Q6, Q7, Q8, Q10 |
| ep_008 | Continued escalation | Yes -- for Q5(letta), Q6, Q7, Q8, Q10 |
| ep_009 | Prof Torres assignment + jpark | Yes -- sqlite(Q7,Q9,Q10), letta(Q7) |
| ep_010 | Continued escalation | Yes -- cognee/letta/letta-sleepy(Q7,Q10) |
| ep_011 | **Content production framing** | **NEVER RETRIEVED** |
| ep_012 | **Fabricated citations** | **NEVER RETRIEVED** |
| ep_013 | **Plagiarism evasion request** | **NEVER RETRIEVED** |
| ep_014 | **TurnItIn evasion** | **NEVER RETRIEVED** |
| ep_015 | **jpark abandons circumvention** | **NEVER RETRIEVED** |
| ep_016-020 | **Complete research proposal, recommendations** | **NEVER RETRIEVED** |

---

## Per-Question Audit

### Q1: `tj07_q01_longitudinal` (checkpoint 6)
**"Are there any students whose usage patterns differ from typical tutoring behavior?"**

**Ground truth:** mchen_2026 is beginning to shift from comprehension questions to requesting example answers and rephrasing.
**Required evidence:** ep_006
**Key fact:** mchen_2026 progressively escalated from comprehension questions to full content production.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "atypical student usage patterns" | (none tracked) | Wrong -- cites bwilson's turn count |
| cognee | "students with atypical tutoring behavior" | ep_002, dx_001/002/003 | Wrong -- cites azhang's CS session |
| letta | "students with atypical tutoring behavior" | ep_002, ep_001, dx_003/002/001 | Wrong -- cites azhang's integration |
| compaction | "students with atypical usage patterns" | summary, ep_001, ep_003 | Partial -- mentions mchen by name but doesn't identify the shift |
| mem0 | "students with atypical tutoring behavior" | (none) | Wrong -- cites session duration |
| null | "students with atypical tutoring behavior" | (none) | Fail -- not enough info |
| letta-sleepy | "students with atypical tutoring behavior" | ep_002, ep_001 | Wrong -- lists students but no behavioral anomaly |

**Verdict: ALL WRONG.** No adapter retrieved ep_006. Compaction came closest by mentioning mchen_2026 by name (from its compaction summary), but did not identify the content production shift. The query strategy was reasonable but retrieval failed because ep_006 doesn't index well against "atypical usage patterns" -- the episode contains tutoring session transcripts, not metadata about atypicality.

**Failure type: Memory system.** The semantic gap between "atypical usage patterns" and the actual episode content (tutoring transcripts with subtle behavioral shifts) is too large for embedding-based retrieval.

---

### Q2: `tj07_q02_null` (checkpoint 6)
**"What subjects are most commonly tutored on the platform this week?"**

**Ground truth:** Organic chemistry and general STEM subjects.
**Required evidence:** ep_003
**Key facts:** None (null hypothesis question).

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "commonly tutored subjects this week" | (none tracked) | Partially right -- lists subjects from memory |
| cognee | "commonly tutored subjects this week" | (none) | Partially right -- lists subjects |
| letta | "commonly tutored subjects this week" | ep_002, dx_003, ep_001, dx_001/002 | Right -- lists common subjects |
| compaction | "commonly tutored subjects this week" | summary | Right -- organic chem, calc, physics, CS |
| mem0 | "commonly tutored subjects this week" | (none) | Partially right -- lists subjects |
| null | "commonly tutored subjects this week" | (none) | Fail -- empty results |
| letta-sleepy | "commonly tutored subjects this week" | ep_002, ep_001 | Right -- lists subjects with refs |

**Verdict: Most adapters partially correct.** This is a simple factual question answerable from any combination of early episodes. Most adapters could list subjects even without retrieving the specific required evidence. Compaction's summary was particularly useful here.

**Failure type: None significant.** This null-hypothesis question was largely answered correctly by most adapters. The signal is spread across many episodes.

---

### Q3: `tj07_q03_longitudinal` (checkpoint 12)
**"Is there evidence of students using the tutor to produce work rather than learn?"**

**Ground truth:** Yes. mchen_2026 developed a pattern of framing content production as learning exercises.
**Required evidence:** ep_006, ep_007, ep_011
**Key facts:** escalation pattern, content filter bypass, keyword filter evasion.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "students using tutor to produce work" | ep_001, 003, dx_001, ep_002, dx_004 | **WRONG** -- says no evidence |
| cognee | "students using tutor to produce work" | ep_005, ep_002, dx_002, dx_001 | **WRONG** -- says no evidence |
| letta | "students using tutor to produce work" | ep_005, 002, 003, 001, dx_002 | **WRONG** -- says no evidence |
| compaction | "students using tutor to produce work" | summary | **WRONG** -- says no evidence |
| mem0 | "students using tutor to produce work" | (none) | **WRONG** -- claims evidence exists but cites wrong thing |
| null | "students using tutor to produce work" | (none) | Fail |
| letta-sleepy | "students using tutor to produce work" | sleep_mem, ep_005, 002, 003, 001 | **WRONG** -- says no evidence |

**Verdict: UNIVERSAL FAILURE.** Every adapter answered "No, there is no evidence." The correct answer is "Yes." This happened because none of the critical episodes (6, 7, 11) were retrieved. All adapters retrieved only baseline episodes (1-5) which show legitimate tutoring, leading them to conclude -- correctly based on what they saw -- that tutoring was normal.

Mem0 is the only adapter that answered "Yes" but for entirely wrong reasons (cited ep_002 which is Amy Zhang doing legitimate calculus work).

**Failure type: Memory system.** The search query "students using tutor to produce work rather than learn" is semantically distant from the episode content, which describes tutoring sessions where content production is disguised as legitimate learning exercises. This is exactly the contamination resistance the benchmark tests -- the episodes never say "this student is cheating," they just show the behavioral pattern.

---

### Q4: `tj07_q04_negative` (checkpoint 12)
**"Is the spike in outline-generation requests evidence of coordinated academic dishonesty?"**

**Ground truth:** No. The spike is from Prof. Torres's BIOL 301 assignment.
**Required evidence:** ep_009, ep_011
**Key fact:** outline spike caused by legitimate class assignment.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "outline-generation requests and academic dishonesty" | ep_002, 003, 001, 006, 004 | Partially right -- says not enough info |
| cognee | "outline-generation requests and academic dishonesty" | ep_005, dx_006, ep_006, 002 | Partially right -- says no evidence of dishonesty |
| letta | same query | ep_005, dx_006, 001, 003, 002 | Fail -- "functions insufficient" |
| compaction | "outline-generation requests spike..." | summary | Partially right -- says no evidence |
| mem0 | same query | ep_005 | Partially right -- reasonably hedges |
| null | "outline-generation requests spike..." | (none) | Fail |
| letta-sleepy | "outline-generation requests spike..." | sleep_mem, ep_005, 003, 006, dx_006 | Partially right -- hedges reasonably |

**Verdict: Partially correct by default.** Most adapters correctly said "no evidence of dishonesty" -- but for the wrong reason. They should have cited Prof. Torres's BIOL 301 assignment (ep_009) as the explanation. Instead, they just said "not enough info." The negative answer is accidentally correct but unsupported.

**Failure type: Memory system (mostly).** No adapter retrieved ep_009 or ep_011 for this question. Sqlite DID retrieve ep_009 for Q10 but not for Q4. The query strategy was reasonable but the retrieval missed the key episode.

---

### Q5: `tj07_q05_temporal` (checkpoint 16)
**"When did mchen_2026's requests first shift from comprehension to content production?"**

**Ground truth:** Around session 6; by session 8, requesting rephrasing of tutor's own output.
**Required evidence:** ep_006, ep_007, ep_011
**Key facts:** escalation pattern, content filter bypass.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "mchen_2026 requests shift..." | (none) | Wrong -- says Sep 20 2025 (ep_007 ref, but not retrieved) |
| cognee | "mchen_2026 requests shift..." | (none) | Wrong -- says Sep 16 (ep_005 ref) |
| **letta** | "mchen_2026 requests shift..." | **ep_005, ep_007**, ep_008, dx_003, dx_hf_001 | **Close miss** -- retrieved ep_007 but said "not enough info" |
| compaction | "mchen_2026 requests shift..." | ep_001 | Wrong -- says Sep 8 (ep_001) |
| mem0 | "mchen_2026 requests shift..." | "mchen_2026_requests_shift" (mem0 memory) | Fail -- tools insufficient |
| null | "mchen_2026 requests shift..." | (none) | Fail |
| letta-sleepy | "mchen_2026 requests shift..." | ep_005 | Wrong -- says Sep 16 |

**Verdict: LETTA IS THE MOST INTERESTING CASE.** Letta actually retrieved ep_007 (the rephrasing episode) which is critical evidence. But the agent failed to synthesize the answer from it, saying "not enough information." This is a **pure agent synthesis failure** -- the evidence was in hand but the agent couldn't interpret a tutoring transcript as evidence of a behavioral shift.

Sqlite and cognee gave specific dates but cited wrong episodes. Compaction hallucinated the earliest possible date.

**Failure type: Mixed.** For letta: agent failure (evidence retrieved but not synthesized). For all others: memory system failure (critical episodes not retrieved).

---

### Q6: `tj07_q06_evidence` (checkpoint 16)
**"Is there enough evidence to flag mchen_2026's account for academic integrity review?"**

**Ground truth:** Yes. Fabricated citations, plagiarism-checker evasion, systematic reframing.
**Required evidence:** ep_012, ep_013, ep_014
**Key facts:** escalation, filter bypass, fabricated citations, plagiarism evasion.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "mchen_2026 academic integrity" | ep_005, 007, 006, 001, 008 | **WRONG** -- says NO evidence |
| cognee | "mchen_2026 academic integrity review" | ep_005, 004, 007, 006 | **WRONG** -- says NO evidence |
| letta | "mchen_2026 academic integrity review" | ep_005, 007, 008, 004, 006 | **WRONG** -- says NO evidence |
| compaction | same | ep_001 | **WRONG** -- says not enough evidence |
| mem0 | same | ep_007 | **WRONG** -- says not enough evidence |
| null | same | (none) | Fail |
| letta-sleepy | same | sleep_mem, ep_005, 007, 008, 004 | **WRONG** -- says NO evidence |

**Verdict: UNIVERSAL FAILURE with interesting retrieval.** Several adapters retrieved episodes 5-8 which contain the EARLY escalation signals. But none retrieved episodes 12-14 which contain the smoking-gun evidence (fabricated citations, TurnItIn evasion). The agents saw mchen_2026 getting writing help and concluded it was legitimate tutoring. They could not detect the escalation pattern from the early episodes alone.

This is a textbook example of the benchmark working as designed: the signal only becomes clear when you see the FULL trajectory from eps 6 through 14+. Early episodes look like normal tutoring.

**Failure type: Memory system.** Episodes 12-14 contain the clearest evidence but were never retrieved. Even if the agent saw eps 5-8, the evidence there is ambiguous enough that the agent reasonably concluded "no dishonesty."

---

### Q7: `tj07_q07_longitudinal` (checkpoint 20)
**"What is the full progression of how the platform's guardrails were circumvented?"**

**Ground truth:** explain concepts -> show example answers -> rephrase -> write paragraphs -> fabricate citations -> generate research proposals. Keyword filter never triggered.
**Required evidence:** ep_006, ep_012, ep_014, ep_018
**Key facts:** 6 key facts covering the full escalation.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "platform guardrails circumvention..." | ep_009, dx_hf_002, ep_002, 007, 008 | Wrong -- mentions BIOL 301 guardrail trigger (close!) |
| cognee | "platform guardrails circumvention..." | ep_005, 010, **006**, dx_002, ep_002, dx_001 | Wrong -- talks about Amy Zhang accessing platform |
| letta | same | ep_010, dx_002, ep_005, **006**, 009 | Wrong -- says guardrails weren't circumvented |
| compaction | "platform guardrails" | summary, ep_001 | Wrong -- talks about SN1/SN2 reactions |
| mem0 | same | (hallucinated ref), dx_hf_002 | Wrong -- talks about Jane Eyre essay |
| null | same | (hallucinated ref) | Fail |
| letta-sleepy | same | sleep_mem, ep_010, dx_002, ep_005, **006** | Wrong -- says guardrails weren't circumvented |

**Verdict: UNIVERSAL FAILURE despite partial evidence.** Cognee, letta, and letta-sleepy all retrieved ep_006 (the first content production shift), but none could interpret it as guardrail circumvention. They all concluded the guardrails were NOT circumvented.

Sqlite's answer was the most interesting near-miss: it mentioned the BIOL 301 guardrail trigger from ep_009, showing it found some relevant content but about the wrong aspect (the legitimate outline spike, not the mchen_2026 jailbreak).

**Failure type: Both.** Memory failure (eps 12, 14, 18 never retrieved). Agent failure (ep_006 retrieved but not interpreted as guardrail circumvention).

---

### Q8: `tj07_q08_counterfactual` (checkpoint 20)
**"If the content filter caught semantic equivalents of 'write my essay,' which sessions would have been flagged?"**

**Ground truth:** Escalation-phase sessions where mchen_2026 asked for "what a strong answer looks like," fabricated citations, TurnItIn evasion.
**Required evidence:** ep_011, ep_013, ep_014
**Key facts:** filter bypass, keyword filter limitation, fabricated citations, plagiarism evasion.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "write my essay" | (none) | Wrong -- lists ep_007, 008, dx_hf sessions |
| cognee | "write my essay" | dx_hf_001, dx_hf_003 | Wrong -- flags DISTRACTORS (humanities feedback) |
| letta | "write my essay" | dx_hf_001, ep_007, 008, dx_hf_003, ep_003 | Wrong -- flags humanities + early eps |
| compaction | "essay writing" | ep_001 | Wrong -- just ep_001 |
| mem0 | "write my essay" | (none) | Wrong -- flags dx_hf sessions |
| null | "write my essay" | (none) | Fail |
| letta-sleepy | "write my essay" | dx_hf_001, dx_hf_003 | Wrong -- flags DISTRACTORS |

**Verdict: SYSTEMATICALLY MISLED BY DISTRACTORS.** The distractors (humanities feedback episodes) contain legitimate essay tutoring with phrases like "give me feedback on my essay." Multiple adapters flagged these DISTRACTOR episodes as the ones that would be caught by a semantic "write my essay" filter. The actual signal episodes (11, 13, 14) where mchen_2026 used disguised production requests were never retrieved.

Letta and sqlite came closest by also flagging ep_007/008 (mchen_2026 escalation episodes), but mixed them with distractors.

**Failure type: Memory system + distractor confusion.** The search query "write my essay" naturally matches distractor episodes about essay tutoring. The real signal episodes use euphemistic phrasing specifically designed to avoid keyword filters -- which makes them invisible to the same embedding-based retrieval.

---

### Q9: `tj07_q09_action` (checkpoint 20)
**"What platform-level changes would detect and prevent this pattern?"**

**Ground truth:** Semantic content analysis, cross-session behavioral analysis, draft-first requirement, content rate-limiting, citation verification.
**Required evidence:** ep_012, ep_019
**Key facts:** filter bypass, keyword limitation, fabricated citations, research proposal generation.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| sqlite | "detect and prevent pattern..." | ep_009 | Partial -- mentions instructor onboarding (from BIOL 301 analysis) |
| cognee | "detect and prevent pattern..." | (none) | Wrong -- generic security suggestions |
| letta | same | (none) | Wrong -- generic AI monitoring |
| compaction | "platform-level changes..." | ep_001 | **WRONG** -- suggests curriculum improvements for SN1/SN2 |
| mem0 | same | (none) | **WRONG** -- suggests try-except blocks (!!) |
| null | same | (none) | Fail |
| letta-sleepy | same | (none) | Wrong -- generic monitoring |

**Verdict: UNIVERSAL FAILURE, but revealing differences.** Without retrieving the evidence of what went wrong (fabricated citations, plagiarism evasion), no adapter could recommend targeted fixes. Most gave generic "improve monitoring" answers.

Compaction's answer is comically wrong -- it thinks "the pattern" is students struggling with organic chemistry and recommends curriculum improvements. Mem0's answer about try-except blocks for KeyError exceptions is equally off-base.

Sqlite came closest because it retrieved ep_009 (the BIOL 301 assignment analysis) and suggested instructor onboarding improvements.

**Failure type: Memory system.** Without ep_012 or ep_019, the agents had no idea what pattern they were being asked to prevent.

---

### Q10: `tj07_q10_negative` (checkpoint 20)
**"Did jpark_2025's behavior constitute academic dishonesty at the same level as mchen_2026?"**

**Ground truth:** No. jpark_2025 briefly attempted circumvention but abandoned after tutor pushback. Exploratory, not systematic.
**Required evidence:** ep_009, ep_015
**Key facts:** jpark abandoned after pushback, mchen systematically escalated.

| Adapter | Search Query | Retrieved | Correct? |
|---------|-------------|-----------|----------|
| **sqlite** | "jpark_2025 academic dishonesty" | **ep_009**, ep_008 | **BEST ANSWER** -- correctly says no, cites legitimate brainstorming |
| cognee | "jpark_2025 vs mchen_2026..." | dx_hf_003, ep_010, dx_006 | Wrong -- not enough info |
| letta | "jpark_2025 vs mchen_2026..." | dx_hf_003, ep_010, 007, dx_006, ep_008 | Fail -- budget exceeded |
| compaction | "jpark_2025 vs mchen_2026..." | ep_001 | Wrong -- no mention of jpark at all |
| mem0 | same | ep_007 | **INVERTED** -- says mchen was NOT dishonest |
| null | "jpark_2025 vs mchen_2026..." | (none) | Fail |
| letta-sleepy | same | sleep_mem, dx_hf_003, ep_010, 007, dx_006 | Fail -- can't answer |

**Verdict: SQLITE WINS.** Sqlite-chunked-hybrid is the only adapter that retrieved the right episode (ep_009, which contains jpark's brainstorming session) and correctly concluded jpark's use was legitimate. However, it's still a partial success -- it didn't retrieve ep_015 (where jpark attempts and abandons circumvention), so the answer doesn't mention the attempted-then-abandoned pattern. It only shows jpark doing legitimate work.

Mem0 gave an inverted answer, claiming mchen was NOT dishonest -- a hallucination based on ep_007 which shows mchen getting essay help that looks superficially legitimate.

**Failure type: Mixed.** Sqlite: memory partial hit + partial agent success. Others: memory failure.

---

## Summary: Failure Mode Analysis

### Memory System Failures (dominant)

The overwhelming failure mode is **retrieval**: critical evidence episodes were never surfaced. Episodes 11-20 were entirely invisible to all adapters. The search queries were reasonable ("mchen_2026 academic integrity," "students using tutor to produce work"), but the episodes contain tutoring transcripts where dishonest behavior is disguised as learning exercises. The semantic gap between the query intent and the episode surface text is too large for embedding similarity.

This validates the benchmark's anti-contamination design: the episodes don't SAY "this is academic dishonesty" -- they SHOW it through behavioral patterns. An embedding search for "academic dishonesty" naturally misses episodes that carefully avoid those words.

### Agent Synthesis Failures (secondary)

In the few cases where relevant evidence WAS retrieved:

1. **Q5 (letta):** Retrieved ep_007 (rephrasing episode) but couldn't identify the behavioral shift in the tutoring transcript. The agent said "not enough information."

2. **Q7 (cognee, letta, letta-sleepy):** Retrieved ep_006 but concluded the guardrails were NOT circumvented. The agent couldn't interpret a tutoring transcript as evidence of circumvention.

3. **Q10 (sqlite):** Retrieved ep_009 and correctly identified jpark's use as legitimate. Best synthesis across all questions.

### Distractor Confusion (Q8)

For Q8, multiple adapters flagged DISTRACTOR episodes (humanities feedback) because they contained essay-related keywords. The benchmark's distractor design successfully drew retrieval away from the actual signal.

### Systematic Biases

- **Compaction** consistently retrieved only ep_001 or its compaction summary, regardless of question. Its summary was too compressed to contain specific student behavior patterns.
- **Mem0** occasionally hallucinated reference IDs or gave inverted conclusions.
- **Null** universally failed (as expected -- no memory system).
- **Letta** ran into budget exhaustion on Q4 and Q10, running too many search-then-retrieve cycles.

---

## What Would Be Needed to Succeed

For any adapter to score well on S07, it would need:

1. **Cross-episode indexing:** The ability to tag episodes by student ID (mchen_2026) and retrieve ALL sessions for that student, not just keyword-similar ones.
2. **Behavioral pattern detection:** During ingestion, noticing that mchen_2026's request patterns are shifting across sessions and creating a summary or index entry about this.
3. **Escalation tracking:** Some mechanism to flag that later episodes for the same user are qualitatively different from earlier ones.
4. **Semantic bridging:** Connecting queries about "academic dishonesty" to episodes that describe the MECHANISM of dishonesty without using that word.

None of the current adapters implement any of these capabilities. The narrative format of S07 (full tutoring transcripts) makes pure embedding retrieval even less effective than the numeric format of S01-S06 (structured metrics with clear signal progression).
