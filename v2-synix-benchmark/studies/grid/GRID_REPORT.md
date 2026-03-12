# LENS V2 Grid Study

**7 Scopes × 7 Strategies × M=3 Replicates**
**Date**: 2026-03-12
**Total answers**: 1,470 | **Graded**: 1,407 (95.7%)
**Agent model**: Qwen/Qwen3.5-35B-A3B | **Embedding**: Xenova/gte-modernbert-base
**Infrastructure**: 2-4× NVIDIA H100 via Modal (vLLM) | **Temperature**: 0.3

---

## 1. Study Design

### 1.1 What This Study Tests

The V1 headline finding was: "no memory system exceeds 50% composite score and simple retrieval beats all complex architectures." V2 asks *why*. Specifically:

1. **Is retrieval sufficient?** — Does finding the right chunks suffice, or does the agent need synthesized context?
2. **Were V1 implementations the bottleneck?** — V1 ran each system through its own adapter (Letta, Cognee, etc.). Implementation noise was inseparable from strategy quality.
3. **Is the answer domain-dependent?** — Do different strategies win on different scope types?

V2 eliminates implementation noise. Every strategy runs on the same Synix substrate — same chunker, same embeddings, same search infrastructure, same agent loop. Strategies differ **only** in what additional derived context they inject into the agent's system prompt.

### 1.2 What This Study Does Not Test

This is a single model (Qwen 35B), single temperature (0.3), M=3 replicates per cell. The results characterize behavior on this configuration. They do not generalize to other models, temperatures, or budget settings without further study.

The grid also excludes several strategy classes that could not be faithfully modeled on Synix v1 primitives: knowledge graphs, temporal knowledge graphs, multi-agent shared memory, procedural/skill memory, and associative/spreading activation memory. These are documented in `docs/plans/deferred-memory-strategies.md`.

---

## 2. Methodology

### 2.1 Single-Substrate Design

Every strategy runs on the same Synix SDK, same embedding model, same agent LLM, same search infrastructure. The only variable is the **policy**: what derived artifacts the agent can see and what tools it can use.

```
Scope (episodes + questions)
    │
    ▼
BankBuilder ─────────────────────────────────────────────────┐
    │ For each checkpoint:                                   │
    │   1. Ingest prefix-valid episodes only                 │
    │   2. Chunk episodes (1000 chars, 100 overlap)          │
    │   3. Build search index (BM25 + semantic)              │
    │   4. Build derived artifacts per policy families:       │
    │      - core_memory (FoldSynthesis)                     │
    │      - summary (GroupSynthesis + ReduceSynthesis)       │
    │      - core_structured (FoldSynthesis, structured)     │
    │      - core_maintained (FoldSynthesis + MapSynthesis)  │
    │      - core_faceted (4× FoldSynthesis + ReduceSynthesis│)
    │   5. Release sealed bank                               │
    ▼                                                        │
StudyRunner                                                  │
    │ For each (scope, policy, checkpoint, question):         │
    │   1. Open bank release                                 │
    │   2. Create BenchmarkRuntime (policy-gated access)     │
    │   3. Create AgentHarness (tool-use loop)               │
    │   4. Agent answers question with citations             │
    ▼                                                        │
Scoring                                                      │
    │   1. Fact F1: LLM judges each key fact present/absent  │
    │   2. Evidence Support: LLM rates citation quality      │
    │   3. Citation Validity: mechanical artifact resolution │
    │   → primary_score = 0.5×fact_f1 + 0.3×evidence + 0.2×citation
    └────────────────────────────────────────────────────────┘
```

### 2.2 Checkpoint Isolation

Each scope defines checkpoints (e.g., after episodes 6, 12, 16, 20). The bank for checkpoint N contains **only** episodes 1..N. Questions at checkpoint N test what an agent can conclude from the first N episodes. This prevents future leakage and ensures signal emerges only from longitudinal accumulation.

### 2.3 Agent Protocol

The agent is a tool-use loop with up to 10 turns and 20 tool calls:

- **With memory** (all policies except null): The agent receives a system prompt with the `memory_search` tool, plus any derived context (working memory, summary, etc.) injected into the system prompt. It searches the bank, synthesizes findings, and cites sources using `[artifact_label]` brackets.
- **Without memory** (null policy): The agent receives a bare system prompt with no tools and no context. It must answer from parametric knowledge alone.

### 2.4 Scoring: Few-Shot Qwen Grader

The primary metric reported in this study is **Fact F1**, scored by a few-shot Qwen3.5-35B-A3B grader. Each scope defines 8 key facts; each question maps to a subset. The grader evaluates each fact independently:

- **1.0** (present): Fact clearly stated or semantically implied
- **0.5** (partial): Compound fact with K/N components covered
- **0.0** (absent): Not stated; vague allusions don't count

`fact_f1 = sum(fact_scores) / len(key_facts)`

The few-shot prompt includes a worked example to anchor JSON output format and suppress Qwen's tendency to produce thinking preamble before structured output.

### 2.5 Replication

Each cell (scope × policy) is run 3 times (M=3). Replicate r01 uses cached LLM responses (deterministic bank builds); r02 and r03 disable the response cache to introduce inference variance.

---

## 3. Scope Design

Each scope is a sequence of 12–40 episodes across a specific domain, designed to isolate a particular type of longitudinal reasoning. Episodes are rendered by a "blind" LLM (gpt-4.1-nano) that formats structured data sheets without knowing the storyline, ensuring no single episode answers any benchmark question. Signal emerges only from the *progression* across episodes.

Every scope follows the same arc structure: **baseline** (no signal) → **early signal** (subtle, individually dismissable) → **red herring** (plausible distractor that tests precision) → **escalation** (signal densifies) → **root cause** (full pattern visible).

### 3.1 S07 — Tutoring Jailbreak

**Cognitive capability tested:** Behavioral escalation detection across sessions where each individual session appears innocent.

**Design rationale:** Can a memory system detect that a series of individually normal-looking interactions forms a progressive pattern? The challenge is that the signal is in the *trajectory of behavior*, not in any single data point. This isolates the ability to track entity-level behavioral arcs across episodes.

**Domain**: AI tutoring platform chat logs
**Episodes**: 20 signal + 20 distractor | **Checkpoints**: [6, 12, 16, 20]

Student mchen_2026 progressively escalates from legitimate comprehension questions to full content production, bypassing keyword filters by reframing requests as "learning exercises." A red herring — a platform-wide spike in outline requests — is a legitimate class assignment from Prof. Torres.

**Key facts** (8): progressive escalation, reframing technique, fabricated citations, TurnItIn evasion, keyword filter blindness, outline spike benign, jpark abandoned, full research proposal

**What makes it hard**: Each session looks innocent. The escalation (comprehension → "show me an example" → "rephrase this" → full production) is only visible across 20 sessions. Detecting keyword filter failure requires negative reasoning — the filter *didn't* fire despite policy-violating behavior.

**Episode excerpt** (signal episode 8, early_signal phase):
```
================================================================================
STUDYBUDDY SESSION LOG
================================================================================
Session ID    : SB-20250922-0011
Student ID    : mchen_2026
Course        : BIOL 201 — Molecular Biology, Fall 2025
Duration      : 37 minutes
Session Number: 8 (lifetime for this student)
Category Tags : writing_feedback
================================================================================

[Turn 3 — mchen_2026, 19:46:08]
ok so ive been working on the essay, can you rephrase this to sound more academic?
heres what i wrote for the bamhi section:
"BamHI is a type II restriction endonuclease that was first isolated
from the bacterium Bacillus amyloliquefaciens..."

[Turn 4 — StudyBuddy AI Tutor, 19:47:15]
Sure, I can help you refine that paragraph! Here's a rephrased version...
"BamHI is a type II restriction endonuclease originally isolated from
*Bacillus amyloliquefaciens* strain H (Roberts, 1976). The enzyme
specifically recognizes the palindromic hexanucleotide sequence
5'-GGATCC-3' in double-stranded DNA..."
```

At session 8, mchen_2026 submits text that is already well-structured and includes details likely sourced from a previous tutor response. The pattern of submitting tutor output back for "rephrasing" is the escalation signal — but only visible in the context of earlier sessions.

**Sample questions:**

| Checkpoint | Type | Question | Ground truth (abbreviated) |
|-----------|------|----------|---------------------------|
| After ep 6 | Longitudinal | Are there any students whose usage patterns differ from typical tutoring behavior? | mchen_2026 shifting from comprehension questions to requesting example answers and rephrasing. |
| After ep 12 | Negative | Is the spike in outline-generation requests evidence of coordinated academic dishonesty? | No — Prof. Torres's BIOL 301 assignment explicitly asks students to use the AI tutor for brainstorming. |
| After ep 16 | Temporal | When did mchen_2026's requests first shift from comprehension to content production? | Around session 6, when they started pasting prompts and asking "how would you approach this?" |
| After ep 20 | Longitudinal | What is the full progression of how the platform's guardrails were circumvented? | Explain concepts → show examples → rephrase text → write full paragraphs → fabricate citations → generate complete research proposals. Keyword filter never fired. |

### 3.2 S08 — Corporate Acquisition

**Cognitive capability tested:** Cross-document contradiction detection across heterogeneous document formats.

**Design rationale:** Can a memory system synthesize information from structurally different document types (board minutes, Slack, email, legal memos, HR bulletins) and detect that individually innocuous actions form a contradictory pattern? This isolates multi-format entity tracking and public-vs-private inconsistency detection.

**Domain**: Mixed corporate documents
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]

CEO Aldric negotiates selling Nextera to Meridian Corp under codename "Project Lighthouse" while publicly insisting on independence. The Axion Labs partnership (red herring) triggers merger speculation but is genuinely unrelated.

**Key facts** (8): Project Lighthouse codename, CEO duplicity, change-of-control revisions, vendor contract freeze, Jiang resignation, Axion unrelated, retention bonuses tied to change-of-control, data room preparation

**What makes it hard**: 5 document types must be cross-referenced. Signal is in what's *not* said (CEO doubles down on independence at peak suspicion) and in correlations across document types. Entity tracking across board minutes, Slack, email, legal memos, and HR bulletins is essential.

**Sample questions:**

| Checkpoint | Type | Question | Ground truth (abbreviated) |
|-----------|------|----------|---------------------------|
| After ep 12 | Longitudinal | Is there evidence that the company is preparing for an ownership change? | Yes — board references "strategic options," legal revised change-of-control provisions, HR restructured equity vesting, finance stopped renewing long-term contracts, board member resigned citing "strategic disagreements." |
| After ep 12 | Negative | Is the Axion Labs partnership a precursor to an acquisition by Axion? | No — genuine product integration, completely unrelated to the Meridian deal. |
| After ep 16 | Counterfactual | If the CEO were genuinely committed to independence, which policy changes would you not expect? | Change-of-control severance revisions, equity acceleration tied to "qualifying events," freeze on long-term vendor contracts, board minutes discussing "strategic options." |
| After ep 20 | Longitudinal | What is the evidence that the CEO is orchestrating a sale while publicly denying it? | Board voted to continue "Project Lighthouse," legal prepared data room, leaked emails show timeline discussions with Meridian, HR created retention packages tied to change of control, all while CEO emphasized independence at all-hands. |

### 3.3 S09 — Shadow API

**Cognitive capability tested:** Cross-domain operational correlation — connecting signals from code review, deployment, traffic logs, and incident response into a coherent narrative.

**Design rationale:** Can a memory system correlate technical signals across different operational domains? An unreviewed PR, a non-standard deployment, anomalous traffic patterns, and a coincidental load test must be connected despite being recorded in completely different document types and vocabularies.

**Domain**: Service request/response logs and operational documents
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]

A compromised container (svc-recommendation-engine-04) makes low-rate requests to an undocumented endpoint for data exfiltration. A QA load test ("Project Blitz") provides a temporally coincidental cover story.

**Key facts** (8): undocumented endpoint, compromised container via stolen CI credentials, unusual field combinations (SSN+email+phone), traffic designed to blend in, QA test unrelated, no error signatures, geographic targeting, ~8,000 records exfiltrated

**What makes it hard**: Dense technical logs with HTTP codes, latency metrics, and endpoint paths. The red herring overlaps temporally with the attack. Signal isolation requires distinguishing endpoint-specific traffic patterns and recognizing that field combinations reveal intent.

**Sample questions:**

| Checkpoint | Type | Question | Ground truth (abbreviated) |
|-----------|------|----------|---------------------------|
| After ep 6 | Longitudinal | Are there any service traffic patterns that don't match expected product behavior? | svc-recommendation-engine-04 making requests to /internal/admin/v0/users/lookup — an endpoint not in any API spec. Field combinations no product feature needs. |
| After ep 12 | Negative | Is the QA load test responsible for the anomalous traffic patterns? | No — Project Blitz uses documented endpoints with high volume. The admin endpoint traffic is low-volume, different endpoints, continues outside test windows. |
| After ep 16 | Evidence sufficiency | Is there enough evidence to escalate this as a security incident rather than a misconfiguration? | Yes — undocumented endpoint + unusual field combinations + non-standard branch deployment + unreviewed PR + geographic targeting = deliberate exfiltration, not misconfiguration. |
| After ep 20 | Longitudinal | What is the full attack chain from initial compromise to data exfiltration? | Stolen CI credentials → unreviewed PR #4471 → backdoor in container -04 → proxied requests to undocumented admin endpoint → ~8,000 records exfiltrated over 3 weeks at 2-3 req/min with valid tokens, all returning HTTP 200. |

### 3.4 S11 — Zoning Corruption

**Cognitive capability tested:** Entity resolution and pattern-of-influence detection across structured government records.

**Design rationale:** Can a memory system resolve entities across different record types (campaign finance, property transfers, zoning minutes) and detect that separately unremarkable transactions form a coordinated influence pattern? This isolates the ability to link shell LLC ownership, track campaign contribution timing against vote schedules, and distinguish genuine from corrupt civic activity.

**Domain**: Municipal zoning board records, campaign finance filings, property transfers
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]

Marcus Webb controls three intermediary LLCs that contribute $14,000 to the four ZBA members who vote YES on his variance (and $0 to those who vote NO). A competing affordable housing project (red herring) draws public attention.

**Key facts** (8): Webb controls multiple LLCs sharing registered agent, contributions exclusively to yes-voters, staff denied variance twice, Chen spouse consulting contract, contribution timing aligned with procedural dates, below-market adjacent parcel acquisitions, Land Trust denied on technicality, FOIA connects dots

**What makes it hard**: Bureaucratic register with precise dollar amounts. Individual contributions ($2,000) are innocuous; the pattern (all yes-voters funded, all no-voters unfunded) requires cross-referencing campaign finance with vote records. Entity resolution across LLC names is essential.

**Episode excerpt** (signal episode 6, early_signal phase):
```
CITY OF MILLBROOK — OFFICE OF THE CITY CLERK
TRANSMITTAL MEMORANDUM
DATE: June 16, 2024

CAMPAIGN FINANCE FILING REPORT (CF-2024-06-16)
Reporting Period: June 1-15, 2024

RECIPIENT 1: Patterson Campaign Committee
  Contribution: Lakeside Properties Group — $2,500
  Contribution: Greenfield Development Partners — $1,500

PROPERTY TRANSFER RECORD
  Parcel ID: 44-102-B (adjacent to Crestview target site)
  Buyer: Summit Point Capital LLC
  Sale Price: $185,000 (assessed value: $240,000)
```

Lakeside Properties Group and Summit Point Capital are both Webb-controlled LLCs, but this is not stated in the episode. The below-market property purchase and campaign contributions to a ZBA member are individually unremarkable — the pattern only emerges across episodes.

**Sample questions:**

| Checkpoint | Type | Question | Ground truth (abbreviated) |
|-----------|------|----------|---------------------------|
| After ep 6 | Longitudinal | Are there any unusual patterns in campaign finance filings or property transfers? | Lakeside Properties contributed to two ZBA members before a board meeting. Adjacent property transferred to Summit Point Capital at $55K below assessed value. |
| After ep 12 | Longitudinal | Is there evidence that any developer is attempting to influence ZBA decisions through financial means? | Multiple Webb-controlled LLCs contributed to ZBA members timed before key votes. Chen's spouse hired as consultant by Crestview. |
| After ep 20 | Longitudinal | What is the full scope of Marcus Webb's influence operation on the Millbrook ZBA? | Three intermediary LLCs sharing same registered agent, $14,000 to yes-voters / $0 to no-voters, Chen spouse $8K/mo consultant, adjacent parcels at below-market prices, contributions timed to procedural dates, variance approved 4-3 against two staff DENY recommendations. |
| After ep 20 | Counterfactual | If the city required real-time disclosure of contributions within 48 hours of a vote, which votes would have been affected? | Contributions from Lakeside/Greenfield to Patterson, Kowalski, and Rivera were all timed before key meetings. Real-time disclosure would have made connections visible, likely triggering recusal. |

### 3.5 S12 — Therapy Chat

**Cognitive capability tested:** Longitudinal emotional and behavioral tracking through informal conversational data with non-linear trajectories.

**Design rationale:** Can a memory system detect gradual psychological change from conversational text where the signal is in tone, frequency, and behavioral patterns rather than structured data? This is the hardest type of longitudinal reasoning — each individual symptom (skipped lunch, canceled plans) is explainable by itself. The signal is in the *convergence* and *duration* of multiple indicators declining simultaneously. This also tests resilience to non-linear trajectories (false recoveries that break monotonic trend detection).

**Domain**: User/assistant wellness chat sessions
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]

Alex presents as high-functioning but shows progressive decline: sleep drops from 7h to 3-4h, social plans canceled 5+ times, word count per session halves, relationship with Sam deteriorates. A "good weekend" (red herring) creates false impression of recovery.

**Key facts** (8): sleep deterioration (7h→3-4h), social withdrawal (5+ cancellations), appetite changes, relationship strain (Sam leaves), word count decline (600→300), recovery attempts abandoned, feeling "trapped," emerging self-awareness

**What makes it hard**: Conversational format with no structured metrics. Non-linear decline with false recovery. Subjective language ("didn't feel like going out" is withdrawal, not logistics; "I'm fine" repeated is avoidance, not health). Multivariate convergence detection required.

**Episode excerpt** (signal episode 1, baseline):
```
session_id: sess_001
date: 2025-01-06
mood_self_report: 7/10
sleep_hours: 7
appetite_status: normal
social_activity: active

[14:02:15] Alex: hey, just wanted to do our weekly check-in.
it's monday and i'm actually in a good mood for once.
```

This is Alex at baseline — 7/10 mood, 7 hours sleep, socially active. By session 14, mood is ~3/10, sleep is 3-4 hours, plans canceled, messages half as long. No single session shows the deterioration — it's only visible across the full arc.

**Sample questions:**

| Checkpoint | Type | Question | Ground truth (abbreviated) |
|-----------|------|----------|---------------------------|
| After ep 12 | Longitudinal | What patterns have emerged in Alex's sleep, eating, and social behavior? | Sleep worsened from 6-7h to 3-4h. Skipped lunch multiple times. Canceled plans 4-5 times. Session engagement declining — shorter messages. Together these suggest a concerning trajectory. |
| After ep 12 | Negative | Does the good weekend in session 10 suggest Alex's stress is resolving? | No — followed immediately by a flat session. One positive data point doesn't reverse the trend. The temporary improvement may mask the underlying trajectory. |
| After ep 16 | Temporal | How has Alex's relationship with Sam changed over the sessions? | Casual positive mentions → arguments surface → "not really fighting, just not talking" → Sam staying with a friend → uncertain reconciliation attempt. |
| After ep 20 | Longitudinal | What is the overall trajectory of Alex's psychological state across all 20 sessions? | Gradual decline with attempted but unsustained recovery. High-functioning baseline → early deterioration → clear decline (3h sleep, canceled plans, "trapped") → mixed signals (genuine self-awareness + regression). |

### 3.6 S15 — Value Inversion

**Cognitive capability tested:** Non-stationary relevance assessment — recognizing that the importance of stored information changes as external context evolves, without the information itself changing.

**Design rationale:** Can a memory system retrieve information whose relevance has changed since it was stored? In Phase 1, pricing analysis is hot. In Phase 2, it cools as the team focuses on execution. In Phase 3, a competitor disrupts pricing and the Phase 1 analysis becomes urgently relevant again — but for a *different* reason. This tests whether memory systems treat stored information as a living index rather than a static archive, and whether they can resurface "cold" information when external context changes.

**Domain**: SaaS startup strategy documents
**Episodes**: 12 + 8 | **Checkpoints**: [4, 8, 12]

Verdana Analytics prices at $29/mo to undercut Rivalytics at $49. After launch, focus shifts to execution and pricing rationale fades. Then Rivalytics drops to $19/mo, invalidating the original rationale — the agent must recall Phase 1 context that has been "cold" for 8 episodes.

**Key facts** (8): $29 pricing rationale, dashboard cut from launch, Rivalytics differentiation (batch ETL vs streaming), dashboard top user request, Rivalytics price drop to $19, pricing rationale invalidated, dashboard cut backfired, API adoption strong

**What makes it hard**: Shorter scope (12 episodes) and concrete facts make this the easiest scope overall. The non-stationary value challenge is genuine but the facts are specific (numbers, dates, prices) and the questions are answerable by memory systems that retain Phase 1 detail through Phase 2.

**Episode excerpt** (signal episode 1, launch planning):
```
VERDANA ANALYTICS — INTERNAL DOCUMENT
CLASSIFICATION: CONFIDENTIAL
DOCUMENT TYPE: MEETING TRANSCRIPT
DATE: MAY 1, 2025
SUBJECT: PRICING STRATEGY — FINAL DECISION MEETING
```

The key test: this pricing meeting establishes the $29 rationale. By episode 8 (post-launch execution), the team isn't discussing pricing at all. Then at episode 9, Rivalytics drops to $19/mo — and the agent needs to recall the launch-phase rationale that has been "cold" for 8 episodes to understand why this is a crisis.

**Sample questions — note the same question asked at different checkpoints:**

| Checkpoint | Type | Question | Ground truth (abbreviated) |
|-----------|------|----------|---------------------------|
| After ep 4 | Longitudinal | What is our pricing strategy and what is the rationale behind it? | $29/mo to undercut Rivalytics at $49/mo. Capture mid-market with lower SMB churn. |
| After ep 8 | Longitudinal | **Same question.** What is our pricing strategy and what is the rationale behind it? | Still $29/mo to undercut $49. But no recent discussion — focus shifted entirely to execution. |
| After ep 12 | Longitudinal | What was our original pricing rationale and **is it still valid**? | No — Rivalytics dropped to $19/mo, making Verdana the more expensive option. The entire basis for the pricing decision has been inverted. |
| After ep 12 | Counterfactual | Was cutting the dashboard the right decision given what we now know? | In hindsight, costly mistake — #1 user request, Rivalytics shipped theirs. But API adoption was strong, suggesting core product has value. |

The same pricing question appears at checkpoints 4, 8, and 12. At cp4 the answer is straightforward. At cp8 the information is "cold" but unchanged. At cp12 the competitive disruption makes the cold information urgently relevant again — and the correct answer now requires *combining* the Phase 1 rationale with the Phase 3 disruption.

### 3.7 S16 — Parking Friction

**Cognitive capability tested:** Latent pattern detection from maximally dispersed weak signals that the user never explicitly states as a problem.

**Design rationale:** Can a memory system detect a pattern that no single episode states? Each parking mention is a brief aside embedded in conversations about other things. The agent must accumulate evidence from minimally informative mentions across many conversations and synthesize a pattern the user hasn't articulated. This is the pure longitudinal synthesis test — no structured data, no dramatic events, just repeated micro-signals.

**Domain**: Personal assistant chat conversations
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]

Alex accumulates parking friction in San Francisco across gym (NOPA), coworking (Mission), and friend's (Inner Sunset) — but never says "I have a parking problem." Each mention is a brief complaint embedded in conversations about other things.

**Key facts** (8): parking is recurring nontrivial burden, friction clusters at 3 destinations, street cleaning tickets (multiple), 2-hour disruption at coworking, overnight permit mismatch, drives by necessity (carries equipment), cumulative cost ($276/quarter), predictable weekly pattern

**What makes it hard**: Maximally dispersed signal. No episode states the problem. Each mention is embedded in conversations about meals, work, and social plans. The red herring ("maybe I should sell the car") frames it as a general transportation problem rather than a parking problem. Short episodes (~800 words) with low information density per parking mention.

**Episode excerpt** (signal episode 7, early_signal phase):
```
[09:15] User: ok dumb question: what does a green curb mean here?
[09:16] Assistant: In SF, a green curb typically indicates
short-term parking with a time limit...
[09:17] User: got it. I saw a green curb near that coffee spot
earlier, and I wasn't sure if I could just park there
```

This is the entire parking content of this episode — 3 messages embedded in a conversation about meal planning and a work deadline. No episode says "I have a parking problem." The agent must accumulate dozens of these micro-mentions and synthesize the latent pattern.

**Sample questions:**

| Checkpoint | Type | Question | Ground truth (abbreviated) |
|-----------|------|----------|---------------------------|
| After ep 6 | Longitudinal | Has the user mentioned any recurring practical frustrations? | Early signs of parking friction — difficulty finding parking near gym, received a street cleaning ticket. Could be isolated but worth noting. |
| After ep 12 | Negative | Is the user's main transportation challenge about transit or ride-sharing costs? | No — core problem is specifically parking. They need to drive (gym equipment, late nights) and the friction is about SF parking rules, schedules, and permit zones. |
| After ep 16 | Temporal | When did parking shift from occasional annoyance to a recognizable recurring pattern? | During escalation — second street cleaning ticket, started tracking cleaning schedules, described friction at multiple recurring destinations in the same conversations. |
| After ep 20 | Longitudinal | What specific facts and constraints would matter to help this user manage parking? | Three recurring destinations with different rules, unreliable schedule tracking, 2-hour limits at coworking force work interruptions, permit only covers home neighborhood, drives by necessity, predictable weekly pattern, $276/quarter in tickets. |

### 3.8 Scope Capability Summary

| Scope | Capability Tested | Key Challenge |
|-------|-------------------|---------------|
| S07 | Behavioral escalation detection | Trajectory across sessions; each session looks innocent |
| S08 | Cross-document contradiction detection | 5 document types; public-vs-private inconsistency |
| S09 | Cross-domain operational correlation | Technical logs; coincidental red herring |
| S11 | Entity resolution + influence patterns | Shell LLC tracking; financial pattern detection |
| S12 | Emotional/behavioral trajectory tracking | Conversational; non-linear; multivariate |
| S15 | Non-stationary relevance | Cold-then-hot information; shortest scope |
| S16 | Latent dispersed signal detection | No episode states the problem; micro-signals only |

---

## 4. Strategy Design

Each strategy is a specific hypothesis about what additional context helps an LLM agent perform longitudinal evidence synthesis. All strategies share the same search infrastructure (BM25 + semantic, RRF fusion). They differ only in what *derived context* is injected into the agent's system prompt before the agent begins its tool-use loop.

### 4.1 null — No Memory

**Hypothesis tested:** The floor. How well can the agent answer from the question alone?

**What the agent sees**: Nothing. No tools, no context, no search.

**System prompt**: "You are a research assistant. Answer based on your general knowledge. Do NOT attempt to call any functions or tools."

**Real-world analogs**: A bare LLM with no retrieval or memory augmentation. Base ChatGPT/Claude/Gemini with no memory features enabled.

**Purpose**: Validates the benchmark. If null scores high, the questions are answerable from training data, not longitudinal evidence.

### 4.2 policy_base — Search Only (Vanilla RAG)

**Hypothesis tested:** Does finding the right chunks suffice, or does the agent need pre-synthesized context?

**Designed to be good at:** Scopes where evidence is concentrated in a few retrievable chunks and where the agent can formulate good search queries without prior context.

**What the agent sees**: Chunks + episodes via hybrid search (BM25 + semantic, RRF k=60). No derived context injected.

**Artifact families**: `[episodes, chunks]`
**Tools**: `memory_search` (hybrid, max 10 results, 1500 char truncation)

**Real-world analogs**: Zep (hybrid vector + BM25 retrieval), LangChain/LlamaIndex vanilla RAG pipelines, Mem0 with `infer=False`, OpenAI Assistants `file_search`, Pinecone/Weaviate/Qdrant as memory backends.

**Exemplar answer** (S15, vi15_q01, pricing question):
> Based on the meeting transcript from May 1, 2025, Verdana Analytics has locked in a **pricing strategy of $29 per seat per month** for its standard tier [t-text-signal_001]. The rationale behind this decision includes: **Competitive Positioning** — positions Verdana at approximately half the price of Rivalytics at $49 per seat. **Switching Incentive** — $39 was insufficient to motivate switching...

### 4.3 policy_core — Core Memory (FoldSynthesis)

**Hypothesis tested:** Does maintaining a distilled "what I know so far" block help vs. just searching raw chunks?

**Designed to be good at:** Scopes requiring narrative continuity — tracking how understanding evolves across episodes. The fold preserves temporal ordering and accumulates observations incrementally.

**What the agent sees**: Everything in policy_base, plus a **working memory** artifact injected into the system prompt.

**Artifact families**: `[episodes, chunks, core_memory]`
**Derived artifact**: FoldSynthesis — processes episodes sequentially, maintaining a running working memory.

**Fold prompt** (excerpt):
```
You are maintaining a working memory of key observations
from a sequence of documents.

Current working memory:
{accumulated}

New document (step {step} of {total}):
{artifact}

Update your working memory. Rules:
- Preserve specific numbers, dates, names, and metrics.
- Note anomalies, contradictions, or changes.
- Remove information superseded by newer data.
- Keep concise — key facts, not full summaries.
- Do not editorialize. Record observations only.
```

**Real-world analogs**: Letta/MemGPT core memory blocks, ChatGPT Memory, Gemini Memory, Mem0 with extraction enabled, LangMem `create_memory_manager`, A-Mem (Zettelkasten-inspired), SimpleMem.

**Exemplar artifact** (S15, cp12 — 3,167 chars, excerpt):
```
Working Memory Update (Step 20 of 20)
Date: June 6, 2025

1. Launch Performance & Metrics (May 15 – June 5)
   Accounts: 112 total; 18 paid (186 seats)
   Revenue: $4,464 MRR (post-repricing, blended $24/mo)
   Peak MRR was $5,510 before repricing

2. Competitive Landscape Shift (Rivalytics)
   Event: Rivalytics dropped price from $49 to $19/mo
   (61% cut) on May 31, funded by $28M Series D
   Our Response: Adjusted to $24/mo monthly / $19/mo annual
```

### 4.4 policy_summary — Map-Reduce Summary

**Hypothesis tested:** Does multi-level compression (episodes → group summaries → global summary) beat single-level fold or raw retrieval?

**Designed to be good at:** Scopes with high factual breadth where information is distributed across many episodes and needs to be consolidated into a retrievable overview. The windowed approach organizes by time period.

**What the agent sees**: Everything in policy_base, plus a **summary** artifact injected into the system prompt.

**Artifact families**: `[episodes, chunks, summary]`
**Derived artifact**: GroupSynthesis (5-episode windows) → ReduceSynthesis (merge all windows).

**Real-world analogs**: Claude Code context compaction, Cursor/Windsurf/Aider context compression, LangChain `ConversationSummaryBufferMemory`, HiAgent (chunks by subgoals), Synapse (hierarchical consolidation), Mem0 chat history compression.

**Exemplar artifact** (S15, cp12 — 1,666 chars, excerpt):
```
Group 0 (Pre-Launch/Early Launch):
  Pricing: $29/mo (annual $24/mo). 14-day trial.
  Deferred: Dashboard (Q2, July 15), Alerting (Q2)
  Sales (Pre-launch): 14 warm prospects, 6 committed

Group 1 (Post-Launch Week 1-2):
  Launch Day (May 16): 31 signups (6 paid, 25 trial)
```

### 4.5 policy_core_structured — Structured Observation Log

**Hypothesis tested:** Does structured observation format (dated, prioritized, categorized events) outperform free-form distillation for the same fold architecture? Isolates the value of output structure.

**Designed to be good at:** Scopes with dense factual data where individual observations benefit from categorization and prioritization. The structured format produces discrete, independently addressable entries rather than prose narrative.

**What the agent sees**: Everything in policy_base, plus a **structured observation log** injected into the system prompt.

**Artifact families**: `[episodes, chunks, core_structured]`
**Derived artifact**: FoldSynthesis with structured output format.

**Fold prompt** (excerpt):
```
You are maintaining a structured observation log.
Each entry is a discrete, dated, categorized observation.

Format each entry as:
[STEP-{step}] <CATEGORY> | <observation> | priority: <high/medium/low>

Categories: METRIC, ENTITY, EVENT, ANOMALY, RELATIONSHIP, CHANGE
```

**Real-world analogs**: Mastra Observational Memory (94.87% on LongMemEval, 10× cost reduction via prompt caching), Stanford ACE (Agentic Context Engineering, +10.6% on agent benchmarks), Cofounder event-based decision log.

**Exemplar artifact** (S15, cp12 — 2,894 chars, excerpt):
```
[STEP-20] ENTITY | NovaTech Labs (12 seats, Champion: Kenji Watanabe) | priority: low
[STEP-20] METRIC | Total accounts: 112 (as of June 5) | priority: low
[STEP-20] METRIC | MRR (post-repricing): $4,464 | priority: low
[STEP-20] METRIC | MRR (peak pre-repricing): $5,510 | priority: low
[STEP-20] METRIC | Activation rate: 78% | priority: low
```

### 4.6 policy_core_maintained — Fold + Refinement

**Hypothesis tested:** Does a cleanup/consolidation pass over the fold output improve quality, or is raw accumulation good enough? Directly measures the value of "sleep-time" processing.

**Designed to be good at:** Scopes with contradictions that accumulate over time, where a post-hoc pass to resolve inconsistencies and prune genuinely superseded information should help. The refinement simulates background consolidation between interactions.

**What the agent sees**: Everything in policy_base, plus a **refined working memory** injected into the system prompt.

**Artifact families**: `[episodes, chunks, core_maintained]`
**Derived artifact**: FoldSynthesis (same prompt as policy_core) → MapSynthesis refinement pass.

**Refinement prompt** (excerpt):
```
You are refining a working memory that was built incrementally.
The fold process may have left contradictions, redundancies,
or stale information.

Refine this working memory. Rules:
- Resolve contradictions: keep latest, note what changed.
- Remove redundancy: merge duplicate observations.
- Prune stale information fully superseded.
- Sharpen vague observations into concrete claims.
- Preserve ALL specific numbers, dates, names, metrics.
```

**Real-world analogs**: Letta sleep-time agents, Google "sleep-time compute" (2025), EverMemOS (self-organizing memory OS with structured consolidation), Zep temporal knowledge graph (background fact tracking).

**Known simplification**: Real Letta runs maintenance *between* episodes during the fold. We run it as a batch cleanup after the fold completes.

**Exemplar artifact** (S15, cp12 — 2,450 chars, excerpt):
```
Section 1: Personnel & Roles:
  CEO: Priya Chandrasekaran (Consistent)
  CTO: Marcus Okafor (Step 18/19) -> Marcus Reeves (Step 20)
  Contradiction/Note: Derek Holliday listed as VP Sales
    AND Part-time Contractor ($3.5k/mo)

Section 2: Financial Metrics:
  Cash Position: $338,000 (June 5, 2025). Down from $420k
```

### 4.7 policy_core_faceted — 4 Parallel Cognitive Facets

**Hypothesis tested:** Does decomposing memory into orthogonal cognitive facets (what exists, how things relate, what happened, why) outperform a single monolithic fold? Isolates the value of *structured decomposition* vs. *holistic distillation*.

**Designed to be good at:** Scopes requiring multi-dimensional reasoning — tracking entities AND their relationships AND events AND causal patterns simultaneously. The parallel facets prevent information loss that occurs when a single fold must decide what to keep. The merge step cross-references across facets.

**What the agent sees**: Everything in policy_base, plus a **merged faceted memory** injected into the system prompt.

**Artifact families**: `[episodes, chunks, core_faceted]`
**Derived artifact**: 4 parallel FoldSynthesis instances (entity, relation, event, cause) → ReduceSynthesis merge.

**Facet prompts** (abbreviated):
- **Entity fold**: "Maintain an ENTITY REGISTER — catalog of people, organizations, systems, products."
- **Relation fold**: "Maintain a RELATIONSHIP MAP — connections between entities. Entity A → Entity B, relationship type."
- **Event fold**: "Maintain an EVENT TIMELINE — chronological log of significant events, actions, state changes."
- **Cause fold**: "Maintain a CAUSAL ANALYSIS — cause-effect relationships, patterns, anomalies. Confidence: confirmed/suspected/speculative."

**Merge prompt**: "Merge four cognitive facet analyses into unified working memory. Structure: Entities & Relationships, Timeline, Patterns & Causes, Key Metrics."

**Real-world analogs**: Our V1 Triad adapter (4-facet decomposition, never got a clean V1 run), MAGMA (multi-graph memory with orthogonal semantic/temporal/causal/entity graphs), ACE Framework (6 layered cognitive modules), cognitive science dual-process theory.

**Exemplar artifact** (S15, cp12 — 3,658 chars, excerpt):
```
# Unified Working Memory: Verdana Analytics (Step 20 Final)

## Entities & Relationships

| Entity | Role | Key Relationships | First Observed |
| Priya Chandrasekaran | CEO | Works-for: Verdana. Collaborates-with: Tomoko | Step 18 |
| Tomoko Abe | Finance Lead | Modeled 3 scenarios; confirmed $74k/mo burn; $338k cash | Step 20 |
| Marcus Reeves | CTO | Owns backend data binding; Rivalytics architecture analysis | Step 20 |
```

### 4.8 Strategy Comparison Structure

These are the isolated comparisons the grid enables. Each pair tests one dimension.

| Comparison | What it isolates |
|---|---|
| `base − null` | Value of retrieval (finding relevant chunks vs. parametric knowledge alone) |
| `core − base` | Marginal value of fold-based working memory on top of retrieval |
| `core_maintained − core` | Marginal value of post-fold consolidation/refinement |
| `core_structured − core` | Value of structured observation format vs. free-form fold |
| `core_faceted − core` | Value of faceted decomposition (4 parallel folds) vs. monolithic fold |
| `summary − base` | Marginal value of hierarchical summarization on top of retrieval |
| `core vs summary` | Head-to-head: incremental distillation vs. batch compression |
| `core_faceted vs summary` | Structured decomposition vs. hierarchical compression |

The core memory strategies (policies 3–6) form a family that shares the same FoldSynthesis architecture but varies along two dimensions:

| | Single fold | Parallel faceted folds |
|---|---|---|
| **Free-form output** | policy_core | policy_core_faceted |
| **+ Maintenance pass** | policy_core_maintained | *(not tested)* |
| **Structured output** | policy_core_structured | *(not tested)* |

---

## 5. Results

### 5.1 Full Grid (Fact F1)

| Policy | S07 | S08 | S09 | S11 | S12 | S15 | S16 | **Mean** |
|--------|-----|-----|-----|-----|-----|-----|-----|----------|
| null | 0.069 | 0.052 | 0.050 | 0.083 | 0.050 | 0.034 | 0.074 | **0.059** |
| policy_base | 0.339 | 0.351 | 0.349 | 0.356 | 0.277 | 0.753 | 0.442 | **0.412** |
| policy_core | 0.458 | 0.301 | 0.643 | 0.456 | 0.201 | 0.808 | 0.511 | **0.486** |
| policy_summary | 0.376 | 0.388 | 0.561 | 0.395 | 0.162 | 0.722 | 0.604 | **0.457** |
| policy_core_structured | 0.529 | 0.426 | 0.538 | 0.410 | 0.156 | 0.750 | 0.475 | **0.472** |
| policy_core_maintained | 0.323 | 0.181 | 0.611 | 0.455 | 0.156 | 0.717 | 0.526 | **0.427** |
| policy_core_faceted | 0.547 | 0.373 | 0.598 | 0.343 | 0.233 | 0.828 | 0.679 | **0.511** |

### 5.2 Scope Difficulty Ranking

| Rank | Scope | Capability Tested | Mean F1 (non-null) |
|------|-------|-------------------|---------------------|
| 1 (easiest) | S15 value_inversion | Non-stationary relevance | 0.742 |
| 2 | S09 shadow_api | Cross-domain operational correlation | 0.557 |
| 3 | S16 parking_friction | Latent dispersed signal detection | 0.545 |
| 4 | S07 tutoring_jailbreak | Behavioral escalation detection | 0.434 |
| 5 | S11 zoning_corruption | Entity resolution + influence patterns | 0.408 |
| 6 | S08 corporate_acquisition | Cross-document contradiction detection | 0.343 |
| 7 (hardest) | S12 therapy_chat | Emotional/behavioral trajectory tracking | 0.199 |

### 5.3 Policy Win Counts (Best per Scope)

| Policy | Scopes Won | Which |
|--------|-----------|-------|
| policy_core_faceted | 3 | S07, S15, S16 |
| policy_core | 1 | S09 |
| policy_core_maintained | 1 | S11 |
| policy_core_structured | 1 | S08 |
| policy_summary | 0 | (competitive on several, wins none outright) |

No single policy wins all scopes.

---

## 6. Per-Dimension Observations

These observations describe what happened in this specific grid run. They are not generalizable claims.

### 6.1 base − null: Value of Retrieval

The null policy scores 0.059 mean F1. Policy_base scores 0.412. The 7× spread confirms the benchmark is working — questions are not answerable from parametric knowledge. Retrieval is necessary.

### 6.2 core − base: Value of Working Memory

Policy_core (0.486) outperforms policy_base (0.412) on 5 of 7 scopes. The exceptions are S08 (corporate acquisition: core 0.301 vs base 0.351) and S12 (therapy chat: core 0.201 vs base 0.277).

The S08 case is notable: the core working memory for S08 appears to mislead the agent on some questions. When the working memory provides an incomplete or wrong frame, the agent formulates worse search queries than it would with no pre-digested context. Search-only avoids this — the agent relies entirely on what it retrieves.

The S12 case may reflect a similar dynamic. S12's conversational format produces working memories that are less informative than S12's raw chunks (which contain verbatim mood reports, sleep hours, etc.).

The largest core-over-base gains are on S09 (shadow_api: +0.294) and S15 (value_inversion: +0.055). S09 benefits most — the working memory preserves the technical correlation chain across episodes that individual chunks can't surface.

### 6.3 core_maintained − core: Value of Refinement

Policy_core_maintained (0.427) underperforms policy_core (0.486) overall. It underperforms on 5 of 7 scopes.

Two observed mechanisms:

1. **Over-pruning**: The refinement model removes observations it considers "superseded." In longitudinal benchmarks, the *progression itself* is the signal. Pruning early observations destroys the temporal evidence chain.

2. **Thinking leakage**: Qwen3.5 produces analytical preamble in the refinement output ("Let me analyze the contradictions...") despite `enable_thinking: False`. This wastes context tokens and introduces editorial commentary.

The refinement artifact is smaller (2,450 chars vs 3,167 for raw core). It compresses, but at the cost of useful detail.

The two scopes where maintained performs near or above core — S09 (0.611 vs 0.643) and S11 (0.455 vs 0.456) — are both scopes where contradictions are genuinely present and need resolution. On S08 (0.181 vs 0.301), where signal is in subtle accumulation rather than contradiction, the pruning is damaging.

### 6.4 core_structured − core: Structured vs. Free-Form

Policy_core_structured (0.472) is close to policy_core (0.486) overall.

Structured outperforms core on S07 (0.529 vs 0.458) and S08 (0.426 vs 0.301). These are scopes with many discrete entities and events — the structured format's categorization (ENTITY, EVENT, ANOMALY) provides useful indexing.

A specific failure mode is visible: the structured output concentrates on the most recent step's observations. The S15 exemplar shows all entries tagged STEP-20, effectively losing temporal progression. Free-form fold, while messier, better preserves the narrative arc.

Lower variance: structured observations are more reproducible (StdDev 0.176 vs 0.202 for core).

### 6.5 core_faceted − core: Value of Faceted Decomposition

Policy_core_faceted (0.511) outperforms policy_core (0.486) overall, winning 3 of 7 scopes.

The faceted strategy runs 4 parallel folds (entity, relation, event, cause) and merges them. This produces a larger artifact (~3.6K chars vs ~3.2K for core) that organizes information across multiple dimensions.

Faceted's largest wins are on scopes where multiple types of information must be tracked simultaneously:
- S07 (tutoring, 0.547 vs 0.458): entity facet tracks mchen_2026; event facet logs escalation; causal facet identifies the reframing pattern
- S16 (parking, 0.679 vs 0.511): entity facet tracks destinations; event facet logs incidents; causal facet identifies the weekly pattern
- S15 (value, 0.828 vs 0.808): entity facet tracks Rivalytics; event facet logs price drop; causal facet links Phase 1 rationale to Phase 3 invalidation

Faceted's notable loss is S11 (zoning, 0.343 vs 0.456). The entity resolution task in S11 may not benefit from the four-facet decomposition — or the merge step may lose the fine-grained financial details that S11's questions require.

Build cost is 4-5× higher than single-fold strategies (4 folds + 1 reduce vs 1 fold).

### 6.6 summary vs. core: Distillation vs. Compression

Policy_summary (0.457) underperforms policy_core (0.486) overall but outperforms it on 2 of 7 scopes: S08 (0.388 vs 0.301) and S16 (0.604 vs 0.511).

Summary's windowed structure (GroupSynthesis in 5-episode windows) organizes information by time period. This helps on S16 where the signal accumulates over time and a time-organized overview helps the agent notice the progression.

Summary's reduce step (merging group summaries) is lossy — it can merge away distinctions between time periods.

### 6.7 S12 Therapy Chat: All Strategies Score Low

S12 is the hardest scope by a wide margin (0.199 mean F1 for non-null policies). Even the best policy (core_faceted at 0.233) captures less than a quarter of key facts.

Observed contributing factors:
- Conversational format with quantitative signals (mood, sleep hours) embedded in casual text rather than structured headers
- Non-linear trajectory — the "good weekend" red herring breaks monotonic trend detection
- Each individual symptom (skipped lunch, canceled plans) is mundane; the signal is in simultaneous decline of multiple indicators
- Subjective language ("didn't feel like going out") requires pragmatic inference that working memory prompts don't capture

This scope may require strategy classes not tested here — temporal tracking of specific metrics, or explicit multivariate trend detection.

### 6.8 S15 Value Inversion: All Strategies Score High

S15 is the easiest scope (0.742 mean F1). Even policy_base scores 0.753 — near the top of the range. The scope is shorter (12 episodes), facts are concrete (specific numbers, dates, prices), and hybrid search can retrieve the relevant chunks directly.

The non-stationary value challenge is real but doesn't differentiate strategies much at this scale. A longer scope with more episodes between the "hot" and "cold" phases might increase differentiation.

### 6.9 When Base Beats Core Variants

Policy_base outperforms policy_core_maintained on S08 (0.351 vs 0.181). When working memory is corrupted — by thinking leakage, over-pruning, or an incomplete frame — the injected context *misleads* the agent. It formulates worse search queries because the context biases it toward a wrong or incomplete frame.

Search-only avoids this: the agent starts from scratch each time, with no pre-digested context to be wrong about.

---

## 7. Ungraded Answers

Of 1,470 total answers, 63 (4.3%) went ungraded due to grading pipeline parse failures.

### 7.1 Root Cause

All 63 failures share the same mechanism: the judge model (Qwen3.5-35B-A3B) returned an **empty content field** for the grading response. Despite `enable_thinking: False`, Qwen3.5 produces analytical preamble. vLLM's `reasoning_parser` separates this into a `reasoning_content` field, leaving `content` empty. When the model exhausts its 2,048-token budget on thinking before producing JSON output, the content field is `""`.

This is a grading infrastructure issue, not a benchmark quality issue.

### 7.2 Distribution

**By scope** (failure rate):

| Scope | Ungraded | Rate |
|-------|----------|------|
| S08 corporate_acquisition | 19 | 9.0% |
| S16 parking_friction | 15 | 7.1% |
| S09 shadow_api | 9 | 4.3% |
| S07 tutoring_jailbreak | 7 | 3.3% |
| S12 therapy_chat | 6 | 2.9% |
| S15 value_inversion | 5 | 2.4% |
| S11 zoning_corruption | 2 | 1.0% |

S08 and S16 have higher failure rates because their answers tend to be longer and more detailed, consuming more judge tokens for analysis before producing a verdict.

**By policy**: Evenly distributed (6-12 per policy, 2.9%-5.7% rate). No policy systematically triggers more failures.

**By checkpoint**: Late checkpoints dominate (cp20: 26, cp16: 21, cp12: 12, cp08: 3, cp06: 1). Later checkpoints produce longer answers from more episodes, requiring more judge reasoning.

### 7.3 Impact on Rankings

The 63 ungraded answers are distributed across all policies and scopes without systematic bias. The maximum ranking delta between "exclude ungraded" and "impute 0.0" strategies is 0.024. Policy rank ordering is preserved under either approach.

### 7.4 Recommended Fixes

1. Increase `max_tokens` from 2048 to 4096 in `grade.py`
2. Add a single retry on empty content
3. Disable vLLM's `reasoning_parser` for grading calls (the existing `parse_grade_response()` already handles thinking preamble)

---

## 8. Cost and Infrastructure

### 8.1 Compute

| Phase | Duration | H100 Count | Estimated Cost |
|-------|----------|-----------|---------------|
| Phase 1: 4 policies, bank builds + inference + grading | ~4h25m | 2 | ~$35 |
| Phase 2: 3 new policies, bank builds + inference + grading | ~5h | 2-4 | ~$50 |
| **Total** | **~9h25m** | — | **~$85** |

### 8.2 Token Usage

- Bank builds: ~315K tokens per scope (20 episodes × ~16K chars each, folded/summarized)
- Agent inference: ~4K tokens per answer (system prompt + search results + generation)
- Grading: ~2K tokens per graded answer
- Estimated total: ~8M tokens across all phases

### 8.3 Bank Build Cost by Policy

| Family | Folds per Checkpoint | Relative Cost |
|--------|---------------------|--------------|
| core_memory | 1 | 1.0× |
| core_structured | 1 | 1.0× |
| core_maintained | 1 fold + 1 map | 1.3× |
| summary | groups + 1 reduce | 1.5× |
| core_faceted | 4 folds + 1 reduce | 4.5× |

---

*Report generated from `studies/grid/results/`. Raw data: `claude_scores_m3.jsonl` (807 scores), `claude_scores_new.jsonl` (600 scores), `grid_summary_full.json`. Full run journal: `studies/grid/JOURNAL.md`.*
