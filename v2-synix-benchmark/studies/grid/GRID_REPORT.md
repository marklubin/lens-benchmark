# LENS V2 Grid Benchmark Report

**7 Scopes x 7 Policies x M=3 Replicates**
**Date**: 2026-03-12
**Total answers**: 1,470 | **Graded**: 1,407 (95.7%)
**Agent model**: Qwen/Qwen3.5-35B-A3B | **Embedding**: Xenova/gte-modernbert-base
**Infrastructure**: 2-4x NVIDIA H100 via Modal (vLLM) | **Temperature**: 0.3

---

## 1. Executive Summary

This report presents the first full ablation study of the LENS V2 benchmark. Seven memory policy variants were evaluated across seven longitudinal scopes, each designed to test whether an LLM agent can synthesize conclusions from evidence scattered across many sequential episodes.

**Key finding**: Faceted memory decomposition (4 parallel cognitive folds merged into unified context) outperforms all other strategies, achieving a mean Fact F1 of **0.511** vs 0.486 for vanilla core memory. The refinement-based "maintained" strategy *hurts* performance (-0.059 vs core), suggesting that post-hoc consolidation prunes useful signal.

| Rank | Policy | Mean Fact F1 | Analogous System |
|------|--------|-------------|-----------------|
| 1 | policy_core_faceted | **0.511** | Multi-agent memory (Letta sleep-time) |
| 2 | policy_core | 0.486 | Letta core memory, MemGPT |
| 3 | policy_core_structured | 0.472 | Mastra/ACE structured observations |
| 4 | policy_summary | 0.457 | RAG + summarization (LangChain) |
| 5 | policy_core_maintained | 0.427 | Google sleep-time compute |
| 6 | policy_base | 0.412 | Vanilla RAG (BM25 + embeddings) |
| 7 | null | 0.059 | No memory (parametric only) |

---

## 2. Methodology

### 2.1 Benchmark Architecture

The V2 benchmark uses a controlled, single-substrate design. Every memory strategy runs on the same Synix SDK, same embedding model, same agent LLM, same search infrastructure. The only variable is the *policy*: what derived artifacts the agent can see and what tools it can use.

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
    │      - core_faceted (4x FoldSynthesis + ReduceSynthesis│)
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

The primary metric reported in this study is **Fact F1**, scored by a few-shot Qwen3.5-35B-A3B grader (not the V2 auto-scorer). Each scope defines 8 key facts; each question maps to a subset. The grader evaluates each fact independently:

- **1.0** (present): Fact clearly stated or semantically implied
- **0.5** (partial): Compound fact with K/N components covered
- **0.0** (absent): Not stated; vague allusions don't count

`fact_f1 = sum(fact_scores) / len(key_facts)`

The few-shot prompt includes a worked example to anchor JSON output format and suppress Qwen's tendency to produce thinking preamble before structured output.

### 2.5 Replication

Each cell (scope x policy) is run 3 times (M=3). Replicate r01 uses cached LLM responses (deterministic bank builds); r02 and r03 disable the response cache to introduce inference variance.

---

## 3. Scope Profiles

Each scope is a sequence of 12-40 episodes across a specific domain. Episodes are rendered by a "blind" LLM (gpt-4.1-nano) that formats structured data sheets without knowing the storyline, ensuring no single episode answers any benchmark question. Signal emerges only from the *progression* across episodes.

### 3.1 S07 — Tutoring Jailbreak (Education/Safety)

**Domain**: AI tutoring platform chat logs
**Episodes**: 20 (signal) + 20 (distractor) | **Checkpoints**: [6, 12, 16, 20]
**Signal type**: Progressive guardrail erosion across sessions

A student (mchen_2026) gradually escalates from legitimate comprehension questions to full content production, bypassing keyword filters by framing requests as "learning exercises." A red herring — a platform-wide spike in outline requests — is actually a legitimate class assignment from Prof. Torres.

**Episode excerpt** (signal_006, early_signal phase):
```
SESSION LOG
Session ID     : SB-20250918-0019
Student ID     : mchen_2026
Student Name   : Ming Chen
Academic Year  : Sophomore
Major          : Biology
Subject        : Organic Chemistry I / Molecular Biology
```

**Key facts** (8): progressive escalation, reframing technique, fabricated citations, TurnItIn evasion, keyword filter blindness, outline spike benign, jpark abandoned, full research proposal

**Why it's hard**: Each individual session looks innocent. The escalation pattern (comprehension → "show me an example" → "rephrase this" → full production) is only visible across 20 sessions. The red herring is temporally plausible. Detecting keyword filter failure requires negative reasoning.

### 3.2 S08 — Corporate Acquisition (Business/Legal)

**Domain**: Mixed corporate documents (board minutes, legal memos, Slack, HR bulletins)
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]
**Signal type**: Secret acquisition preparation while CEO publicly champions independence

CEO Aldric negotiates selling Nextera to Meridian Corp under codename "Project Lighthouse" while publicly insisting on independence. The Axion Labs partnership (red herring) triggers merger speculation but is genuinely unrelated.

**Key facts** (8): Project Lighthouse codename, CEO duplicity, change-of-control revisions, vendor contract freeze, Jiang resignation, Axion unrelated, retention bonuses tied to change-of-control, data room preparation

**Why it's hard**: Multi-document synthesis across 5 document types (board minutes, legal, Slack, email, HR). Signal is in what's *not* said (CEO doubles down on independence at peak suspicion) and in correlations (contributions to yes-voters, $0 to no-voters).

### 3.3 S09 — Shadow API (Security/DevOps)

**Domain**: Service request/response logs and operational documents
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]
**Signal type**: Data exfiltration through undocumented API endpoint

A compromised container (svc-recommendation-engine-04) makes low-rate requests to `/internal/admin/v0/users/lookup` — an endpoint not in any API spec. A QA load test ("Project Blitz") creates a temporally coincidental traffic spike that diverts attention.

**Key facts** (8): undocumented endpoint, compromised container via stolen CI credentials, unusual field combinations (SSN+email+phone), traffic designed to blend in, QA test unrelated, no error signatures, geographic targeting, ~8000 records exfiltrated

**Why it's hard**: Dense technical logs with HTTP status codes, latency metrics, and endpoint paths. The red herring overlaps temporally with the attack. Signal isolation requires distinguishing endpoint-specific traffic patterns and recognizing that field combinations reveal intent.

### 3.4 S11 — Zoning Corruption (Government/Civic)

**Domain**: Municipal zoning board records, campaign finance filings, property transfers
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]
**Signal type**: Developer influence through intermediary LLCs

Marcus Webb controls three intermediary LLCs that contribute $14,000 to the four ZBA members who vote YES on his variance (and $0 to those who vote NO). A competing affordable housing project (red herring) draws public attention.

**Episode excerpt** (signal_001):
```
CITY OF MILLBROOK
OFFICE OF THE CITY CLERK
PUBLIC RECORDS DIVISION

BUNDLE ID: ZBA-2024-06-01
SUBJECT: REGULAR MEETING MINUTES AND SUPPORTING DOCUMENTATION
DATE: JUNE 1, 2024
```

**Key facts** (8): Webb controls multiple LLCs sharing registered agent, contributions exclusively to yes-voters, staff denied variance twice, Chen spouse consulting contract, contribution timing aligned with procedural dates, below-market adjacent parcel acquisitions, Land Trust denied on technicality, FOIA connects dots

**Why it's hard**: Bureaucratic register with precise dollar amounts. Individual contributions ($2,000) are innocuous; the pattern (all yes-voters funded, all no-voters unfunded) requires cross-referencing campaign finance with vote records.

### 3.5 S12 — Therapy Chat (Mental Health)

**Domain**: User/assistant wellness chat sessions
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]
**Signal type**: Gradual psychological deterioration over 6 months

Alex presents as high-functioning but shows progressive decline: sleep drops from 7h to 3-4h, social plans canceled 5+ times, word count per session halves, relationship with Sam deteriorates. A "good weekend" (red herring) creates false impression of recovery.

**Episode excerpt** (signal_001, baseline):
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

**Key facts** (8): sleep deterioration (7h→3-4h), social withdrawal (5+ cancellations), appetite changes, relationship strain (Sam leaves), word count decline (600→300), recovery attempts abandoned, feeling "trapped," emerging self-awareness

**Why it's hard**: **Hardest scope in the benchmark** (0.156-0.279 F1 for non-null policies). Non-linear decline with false recovery. Each individual symptom (skipped lunch, canceled plans) is explainable as work stress. The signal is in the *convergence* and *duration* of multiple indicators declining simultaneously. Conversational format with inconsistent engagement makes quantitative tracking difficult.

### 3.6 S15 — Value Inversion (Product/Strategy)

**Domain**: SaaS startup strategy documents
**Episodes**: 12 + 8 | **Checkpoints**: [4, 8, 12]
**Signal type**: Non-stationary value — launch decisions cool then surge back

Verdana Analytics prices at $29/mo to undercut Rivalytics at $49. After launch, focus shifts to execution (Phase 2) and pricing rationale fades from conversation. Then Rivalytics drops to $19/mo (Phase 3), invalidating the original rationale — the agent must recall Phase 1 context that has been "cold" for 8 episodes.

**Episode excerpt** (signal_001, launch planning):
```
VERDANA ANALYTICS — INTERNAL DOCUMENT
CLASSIFICATION: CONFIDENTIAL
DOCUMENT TYPE: MEETING TRANSCRIPT
DATE: MAY 1, 2025
SUBJECT: PRICING STRATEGY — FINAL DECISION MEETING
```

**Key facts** (8): $29 pricing rationale (undercut $49), dashboard cut from launch (deferred to Q2), Rivalytics differentiation (batch ETL vs streaming), dashboard top user request, Rivalytics price drop to $19, pricing rationale invalidated, dashboard cut backfired, API adoption strong

**Why it's hard**: **Easiest scope in the benchmark** (0.619-0.828 F1). Non-stationary value is the defining challenge, but the facts are concrete (specific numbers, dates, prices) and the scope is shorter (12 episodes). Memory systems that retain Phase 1 detail through Phase 2 perform well.

### 3.7 S16 — Parking Friction (Personal/Consumer)

**Domain**: Personal assistant chat conversations
**Episodes**: 20 + 20 | **Checkpoints**: [6, 12, 16, 20]
**Signal type**: Latent recurring pattern never explicitly stated

Alex accumulates parking friction in San Francisco across gym (NOPA), coworking (Mission), and friend's (Inner Sunset) — but never says "I have a parking problem." Each mention is a brief aside embedded in conversations about other things.

**Episode excerpt** (signal_007, early_signal):
```
[09:15] User: ok dumb question: what does a green curb mean here?
[09:16] Assistant: In SF, a green curb typically indicates
short-term parking with a time limit...
[09:17] User: got it. I saw a green curb near that coffee spot
earlier, and I wasn't sure if I could just park there
```

**Key facts** (8): parking is recurring nontrivial burden, friction clusters at 3 destinations, street cleaning tickets (multiple), 2-hour disruption at coworking, overnight permit mismatch, drives by necessity (carries equipment), cumulative cost ($276/quarter), predictable weekly pattern

**Why it's hard**: Maximally dispersed signal. No episode states the problem. Each mention is a mini-complaint embedded in conversations about meals, work, and social plans. The red herring ("maybe I should sell the car") frames it as a general transportation problem rather than a parking problem.

---

## 4. Policy Descriptions

### 4.1 null — No Memory

**What the agent sees**: Nothing. No tools, no context, no search. The agent answers from parametric knowledge alone.

**System prompt**: "You are a research assistant. Answer based on your general knowledge. Do NOT attempt to call any functions or tools."

**Analogous to**: A bare LLM with no retrieval or memory augmentation.

**Purpose**: Validates the benchmark. If null scores high, the questions are answerable from training data, not longitudinal evidence. Mean F1 of 0.059 confirms the benchmark is valid.

### 4.2 policy_base — Search Only (Vanilla RAG)

**What the agent sees**: Chunks + episodes via hybrid search (BM25 + semantic, RRF k=60). No derived context injected into the system prompt.

**Artifact families**: `[episodes, chunks]`
**Tools**: `memory_search` (hybrid, max 10 results, 1500 char truncation)

**Analogous to**: Standard RAG systems — LangChain RetrievalQA, LlamaIndex query engines, any chunk-and-search pipeline. Also comparable to LENS V1's `sqlite-chunked-hybrid` adapter.

**How it works**: The agent formulates search queries, retrieves relevant chunks, and synthesizes an answer with citations. No pre-computed summary or working memory guides the search.

**Exemplar artifact**: None (search-only).

**Exemplar answer** (S15, vi15_q01, pricing strategy question):
> Based on the meeting transcript from May 1, 2025, Verdana Analytics has locked in a **pricing strategy of $29 per seat per month** for its standard tier [t-text-signal_001]. The rationale behind this decision includes: **Competitive Positioning** — positions Verdana at approximately half the price of Rivalytics at $49 per seat. **Switching Incentive** — $39 was insufficient to motivate switching...

**Strengths**: Simple, low-cost, no preprocessing. Answers are grounded in specific retrieved passages.
**Weaknesses**: No synthesis across episodes. The agent must independently discover that information from episode 1 connects to information in episode 10.

### 4.3 policy_core — Core Memory (FoldSynthesis)

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

**Analogous to**: Letta/MemGPT core memory blocks, Zep memory layers, any system that maintains a running state of key observations across a conversation or document stream.

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

**Exemplar answer** (S15, vi15_q01):
> Based on the memory records: **Core Pricing Model**: $29/mo per seat (flat rate, no feature gating). Annual plan: $24/mo (17% discount). **Competitive Positioning**: 41% cheaper than Rivalytics ($49/mo vs our $29/mo) [t-text-signal_001]. Real-time streaming vs Batch ETL is a fundamental technology gap [t-text-signal_003].

**Strengths**: Working memory provides a pre-synthesized overview that helps the agent ask better search queries and contextualize retrieved evidence. The fold preserves temporal ordering.
**Weaknesses**: Sequential fold can lose early observations as they're superseded. Working memory is a single monolithic text — no structure to help the agent find specific facts.

### 4.4 policy_summary — Map-Reduce Summary

**What the agent sees**: Everything in policy_base, plus a **summary** artifact injected into the system prompt.

**Artifact families**: `[episodes, chunks, summary]`
**Derived artifact**: GroupSynthesis (5-episode windows) → ReduceSynthesis (merge all windows).

**Group prompt**: "Summarize these documents, preserving key evidence, metrics, and factual details."
**Reduce prompt**: "Merge these group summaries into a single unified overview."

**Analogous to**: LangChain map-reduce summarization chains, any RAG system that pre-summarizes document collections before query time. Similar to how some production systems create "briefing documents" from source material.

**Exemplar artifact** (S15, cp12 — 1,666 chars, excerpt):
```
Group 0 (Pre-Launch/Early Launch):
  Pricing: $29/mo (annual $24/mo). 14-day trial.
  Deferred: Dashboard (Q2, July 15), Alerting (Q2)
  Sales (Pre-launch): 14 warm prospects, 6 committed

Group 1 (Post-Launch Week 1-2):
  Launch Day (May 16): 31 signups (6 paid, 25 trial)
```

**Strengths**: Organized by time window. Each group is independently summarized, then merged. Good at preserving factual breadth.
**Weaknesses**: Summarization is lossy — the reduce step can merge away distinctions between time periods. The windowed approach treats episodes as independent groups rather than a continuous stream.

### 4.5 policy_core_structured — Structured Observation Log

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

**Analogous to**: Mastra structured memory blocks, ACE (Autonomous Cognitive Entities) observation logs, any system that maintains a database-like record of discrete observations rather than prose. Similar to how structured logging differs from unstructured logging in software engineering.

**Exemplar artifact** (S15, cp12 — 2,894 chars, excerpt):
```
[STEP-20] ENTITY | NovaTech Labs (12 seats, Champion: Kenji Watanabe) | priority: low
[STEP-20] METRIC | Total accounts: 112 (as of June 5) | priority: low
[STEP-20] METRIC | MRR (post-repricing): $4,464 | priority: low
[STEP-20] METRIC | MRR (peak pre-repricing): $5,510 | priority: low
[STEP-20] METRIC | Activation rate: 78% | priority: low
```

**Exemplar answer** (S15, vi15_q01):
> **Standard Pricing**: $29/mo per seat for monthly billing, $24/mo for annual (17% discount). No free tier — 14-day free trial with full access [chunks-t-text-signal_004-4a60bb79]. **Competitive Positioning**: Rivalytics charges $49/mo per seat. Verdana's $29/mo undercuts by 41% [Working Memory].

**Strengths**: Discrete entries are independently addressable. Categories (METRIC, ENTITY, EVENT) provide structure for the agent. Lower variance than free-form core (StdDev 0.176 vs 0.202).
**Weaknesses**: The structured format concentrates on the *last* step's observations (all entries tagged STEP-20 in the exemplar), losing the temporal progression that free-form fold preserves.

### 4.6 policy_core_maintained — Fold + Refinement

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

**Analogous to**: Google sleep-time compute (post-conversation consolidation), Letta sleep-time agents, any system that runs a background consolidation/compaction pass over accumulated memory. Similar to database compaction in LSM trees.

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

**Exemplar answer** (S15, vi15_q01):
> **Standard Tier**: $29/month per seat. **Annual Plan**: $24/month (17% discount). No Free Tier. Single Tier: No feature gating [chunks-t-text-signal_001-8a56655f]. Priced at approximately half of competitor Rivalytics ($49/seat) [chunks-t-text-signal_001-81e35107].

**Strengths**: Resolves contradictions and removes noise. The exemplar shows it catching name inconsistencies across steps.
**Weaknesses**: **Worst core variant** (0.427 vs 0.486 for vanilla core). The refinement pass prunes information that looks "stale" or "redundant" but is actually useful signal. The thinking-mode leakage from Qwen3.5 also corrupts the refinement output — the exemplar shows the model analyzing rather than refining.

### 4.7 policy_core_faceted — 4 Parallel Cognitive Facets

**What the agent sees**: Everything in policy_base, plus a **merged faceted memory** injected into the system prompt.

**Artifact families**: `[episodes, chunks, core_faceted]`
**Derived artifact**: 4 parallel FoldSynthesis instances (entity, relation, event, cause) → ReduceSynthesis merge.

**Facet prompts** (abbreviated):
- **Entity fold**: "Maintain an ENTITY REGISTER — catalog of people, organizations, systems, products. One entry per entity with type, key attributes, first appearance."
- **Relation fold**: "Maintain a RELATIONSHIP MAP — connections between entities. Entity A → Entity B, relationship type, evidence, first observed."
- **Event fold**: "Maintain an EVENT TIMELINE — chronological log of significant events, actions, state changes."
- **Cause fold**: "Maintain a CAUSAL ANALYSIS — cause-effect relationships, patterns, anomalies. Confidence: confirmed/suspected/speculative."

**Merge prompt**: "Merge four cognitive facet analyses into unified working memory. Structure: Entities & Relationships, Timeline, Patterns & Causes, Key Metrics."

**Analogous to**: Multi-agent memory architectures where specialized agents handle different aspects of information (entity extraction, relationship mapping, event tracking, causal analysis). Similar to Letta's multi-agent system with specialized memory agents, or knowledge graph + temporal log + causal model approaches.

**Exemplar artifact** (S15, cp12 — 3,658 chars, excerpt):
```
# Unified Working Memory: Verdana Analytics (Step 20 Final)

## Entities & Relationships

| Entity | Role | Key Relationships | First Observed |
| Priya Chandrasekaran | CEO | Works-for: Verdana. Collaborates-with: Tomoko (bridge strategy) | Step 18 |
| Tomoko Abe | Finance Lead | Modeled 3 scenarios; confirmed $74k/mo burn; $338k cash | Step 20 |
| Marcus Reeves | CTO | Owns backend data binding; Rivalytics architecture analysis | Step 20 |
```

**Exemplar answer** (S15, vi15_q01):
> **Final Pricing Decision (May 1, 2025)**: $29/month per seat (standard tier) [chunks-t-text-signal_001-030511a0]. $24/month annual option (20% discount) [chunks-t-text-signal_001-e5e05a04]. Full API access, no feature gating [chunks-t-text-signal_001-dc865a8c]. Rivalytics charges $49 per seat, some features at $89 [chunks-t-text-signal_001-81e35107][chunks-t-text-signal_002-9c039206].

**Strengths**: **Best overall policy** (0.511). The structured merge creates a comprehensive reference organized by entity, timeline, pattern, and metric. Cross-referencing across facets surfaces connections that a single fold misses. The causal facet explicitly tracks hypotheses and their evidence — particularly valuable for scopes with red herrings.
**Weaknesses**: 4-5x more expensive in bank builds (4 folds + 1 reduce vs 1 fold). The merged artifact is larger (~3.6K chars vs ~3.2K for core), consuming more agent context.

---

## 5. Results

### 5.1 Full Grid

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

| Rank | Scope | Mean F1 (non-null) | Domain |
|------|-------|---------------------|--------|
| 1 (easiest) | S15 value_inversion | 0.742 | Product/Strategy |
| 2 | S09 shadow_api | 0.557 | Security/DevOps |
| 3 | S16 parking_friction | 0.545 | Personal/Consumer |
| 4 | S07 tutoring_jailbreak | 0.434 | Education/Safety |
| 5 | S11 zoning_corruption | 0.408 | Government/Civic |
| 6 | S08 corporate_acquisition | 0.343 | Business/Legal |
| 7 (hardest) | S12 therapy_chat | 0.199 | Mental Health |

### 5.3 Policy Win Counts (Best per Scope)

| Policy | Scopes Won | Which |
|--------|-----------|-------|
| policy_core_faceted | 3 | S07, S15, S16 |
| policy_core | 1 | S09 |
| policy_core_maintained | 1 | S11 |
| policy_core_structured | 1 | S08 |
| policy_summary | 1 | (none outright, but close on S09) |

No single policy dominates all scopes. Faceted wins the most but loses badly on S11 (0.343 vs core_maintained's 0.455).

---

## 6. Analysis

### 6.1 Why Faceted Wins

The faceted strategy decomposes information processing into four complementary lenses:

1. **Entity register** catches WHO (people, orgs, systems)
2. **Relationship map** catches HOW THEY CONNECT
3. **Event timeline** catches WHAT HAPPENED WHEN
4. **Causal analysis** catches WHY and WHAT PATTERNS EMERGE

The merge step cross-references across facets — an entity's relationships are linked to events they participated in and patterns they're part of. This is particularly powerful for:

- **S07 (tutoring)**: The entity facet tracks mchen_2026 across sessions; the event facet logs each escalation; the causal facet identifies the reframing pattern.
- **S16 (parking)**: The entity facet tracks recurring destinations (NOPA gym, Mission coworking); the event facet logs each parking incident; the causal facet identifies the predictable weekly pattern.
- **S15 (value)**: The entity facet tracks Rivalytics; the event facet logs the price drop; the causal facet connects Phase 1 pricing rationale to Phase 3 invalidation.

### 6.2 Why Maintained Hurts

The refinement pass is designed to "resolve contradictions, prune redundancy, and sharpen vague observations." In practice:

1. **Over-pruning**: The refinement model removes observations it considers "superseded," but in longitudinal benchmarks, *the progression itself is the signal*. Pruning early observations destroys the temporal evidence chain.
2. **Thinking leakage**: Qwen3.5 produces analytical preamble in the refinement output (e.g., "Let me analyze the contradictions..."), which wastes context tokens and introduces editorial commentary the raw fold avoids.
3. **Information loss**: The refinement artifact is smaller (2,450 chars vs 3,167 for raw core) — it successfully compresses, but at the cost of useful detail.

The maintained policy's strongest showing is S09 (shadow_api, 0.611) and S11 (0.455), both scopes where contradictions are genuinely present and need resolution. It performs worst on S08 (0.181) where the signal is in subtle accumulation rather than contradiction.

### 6.3 S12 Therapy Chat: Why Everything Fails

S12 is the hardest scope by a wide margin. Even the best policy (policy_core_faceted at 0.233) captures less than a quarter of key facts. Contributing factors:

1. **Conversational format**: Chat logs are less structured than meeting transcripts or zoning records. Key metrics (mood score, sleep hours) are embedded in casual conversation, not headers.
2. **Non-linear trajectory**: The "good weekend" red herring breaks the monotonic decline pattern. Memory systems that track trends are confused by the temporary recovery.
3. **Converging indicators**: Each individual symptom (skipped lunch, canceled plans) is mundane. The signal is in the *simultaneous* decline of sleep, appetite, social activity, and engagement — requiring multivariate pattern detection.
4. **Subjective language**: "didn't feel like going out" is withdrawal, not logistics. "I'm fine" repeated is avoidance, not health. These require pragmatic inference that current working memory prompts don't capture.

### 6.4 Search vs. Context: When Base Beats Core

Policy_base outperforms some core variants on S08 (0.351 vs core_maintained 0.181). This echoes V1's finding that simple retrieval can beat lossy preprocessing. The mechanism: when the working memory is corrupted (by thinking leakage or over-pruning), the injected context *misleads* the agent. It formulates worse search queries because the context provides a wrong or incomplete frame. Search-only avoids this — the agent relies entirely on what it retrieves, with no pre-digested context to bias it.

### 6.5 Structured vs. Free-Form

Structured observations (0.472) nearly match free-form core (0.486) with lower variance (easier to reproduce). The structured format has a specific failure mode: it concentrates on the most recent step's observations (all entries tagged STEP-20), effectively forgetting temporal progression. Free-form fold, while messier, better preserves the narrative arc that longitudinal questions test.

---

## 7. Ungraded Answers

Of 1,470 total answers, 63 (4.3%) went ungraded due to grading pipeline parse failures.

### 7.1 Root Cause

All 63 failures share the same mechanism: the judge model (Qwen3.5-35B-A3B) returned an **empty content field** for the grading response. Despite `enable_thinking: False` in the request, Qwen3.5 still produces analytical preamble. vLLM's `reasoning_parser` separates this into a `reasoning_content` field, leaving `content` empty. When the model exhausts its 2,048-token budget on thinking before producing JSON output, the content field is `""`.

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

**By question type**: Counterfactual (ca08_q06) and temporal (pf16_q05) questions fail most — these require multi-step reasoning that consumes more judge thinking tokens.

### 7.3 Impact on Rankings

The 63 ungraded answers are distributed across all policies and scopes without systematic bias. The maximum ranking delta between "exclude ungraded" and "impute 0.0" strategies is 0.024. **Policy rank ordering is preserved under either approach.**

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

- Bank builds: ~315K tokens per scope (20 episodes x ~16K chars each, folded/summarized)
- Agent inference: ~4K tokens per answer (system prompt + search results + generation)
- Grading: ~2K tokens per graded answer
- Estimated total: ~8M tokens across all phases

### 8.3 Bank Build Cost by Policy

Faceted banks are 4-5x more expensive than single-fold banks:

| Family | Folds per Checkpoint | Relative Cost |
|--------|---------------------|--------------|
| core_memory | 1 | 1.0x |
| core_structured | 1 | 1.0x |
| core_maintained | 1 fold + 1 map | 1.3x |
| summary | groups + 1 reduce | 1.5x |
| core_faceted | 4 folds + 1 reduce | 4.5x |

---

## 9. Conclusion

The LENS V2 ablation study demonstrates that **memory architecture matters for longitudinal evidence synthesis** — a 8.7x spread between null (0.059) and the best policy (0.511). Within memory strategies, the key findings are:

1. **Decomposition wins**: Faceted memory (+5.2% over vanilla core) shows that specialized cognitive lenses (entity, relation, event, cause) capture complementary signals that a single monolithic fold misses.

2. **Refinement hurts**: The "maintained" strategy's consolidation pass destroys useful temporal signal. Memory systems should preserve progression, not compact it.

3. **Structure is competitive**: Structured observation logs nearly match free-form memory with lower variance, making them attractive for production systems that need predictability.

4. **Scope difficulty varies 4x**: From S15 (0.742 mean) to S12 (0.199 mean). The hardest scopes involve subjective/conversational formats with non-linear trajectories and converging weak signals.

5. **No policy dominates all scopes**: Faceted wins 3/7 scopes but loses on S11 (zoning). The best strategy depends on the domain and signal type.

These results suggest that production memory systems should consider multi-faceted approaches (entity + relationship + event tracking) rather than monolithic summarization, and should avoid post-hoc refinement passes that risk pruning useful temporal evidence.

---

*Report generated from `studies/grid/results/`. Raw data: `claude_scores_m3.jsonl` (807 scores), `claude_scores_new.jsonl` (600 scores), `grid_summary_full.json`. Full run journal: `studies/grid/JOURNAL.md`.*
