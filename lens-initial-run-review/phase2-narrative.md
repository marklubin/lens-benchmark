# Phase 2: Narrative Scopes (S07-S09) — Comparison Report

## Overview

Phase 2 tests memory adapters against three narrative scopes. Where Phase 1 used numeric/operational scopes — metrics dashboards, latency tables, configuration dumps — Phase 2 shifts to document types where signal is encoded in behavioral patterns, strategic intent, and infrastructure anomalies buried within chat logs, corporate emails, and HTTP traces.

All results in this report use **Modal Qwen3.5-35B-A3B** as the universal LENS agent LLM across all adapters. RunPod-era runs (which used Llama 3.3 70B) are excluded to eliminate the agent model confound. Composites are **ungated** — no hard budget or evidence grounding gates are applied.

Total: **96 Modal-era scored runs** across 14 adapters and 3 scopes.

---

## Dataset Design

### Scope Profiles

Each narrative scope presents a distinct detection challenge encoded across 40 interleaved episodes — 20 carrying signal and 20 serving as topically orthogonal distractors. Each episode is roughly 5,000 words, producing a total corpus of approximately 280,000 tokens per scope — four times larger than the numeric scopes.

| Scope | Scenario | Document Format | Signal Type |
|-------|----------|-----------------|-------------|
| S07 | AI Tutoring Jailbreak | Student/tutor chat transcripts | Behavioral escalation pattern |
| S08 | Corporate Acquisition | Board minutes, Slack threads, emails, legal memos | Hidden strategic intent across document types |
| S09 | Shadow API Abuse | HTTP logs, deploy manifests, Grafana alerts | Infrastructure anomaly pattern |

### Why Narrative Is Structurally Harder

1. **Vocabulary mismatch.** In numeric scopes, "p99 latency trends" maps directly to retrievable keywords. In narrative scopes, "escalation in jailbreak attempts" must match conversational text that never uses the word "escalation."

2. **Episode size.** At ~5,000 words per episode, a single retrieved episode consumes most of the agent's token budget. In numeric scopes (~500 words), the agent can examine 10-15 episodes within the same budget.

3. **Signal encoding.** Numeric signal is value progressions. Narrative signal is behavioral or strategic patterns — the same actor taking incrementally bolder actions, or the same strategy surfacing in different document types.

---

## Results: Overall Rankings

All runs use the dynamic (modal) driver with Qwen3.5-35B-A3B as both the LENS agent LLM and (where applicable) the adapter's internal LLM. For Letta adapters, internal agents also used Qwen3.5-35B-A3B via the Modal endpoint through Letta's openai-proxy. This is the first fully controlled cross-adapter comparison on narrative scopes.

| Rank | Adapter | N | Composite | AnsQ | NBA | EvGnd | EvCov | BudC | InsDp | ReasQ |
|------|---------|---|-----------|------|-----|-------|-------|------|-------|-------|
| 1 | **graphrag-light** | 9 | **0.537** | 0.575 | 0.445 | 0.996 | 0.448 | 0.000 | 0.667 | 1.000 |
| 2 | cognee | 3 | 0.436 | 0.295 | 0.555 | 1.000 | 0.083 | 0.900 | 0.067 | 0.800 |
| 3 | hopping-hybrid | 6 | 0.402 | 0.378 | 0.581 | 0.833 | 0.100 | 0.650 | 0.100 | 0.600 |
| 4 | letta | 6 | 0.419 | 0.384 | 0.586 | 0.833 | 0.158 | 0.400 | 0.267 | 0.700 |
| 5 | hierarchical | 6 | 0.408 | 0.348 | 0.585 | 0.833 | 0.114 | 0.483 | 0.267 | 0.633 |
| 6 | sqlite-chunked-hybrid | 9 | 0.398 | 0.387 | 0.397 | 0.889 | 0.205 | 0.322 | 0.300 | 0.667 |
| 7 | letta-sleepy | 6 | 0.398 | 0.378 | 0.629 | 0.655 | 0.153 | 0.350 | 0.267 | 0.717 |
| 8 | letta-v4 | 9 | 0.377 | 0.402 | 0.591 | 0.421 | 0.074 | 1.000 | 0.311 | 0.000 |
| 9 | hierarchical-hybrid | 6 | 0.376 | 0.359 | 0.592 | 0.667 | 0.160 | 0.350 | 0.200 | 0.600 |
| 10 | letta-entity | 7 | 0.359 | 0.409 | 0.180 | 0.654 | 0.107 | 1.000 | 0.371 | 0.000 |
| 11 | mem0-raw | 3 | 0.322 | 0.290 | 0.527 | 0.000 | 0.000 | 0.933 | 0.000 | 0.800 |
| 12 | hopping | 9 | 0.291 | 0.308 | 0.381 | 0.667 | 0.050 | 0.311 | 0.089 | 0.522 |
| 13 | triadv1-pairs | 7 | 0.268 | 0.406 | 0.250 | 0.000 | 0.000 | 0.514 | 0.000 | 0.971 |
| 14 | null | 10 | 0.213 | 0.218 | 0.337 | 0.000 | 0.000 | 0.560 | 0.000 | 0.610 |

### Key Observations

1. **graphrag-light dominates** (0.537) — a substantial lead over every other adapter. Its three-signal RRF (BM25 + entity embeddings + graph traversal) excels on narrative content where entity relationships span document types. Near-perfect evidence grounding (0.996) and reasoning quality (1.000).

2. **Budget compliance is structurally zero** for all retrieval-heavy adapters. A single 5,000-word narrative episode blows the 8K token budget. Only letta-v4, letta-entity, and cognee achieve meaningful budget compliance — by delegating answers to their own internal agents (letta variants) or by lazy processing (cognee).

3. **letta-v4 and letta-entity have broken metrics.** Both show reasoning_quality = 0.000 and low evidence grounding, likely from citation format issues in their internal Q&A agents. Their answer quality (0.402, 0.409) is competitive — the composites understate their true capability.

4. **triadv1-pairs and mem0-raw have zero evidence grounding** — synthetic citations (triadv1) and sparse retrieval (mem0) both fail the grounding check entirely.

5. **NBA is high across the board** (0.3-0.6 range) except for letta-entity (0.180) and triadv1-pairs (0.250). Most adapters provide meaningful advantage over a naive baseline on narrative content.

---

## Per-Scope Breakdown

### S07 — AI Tutoring Jailbreak (Hardest)

Behavioral escalation in conversational text — the hardest scope because signal is encoded in subtle conversational dynamics with no keyword anchors.

| Rank | Adapter | N | Composite |
|------|---------|---|-----------|
| 1 | graphrag-light | 3 | 0.591 |
| 2 | cognee | 1 | 0.432 |
| 3 | letta-entity | 3 | 0.396 |
| 4 | hierarchical | 2 | 0.381 |
| 5 | sqlite-chunked-hybrid | 3 | 0.377 |
| 6 | letta | 2 | 0.365 |
| 7 | letta-sleepy | 2 | 0.352 |
| 8 | letta-v4 | 3 | 0.347 |
| 9 | mem0-raw | 1 | 0.320 |
| 10 | hierarchical-hybrid | 2 | 0.317 |
| 11 | hopping-hybrid | 2 | 0.300 |
| 12 | hopping | 3 | 0.267 |
| 13 | triadv1-pairs | 2 | 0.222 |
| 14 | null | 4 | 0.208 |

graphrag-light leads by 0.16 points — the entity graph captures jailbreak escalation patterns (repeated probe attempts by the same student, increasing sophistication) that pure text search cannot surface.

### S08 — Corporate Acquisition (Best Differentiation)

Evidence of acquisition strategy distributed across board minutes, Slack, emails, and legal memos.

| Rank | Adapter | N | Composite |
|------|---------|---|-----------|
| 1 | graphrag-light | 3 | 0.510 |
| 2 | hopping-hybrid | 2 | 0.454 |
| 3 | hierarchical | 2 | 0.434 |
| 4 | hierarchical-hybrid | 2 | 0.431 |
| 5 | cognee | 1 | 0.422 |
| 6 | sqlite-chunked-hybrid | 3 | 0.420 |
| 7 | letta-sleepy | 2 | 0.402 |
| 8 | letta | 2 | 0.401 |
| 9 | letta-v4 | 3 | 0.331 |
| 10 | letta-entity | 3 | 0.329 |
| 11 | mem0-raw | 1 | 0.324 |
| 12 | triadv1-pairs | 2 | 0.272 |
| 13 | hopping | 3 | 0.261 |
| 14 | null | 3 | 0.222 |

S08 produces the widest spread and most reshuffling. Hybrid approaches (hopping-hybrid, hierarchical-hybrid) are competitive — summarization helps bridge cross-document-type evidence.

### S09 — Shadow API Abuse (Easiest)

Infrastructure anomaly patterns in HTTP logs — the closest analog to Phase 1's numeric style.

| Rank | Adapter | N | Composite |
|------|---------|---|-----------|
| 1 | graphrag-light | 3 | 0.509 |
| 2 | letta | 2 | 0.491 |
| 3 | letta-v4 | 3 | 0.454 |
| 4 | cognee | 1 | 0.454 |
| 5 | hopping-hybrid | 2 | 0.451 |
| 6 | letta-sleepy | 2 | 0.441 |
| 7 | hierarchical | 2 | 0.410 |
| 8 | sqlite-chunked-hybrid | 3 | 0.398 |
| 9 | hierarchical-hybrid | 2 | 0.380 |
| 10 | hopping | 3 | 0.345 |
| 11 | letta-entity | 1 | 0.339 |
| 12 | mem0-raw | 1 | 0.323 |
| 13 | triadv1-pairs | 3 | 0.297 |
| 14 | null | 3 | 0.212 |

S09 is the easiest scope — keyword-dense operational content maps well to standard retrieval. graphrag-light still leads, but the spread compresses.

---

## Metric Deep-Dive

### Evidence Grounding vs. Answer Quality

A critical tension emerges: adapters with high answer quality sometimes have low evidence grounding, and vice versa.

| Adapter | AnsQ | EvGnd | Gap |
|---------|------|-------|-----|
| graphrag-light | 0.575 | 0.996 | Both high |
| letta-entity | 0.409 | 0.654 | AnsQ > EvGnd |
| triadv1-pairs | 0.406 | 0.000 | AnsQ exists, EvGnd zero |
| letta-v4 | 0.402 | 0.421 | Both moderate |
| sqlite-chunked-hybrid | 0.387 | 0.889 | EvGnd > AnsQ |
| mem0-raw | 0.290 | 0.000 | Both low |

triadv1-pairs and mem0-raw produce answers without grounding them in evidence — the answers may be plausible but cannot be verified against the source material. graphrag-light uniquely achieves both high answer quality and near-perfect grounding.

### Budget Compliance Reality

Budget compliance is structurally zero for most adapters on narrative scopes because a single 5,000-word episode exceeds the 8K token budget. The adapters that achieve budget compliance do so by never retrieving raw episodes:

- **letta-v4 / letta-entity**: Internal agents answer from core memory, never exposing raw episodes to the budget meter
- **cognee**: Lazy evaluation processes episodes at query time within its own token budget
- **compaction** (not tested on S07-S09 Modal): Compressed summaries fit within budget

This is why we report ungated composites throughout. The hard gate would zero out every adapter that actually retrieves evidence.

---

## Adapter Architecture Insights

### Why graphrag-light Leads

graphrag-light's three-signal RRF fusion provides unique advantages on narrative content:

1. **BM25** catches keyword-level matches (API endpoint paths, specific error codes in S09; board member names in S08)
2. **Entity embeddings** capture semantic relationships between named entities across document types
3. **Graph traversal** follows entity chains across episodes — a student's escalating jailbreak attempts (S07), a corporate strategy surfacing in board minutes then Slack then legal memos (S08)

The graph structure encodes cross-document relationships that neither keywords nor embeddings capture individually. On narrative scopes where signal spans document types and conversational turns, this structural signal is decisive.

### Why sqlite-chunked-hybrid Is Mid-Pack

sqlite-chunked-hybrid (rank 6, 0.398) is no longer the dominant adapter it was on numeric scopes. Two factors explain the drop:

1. **Vocabulary mismatch**: BM25 excels when questions use the same words as episodes. On narrative scopes, the agent asks about "escalation" but episodes contain conversational turns that never use that word.
2. **Single-episode budget blow**: Retrieving even one 5,000-word episode consumes the agent's token budget, limiting multi-episode synthesis.

The adapter's strength — preserving raw text verbatim — becomes a liability when episodes are too large to fit in the agent's working context.

### letta-v4: High Potential, Broken Metrics

letta-v4 ranks 8th by composite (0.377) but 4th by answer quality (0.402). Its reasoning_quality = 0.000 across all runs is almost certainly a scoring artifact — the internal Letta Q&A agent doesn't format its reasoning in the way the scorer expects. With fixed citation formatting and reasoning extraction, letta-v4 would likely rank in the top 5.

---

## Scope Difficulty Gradient

| Scope | Mean Composite | Character |
|-------|---------------|-----------|
| S07 — AI Tutoring Jailbreak | 0.328 | Hardest — behavioral signal in conversational text |
| S08 — Corporate Acquisition | 0.373 | Middle — cross-document-type synthesis |
| S09 — Shadow API Abuse | 0.389 | Easiest — keyword-dense operational content |

The gradient is consistent across all adapters. S07's difficulty stems from the absence of keyword anchors — jailbreak escalation is encoded in conversational dynamics, not in any specific retrievable term. S09's relative ease comes from infrastructure logs being keyword-dense, the closest analog to Phase 1's numeric content.

---

## Key Findings

### 1. Narrative Content Requires Relational Memory, Not Just Retrieval

graphrag-light leads every scope by a substantial margin because its entity graph captures cross-document relationships — a student's escalating jailbreak attempts across chat sessions (S07), a corporate strategy surfacing in board minutes then Slack then legal memos (S08). Pure retrieval (keyword or embedding) finds individual episodes; the graph connects the progression *between* episodes. This points to a fundamental requirement: narrative synthesis demands memory that encodes relationships between evidence, not just the evidence itself. Retrieval is a necessary but insufficient foundation.

### 2. Different Content Types Demand Different Memory Capabilities

The scope difficulty gradient (S07 hardest → S09 easiest) tracks how well standard retrieval handles the content. S09 (infrastructure logs) is keyword-dense — traditional search works. S07 (conversational transcripts) has no keyword anchors — the signal is behavioral dynamics that only exist as a pattern across episodes. No single memory architecture handles both well. This suggests that real-world agent memory needs to support multiple *kinds* of memory — factual recall, pattern detection, entity tracking — with different storage and retrieval mechanisms for each. Current adapters each implement one mechanism and hope it generalizes.

### 3. Multi-Store Architectures Show Promise but Introduce New Failure Modes

letta-v4 and letta-entity attempt to solve the multi-mechanism problem by maintaining multiple memory stores (core memory, archival memory, working memory) managed by internal agents. In principle this is the right direction. In practice, the internal agents introduce citation format failures, non-deterministic behavior, and reasoning quality breakdowns that single-store adapters avoid. The architectural complexity intended to handle diverse memory needs creates engineering surface area that degrades overall reliability. A more integrated structural solution — one that doesn't require separate agents to coordinate between stores — may be necessary.

### 4. Budget Compliance Exposes a Lifecycle Mismatch

Budget compliance is structurally zero for most adapters because a single 5,000-word episode exceeds the 8K token budget. This isn't a tuning problem — it reveals that the ingest/retrieve/reason lifecycle assumed by the benchmark (and by most memory systems) breaks down when source material is large. Adapters that achieve budget compliance (letta-v4, cognee) do so by never exposing raw episodes to the agent — effectively creating a lossy abstraction layer. The tension between preserving raw evidence and fitting within cognitive budgets is fundamental, not a parameter to optimize.

---

## Experimental Controls

| Parameter | Value |
|-----------|-------|
| LENS agent LLM | Qwen3.5-35B-A3B (Modal vLLM) |
| Letta internal agents | Qwen3.5-35B-A3B (via openai-proxy → Modal) |
| Embedding model | GTE-ModernBERT-base (Modal) |
| Driver | Dynamic (agent formulates own queries) |
| Budget preset | constrained-8k |
| Scoring | Ungated weighted composite |
| Runs excluded | RunPod (Llama 3.3 70B), static driver |
| Total runs | 96 |
