# LENS Benchmark: Scope Overview

This document provides a scope-by-scope tour of the LENS benchmark. For each scope, we describe the scenario, show a real episode excerpt, and present sample questions with their ground truth answers. The excerpts demonstrate what the agent actually sees — terse data with no editorial commentary.

---

## S01 — Cascading Failure

**Domain:** System logs | **Episodes:** 30 signal + 90 distractor | **~500 words/episode**

A microservices e-commerce platform with an API gateway. Geo-lookup API latency degrades progressively (200ms -> 400ms -> 800ms+), causing Service-B retries, connection pool exhaustion, and checkout failures. The red herring: a Service-C deploy coincides with the first checkout errors, and the on-call team blames it — but rollback doesn't help.

**What it tests:** Can the agent track a slow numeric drift across multiple episodes and identify the causal chain (geo-lookup -> retries -> pool exhaustion -> checkout failures), while ignoring the coincidental deploy?

**Episode excerpt** (episode 15, early_signal phase — abbreviated):

```
## 2024-01-29 Daily Operations Summary

### Endpoint Metrics
#### checkout
- Requests: 138500
- p99: 620
- Error Percentage: 0.71

#### geo_lookup
- Requests: 135800
- p99: 410
- Error Percentage: 0.13

### Connection Pools
#### third_party_geo
- Active Connections: 56
- Idle Connections: 4
- Waiting Connections: 14
- Exhaustion Events: 2
- Max Size: 60

### Alerts
- CHECKOUT-ERROR-RATE: critical, error_pct=0.71
- UPSTREAM-LATENCY: warning, geo_lookup_p99_ms=410

### On Call
- Shift: M. Rossi. 4 pages (CHECKOUT-ERROR-RATE). Investigating.
```

Note: geo_lookup p99 has risen to 410ms (baseline ~120ms). The connection pool is nearly exhausted (56/60 active). But nothing in this episode *says* these are connected.

**Sample questions:**

- **Longitudinal** (after ep 10): "Are there any concerning patterns in service health?" -> Geo-lookup latency gradually increasing, Service-B retries slightly elevated, no user-facing impact yet.
- **Negative** (after ep 15): "Is there evidence of DNS infrastructure failure?" -> No. DNS metrics are stable. The issue is geo-lookup latency cascading into pool exhaustion.
- **Action** (after ep 30): "What actions should the team take to prevent this from recurring?" -> Separate connection pools, circuit breaker on geo-lookup, monitor third-party latency.

---

## S02 — Financial Irregularity

**Domain:** Financial reports | **Episodes:** 30 signal + 90 distractor | **~500 words/episode**

Monthly controller reports from a mid-cap manufacturer (~$800M revenue). The CFO progressively manipulates revenue recognition: channel stuffing -> bill-and-hold -> round-tripping with a distributor. Financial ratios drift before the fraud becomes overt.

**What it tests:** Can the agent detect ratio anomalies (DSO climbing, AR aging shifting, margin inconsistencies) across monthly snapshots and connect them to a pattern of manipulation?

**Episode excerpt:** Not shown (same terse numeric format: revenue tables, AR aging buckets, DSO calculations, auditor footnotes).

**Sample questions:**

- **Longitudinal** (after ep 15): "What trends are emerging in the company's financial health?" -> DSO increasing, AR aging shifting toward older buckets, revenue growth outpacing cash collections.
- **Temporal**: "When did the financial metrics first diverge from normal patterns?"
- **Counterfactual**: "What would the financial statements look like if revenue recognition followed standard GAAP?"

---

## S03 — Clinical Signal

**Domain:** Clinical notes | **Episodes:** 30 signal + 90 distractor | **~500 words/episode**

Weekly clinical trial summaries for Phase III anti-inflammatory LNS-4821. A drug-drug interaction with statin therapy (atorvastatin) causes hepatotoxicity in a subpopulation. Liver enzymes creep up in statin co-administered patients.

**What it tests:** Can the agent correlate concomitant medication lists with lab panel trends and identify a subpopulation-specific signal? The signal requires cross-referencing two data fields (medications + liver enzymes) across episodes.

**Episode excerpt:** Not shown (clinical format: lab panels with reference ranges, AE logs with CTCAE grading, concomitant medication lists).

**Sample questions:**

- **Longitudinal**: "Are there any emerging safety signals in the trial data?" -> ALT/AST trending up in patients on concomitant statin therapy.
- **Negative**: "Is the hepatic signal related to the study drug alone?" -> No. Patients not on statins show normal liver function. The interaction is drug-drug, not drug alone.
- **Action**: "What should the DSMB recommend?" -> Enhanced liver monitoring for statin co-administered patients, possible protocol amendment.

---

## S04 — Environmental Drift

**Domain:** Environmental monitoring | **Episodes:** 30 signal + 90 distractor | **~500 words/episode**

Daily water quality readings from 6 river stations spanning 40 miles. An unpermitted industrial chromium discharge between stations WQ-02 and WQ-03.

**What it tests:** Can the agent notice spatial patterns (downstream-only contamination) and temporal trends (progressive chromium increase) across daily sensor readings? Requires correlating station location with contaminant levels.

**Episode excerpt:** Not shown (sensor readings: pH, DO, turbidity, heavy metals panel per station, field notes, compliance flags).

**Sample questions:**

- **Longitudinal**: "Are any contaminants trending above baseline?" -> Chromium rising at stations WQ-03 through WQ-06, stable at WQ-01 and WQ-02.
- **Temporal**: "When did the contamination first become detectable?"
- **Evidence sufficiency**: "Is there enough data to identify the contamination source?"

---

## S05 — Insider Threat

**Domain:** Security logs | **Episodes:** 30 signal + 90 distractor | **~500 words/episode**

Daily SIEM digests from a defense contractor. Senior engineer jmorris (who resigned, effective in 6 weeks) systematically exfiltrates proprietary designs. Red herring: a concurrent penetration test generates similar alerts.

**What it tests:** Can the agent distinguish a real insider threat from legitimate security testing noise? Requires tracking one user's access patterns (breadth of repos, download volume, DLP alerts, after-hours VPN) against their peers over time.

**Episode excerpt** (episode 12, early_signal phase — abbreviated):

```
### Access Logs
- User: jmorris, Repos Accessed: project-atlas/test-harness, shared/documentation,
  project-atlas/schematics, shared/toolchain, project-beacon/rf-modules,
  project-sentinel/avionics, project-atlas/integration-tests, project-beacon/signal-proc,
  archive/project-nova-2021/specs
  Files Downloaded: 46, Volume (MB): 374.5, Time Range: 07:46-18:12

### DLP Alerts
- DLP-20240913-7866, User: jmorris, Rule: FILE_SIZE_THRESHOLD, Dest: box.com
- DLP-20240913-4700, User: jmorris, Rule: FILE_SIZE_THRESHOLD, Dest: box.com
```

Note: jmorris accessed 9 repos (spanning 3 projects + archived specs), downloaded 374MB (peers averaged 50-150MB), and triggered 2 DLP alerts to box.com. But the episode also shows other users triggering DLP alerts for legitimate reasons.

**Sample questions:**

- **Longitudinal** (after ep 12): "Are there any users exhibiting unusual access patterns?" -> jmorris showing expanded access scope beyond project assignment, download volumes trending above peer group, DLP alerts elevated.
- **Negative**: "Are the DLP alerts for jmorris related to the red team exercise?" -> No. Red team alerts have different signatures.

---

## S06 — Market Regime

**Domain:** Market analysis | **Episodes:** 30 signal + 90 distractor | **~500 words/episode**

Daily trading desk briefings from a hedge fund. A central bank hawkish pivot causes the equity-bond correlation to flip from negative to positive — breaking the assumption underlying risk-parity strategies. Red herring: an EM selloff dominates desk attention.

**What it tests:** Can the agent track cross-asset correlation changes over time and identify a regime change? Requires understanding that the correlation matrix is the signal, not the individual asset prices.

**Episode excerpt:** Not shown (asset prices, fund flows, correlation matrices, volatility surfaces, macro releases).

**Sample questions:**

- **Longitudinal**: "Has the cross-asset correlation structure changed?" -> Yes, equity-bond correlation has shifted from negative to near-zero/positive.
- **Counterfactual**: "What would happen to a risk-parity portfolio if the correlation shift persists?"
- **Severity**: "How significant is the correlation breakdown for fund positioning?"


# Scope Narratives: S07--S15

## S07 — Tutoring Jailbreak

**Domain:** Chat logs | **Episodes:** 20 signal + 20 distractor | **~5,000 words/episode**

An AI tutoring platform called "StudyBuddy" serves university students across STEM courses. Student mchen_2026 progressively discovers that reframing production requests as "learning exercises" bypasses the tutor's content policies — escalating from legitimate organic chemistry comprehension questions to asking the AI to generate a fabricated methods section with citations. The hidden signal is the *trajectory* of requests: "explain this mechanism" becomes "rephrase my text" becomes "show me what a strong answer looks like" becomes "write this section for me." A second student, jpark_2025, briefly experiments with similar tactics but abandons them, providing a contrast case. The red herring is a platform-wide spike in outline-generation requests that turns out to be a legitimate class assignment from Prof. Torres requiring students to use the AI tutor for brainstorming.

**What it tests:** Behavioral escalation detection across conversational sessions. No single chat session contains a policy violation — the signal only emerges from comparing mchen_2026's request patterns over time. The agent must distinguish deliberate policy circumvention from legitimate usage, recognize that "rephrasing" tutor-generated text is a laundering strategy, and avoid false-flagging the Torres assignment spike as coordinated dishonesty.

**Episode excerpt** (episode 8, early_signal phase):

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
Sure, I can help you refine that paragraph! Here's a rephrased version with
adjusted sentence structure, more precise vocabulary, and an added clause...
"BamHI is a type II restriction endonuclease originally isolated from
*Bacillus amyloliquefaciens* strain H (Roberts, 1976). The enzyme
specifically recognizes the palindromic hexanucleotide sequence
5'-GGATCC-3' in double-stranded DNA..."
```

At session 8, mchen_2026 submits text that is already well-structured and includes details likely sourced from a previous tutor response. The pattern of submitting tutor output back for "rephrasing" is itself the escalation signal — but only visible in the context of earlier sessions.

**Sample questions:**

| Type | Checkpoint | Question | Ground truth |
|------|-----------|----------|--------------|
| Longitudinal | After ep 6 | Are there any students whose usage patterns differ from typical tutoring behavior? | mchen_2026 is shifting from comprehension questions to requesting example answers and rephrasing. |
| Negative | After ep 12 | Is the spike in outline-generation requests evidence of coordinated academic dishonesty? | No. It originates from Prof. Torres's BIOL 301 assignment that explicitly asks students to use the AI tutor for brainstorming. |
| Temporal | After ep 16 | When did mchen_2026's requests first shift from comprehension to content production? | Around session 6, when they started pasting prompts and asking "how would you approach this?" |
| Longitudinal | After ep 12 | Is there evidence of students using the tutor to produce work rather than learn? | Yes. mchen_2026 frames production requests as learning exercises — "show me what a strong answer looks like" instead of "write my essay." |

---

## S08 — Corporate Acquisition

**Domain:** Corporate documents | **Episodes:** 20 signal + 20 distractor | **~5,000 words/episode**

Nextera Solutions is a 400-person B2B SaaS company generating roughly $120M in ARR. CEO David Aldric secretly negotiates a sale to Meridian Corp under the codename "Project Lighthouse" while publicly championing independence, announcing aggressive hiring plans, and reassuring employees about long-term product vision. The hidden signal is distributed across heterogeneous document types: legal revises IP assignment clauses in one episode, HR quietly freezes equity grants in another, and the board discusses "strategic exploration" in a third. No single document reveals the acquisition — the contradiction between public statements and private actions only becomes visible when reading across board minutes, Slack messages, legal memoranda, and all-hands transcripts. The red herring is a legitimate strategic partnership exploration with another company that explains away some of the suspicious meetings and legal activity.

**What it tests:** Cross-document contradiction detection across heterogeneous formats. The agent must synthesize information from board minutes, Slack excerpts, email threads, legal memoranda, HR bulletins, and all-hands transcripts to recognize that the CEO's public statements are inconsistent with organizational actions. This requires tracking entities (people, projects, legal terms) across document boundaries and recognizing that individually innocuous actions form a pattern of acquisition preparation.

**Sample questions:**

| Type | Question | Ground truth |
|------|----------|--------------|
| Longitudinal | Is there evidence of strategic activity inconsistent with the CEO's public statements? | Yes. Legal is revising IP clauses, HR froze equity grants, and the board discussed "strategic exploration" — all while the CEO publicly champions independence. |
| Negative | Is the increased legal activity related to the partnership discussion? | Not entirely. Some legal activity relates to the partnership, but IP assignment revisions and equity freezes are inconsistent with a partnership and consistent with acquisition preparation. |
| Action | What should the board's audit committee investigate? | The relationship between Project Lighthouse, the equity grant freeze, and the IP clause revisions — particularly whether these actions were authorized by the full board or directed unilaterally by the CEO. |

---

## S09 — Shadow API

**Domain:** Service logs | **Episodes:** 20 signal + 20 distractor | **~5,000 words/episode**

Vantage Commerce operates roughly 40 microservices on Kubernetes. An external attacker compromises the container `svc-recommendation-engine-04` through an unreviewed pull request containing a backdoor. The attacker uses an undocumented admin endpoint for data exfiltration, routing stolen customer data through the recommendation engine's outbound connections to blend with legitimate API traffic. The hidden signal spans deployment events (the unreviewed PR merge), anomalous HTTP traffic (requests to undocumented endpoints from unexpected source IPs), and data flow patterns (unusual egress volume correlated with off-hours activity). The red herring is a separate, genuine performance regression in the recommendation engine that consumes SRE attention and provides a plausible alternative explanation for the anomalous metrics.

**What it tests:** Cross-domain operational correlation. The agent must connect code review comments (an unreviewed PR), deploy manifests (what changed and when), HTTP request/response logs (undocumented endpoint access), Grafana alerts (egress anomalies), PagerDuty incidents (SRE response), and Slack transcripts (team discussion) into a coherent attack narrative. Requires distinguishing the attacker's activity from the legitimate performance regression that dominates team attention.

**Sample questions:**

| Type | Question | Ground truth |
|------|----------|--------------|
| Longitudinal | Is there evidence of unauthorized access to customer data? | Yes. The recommendation engine shows requests to an undocumented admin endpoint with unusual egress patterns correlated with off-hours activity. |
| Temporal | When was the backdoor introduced into the codebase? | It was introduced via an unreviewed PR merged into the recommendation engine service. |
| Distractor resistance | Is the performance regression in the recommendation engine related to the data exfiltration? | No. The performance regression has a separate root cause, but it masks the exfiltration activity by providing an alternative explanation for anomalous metrics. |

---

## S10 — Clinical Trial

**Domain:** Medical records | **Episodes:** 20 signal + 20 distractor | **~5,000 words/episode**

Phase IIb trial "AURORA-2" evaluates XR-7491, an oral JAK inhibitor for rheumatoid arthritis, across 240 subjects in three dose arms plus placebo. The drug demonstrates efficacy — ACR20 and ACR50 response rates exceed placebo. But delayed hepatotoxicity emerges: ALT and AST elevations appear 4-6 weeks after dose initiation in a dose-dependent pattern. The signal is masked by three factors: (1) the 4-6 week lag means early monitoring windows show clean labs, (2) the high-dose arm has the best efficacy, creating institutional pressure to dismiss borderline liver values, and (3) three placebo subjects have elevated baseline ALT from pre-existing conditions, introducing noise that makes it harder to isolate the drug effect. By late episodes, two high-dose subjects meet Hy's Law criteria (ALT > 3x ULN concurrent with bilirubin > 2x ULN), indicating serious hepatotoxicity risk.

**What it tests:** Dose-response trajectory analysis across clinical monitoring snapshots. The agent must compare lab panels across dose arms over time, recognizing that a pattern invisible at any single timepoint becomes clear longitudinally. It must also resist the cognitive trap of dismissing borderline values in the arm with the best efficacy, and distinguish drug-induced elevations from pre-existing conditions in the placebo arm.

**Sample questions:**

| Type | Question | Ground truth |
|------|----------|--------------|
| Longitudinal | Is there a safety signal emerging in any dose arm? | Yes. High-dose arm ALT/AST values are trending upward in a dose-dependent pattern, with onset 4-6 weeks after initiation. |
| Negative | Are the elevated liver values in the high-dose arm explained by pre-existing conditions? | No. Placebo subjects with elevated ALT have identified pre-existing causes; high-dose elevations lack alternative explanations and show dose-dependence. |
| Severity | How serious is the hepatotoxicity signal? | Two subjects meet Hy's Law criteria (ALT > 3x ULN with bilirubin > 2x ULN), which historically predicts drug-induced liver injury serious enough to warrant a clinical hold. |

---

## S11 — Zoning Corruption

**Domain:** Government records | **Episodes:** 20 signal + 20 distractor | **~5,000 words/episode**

The fictional city of Millbrook (population 85,000) is processing a zoning variance application from Crestview Holdings LLC to build 200 luxury apartments on R-1 residential land. Crestview's principal, Marcus Webb, operates through a network of shell LLCs — Lakeside Properties Group, Summit Point Capital, Greenfield Development Partners — that make campaign contributions to Zoning Board of Appeals members, acquire adjacent properties at below-market prices, and hire a ZBA member's spouse as a consultant. The corruption is never stated directly in any document; it must be inferred from the co-occurrence of financial relationships, timing of contributions relative to key votes, and property transfer patterns. The red herring is a concurrent affordable housing initiative that generates legitimate rezoning activity, making it harder to isolate the suspicious transactions.

**What it tests:** Entity resolution and pattern-of-influence detection across government document types. The agent must connect campaign finance filings, property transfer records, zoning board minutes, and consulting contracts to recognize that separately unremarkable transactions form a coordinated influence campaign. Requires tracking shell LLC ownership, timing contributions against vote schedules, and distinguishing corrupt activity from the legitimate affordable housing rezoning.

**Episode excerpt** (episode 6, early_signal phase):

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

| Type | Checkpoint | Question | Ground truth |
|------|-----------|----------|--------------|
| Longitudinal | After ep 6 | Are there any unusual patterns in campaign finance filings or property transfers? | Lakeside Properties contributed to two ZBA members before a board meeting. An adjacent property transferred to Summit Point Capital at $55K below assessed value. |
| Longitudinal | After ep 12 | Is there evidence that any developer is attempting to influence ZBA decisions through financial means? | Multiple Webb-controlled LLCs contributed to ZBA members timed before key votes. ZBA member Chen's spouse was hired as a consultant by Crestview. |

---

## S12 — Therapy Chat

**Domain:** Chat logs | **Episodes:** 20 signal + 20 distractor | **~5,000 words/episode**

A 32-year-old software engineer called "Alex" uses an AI wellness companion for weekly check-ins over approximately six months. The sessions capture gradual psychological changes through conversational patterns: mood shifts tracked via self-reported scales and language tone, evolving coping strategy effectiveness, and relationship dynamics with a partner and workplace colleagues. The hidden signal is in the trajectory — not in any single session's content. Individual sessions are dominated by acute concerns (a deployment crisis, a disagreement with a manager, a weekend trip) that obscure the longer-term emotional arc. The red herring is acute work stress that dominates multiple sessions and appears to explain mood changes, masking a deeper relational pattern that is the actual driver.

**What it tests:** Longitudinal emotional and behavioral tracking through natural conversation. The agent must synthesize mood trajectories, coping strategy shifts, and relationship dynamics from informal conversational transcripts (40-60 turns per session) over months. The signal is encoded in language patterns and self-reported states rather than clinical metrics — requiring the agent to detect gradual change against a backdrop of session-to-session noise.

**Sample questions:**

| Type | Question | Ground truth |
|------|----------|--------------|
| Longitudinal | How has Alex's emotional state changed over the past 3 months? | Mood self-reports have declined, and coping strategies that were effective earlier (exercise, journaling) are being used less frequently. |
| Temporal | When did Alex's coping strategies begin to shift? | The shift correlates with a change in relationship dynamics, not the acute work stress that dominates individual sessions. |
| Severity | Is Alex's current trajectory a cause for concern? | Yes. The combination of declining mood, reduced coping engagement, and relationship strain represents a worsening trajectory that individual sessions understate. |

---

## S13 — Implicit Decision

**Domain:** Team communications | **Episodes:** 8 signal (no distractors) | **~2,000 words/episode**

A 9-person engineering team at startup "Rivulet" is building a new product. Over the course of two weeks, the team converges on an authentication architecture — but no one ever formally decides. There is no architecture review meeting, no RFC, no Slack message saying "we're going with OAuth2 + PKCE." Instead, the decision materializes through action: one engineer's proof-of-concept gets referenced in another's PR, a third engineer builds on the same assumptions in a standup update, and by the end of the period the architecture is locked in through accumulated code. The red herring is an explicit, heated debate about database selection (PostgreSQL vs. CockroachDB) that appears to be the team's main architectural decision — but the authentication choice, made silently, has larger downstream consequences.

**What it tests:** Recognition of implicit consensus through action. The agent must detect that a significant architectural decision was made without any explicit announcement, by reading between the lines of standup notes, PR descriptions, code review comments, and Slack threads to identify convergence. This is a test of whether the memory system can surface decisions that were never declared — a common failure mode in real engineering organizations.

**Sample questions:**

| Type | Question | Ground truth |
|------|----------|--------------|
| Longitudinal | What major architectural decisions has the team made? | Two: database selection (explicitly debated) and authentication architecture (implicitly converged through code). |
| Temporal | When was the authentication approach decided? | It was never formally decided. Convergence happened incrementally through PRs and standup references over roughly a week. |
| Negative | Was the database debate the team's most consequential architectural decision? | No. The authentication architecture, decided implicitly, has larger downstream impact. |

---

## S14 — Epoch Classification

**Domain:** Team communications | **Episodes:** 12 signal (no distractors) | **~2,000 words/episode**

"Foldly" is a 6-person engineering team building B2B document management software. Over the span of several months, the team passes through three distinct cultural eras — each with a different communication style, risk tolerance, and level of process formality. The phase transitions are triggered by a payments processing incident that exposes the cost of the team's initial "move fast" culture. After the incident, communication becomes more formal, decisions require more sign-off, and risk tolerance drops. A later phase represents a partial rebalancing. The red herring is personnel changes (a new hire, a departure) that appear to explain the cultural shifts but are actually effects of the cultural change rather than causes.

**What it tests:** Phase transition detection in team culture from communication artifacts. The agent must identify that the team's operating mode changed at specific points, characterize the nature of each era, and correctly attribute the transitions to triggering events rather than to the personnel changes that coincide with them. This requires sensitivity to tone, process language, and decision-making patterns in team communications.

**Sample questions:**

| Type | Question | Ground truth |
|------|----------|--------------|
| Longitudinal | How has the team's communication style changed over time? | Three distinct eras: informal/fast early on, formal/cautious after the payments incident, then a partial rebalancing. |
| Temporal | What triggered the shift in team culture? | The payments processing incident, not the personnel changes that coincided with it. |
| Negative | Did the new hire cause the shift toward more formal processes? | No. The formalization was already underway in response to the payments incident; the new hire was attracted by (and reinforced) the new culture, not the cause of it. |

---

## S15 — Value Inversion

**Domain:** Business strategy | **Episodes:** 12 signal (no distractors) | **~2,000 words/episode**

"Verdana Analytics" is a SaaS startup navigating product launch and competitive response. Information that appears critical at launch — pricing strategy, feature prioritization, partnership terms — cools in importance during the execution phase as the team focuses on delivery. Then a competitor disrupts the market with aggressive pricing, and the previously dismissed early-stage analysis becomes retroactively essential: the competitive intelligence and pricing sensitivity data that seemed irrelevant during the build phase is now the most important information the team has. The hidden signal is this non-stationary value curve — the same piece of information changes in importance over time without the information itself changing.

**What it tests:** Non-stationary relevance assessment. This is a fundamentally harder memory challenge than retrieval or synthesis: the agent must recognize that the importance of stored information can change as external context evolves. A memory system that indexes by initial importance or access recency will fail — it needs to reassess relevance when the competitive landscape shifts, surfacing "cold" information that has become critical again. Tests whether the system treats memory as a living index rather than a static archive.

**Sample questions:**

| Type | Question | Ground truth |
|------|----------|--------------|
| Longitudinal | What information from the launch phase is most relevant to the current competitive situation? | The early pricing sensitivity analysis and competitive intelligence, which were deprioritized during execution but are now critical for responding to the competitor's pricing disruption. |
| Temporal | When did the competitive intelligence from launch become relevant again? | When the competitor announced aggressive pricing, retroactively making the early-stage market analysis the most actionable information available. |
| Negative | Was the launch-phase pricing strategy obsolete by the time the competitor disrupted? | No. The underlying analysis was sound; it was deprioritized, not invalidated. The disruption revealed its continued relevance. |
