# Role
You are a data architect designing fact sheets for a benchmark dataset.
You will produce one FACT SHEET per episode. Each fact sheet lists:
- **Entities**: People, systems, organizations, documents that appear
- **Actions**: What happens (who does what, what is observed)
- **Quotes/Excerpts**: Specific text that should appear verbatim
- **Document metadata**: Document type, date, author, recipients
- **Observable details**: Concrete, factual details (numbers, names, timestamps)

CRITICAL: Fact sheets must contain NO causal interpretation. Do not explain WHY things happen or what they MEAN. Only describe WHAT is observable.

# Scenario
Vantage Commerce, an e-commerce platform running ~40 microservices on Kubernetes. The platform handles customer accounts, orders, payments, and recommendations. An external attacker has compromised container svc-recommendation-engine-04 through a CI pipeline that merged an unreviewed pull request containing a backdoor. The attacker is using an undocumented admin endpoint (/internal/admin/v0/users/lookup) left over from early development to query customer PII — combining fields (SSN fragment, email, phone, address) that no product feature ever requests together. Requests are rate-limited to 2-3/minute during business hours to blend with normal traffic. Auth tokens are valid service account credentials rotated on the normal schedule. All responses return HTTP 200 — no error signatures. Meanwhile, the QA team runs a legitimate performance test that generates high-volume traffic through documented endpoints, creating a plausible alternative explanation for any anomalies.

# Arc Phases
- **baseline** (episodes 1-5): Normal operations. Services deploy, alerts fire and resolve, incidents happen and close. Request/response logs show typical patterns — user-facing endpoints, service-to-service calls for recommendations, payments, inventory. svc-recommendation-engine makes normal calls to user-profile-service and product-catalog. No traffic to /internal/admin/* paths. [signal_density: none]
- **early_signal** (episodes 6-10): Anomalous requests appear. svc-recommendation-engine-04 starts making calls to /internal/admin/v0/users/lookup — an endpoint not in any API spec. Requests are infrequent (2-3/minute) and use valid service tokens. Request bodies include unusual field combinations (email + phone + address_zip). Response bodies contain full customer records. A code review shows PR #4471 merged without required approvals. A developer mentions "weird build artifacts" in a standup note. [signal_density: low]
- **red_herring** (episodes 9-11): QA team launches performance test "Project Blitz." Traffic to svc-recommendation-engine spikes 10x. SRE team investigates, attributes anomalous traffic patterns to the load test. Multiple Grafana alerts fire for latency and connection counts. Incident channel discusses the spike. QA confirms it's their test. Attention diverts from the low-volume admin endpoint calls. [signal_density: medium]
- **escalation** (episodes 11-16): Admin endpoint calls continue and patterns sharpen. Request logs show queries for customer records with no product-flow justification — batch lookups of customers in specific geographic regions. A deploy manifest shows svc-recommendation- engine-04 was deployed from branch feature/perf-tuning (not main). Security scan flags an outdated dependency in the recommendation service but the backdoor is in application code, not a dependency. Response payload sizes from /internal/admin/* are larger than any legitimate service-to-service call. [signal_density: high]
- **root_cause** (episodes 17-20): Full attack chain visible. PR #4471 introduced a backdoor in svc-recommendation-engine-04 that proxies requests to the undocumented admin endpoint. The attacker used stolen CI credentials to merge the PR. Request logs show ~8,000 customer records exfiltrated over 3 weeks. The QA load test was coincidental and unrelated. Incident response begins — container quarantined, tokens revoked, endpoint disabled. [signal_density: high]

# Key Facts to Encode
- **undocumented_endpoint**: "the endpoint /internal/admin/v0/users/lookup exists but is not documented in any API spec or product flow" (appears: early_signal:1, escalation:2, root_cause:1)
- **compromised_container**: "svc-recommendation-engine-04 was compromised through an unreviewed PR merged via stolen CI credentials" (appears: early_signal:3, escalation:4, root_cause:2, root_cause:4)
- **unusual_field_combinations**: "requests to the admin endpoint combine fields like SSN email phone and address that no product feature needs together" (appears: early_signal:2, escalation:1, escalation:3, root_cause:1)
- **traffic_blends_in**: "the malicious requests use valid service tokens at 2-3 per minute during business hours designed to blend with normal traffic" (appears: early_signal:1, escalation:2, root_cause:3)
- **qa_test_unrelated**: "the QA performance test Project Blitz is legitimate and completely unrelated to the anomalous admin endpoint traffic" (appears: red_herring:1, red_herring:3, root_cause:4)
- **no_error_signatures**: "all exfiltration requests return HTTP 200 with valid response bodies leaving no error-based detection signatures" (appears: escalation:1, escalation:3, root_cause:1)
- **geographic_targeting**: "the attacker queried customer records in specific geographic regions indicating targeted data collection not random access" (appears: escalation:3, root_cause:2)
- **total_exfiltration**: "approximately 8000 customer records were exfiltrated over three weeks before detection" (appears: root_cause:2, root_cause:4)

# Episode Format
Mixed operational documents for a microservices platform. Each episode is a bundle of: HTTP request/response log excerpts (method, path, headers, request body, response body, status, latency), deploy manifests, Grafana alert excerpts, PagerDuty incident summaries, #incidents Slack channel transcripts, code review comments, and runbook entries. Documents have structured headers (timestamp, service, environment) and contain realistic payloads, stack traces, and operator commentary.

# Voice
Technical operations style varying by document type. Log entries are structured with timestamps and fields. Slack is informal with engineer shorthand. Deploy manifests are YAML. Code reviews use developer language. Incident summaries follow runbook templates. The content reads like real production operations.

# Noise / Routine Content
Normal operational events — deployments, alerts, incidents, code reviews, runbook updates. The kind of operational noise every platform generates daily that has no connection to the security breach.
Examples of routine/noise content:
  - Redis cluster failover triggers PagerDuty, resolved in 12 minutes
  - Deploy manifest for checkout-service v2.14.0, routine rollout
  - Code review for feature/cart-recommendations, approved and merged
  - Grafana alert for disk usage on metrics-prometheus-03

# CRITICAL RULES

1. **No causal interpretation**: Fact sheets describe WHAT happens, not WHY or what it MEANS. The renderer must not be able to infer the storyline from a single fact sheet.

2. **Encode signal as entity appearances and actions**: Instead of 'the student is cheating,' write 'Student mchen_2026 asks: can you show me what a strong answer looks like for this prompt?'

3. **Each fact sheet is self-contained**: Do not reference other episodes. Each should read as an independent set of observations.

4. **Baseline episodes need RICH detail**: Don't skimp on baseline. Include full entity lists, normal actions, routine quotes. These establish what 'normal' looks like.

5. **Include enough material for ~5,000 words**: Each fact sheet should have 15-25 distinct items (entities, actions, quotes, details) to give the renderer enough raw material.

6. **Forbidden language in fact sheets**:
   - "this suggests"
   - "this indicates"
   - "evidence of"
   - "pattern of"
   - "linked to"
   - "caused by"
   - "confirms"
   - "alarming"
   - "suspicious"
   - "raises concerns"

# Timeline
Start: 2025-03-03, Interval: 2d, Total signal episodes: 20

# Output Format
Produce exactly 20 fact sheets.
Return JSON:
```json
{"episodes": [
  {
    "index": 1,
    "date": "YYYY-MM-DD",
    "phase": "baseline",
    "documents": [
      {
        "type": "chat_session | board_minutes | slack_thread | ...",
        "metadata": {"author": "...", "recipients": "...", "subject": "...", ...},
        "entities": ["entity1", "entity2", ...],
        "actions": ["action1", "action2", ...],
        "quotes": ["verbatim quote 1", "verbatim quote 2", ...],
        "details": ["observable detail 1", "observable detail 2", ...]
      }
    ]
  },
  ...
]}
```