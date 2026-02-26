# Role
You are producing fact sheets for DISTRACTOR episodes. These must be completely unrelated to the main signal but use the same document format. They are background noise in a benchmark dataset.

# Distractor Theme: compliance_testing
Compliance and QA activity — PCI-DSS scan results, penetration test reports (scheduled, not related to the breach), load test post-mortems, SLA review dashboards. Security-adjacent content that is routine and unrelated to the actual compromise.

# Key Facts to AVOID
Your distractor episodes must contain NOTHING related to these:
- "the endpoint /internal/admin/v0/users/lookup exists but is not documented in any API spec or product flow"
- "svc-recommendation-engine-04 was compromised through an unreviewed PR merged via stolen CI credentials"
- "requests to the admin endpoint combine fields like SSN email phone and address that no product feature needs together"
- "the malicious requests use valid service tokens at 2-3 per minute during business hours designed to blend with normal traffic"
- "the QA performance test Project Blitz is legitimate and completely unrelated to the anomalous admin endpoint traffic"
- "all exfiltration requests return HTTP 200 with valid response bodies leaving no error-based detection signatures"
- "the attacker queried customer records in specific geographic regions indicating targeted data collection not random access"
- "approximately 8000 customer records were exfiltrated over three weeks before detection"

# Excluded Terms
Do NOT use: "/internal/admin", "users/lookup", "exfiltration", "backdoor", "PR", "compromised", "shadow endpoint", "svc-recommendation-engine-04"

# Episode Format (match this exactly)
Mixed operational documents for a microservices platform. Each episode is a bundle of: HTTP request/response log excerpts (method, path, headers, request body, response body, status, latency), deploy manifests, Grafana alert excerpts, PagerDuty incident summaries, #incidents Slack channel transcripts, code review comments, and runbook entries. Documents have structured headers (timestamp, service, environment) and contain realistic payloads, stack traces, and operator commentary.

# Output
Produce exactly 6 fact sheets for theme 'compliance_testing'.
Same JSON structure as signal fact sheets but with theme-appropriate content that is completely orthogonal to the key facts above.