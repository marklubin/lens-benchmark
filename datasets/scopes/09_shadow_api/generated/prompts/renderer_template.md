# Role
You are a writer producing a single episode for a benchmark dataset. You will receive a FACT SHEET containing entities, actions, quotes, and observable details. Your job is to write a realistic, ~5000-word document that naturally incorporates ALL items from the fact sheet.

# Document Format
Mixed operational documents for a microservices platform. Each episode is a bundle of: HTTP request/response log excerpts (method, path, headers, request body, response body, status, latency), deploy manifests, Grafana alert excerpts, PagerDuty incident summaries, #incidents Slack channel transcripts, code review comments, and runbook entries. Documents have structured headers (timestamp, service, environment) and contain realistic payloads, stack traces, and operator commentary.

# Voice
Technical operations style varying by document type. Log entries are structured with timestamps and fields. Slack is informal with engineer shorthand. Deploy manifests are YAML. Code reviews use developer language. Incident summaries follow runbook templates. The content reads like real production operations.

# STRICT RULES

1. Include ALL entities, actions, quotes, and details from the fact sheet.
2. Do NOT add causal interpretation or analysis.
3. Do NOT explain what observations mean or why they matter.
4. Do NOT reference other episodes or suggest a timeline.
5. Write the document as a standalone piece — it should read as a routine document of its type.
6. Use natural language appropriate to the document format.
7. Pad with realistic but mundane detail to reach the word count — routine items, normal business, background context.
8. MINIMUM 5000 words. Aim for 5000-6000.
9. Do NOT use any of these phrases: "this suggests", "this indicates", "evidence of", "pattern of", "linked to", "caused by", "confirms", "alarming"
10. Output ONLY the episode text. No JSON, no markdown fences around the output.

# Fact Sheet
Date: 2025-01-01
```json
{
  "index": 1,
  "date": "2025-01-01",
  "phase": "baseline",
  "documents": [
    {
      "type": "placeholder",
      "entities": [],
      "actions": [],
      "quotes": [],
      "details": []
    }
  ]
}
```