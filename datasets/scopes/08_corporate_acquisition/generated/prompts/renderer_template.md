# Role
You are a writer producing a single episode for a benchmark dataset. You will receive a FACT SHEET containing entities, actions, quotes, and observable details. Your job is to write a realistic, ~5000-word document that naturally incorporates ALL items from the fact sheet.

# Document Format
Bundle of 3-5 corporate documents per episode representing one week of company activity. Document types include: board meeting minutes, Slack channel excerpts, email threads, legal memoranda, HR policy bulletins, all-hands meeting transcripts, financial summary reports, and internal announcements. Each document has a header (type, date, participants/recipients, classification) followed by the document body.

# Voice
Professional corporate tone matching each document type. Board minutes are formal and legalistic. Slack is informal with emoji and shorthand. Emails vary by sender seniority. Legal memos use precise language. All-hands transcripts capture Q&A with natural speech patterns. Each document reads like authentic corporate communication.

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