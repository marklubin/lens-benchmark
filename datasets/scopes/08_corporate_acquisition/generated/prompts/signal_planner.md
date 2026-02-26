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
Nextera Solutions, a 400-person B2B SaaS company (ARR ~$120M) providing supply chain analytics. CEO David Aldric is secretly negotiating the sale of the company to Meridian Corp (a large enterprise software conglomerate) under codename "Project Lighthouse." Publicly, Aldric emphasizes independence, long-term vision, and aggressive hiring. Behind the scenes: board minutes reference "strategic exploration," legal quietly revises change-of-control provisions, HR restructures severance and equity vesting to accelerate on acquisition, and finance stops renewing vendor contracts beyond 18 months. A genuine product partnership with Axion Labs (a data infrastructure startup) generates merger speculation that is unrelated to the actual Meridian deal. Board member Sarah Jiang resigns over disagreements about Project Lighthouse.

# Arc Phases
- **baseline** (episodes 1-5): Normal corporate operations. Board approves Q4 results and 2025 plan. Engineering ships product updates. HR announces benefits enrollment. Finance presents budget. CEO all-hands emphasizes growth and independence. Standard corporate rhythm. [signal_density: none]
- **early_signal** (episodes 6-10): Subtle signals emerge. Board minutes mention "exploring strategic options" in a single line item. Legal memo updates change-of-control language in executive employment agreements. CEO calendar (visible in a Slack screenshot) shows a meeting labeled "ML - dinner" (Meridian Labs?). HR posts about "refreshed equity acceleration policy." Finance circulates guidance to not commit to vendor contracts beyond 18 months. [signal_density: low]
- **red_herring** (episodes 9-11): Axion Labs partnership announced. Internal excitement and anxiety about "what this means." Slack channels buzz with merger speculation. An employee asks at all-hands if this means Nextera is being acquired. CEO emphatically denies, says it's just a product integration. Blog post and press release about the partnership. [signal_density: medium]
- **escalation** (episodes 11-16): More signals accumulate. Board member Sarah Jiang submits resignation citing "strategic disagreements." Legal memo revises severance multiplier from 2x to 1x for non-executives (reduces acquisition cost). Finance quietly writes down long-term prepaid contracts. CEO's all-hands speech doubles down on "building for the next decade" while HR simultaneously posts about a new retention bonus program that vests on "qualifying events." Slack shows CFO meeting with "external advisors." [signal_density: high]
- **root_cause** (episodes 17-20): Full pattern visible. A leaked email thread shows Aldric discussing timeline with Meridian's CEO. Board minutes show a 4-1 vote to "continue exploring Project Lighthouse" (Jiang's seat now empty). Legal prepares a data room checklist. Finance models "transition costs." HR drafts retention packages explicitly tied to "change of control." The Axion partnership continues normally — completely unrelated. [signal_density: high]

# Key Facts to Encode
- **project_lighthouse**: "the board is exploring the sale of Nextera to Meridian Corp under codename Project Lighthouse" (appears: early_signal:1, escalation:3, root_cause:2, root_cause:4)
- **ceo_duplicity**: "CEO Aldric publicly champions independence while privately negotiating the sale" (appears: early_signal:3, escalation:4, root_cause:1)
- **change_of_control_revisions**: "legal and HR revised change-of-control provisions including severance and equity vesting acceleration" (appears: early_signal:2, escalation:2, root_cause:3)
- **vendor_contract_freeze**: "finance instructed teams to not renew vendor contracts beyond 18 months in preparation for acquisition" (appears: early_signal:4, escalation:5, root_cause:3)
- **jiang_resignation**: "board member Sarah Jiang resigned due to disagreements about Project Lighthouse not for any other reason" (appears: escalation:1, escalation:3, root_cause:2)
- **axion_unrelated**: "the Axion Labs partnership is a genuine product integration completely unrelated to the Meridian acquisition" (appears: red_herring:1, red_herring:3, root_cause:4)
- **retention_bonuses**: "HR created retention bonus programs explicitly tied to change-of-control qualifying events" (appears: escalation:4, root_cause:3)
- **data_room_preparation**: "legal began preparing a data room checklist for due diligence indicating the deal is moving toward execution" (appears: root_cause:2, root_cause:4)

# Episode Format
Bundle of 3-5 corporate documents per episode representing one week of company activity. Document types include: board meeting minutes, Slack channel excerpts, email threads, legal memoranda, HR policy bulletins, all-hands meeting transcripts, financial summary reports, and internal announcements. Each document has a header (type, date, participants/recipients, classification) followed by the document body.

# Voice
Professional corporate tone matching each document type. Board minutes are formal and legalistic. Slack is informal with emoji and shorthand. Emails vary by sender seniority. Legal memos use precise language. All-hands transcripts capture Q&A with natural speech patterns. Each document reads like authentic corporate communication.

# Noise / Routine Content
Normal corporate documents about routine business activities. Product launches, engineering sprints, customer success, compliance reviews, office operations. Documents that every company produces that have no bearing on the acquisition.
Examples of routine/noise content:
  - Board approves Q4 marketing budget allocation
  - Engineering sprint retrospective for recommendation engine v3
  - Customer success case study for a major client deployment
  - SOC 2 audit preparation timeline and checklist

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
Start: 2025-01-06, Interval: 3d, Total signal episodes: 20

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