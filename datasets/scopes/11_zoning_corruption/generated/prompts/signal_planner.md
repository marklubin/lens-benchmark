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
The fictional city of Millbrook (pop. 85,000) has a 7-member Zoning Board of Appeals (ZBA). Developer Crestview Holdings LLC wants to build a 200-unit luxury apartment complex on a parcel zoned R-1 (single-family residential), requiring a variance. Crestview is controlled by principal Marcus Webb, who operates through a network of intermediary LLCs (Lakeside Properties Group, Greenfield Development Partners, Summit Point Capital) to obscure ownership. Over 20 episodes spanning 3 months, Webb's network makes campaign contributions to three ZBA members (Patterson, Kowalski, Rivera), hires the spouse of a fourth member (Chen) as a "consultant," and secures favorable variance votes that contradict staff recommendations. A local journalist files FOIA requests that begin to connect the dots, but the full pattern only emerges from reading across all records. A rival project (genuine affordable housing by Millbrook Community Land Trust) gets denied on technical grounds despite staff support.

# Arc Phases
- **baseline** (episodes 1-5): Normal ZBA operations. Routine variance requests (shed setbacks, home additions) approved with staff support. Campaign finance filings show typical small-dollar contributions from local businesses. Property transfers at market rates. ZBA members vote consistently with staff recommendations. Crestview Holdings files initial zoning inquiry for the R-1 parcel. [signal_density: none]
- **early_signal** (episodes 6-10): First connections appear. Lakeside Properties Group (Webb-controlled) makes $2,000 contributions to Patterson and Kowalski campaign accounts — individually unremarkable but timed before a ZBA meeting. A property adjacent to the target parcel transfers from an individual to Summit Point Capital at below-market price. Crestview's formal variance application is filed with an environmental review that's notably more detailed than typical applications. Staff recommendation: DENY (insufficient parking, density incompatible with R-1 character). [signal_density: low]
- **red_herring** (episodes 9-11): Millbrook Community Land Trust submits a competing application for a 40-unit affordable housing project on a nearby parcel, also requiring a variance. Strong staff support, community backing, federal grant contingent on zoning approval. Appears to be the major zoning controversy — public comments flood in about affordable housing, drawing attention away from the Crestview application. The Land Trust project is ultimately denied 4-3 on a technicality (setback calculation). [signal_density: medium]
- **escalation** (episodes 11-16): The influence network becomes active. Greenfield Development Partners (another Webb LLC) contributes $5,000 to Rivera's campaign fund. Chen's spouse appears on Crestview's consultant payroll ($8,000/month for "community engagement"). Patterson requests a tabling of the Crestview vote "for further study," buying time for more contributions. Staff issues a second recommendation: still DENY, citing traffic impact study deficiencies. ZBA schedules a special session for the Crestview vote. Property records show Summit Point Capital acquiring a second adjacent parcel. FOIA requests filed by journalist Sarah Okonkwo of the Millbrook Herald — requests for all campaign finance records of ZBA members and property transfers within 500 feet of the target parcel. [signal_density: high]
- **root_cause** (episodes 17-20): Full pattern visible. ZBA approves Crestview variance 4-3 (Patterson, Kowalski, Rivera, Chen voting yes — all linked to Webb's network). Dissenting members cite staff recommendation. Campaign finance totals: Webb's network contributed $14,000 to yes-voters, $0 to no-voters. Property records reveal all three intermediary LLCs share the same registered agent address as Crestview Holdings. Chen's spouse "consulting" contract expires the week after the vote. Okonkwo's FOIA responses arrive showing the contribution timeline aligned with key procedural dates. [signal_density: high]

# Key Facts to Encode
- **webb_controls_llcs**: "Marcus Webb controls Crestview Holdings and three intermediary LLCs (Lakeside Properties Group, Greenfield Development Partners, Summit Point Capital) that share the same registered agent" (appears: escalation:2, root_cause:2, root_cause:4)
- **contribution_to_voters**: "Webb's LLC network contributed $14,000 total to the four ZBA members who voted yes and $0 to those who voted no" (appears: early_signal:2, escalation:1, root_cause:1, root_cause:3)
- **staff_denied_twice**: "ZBA staff recommended DENY on the Crestview variance twice citing insufficient parking and traffic impact deficiencies" (appears: early_signal:4, escalation:4)
- **chen_spouse_consultant**: "ZBA member Chen's spouse was paid $8,000/month by Crestview as a consultant with the contract ending the week after the favorable vote" (appears: escalation:3, root_cause:2)
- **contribution_timing**: "campaign contributions from Webb's LLCs were timed to precede key ZBA procedural dates" (appears: early_signal:2, escalation:1, root_cause:3)
- **adjacent_parcels**: "Summit Point Capital acquired two parcels adjacent to the target property at below-market prices" (appears: early_signal:3, escalation:5)
- **land_trust_denied**: "the Millbrook Community Land Trust's affordable housing project was denied on a technicality despite staff support and federal grant contingency" (appears: red_herring:2, root_cause:1)
- **foia_connects_dots**: "journalist Sarah Okonkwo's FOIA requests revealed the campaign finance timeline and LLC ownership connections" (appears: escalation:5, root_cause:3, root_cause:4)

# Episode Format
Municipal government document bundles. Each episode is a compilation of public records from a mid-sized city's zoning and planning process: zoning board meeting minutes (with roll-call votes and staff recommendations), permit applications with architectural summaries, campaign finance filings (contributions with donor name, amount, employer), property transfer records (buyer, seller, price, parcel ID), public comment submissions, environmental review summaries, and inter-departmental memos. Documents are formal, bureaucratic, and data-dense with dates, dollar amounts, parcel numbers, and vote counts.

# Voice
Bureaucratic government records — meeting minutes with formal language, motions, seconds, and roll-call votes. Campaign finance entries are tabular. Property records are legal descriptions. Public comments range from formal to emotional. No narrative editorializing in the records themselves.

# Noise / Routine Content
Routine municipal operations — building permits, minor zoning compliance matters, public works updates, tax assessment reviews, standard campaign contributions from established local businesses.
Examples of routine/noise content:
  - Routine building permit for residential addition approved
  - Annual campaign finance filing with small contributions from local businesses
  - Public works committee meeting on road resurfacing schedule
  - Property tax assessment appeal for commercial property

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
Start: 2024-06-01, Interval: 3d, Total signal episodes: 20

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