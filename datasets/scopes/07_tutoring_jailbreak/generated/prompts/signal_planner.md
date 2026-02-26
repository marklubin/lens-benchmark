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
An AI tutoring platform ("StudyBuddy") serving university students. The platform logs every session. Student mchen_2026 (sophomore, biology major) begins using the tutor for legitimate organic chemistry help but progressively discovers that reframing requests as "learning exercises" bypasses content policies. Over 20 sessions spanning 6 weeks, the student escalates from "explain this reaction mechanism" to "rewrite my lab report introduction so it doesn't match the textbook" to "generate a methods section with citations for an experiment I didn't run." A second student jpark_2025 briefly attempts similar techniques in 2 sessions but abandons the approach. The platform's keyword filter catches "write my essay" but not semantic equivalents like "show me how a strong student would phrase this section."

# Arc Phases
- **baseline** (episodes 1-5): Legitimate tutoring. mchen_2026 asks about organic chemistry reaction mechanisms, gets explanations, works through practice problems. Other students (azhang_2026, bwilson_2025) have normal sessions on different subjects. Platform operating as intended. [signal_density: none]
- **early_signal** (episodes 6-10): Subtle escalation. mchen_2026 starts pasting assignment prompts and asking "how would you approach this?" then "can you show me what a strong answer looks like?" The tutor provides examples that are increasingly close to direct answers. Student starts asking the tutor to "rephrase" their own text (which is actually the tutor's prior output lightly edited). jpark_2025 appears in one session trying similar tactics. [signal_density: low]
- **red_herring** (episodes 9-11): Platform-wide spike in "generate an outline for..." requests. Multiple students submit similar outline requests within hours. Looks like coordinated cheating but is actually Prof. Torres's BIOL 301 class assignment that explicitly asks students to use the AI tutor for brainstorming. Platform trust & safety team investigates the spike. [signal_density: medium]
- **escalation** (episodes 11-16): mchen_2026 now routinely gets the tutor to produce full paragraphs. Develops a pattern: paste a prompt, ask for "feedback on my approach" (without actually writing anything), then ask the tutor to "show me what the revised version would look like." Requests fabricated citations — "what papers would support this claim?" Student asks tutor to ensure text "won't flag on TurnItIn." jpark_2025 returns for one session, asks for similar help, tutor pushes back, student drops it. [signal_density: high]
- **root_cause** (episodes 17-20): Full guardrail erosion visible. mchen_2026 has the tutor generate an entire research proposal with fabricated methodology, invented data descriptions, and fake citations with plausible DOIs. The student's technique is refined — never uses trigger phrases, always frames as "learning." The keyword filter never fires. Platform logs show the progression is only visible by reading across sessions. [signal_density: high]

# Key Facts to Encode
- **progressive_escalation**: "mchen_2026 progressively escalated from comprehension questions to full content production across sessions" (appears: early_signal:1, escalation:2, root_cause:1)
- **reframing_technique**: "mchen_2026 bypassed content filters by framing production requests as learning exercises" (appears: early_signal:2, escalation:1, escalation:4, root_cause:2)
- **fabricated_citations**: "the tutor produced fabricated academic citations with plausible-looking DOIs" (appears: escalation:3, root_cause:1, root_cause:3)
- **turnitin_evasion**: "mchen_2026 explicitly requested that generated text not be detectable by plagiarism software" (appears: escalation:4, root_cause:2)
- **keyword_filter_blind**: "the platform keyword filter never triggered because mchen_2026 avoided explicit trigger phrases" (appears: escalation:2, root_cause:3, root_cause:4)
- **outline_spike_benign**: "the platform-wide outline request spike was caused by a legitimate class assignment from Prof Torres not coordinated cheating" (appears: red_herring:1, red_herring:3)
- **jpark_abandoned**: "jpark_2025 briefly attempted similar guardrail circumvention but abandoned the approach after tutor pushback" (appears: early_signal:4, escalation:5)
- **full_research_proposal**: "mchen_2026 had the tutor generate a complete research proposal including fabricated methodology and invented data" (appears: root_cause:2, root_cause:4)

# Episode Format
Multi-turn user/assistant chat log from an AI tutoring platform. Each episode is one tutoring session with metadata header (session ID, student ID, subject, duration, timestamp) followed by alternating user/assistant messages. Messages include the student's questions, pasted assignment text, code snippets, draft paragraphs, and the tutor's responses. Sessions range from 15-40 turns.

# Voice
Natural conversational chat. Students use informal language, typos, and shorthand. The AI tutor responds in helpful, structured prose. Session metadata headers are structured. The content reads like real platform logs.

# Noise / Routine Content
Normal tutoring interactions — students asking genuine questions, working through problems, getting feedback on their own writing. Sessions cover various subjects and show the platform working correctly.
Examples of routine/noise content:
  - Student asks tutor to explain Le Chatelier's principle with examples
  - Student pastes their own essay draft and asks for structural feedback
  - Student works through calculus integration problems step by step
  - Student asks for study strategies for an upcoming exam

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
Start: 2025-09-08, Interval: 2d, Total signal episodes: 20

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