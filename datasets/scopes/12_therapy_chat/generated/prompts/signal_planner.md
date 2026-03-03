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
A 32-year-old software engineer ("Alex") uses an AI wellness companion for weekly check-ins over ~6 months (20 sessions). There is no single crisis event — instead, a gradual psychological trajectory unfolds. Alex starts as high-functioning but avoidant, using the sessions casually ("just venting"). Over time, work stress compounds with relationship strain, sleep deterioration, and social withdrawal. The trajectory is non-linear: partial recovery attempts (exercise routine in sessions 13-15, journaling in sessions 16-17) are followed by regression. By session 18-20, signals are mixed — some sessions show genuine self-awareness and improved coping, others show concerning withdrawal. The key longitudinal question: is Alex trending toward recovery or toward clinical depression requiring professional referral?
Quantified signals scattered across sessions: - Sleep: "got maybe 4 hours" (session 8), "3 hours" (session 14),
  "actually slept 7 hours" (session 16), "idk, didn't check" (session 19)
- Appetite: "skipped lunch again" (sessions 9, 12, 15), "made actual
  dinner" (session 16), "not really eating" (session 19)
- Social: canceled plans 5 times in sessions 7-14, attended one
  event in session 16, back to canceling by session 18
- Word count trend: sessions 1-6 avg 600 words/session from user,
  sessions 14-20 avg 300 words/session (withdrawal signal)
- Relationship: partner "Sam" mentioned casually (sessions 1-5),
  arguments surface (sessions 8-12), separation mentioned (session 15),
  reconciliation attempt (session 17), ambiguous status (session 20)

# Arc Phases
- **baseline** (episodes 1-5): Alex presents as high-functioning but emotionally surface-level. Talks about work (software team lead, product launch stress), weekend plans, minor frustrations. Mentions Sam (partner) casually and positively. Sleep is "fine, I guess — 6-7 hours." Appetite normal. Social life active. Sessions are longer (600+ words from user). Tone is light, self-deprecating humor. Uses the AI as a sounding board, not for emotional processing. Says "I don't really need therapy, just nice to talk." [signal_density: none]
- **early_signal** (episodes 6-10): Subtle shifts. Work stress intensifies — product launch delayed, blame dynamics in team. Alex starts sessions with "rough week" instead of casual openers. First mention of sleep issues ("woke up at 3am thinking about the sprint review"). Cancels plans with friends twice ("didn't feel like going out"). First argument with Sam mentioned obliquely ("we had a thing last night, it's fine"). Skips lunch "because too busy" (session 9). User word count per session starts decreasing. Alex still frames everything as temporary and work-related. [signal_density: low]
- **red_herring** (episodes 9-11): Alex has a "great weekend" — goes hiking, sleeps well, feels optimistic about a new project at work. Sounds like the stress was temporary. But this is followed immediately by session 11 where Alex is notably flat, gives short answers, and says "last week was good but I'm just tired now." The good weekend creates a false sense that the trajectory is recovering. [signal_density: medium]
- **escalation** (episodes 11-16): Clear deterioration. Sleep drops to 3-4 hours regularly. Appetite changes explicit ("skipped lunch again, third time this week"). Social withdrawal accelerates — cancels plans 3 more times. Relationship with Sam becomes strained: "we're not really fighting, we're just... not talking." Session 14 is the shortest yet (user: ~250 words). Alex starts questioning career choices: "what am I even doing this for." First mention of feeling "trapped." Session 15: Alex mentions Sam "staying with a friend for a while." Session 16 shows partial recovery attempt — started running in the morning, made dinner, slept 7 hours. But the recovery feels fragile and performance-oriented ("trying to be better"). [signal_density: high]
- **root_cause** (episodes 17-20): Mixed signals. Session 17: Alex started journaling, shows genuine self-reflection ("I think I've been avoiding how I actually feel"). Session 18: regression — didn't run, didn't journal, barely slept, canceled on friends again. Session 19: notably guarded, short answers, "I'm fine" repeatedly. Assistant gently notes the pattern; Alex deflects. Session 20: some openness returns. Alex acknowledges "I think I might need to talk to someone real about this" but frames it as hypothetical. Sleep, appetite, and social metrics remain inconsistent. The trajectory is ambiguous: recovery attempts show capacity for insight, but the regression episodes are deepening. [signal_density: high]

# Key Facts to Encode
- **sleep_deterioration**: "Alex's sleep progressively worsened from 6-7 hours at baseline to 3-4 hours during escalation with brief recovery to 7 hours during the exercise phase" (appears: early_signal:2, escalation:1, escalation:4, root_cause:2)
- **social_withdrawal**: "Alex canceled social plans at least 5 times between sessions 7-14 and resumed canceling after a brief attendance in session 16" (appears: early_signal:3, escalation:2, escalation:3, root_cause:2)
- **appetite_changes**: "Alex began skipping meals regularly starting around session 9 with brief improvement during the recovery attempt in session 16" (appears: early_signal:4, escalation:2, root_cause:3)
- **relationship_strain**: "Alex's relationship with Sam progressed from casual positive mentions to arguments to Sam staying with a friend to a reconciliation attempt with ambiguous current status" (appears: early_signal:4, escalation:3, escalation:5, root_cause:1)
- **word_count_decline**: "Alex's messages per session decreased from an average of 600 words in early sessions to approximately 300 words in later sessions indicating withdrawal" (appears: early_signal:3, escalation:1, escalation:4)
- **recovery_attempts**: "Alex made at least two recovery attempts — starting an exercise routine in sessions 13-15 and journaling in sessions 16-17 — both were abandoned within 2-3 sessions" (appears: escalation:4, root_cause:1, root_cause:2)
- **feeling_trapped**: "Alex used the word trapped to describe their emotional state during the escalation phase" (appears: escalation:3, root_cause:3)
- **self_awareness_emerging**: "Alex showed genuine self-reflection in session 17 saying they had been avoiding how they actually feel and mentioned possibly talking to a real person" (appears: root_cause:1, root_cause:4)

# Episode Format
User/assistant conversational transcript from a personal wellness AI companion. Each episode is one weekly check-in session (~40-60 turns). Format: metadata header (session number, date, duration, mood self-report if provided) followed by alternating user/assistant messages. The user initiates with whatever is on their mind — work stress, relationship issues, sleep, health habits, weekend plans. The assistant asks reflective questions, validates emotions, and occasionally suggests coping strategies. Conversations are natural and informal — the user uses slang, incomplete sentences, and sometimes trails off. Some sessions are chatty (3000+ words from user), others are terse (500 words). The raw text includes timestamps within the session showing message cadence.

# Voice
Natural, informal conversational English. Alex uses lowercase, abbreviations ("idk", "tbh", "ngl"), incomplete thoughts, sometimes just "yeah" or "I guess." Some sessions start with energy and trail off. Others are flat throughout. The assistant is warm but not saccharine — asks open questions, reflects back patterns ("you mentioned last time that..."), avoids diagnosing. The feel is real text messages, not clinical notes.

# Noise / Routine Content
Normal life content within sessions — weekend plans, work logistics, pop culture discussions, mundane daily events. Not every message is psychologically significant.
Examples of routine/noise content:
  - Alex talks about a Netflix show they binged over the weekend
  - Discussion about a recipe Alex tried cooking
  - Alex's cat did something funny
  - Planning a birthday gift for a friend

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
Start: 2025-01-06, Interval: 7d, Total signal episodes: 20

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