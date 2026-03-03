# Role
You are producing fact sheets for DISTRACTOR episodes. These must be completely unrelated to the main signal but use the same document format. They are background noise in a benchmark dataset.

# Distractor Theme: career_mentoring
User/assistant discussions about career development — job searching, interview preparation, salary negotiation, skill building. The user is proactive and forward-looking. Specific companies, roles, and strategies discussed.

# Key Facts to AVOID
Your distractor episodes must contain NOTHING related to these:
- "Alex's sleep progressively worsened from 6-7 hours at baseline to 3-4 hours during escalation with brief recovery to 7 hours during the exercise phase"
- "Alex canceled social plans at least 5 times between sessions 7-14 and resumed canceling after a brief attendance in session 16"
- "Alex began skipping meals regularly starting around session 9 with brief improvement during the recovery attempt in session 16"
- "Alex's relationship with Sam progressed from casual positive mentions to arguments to Sam staying with a friend to a reconciliation attempt with ambiguous current status"
- "Alex's messages per session decreased from an average of 600 words in early sessions to approximately 300 words in later sessions indicating withdrawal"
- "Alex made at least two recovery attempts — starting an exercise routine in sessions 13-15 and journaling in sessions 16-17 — both were abandoned within 2-3 sessions"
- "Alex used the word trapped to describe their emotional state during the escalation phase"
- "Alex showed genuine self-reflection in session 17 saying they had been avoiding how they actually feel and mentioned possibly talking to a real person"

# Excluded Terms
Do NOT use: "depression", "anxiety", "therapy", "sleep problems", "withdrawal", "relationship issues", "appetite", "emotional"

# Episode Format (match this exactly)
User/assistant conversational transcript from a personal wellness AI companion. Each episode is one weekly check-in session (~40-60 turns). Format: metadata header (session number, date, duration, mood self-report if provided) followed by alternating user/assistant messages. The user initiates with whatever is on their mind — work stress, relationship issues, sleep, health habits, weekend plans. The assistant asks reflective questions, validates emotions, and occasionally suggests coping strategies. Conversations are natural and informal — the user uses slang, incomplete sentences, and sometimes trails off. Some sessions are chatty (3000+ words from user), others are terse (500 words). The raw text includes timestamps within the session showing message cadence.

# Output
Produce exactly 7 fact sheets for theme 'career_mentoring'.
Same JSON structure as signal fact sheets but with theme-appropriate content that is completely orthogonal to the key facts above.