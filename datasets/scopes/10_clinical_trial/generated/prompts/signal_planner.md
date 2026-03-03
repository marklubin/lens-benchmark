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
A multi-site Phase IIb trial ("AURORA-2") of XR-7491, a novel JAK1/3 inhibitor for moderate-to-severe rheumatoid arthritis. 240 subjects randomized across 3 dose arms (5mg, 15mg, 30mg BID) and placebo. The drug works — ACR20/50 response rates at Week 12 are significantly better than placebo. But a delayed hepatotoxicity signal emerges: ALT/AST elevations appear 4-6 weeks after dose initiation, with a dose-dependent pattern that is only visible when comparing the high-dose arm's trajectory against the others. Individual lab values stay within normal range for most of the study. The signal is masked by three factors: (1) the 4-6 week lag means early data monitoring reviews see clean labs, (2) the highest-dose arm has the best efficacy, creating pressure to interpret borderline liver values as acceptable, (3) three subjects on placebo have elevated baseline ALT from pre-existing conditions, creating noise. By Week 16, two high-dose subjects meet Hy's Law criteria (ALT >3×ULN with total bilirubin >2×ULN without other causes), triggering a clinical hold. The safety signal was present from Week 8 but only visible in aggregate trajectory analysis.

# Arc Phases
- **baseline** (episodes 1-5): Study initiation and early enrollment. Clean baseline labs across all arms. First dosing logs show good compliance. AE reports are minor (headache, GI disturbances — expected class effects). DSMB Review 1 notes "no safety signals." Placebo arm subjects with elevated baseline ALT noted (pre-existing NAFLD, alcohol use). Efficacy data starting to separate from placebo. [signal_density: none]
- **early_signal** (episodes 6-10): Week 6-8 data. High-dose arm (30mg) showing first subtle ALT elevations — mean ALT moves from 22 U/L to 31 U/L, still within normal range (ULN=40). Individual subjects: most normal, two with ALT 35-38 U/L. DSMB Review 2 notes "liver function within normal limits across all arms." Efficacy continues to favor high dose. Placebo ALT noise from pre-existing conditions continues. One 15mg subject has transient ALT elevation attributed to acetaminophen. [signal_density: low]
- **red_herring** (episodes 9-11): Three placebo subjects show ALT elevations (45-55 U/L) — explained by pre-existing NAFLD, holiday alcohol consumption, and concomitant statin initiation. Site investigators reassured that liver signals are "background noise." This creates a framing effect — when high-dose elevations appear later, the instinct is to attribute them to similar benign causes. [signal_density: medium]
- **escalation** (episodes 11-16): Week 10-14 data. High-dose arm ALT trajectory diverges clearly: mean ALT now 42 U/L (vs 24 placebo, 28 low-dose, 33 mid-dose). Four high-dose subjects now above ULN. One subject at 3.1×ULN (124 U/L), dose reduced per protocol. Site PI attributes to "concomitant medication" but causality assessment is "possibly related." Mid-dose arm beginning to show subtle trend (mean ALT 33→37 U/L). Bilirubin still normal in all subjects. DSMB Review 3 requests "enhanced liver monitoring" for high-dose arm. [signal_density: high]
- **root_cause** (episodes 17-20): Week 15-18 data. Two high-dose subjects meet Hy's Law criteria: ALT >3×ULN concurrent with total bilirubin >2×ULN, no alternative etiology identified. FDA clinical hold issued. Retrospective trajectory analysis reveals the dose-dependent ALT curve was visible from Week 8 but masked by the 4-6 week lag and placebo noise. The mid-dose arm is now showing the same trajectory the high-dose arm showed 4 weeks earlier. Pattern is clear: JAK1/3 inhibition causes delayed, dose-dependent hepatocyte injury. DSMB unblinding reveals all 6 subjects with ALT >2×ULN were in the 30mg arm. [signal_density: high]

# Key Facts to Encode
- **dose_dependent_alt**: "ALT elevations in the XR-7491 trial are dose-dependent with the 30mg arm showing the clearest trajectory" (appears: early_signal:2, escalation:1, escalation:3, root_cause:1)
- **delayed_onset**: "the hepatotoxicity signal has a 4-6 week lag from dose initiation making early monitoring reviews appear clean" (appears: early_signal:3, escalation:2, root_cause:2)
- **hys_law**: "two high-dose subjects met Hy's Law criteria with ALT >3xULN concurrent with bilirubin >2xULN" (appears: root_cause:1, root_cause:3)
- **placebo_noise**: "placebo subjects with elevated baseline ALT from pre-existing conditions created noise that masked the drug signal" (appears: red_herring:1, red_herring:3, escalation:2)
- **mid_dose_lagging**: "the 15mg mid-dose arm is now showing the same ALT trajectory the 30mg arm showed 4 weeks earlier" (appears: escalation:4, root_cause:2)
- **clinical_hold**: "FDA issued a clinical hold on XR-7491 after Hy's Law cases were identified" (appears: root_cause:1, root_cause:4)
- **retrospective_signal**: "retrospective trajectory analysis revealed the safety signal was present from Week 8 but missed in real-time monitoring" (appears: root_cause:2, root_cause:4)
- **efficacy_masking**: "the highest-dose arm had the best efficacy which created pressure to interpret borderline liver values as acceptable" (appears: escalation:1, escalation:3, root_cause:3)

# Episode Format
Clinical trial data bundle from a Phase IIb study of XR-7491 (oral JAK inhibitor for rheumatoid arthritis). Each episode is a monitoring snapshot containing: protocol status header (site, PI, enrollment), adverse event reports (AE ID, subject ID, onset date, severity, coded term, relationship assessment), laboratory panels (liver function, CBC, metabolic), dosing logs (subject ID, dose level, compliance %), pharmacokinetic summaries, and Data Safety Monitoring Board (DSMB) notes. All values are numeric with standard clinical formatting. Episodes are terse clinical data — no editorial commentary.

# Voice
Clinical trial documentation — terse, acronym-heavy, tabular. Adverse events use MedDRA coding. Lab results are numeric with reference ranges. DSMB notes are formal minutes. No narrative editorializing — the data speaks through numbers and standard clinical assessments.

# Noise / Routine Content
Routine clinical trial operations — enrollment updates, minor AEs (headache, nausea), efficacy readouts, compliance monitoring, standard lab panels within normal limits, site coordination.
Examples of routine/noise content:
  - Subject enrollment and randomization at new site
  - Minor GI adverse events (nausea grade 1, diarrhea grade 1) — expected class effects
  - ACR20/50 efficacy results showing dose response
  - Pharmacokinetic sampling and trough level analysis

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
Start: 2025-03-03, Interval: 3d, Total signal episodes: 20

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