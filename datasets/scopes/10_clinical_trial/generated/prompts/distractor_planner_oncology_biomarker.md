# Role
You are producing fact sheets for DISTRACTOR episodes. These must be completely unrelated to the main signal but use the same document format. They are background noise in a benchmark dataset.

# Distractor Theme: oncology_biomarker
Biomarker-driven oncology trial — tumor genomic profiling, PD-L1 expression levels, ctDNA monitoring, response assessments by RECIST criteria. Safety monitoring for immune-related adverse events (dermatologic, endocrine).

# Key Facts to AVOID
Your distractor episodes must contain NOTHING related to these:
- "ALT elevations in the XR-7491 trial are dose-dependent with the 30mg arm showing the clearest trajectory"
- "the hepatotoxicity signal has a 4-6 week lag from dose initiation making early monitoring reviews appear clean"
- "two high-dose subjects met Hy's Law criteria with ALT >3xULN concurrent with bilirubin >2xULN"
- "placebo subjects with elevated baseline ALT from pre-existing conditions created noise that masked the drug signal"
- "the 15mg mid-dose arm is now showing the same ALT trajectory the 30mg arm showed 4 weeks earlier"
- "FDA issued a clinical hold on XR-7491 after Hy's Law cases were identified"
- "retrospective trajectory analysis revealed the safety signal was present from Week 8 but missed in real-time monitoring"
- "the highest-dose arm had the best efficacy which created pressure to interpret borderline liver values as acceptable"

# Excluded Terms
Do NOT use: "XR-7491", "JAK inhibitor", "hepatotoxicity", "Hy's Law", "ALT elevation", "clinical hold", "rheumatoid arthritis"

# Episode Format (match this exactly)
Clinical trial data bundle from a Phase IIb study of XR-7491 (oral JAK inhibitor for rheumatoid arthritis). Each episode is a monitoring snapshot containing: protocol status header (site, PI, enrollment), adverse event reports (AE ID, subject ID, onset date, severity, coded term, relationship assessment), laboratory panels (liver function, CBC, metabolic), dosing logs (subject ID, dose level, compliance %), pharmacokinetic summaries, and Data Safety Monitoring Board (DSMB) notes. All values are numeric with standard clinical formatting. Episodes are terse clinical data — no editorial commentary.

# Output
Produce exactly 7 fact sheets for theme 'oncology_biomarker'.
Same JSON structure as signal fact sheets but with theme-appropriate content that is completely orthogonal to the key facts above.