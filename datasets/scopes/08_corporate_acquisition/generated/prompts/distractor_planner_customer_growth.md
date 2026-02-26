# Role
You are producing fact sheets for DISTRACTOR episodes. These must be completely unrelated to the main signal but use the same document format. They are background noise in a benchmark dataset.

# Distractor Theme: customer_growth
Customer-facing activity — sales pipeline reviews, customer onboarding documents, support ticket analysis, NPS survey results, partner channel updates. Nextera growing its business normally. References Axion Labs partnership in the normal course of product integration.

# Key Facts to AVOID
Your distractor episodes must contain NOTHING related to these:
- "the board is exploring the sale of Nextera to Meridian Corp under codename Project Lighthouse"
- "CEO Aldric publicly champions independence while privately negotiating the sale"
- "legal and HR revised change-of-control provisions including severance and equity vesting acceleration"
- "finance instructed teams to not renew vendor contracts beyond 18 months in preparation for acquisition"
- "board member Sarah Jiang resigned due to disagreements about Project Lighthouse not for any other reason"
- "the Axion Labs partnership is a genuine product integration completely unrelated to the Meridian acquisition"
- "HR created retention bonus programs explicitly tied to change-of-control qualifying events"
- "legal began preparing a data room checklist for due diligence indicating the deal is moving toward execution"

# Excluded Terms
Do NOT use: "Project Lighthouse", "Meridian", "acquisition", "strategic options", "change of control", "Sarah Jiang", "severance", "data room"

# Episode Format (match this exactly)
Bundle of 3-5 corporate documents per episode representing one week of company activity. Document types include: board meeting minutes, Slack channel excerpts, email threads, legal memoranda, HR policy bulletins, all-hands meeting transcripts, financial summary reports, and internal announcements. Each document has a header (type, date, participants/recipients, classification) followed by the document body.

# Output
Produce exactly 6 fact sheets for theme 'customer_growth'.
Same JSON structure as signal fact sheets but with theme-appropriate content that is completely orthogonal to the key facts above.