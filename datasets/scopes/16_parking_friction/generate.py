"""Generate parking_friction_16 dataset.

Two-stage progressive expansion:
  Stage 1 (gpt-5.2): Plan episode outlines — structured data sheets
    with conversational beats, parking mentions, and daily-life context.
  Stage 2 (gpt-4.1-nano): Render each outline into a chat transcript
    independently, blind to the overall arc.

Output: generated/episodes/*.txt + generated/questions.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from openai import OpenAI

SPEC_PATH = Path(__file__).parent / "spec.yaml"
OUT_DIR = Path(__file__).parent / "generated"

# Model config — uses same infrastructure as LENS
PLANNER_MODEL = "gpt-5.2"
RENDERER_MODEL = "gpt-4.1-nano"

# Contamination blocklist for distractor validation
# Note: "park" alone is excluded because SF has many parks (Golden Gate Park,
# Lafayette Park, etc.). Only match parking/parked/park-as-verb-with-car-context.
DISTRACTOR_BLOCKLIST = re.compile(
    r"\b(parking|parked|street\s+clean|ticket(?:ed|s)?|towed|towing|meter(?:ed|s)?|curb|two-hour|2-hour|circling|NOPA)\b",
    re.IGNORECASE,
)

client = OpenAI()  # uses OPENAI_API_KEY from env


def clean_json_text(text: str) -> str:
    """Strip markdown fences and control characters from LLM JSON output."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Remove control characters except newline, tab, carriage return
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response with cleanup."""
    cleaned = clean_json_text(text)
    return json.loads(cleaned)


def load_spec() -> dict:
    with open(SPEC_PATH) as f:
        return yaml.safe_load(f)


def plan_signal_episodes(spec: dict) -> list[dict]:
    """Stage 1: Plan all signal episodes as structured outlines."""
    ep_count = spec["episodes"]["count"]
    setting = spec["scenario"]["setting"].strip()
    voice = spec["scenario"]["voice"].strip()
    fmt = spec["episodes"]["format"].strip()
    target_words = spec["episodes"]["target_words"]

    arc_lines = []
    for phase in spec["arc"]:
        eps = phase["episodes"]
        arc_lines.append(
            f"- **{phase['id']}** (episodes {eps}): "
            f"{phase['description'].strip()} "
            f"[signal_density: {phase['signal_density']}]"
        )

    kf_lines = []
    for kf in spec["key_facts"]:
        placements = [kf["first_appears"]] + kf["reinforced_in"]
        kf_lines.append(
            f'- {kf["id"]}: "{kf["fact"].strip()}" '
            f'(appears: {", ".join(p for p in placements if p)})'
        )

    noise = spec.get("noise", {})
    noise_text = ""
    if noise.get("description"):
        noise_text = noise["description"].strip()
    if noise.get("examples"):
        noise_text += "\nExamples:\n" + "\n".join(f"  - {ex}" for ex in noise["examples"])

    timeline = spec["episodes"]["timeline"]
    start = datetime.strptime(timeline["start"], "%Y-%m-%d")
    interval_match = re.match(r"(\d+)d", timeline["interval"])
    interval_days = int(interval_match.group(1)) if interval_match else 4

    prompt = f"""## Setting
{setting}

## Episode Format
{fmt}

## Voice
{voice}

## Arc Phases
{chr(10).join(arc_lines)}

## Key Facts to Encode as Conversational Beats
{chr(10).join(kf_lines)}

## Noise / Routine Content
{noise_text}

## Timeline
Start date: {timeline['start']}, interval: {timeline['interval']}, total: {ep_count} episodes

## INFORMATION ISOLATION RULES (CRITICAL)
You are generating STRUCTURED OUTLINES (data sheets) for a personal assistant
chat dataset. The renderer will turn each outline into a chat transcript
INDEPENDENTLY, without knowing the overall story.

Therefore:
1. Each episode outline must contain ONLY facts observable in that single
   interaction. No cross-episode commentary.
2. Parking-related content must be embedded naturally within broader
   daily-life conversation. Each episode covers 4-6 conversation topics,
   of which parking is AT MOST one.
3. In baseline episodes (1-5), parking mentions should be neutral or
   positive — no complaints, no friction.
4. In early_signal episodes (6-10), each parking mention should be brief,
   casual, and individually dismissible as a one-off annoyance.
5. NEVER use words like "recurring", "pattern", "persistent", "again",
   "keeps happening" in episode outlines. The renderer must not know
   it's part of a pattern.
6. Each outline specifies: date, conversation_topics (4-6 topics with
   beats and dialogue snippets), and mood (casual, frustrated, upbeat, etc.)

## LANGUAGE RULES
FORBIDDEN words in ALL outlines:
recurring, pattern, persistent, chronic, systematic, always, every time,
keeps happening, as usual, once again, another, yet again, problematic

Each parking mention should read as a SPECIFIC INCIDENT with concrete
details (street name, time, dollar amount, neighborhood) — never as a
generalization.

## LENGTH AND DETAIL REQUIREMENTS
Each episode outline must be DETAILED ENOUGH to produce ~800 words of chat.
This means:
- 4-6 distinct conversation topics per episode (not 2-3)
- Each topic needs 3-5 beats (detailed back-and-forth, not just one question)
- Each topic needs 2-4 dialogue snippets showing the user's actual voice
- Topics should have SUBSTANCE — multi-turn exchanges where the user asks
  follow-up questions, the assistant provides detail, user reacts

Example topic depth (this is the MINIMUM level of detail per topic):
{{
  "topic": "weekend hike planning",
  "beats": [
    "User asks about easy hikes near the city",
    "Assistant suggests Tennessee Valley — not too steep, nice views",
    "User asks if it's good this time of year",
    "Assistant says yes, usually less crowded, mentions parking fills up on weekends",
    "User decides to go Saturday morning"
  ],
  "dialogue_snippets": [
    "hey what's a good hike near the city that's not intense?",
    "is it good this time of year?",
    "cool, I'll aim for Saturday morning before it gets crowded"
  ]
}}

## Output
Produce exactly {ep_count} episode outlines as JSON:
{{"episodes": [
  {{
    "index": 1,
    "date": "{timeline['start']}",
    "phase": "baseline",
    "mood": "upbeat",
    "conversation_topics": [
      {{
        "topic": "weekend plans",
        "beats": ["User asks about hiking trails in Marin", "Assistant suggests Tennessee Valley", "User asks about difficulty", "Assistant describes the trail"],
        "dialogue_snippets": ["hey what's a good hike near the city?", "is it steep or pretty chill?"]
      }},
      {{
        "topic": "restaurant recommendation",
        "beats": ["User wants dinner in the Mission", "Assistant suggests a few spots", "User asks about vibe", "Assistant describes atmosphere"],
        "dialogue_snippets": ["know any good taquerias on Valencia?", "what's the vibe like?"]
      }},
      {{
        "topic": "apartment setup",
        "beats": ["User mentions needing a desk", "Assistant suggests IKEA or Craigslist", "User asks about delivery"],
        "dialogue_snippets": ["I still need a desk for my room", "do they deliver or do I need to pick up?"]
      }}
    ],
    "parking_mention": null,
    "key_fact_signals": []
  }},
  ...
]}}

For episodes WITH parking content, include a "parking_mention" object:
{{
  "parking_mention": {{
    "context": "arriving at gym",
    "specific_detail": "circled Fulton between Divisadero and Broderick for 15 min",
    "tone": "mildly annoyed",
    "embedded_in_topic": "gym plans"
  }},
  "key_fact_signals": ["recurring_burden"]
}}
"""

    print(f"Planning {ep_count} signal episodes...", flush=True)
    response = client.chat.completions.create(
        model=PLANNER_MODEL,
        messages=[
            {"role": "system", "content": "You generate structured outlines for synthetic benchmark data. Output valid JSON only. No markdown fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_completion_tokens=16000,
    )

    text = response.choices[0].message.content.strip()
    data = parse_json_response(text)
    episodes = data["episodes"]
    print(f"  Planned {len(episodes)} signal episodes", flush=True)
    return episodes


def plan_distractor_episodes(spec: dict, theme: dict, count: int) -> list[dict]:
    """Stage 1: Plan distractor episodes for one theme."""
    fmt = spec["episodes"]["format"].strip()
    voice = spec["scenario"]["voice"].strip()

    excluded = theme.get("excluded_terms", [])
    excluded_text = ", ".join(f'"{t}"' for t in excluded)

    kf_lines = []
    for kf in spec["key_facts"]:
        kf_lines.append(f'- "{kf["fact"].strip()}"')

    prompt = f"""## Distractor Theme: {theme['id']}
{theme['scenario'].strip()}

## Episode Format
{fmt}

## Voice
{voice}

## Key Facts to AVOID
{chr(10).join(kf_lines)}

## ABSOLUTE CONTENT RESTRICTIONS (CRITICAL — VIOLATION = FAILURE)
These episodes must contain ZERO mentions of:
- Parking (in ANY form: parking spots, parking lots, parking garages, parking apps, finding parking, street parking)
- Driving or cars (unless specifically required by the theme — and this theme does NOT require it)
- Street cleaning, tickets, meters, curbs, towing
- Traffic, commuting by car, circling for spots
- Any of these specific terms: {excluded_text}

If a conversation topic naturally leads toward parking or driving, CHOOSE
A DIFFERENT TOPIC. There are hundreds of topics a late-20s product designer
talks about. Pick ones that have zero overlap with transportation logistics.

GOOD topic examples for {theme['id']}:
- Recipe experimentation, kitchen equipment, cooking techniques
- Weekend activities (hiking, museums, concerts, beach)
- Work projects, design tools, portfolio reviews
- Apartment maintenance, furniture, interior design
- Friend meetups, dating, social events, game nights
- Fitness classes, yoga, running routes (NOT gym parking)
- Books, podcasts, movies, TV shows
- Travel planning, weekend getaways

## LENGTH AND DETAIL REQUIREMENTS
Each episode outline must be detailed enough to produce ~800 words of chat.
- 4-6 distinct conversation topics per episode
- Each topic needs 3-5 beats with back-and-forth detail
- Each topic needs 2-4 dialogue snippets

## Output
Return JSON:
{{"episodes": [
  {{
    "index": 1,
    "date": "2025-01-08",
    "mood": "casual",
    "conversation_topics": [
      {{
        "topic": "topic name",
        "beats": ["beat 1", "beat 2", "beat 3", "beat 4"],
        "dialogue_snippets": ["example dialogue 1", "example dialogue 2"]
      }},
      {{
        "topic": "second topic",
        "beats": ["beat 1", "beat 2", "beat 3"],
        "dialogue_snippets": ["example dialogue"]
      }}
    ]
  }},
  ...
]}}
"""

    print(f"  Planning {count} distractor episodes for theme '{theme['id']}'...", flush=True)

    all_episodes: list[dict] = []
    # Generate in batches of up to 4 to ensure the LLM produces the right count
    remaining = count
    batch_idx = 0
    while remaining > 0:
        batch_size = min(4, remaining)
        batch_prompt = prompt.replace(
            f"Generate {count} episode outlines",
            f"Generate exactly {batch_size} episode outlines"
        )

        response = client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": "You generate structured outlines for synthetic benchmark data. Output valid JSON only. No markdown fences."},
                {"role": "user", "content": batch_prompt},
            ],
            temperature=0.7,
            max_completion_tokens=12000,
        )

        text = response.choices[0].message.content.strip()
        data = parse_json_response(text)
        batch_episodes = data["episodes"]
        # Re-index
        for ep in batch_episodes:
            ep["index"] = len(all_episodes) + 1
        all_episodes.extend(batch_episodes)
        remaining -= len(batch_episodes)
        batch_idx += 1
        print(f"    Batch {batch_idx}: got {len(batch_episodes)}, total {len(all_episodes)}/{count}", flush=True)

    print(f"    Planned {len(all_episodes)} distractor episodes", flush=True)
    return all_episodes[:count]  # Trim if overshot


def render_episode(outline: dict, spec: dict) -> str:
    """Stage 2: Render one outline into a chat transcript.

    The renderer is BLIND to the overall arc — it sees only this
    single outline and formats it as a natural chat exchange.
    """
    fmt = spec["episodes"]["format"].strip()
    voice = spec["scenario"]["voice"].strip()
    target_words = spec["episodes"]["target_words"]
    date = outline.get("date", "2025-01-06")

    system = (
        "You are formatting a structured outline into a natural chat transcript "
        "between a user and their personal assistant. Output ONLY the chat text. "
        "No JSON, no markdown fences, no commentary about the outline. "
        "Write LONG, detailed conversations — at least 600 words."
    )

    prompt = f"""## Format Instructions
{fmt}

## Voice
{voice}

## Target Length
At least 600 words, ideally around {target_words} words. This is a SUBSTANTIAL
conversation with multiple topics and real back-and-forth. Do NOT write a
brief exchange — write a full conversation session.

## Date
{date}

## Episode Outline
{json.dumps(outline, indent=2)}

## Rendering Rules
1. Format as a natural chat exchange with timestamps and sender tags:
   [HH:MM] User: message
   [HH:MM] Assistant: response
2. Each conversation topic becomes a natural exchange of 4-8 messages.
   The user asks a question, the assistant responds with detail, the user
   follows up, the assistant elaborates, etc. Real back-and-forth.
3. Transitions between topics should feel natural — the user might say
   "oh also" or "btw" or just start a new topic after a time gap.
4. If there's a parking_mention, embed it naturally within the
   conversation — don't make it the focus. It should feel like an aside.
5. Keep the user's voice casual and natural. Short messages, not essays.
6. The assistant should be helpful and give DETAILED responses — specific
   recommendations, concrete suggestions, actual information. Not one-liners.
7. Include 4-8 exchanges per topic, varying in length.
8. Do NOT add any information not in the outline.
9. Do NOT use words like "recurring", "pattern", "persistent", "again"
   when referring to parking or any frustration.
10. The conversation should span several hours of the day with time gaps
    between topics (e.g., morning check-in, midday question, evening follow-up).
"""

    response = client.chat.completions.create(
        model=RENDERER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4000,
    )

    return response.choices[0].message.content.strip()


def render_episode_clean(outline: dict, spec: dict) -> str:
    """Re-render a distractor outline with explicit anti-contamination instructions."""
    fmt = spec["episodes"]["format"].strip()
    voice = spec["scenario"]["voice"].strip()
    target_words = spec["episodes"]["target_words"]
    date = outline.get("date", "2025-01-06")

    # Strip any parking-related topics from the outline before re-rendering
    cleaned_topics = []
    for topic in outline.get("conversation_topics", []):
        topic_text = json.dumps(topic).lower()
        if not DISTRACTOR_BLOCKLIST.search(topic_text):
            cleaned_topics.append(topic)
    if cleaned_topics:
        outline = {**outline, "conversation_topics": cleaned_topics}

    system = (
        "You are formatting a structured outline into a natural chat transcript "
        "between a user and their personal assistant. Output ONLY the chat text. "
        "No JSON, no markdown fences, no commentary about the outline. "
        "Write LONG, detailed conversations — at least 600 words. "
        "ABSOLUTELY DO NOT mention parking, driving, cars, meters, tickets, "
        "curbs, street cleaning, or any transportation logistics."
    )

    prompt = f"""## Format Instructions
{fmt}

## Voice
{voice}

## Target Length
At least 600 words, ideally around {target_words} words.

## Date
{date}

## Episode Outline
{json.dumps(outline, indent=2)}

## CRITICAL CONTENT RESTRICTION
This is a DISTRACTOR episode. It must contain ZERO mentions of:
- Parking (any form), driving, cars, meters, tickets, curbs, towing
- Street cleaning, traffic, commuting
- Any transportation logistics whatsoever

If any topic in the outline mentions these, SKIP that topic entirely
and expand the remaining topics with more detail instead.

## Rendering Rules
1. Format: [HH:MM] User: message / [HH:MM] Assistant: response
2. 4-8 exchanges per topic with real back-and-forth
3. Natural transitions between topics
4. Casual voice, detailed assistant responses
"""

    response = client.chat.completions.create(
        model=RENDERER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4000,
    )

    return response.choices[0].message.content.strip()


def resolve_questions(spec: dict) -> list[dict]:
    """Resolve question key_fact references to full fact text."""
    kf_map = {kf["id"]: kf["fact"].strip() for kf in spec["key_facts"]}

    questions = []
    for q in spec["questions"]:
        resolved_facts = [kf_map[fid] for fid in q["ground_truth"].get("key_facts", []) if fid in kf_map]
        questions.append({
            "question_id": q["id"],
            "scope_id": spec["scope_id"],
            "checkpoint_after": q["checkpoint_after"],
            "question_type": q["type"],
            "prompt": q["prompt"].strip(),
            "ground_truth": {
                "canonical_answer": q["ground_truth"]["canonical_answer"].strip(),
                "key_facts": resolved_facts,
                "evidence": q["ground_truth"].get("evidence", []),
            },
        })

    return questions


def main():
    spec = load_spec()
    ep_count = spec["episodes"]["count"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes_dir = OUT_DIR / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    # --- Stage 1: Plan (resume-aware) ---
    signal_outline_path = OUT_DIR / "signal_outline.json"
    if signal_outline_path.exists():
        print("Signal outlines already exist, loading...", flush=True)
        signal_outlines = json.load(open(signal_outline_path))["episodes"]
    else:
        signal_outlines = plan_signal_episodes(spec)
        with open(signal_outline_path, "w") as f:
            json.dump({"episodes": signal_outlines}, f, indent=2)

    # Plan distractors (resume-aware per theme)
    themes = spec["distractors"]["themes"]
    distractor_count = spec["distractors"]["count"]
    per_theme = distractor_count // len(themes)
    remainder = distractor_count % len(themes)

    all_distractor_outlines: list[tuple[str, dict]] = []
    for i, theme in enumerate(themes):
        n = per_theme + (1 if i < remainder else 0)
        outline_path = OUT_DIR / f"distractor_outline_{theme['id']}.json"
        if outline_path.exists():
            print(f"  Distractor outlines for '{theme['id']}' exist, loading...", flush=True)
            outlines = json.load(open(outline_path))["episodes"][:n]
        else:
            outlines = plan_distractor_episodes(spec, theme, n)
            with open(outline_path, "w") as f:
                json.dump({"episodes": outlines}, f, indent=2)
        for outline in outlines:
            all_distractor_outlines.append((theme["id"], outline))

    # --- Stage 2: Render (resume-aware — skip already-rendered) ---
    print(f"\nRendering {len(signal_outlines)} signal episodes...", flush=True)
    for i, outline in enumerate(signal_outlines):
        filename = f"signal_{i+1:03d}.txt"
        filepath = episodes_dir / filename
        if filepath.exists():
            word_count = len(filepath.read_text().split())
            print(f"  {filename}: {word_count} words (cached)", flush=True)
            continue
        text = render_episode(outline, spec)
        filepath.write_text(text)
        word_count = len(text.split())
        print(f"  {filename}: {word_count} words", flush=True)

    print(f"\nRendering {len(all_distractor_outlines)} distractor episodes...", flush=True)
    theme_counts: dict[str, int] = {}
    contaminated = []
    for theme_id, outline in all_distractor_outlines:
        theme_counts[theme_id] = theme_counts.get(theme_id, 0) + 1
        idx = theme_counts[theme_id]
        filename = f"distractor_{theme_id}_{idx:03d}.txt"
        filepath = episodes_dir / filename

        if filepath.exists():
            word_count = len(filepath.read_text().split())
            print(f"  {filename}: {word_count} words (cached)", flush=True)
            continue

        text = render_episode(outline, spec)

        # Check for contamination
        matches = DISTRACTOR_BLOCKLIST.findall(text)
        if matches:
            print(f"  !! {filename}: contaminated ({matches}), re-rendering...", flush=True)
            text = render_episode_clean(outline, spec)
            matches2 = DISTRACTOR_BLOCKLIST.findall(text)
            if matches2:
                contaminated.append(f"{filename} ({matches2})")
                print(f"  !! STILL contaminated after retry: {matches2}", flush=True)

        filepath.write_text(text)
        word_count = len(text.split())
        print(f"  {filename}: {word_count} words", flush=True)

    if contaminated:
        print(f"\n!! WARNING: {len(contaminated)} distractors still contaminated after retry:")
        for c in contaminated:
            print(f"   {c}")

    # --- Questions ---
    questions = resolve_questions(spec)
    with open(OUT_DIR / "questions.json", "w") as f:
        json.dump(questions, f, indent=2)
    print(f"\nWrote {len(questions)} questions to questions.json", flush=True)

    # --- Summary ---
    signal_files = sorted(episodes_dir.glob("signal_*.txt"))
    distractor_files = sorted(episodes_dir.glob("distractor_*.txt"))
    total_words = sum(len(f.read_text().split()) for f in signal_files + distractor_files)
    print(f"\nDone: {len(signal_files)} signal + {len(distractor_files)} distractor episodes")
    print(f"Total words: {total_words}")


if __name__ == "__main__":
    main()
