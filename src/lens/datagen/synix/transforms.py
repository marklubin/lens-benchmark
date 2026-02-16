"""Custom synix transforms for the LENS datagen pipeline.

Transforms:
  - LoadSpec: Level 0 (Source) — reads spec.yaml from source_dir
  - PlanOutline: Level 1 — planner that produces per-episode data sheets
  - RenderSignalEpisodes: Level 1 — blind renderer for signal data sheets
  - RenderDistractorEpisodes: Level 1 — blind renderer for distractor data sheets
  - ResolveQuestions: Level 2 — resolves phase-relative refs to episode IDs
  - AuditKeyFacts: Level 2 — checks key fact coverage across episodes
"""
from __future__ import annotations

import json
from datetime import timedelta

from synix import Source, Transform
from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact

import spec_utils
import prompt_utils
import scoring


MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Level 0: LoadSpec (Source — no inputs, reads from source_dir)
# ---------------------------------------------------------------------------


class LoadSpec(Source):
    """Read spec.yaml from source_dir, validate, and emit a single spec artifact."""

    def load(self, config: dict) -> list[Artifact]:
        source_dir = config.get("source_dir", ".")
        from pathlib import Path
        spec_path = Path(source_dir) / "spec.yaml"

        spec = spec_utils.load_spec(spec_path)
        spec_utils.validate_spec_or_raise(spec)

        spec_hash = spec_utils.compute_spec_hash(spec_path)
        spec["_spec_hash"] = spec_hash

        content = json.dumps(spec, indent=2)
        return [Artifact(
            label="spec",
            artifact_type="scope_spec",
            content=content,
            input_ids=[spec_hash],
            metadata={"scope_id": spec["scope_id"], "domain": spec.get("domain", "")},
        )]


# ---------------------------------------------------------------------------
# Level 1: PlanOutline (full-context planner → per-episode data sheets)
# ---------------------------------------------------------------------------


class PlanOutline(Transform):
    """Stage 1: Full-context planner that produces per-episode data sheets.

    The planner sees the complete spec (arc, key facts, signal placements)
    and outputs structured JSON data sheets — one per episode — with concrete
    metric values. Signal is encoded as metric progressions (numbers only).

    split() produces one group for signal planning, plus one group per
    distractor theme.
    """

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        spec = json.loads(_find_artifact(inputs, "scope_spec").content)
        groups: list[tuple[list[Artifact], dict]] = [
            (inputs, {"_plan_type": "signal"}),
        ]
        dc = spec.get("distractors")
        if dc and dc["count"] > 0:
            themes = dc["themes"]
            per_theme = [dc["count"] // len(themes)] * len(themes)
            for i in range(dc["count"] % len(themes)):
                per_theme[i] += 1
            for idx, theme in enumerate(themes):
                if per_theme[idx] > 0:
                    groups.append((inputs, {
                        "_plan_type": "distractor",
                        "_theme_idx": idx,
                        "_theme_count": per_theme[idx],
                    }))
        return groups

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        if config.get("_plan_type") == "signal":
            return self._plan_signal(inputs, config)
        else:
            return self._plan_distractor(inputs, config)

    def _plan_signal(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        """One LLM call → signal_outline artifact with all episode data sheets."""
        spec_artifact = _find_artifact(inputs, "scope_spec")
        spec = json.loads(spec_artifact.content)
        client = _get_llm_client(config)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt = prompt_utils.build_plan_signal_prompt(spec)
                response = _logged_complete(
                    client, config,
                    messages=[
                        {"role": "system", "content": prompt_utils.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    artifact_desc="plan-signal-outline",
                )

                parsed = _parse_json_response(response.content)
                if "episodes" not in parsed or not isinstance(parsed["episodes"], list):
                    raise ValueError("LLM response missing 'episodes' list")

                return [Artifact(
                    label="signal_outline",
                    artifact_type="signal_outline",
                    content=json.dumps(parsed, indent=2),
                    input_ids=[spec_artifact.artifact_id],
                    metadata={"episode_count": len(parsed["episodes"])},
                )]

            except (ValueError, json.JSONDecodeError, KeyError) as e:
                import logging
                logging.warning(
                    "Signal outline attempt %d/%d failed: %s (response length: %d chars)",
                    attempt, MAX_RETRIES, e,
                    len(response.content) if response else 0,
                )
                if attempt == MAX_RETRIES:
                    return [Artifact(
                        label="signal_outline",
                        artifact_type="signal_outline",
                        content=json.dumps({"episodes": []}, indent=2),
                        input_ids=[spec_artifact.artifact_id],
                        metadata={"episode_count": 0},
                    )]

        return []  # unreachable but satisfies type checker

    def _plan_distractor(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        """One LLM call per theme → distractor_outline artifact."""
        spec_artifact = _find_artifact(inputs, "scope_spec")
        spec = json.loads(spec_artifact.content)
        dc = spec["distractors"]
        theme_idx = config["_theme_idx"]
        theme = dc["themes"][theme_idx]
        count = config.get("_theme_count", 0)
        client = _get_llm_client(config)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt = prompt_utils.build_plan_distractor_prompt(spec, theme, count)
                response = _logged_complete(
                    client, config,
                    messages=[
                        {"role": "system", "content": prompt_utils.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    artifact_desc=f"plan-distractor-{theme['id']}",
                )

                parsed = _parse_json_response(response.content)
                if "episodes" not in parsed or not isinstance(parsed["episodes"], list):
                    raise ValueError("LLM response missing 'episodes' list")

                return [Artifact(
                    label=f"distractor_outline_{theme['id']}",
                    artifact_type="distractor_outline",
                    content=json.dumps(parsed, indent=2),
                    input_ids=[spec_artifact.artifact_id],
                    metadata={
                        "theme": theme["id"],
                        "theme_index": theme_idx,
                        "episode_count": len(parsed["episodes"]),
                    },
                )]

            except (ValueError, json.JSONDecodeError, KeyError):
                if attempt == MAX_RETRIES:
                    return [Artifact(
                        label=f"distractor_outline_{theme['id']}",
                        artifact_type="distractor_outline",
                        content=json.dumps({"episodes": []}, indent=2),
                        input_ids=[spec_artifact.artifact_id],
                        metadata={
                            "theme": theme["id"],
                            "theme_index": theme_idx,
                            "episode_count": 0,
                        },
                    )]

        return []  # unreachable


# ---------------------------------------------------------------------------
# Level 1: RenderSignalEpisodes (blind renderer — one call per episode)
# ---------------------------------------------------------------------------


class RenderSignalEpisodes(Transform):
    """Stage 2: Blind renderer that formats signal data sheets into log entries.

    Receives ONLY shared context (voice, format, entity names) and one data sheet.
    Does NOT receive the arc, key facts, signal placements, or questions.

    split() returns one group per episode brief for parallel execution.
    """

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        outline = _find_artifact(inputs, "signal_outline")
        briefs = json.loads(outline.content).get("episodes", [])
        return [(inputs, {"_brief_idx": i}) for i in range(len(briefs))]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        spec_artifact = _find_artifact(inputs, "scope_spec")
        spec = json.loads(spec_artifact.content)
        outline = _find_artifact(inputs, "signal_outline")
        briefs = json.loads(outline.content).get("episodes", [])

        brief_idx = config.get("_brief_idx", 0)
        if brief_idx >= len(briefs):
            return []

        brief = briefs[brief_idx]
        global_idx = brief.get("index", brief_idx + 1)
        client = _get_llm_client(config)

        # Determine phase for this episode
        phase = spec_utils.get_phase_for_episode(spec, global_idx)
        phase_id = phase["id"] if phase else "unknown"
        signal_density = phase["signal_density"] if phase else "none"

        text = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt = prompt_utils.build_render_prompt(spec, brief, episode_type="signal")
                response = _logged_complete(
                    client, config,
                    messages=[
                        {"role": "system", "content": prompt_utils.RENDER_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    artifact_desc=f"render-signal-ep-{global_idx:03d}",
                )
                text = response.content.strip()
                break
            except Exception:
                if attempt == MAX_RETRIES:
                    text = ""

        eid = spec_utils.make_episode_id(spec["scope_id"], global_idx)
        return [Artifact(
            label=eid,
            artifact_type="signal_episode",
            content=text,
            input_ids=[spec_artifact.artifact_id, outline.artifact_id],
            metadata={
                "episode_id": eid,
                "scope_id": spec["scope_id"],
                "timestamp": spec_utils.make_episode_timestamp(spec, global_idx),
                "phase": phase_id,
                "signal_density": signal_density,
                "episode_type": "signal",
            },
        )]


# ---------------------------------------------------------------------------
# Level 1: RenderDistractorEpisodes (blind renderer for distractors)
# ---------------------------------------------------------------------------


class RenderDistractorEpisodes(Transform):
    """Stage 2: Blind renderer that formats distractor data sheets into log entries.

    Same isolation as RenderSignalEpisodes but for distractor themes.
    split() returns one group per episode brief across all distractor outlines.
    """

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        outlines = [a for a in inputs if a.artifact_type == "distractor_outline"]
        groups: list[tuple[list[Artifact], dict]] = []

        for outline in outlines:
            briefs = json.loads(outline.content).get("episodes", [])
            theme = outline.metadata.get("theme", "unknown")
            theme_idx = outline.metadata.get("theme_index", 0)
            for i in range(len(briefs)):
                groups.append((inputs, {
                    "_outline_label": outline.label,
                    "_brief_idx": i,
                    "_theme": theme,
                    "_theme_idx": theme_idx,
                }))

        return groups if groups else [(inputs, {"_no_distractors": True})]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        if config.get("_no_distractors"):
            return []

        spec_artifact = _find_artifact(inputs, "scope_spec")
        spec = json.loads(spec_artifact.content)

        outline_label = config["_outline_label"]
        outline = None
        for a in inputs:
            if a.label == outline_label:
                outline = a
                break
        if outline is None:
            return []

        briefs = json.loads(outline.content).get("episodes", [])
        brief_idx = config.get("_brief_idx", 0)
        if brief_idx >= len(briefs):
            return []

        brief = briefs[brief_idx]
        theme = config.get("_theme", "unknown")
        theme_idx = config.get("_theme_idx", 0)
        local_idx = brief_idx + 1
        client = _get_llm_client(config)

        text = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt = prompt_utils.build_render_prompt(spec, brief, episode_type="distractor")
                response = _logged_complete(
                    client, config,
                    messages=[
                        {"role": "system", "content": prompt_utils.RENDER_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    artifact_desc=f"render-distractor-{theme}-{local_idx:03d}",
                )
                text = response.content.strip()
                break
            except Exception:
                if attempt == MAX_RETRIES:
                    text = ""

        scope_id = spec["scope_id"]
        label = f"{scope_id}_dx_{theme}_{local_idx:03d}"

        return [Artifact(
            label=label,
            artifact_type="distractor_episode",
            content=text,
            input_ids=[spec_artifact.artifact_id, outline.artifact_id],
            metadata={
                "episode_type": "distractor",
                "theme": theme,
                "theme_index": theme_idx,
                "local_index": local_idx,
                "scope_id": scope_id,
            },
        )]


# ---------------------------------------------------------------------------
# Level 2: resolve_questions
# ---------------------------------------------------------------------------


class ResolveQuestions(Transform):
    """Resolve phase-relative references in questions to concrete episode IDs."""

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        return [(inputs, {})]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        spec_artifact = _find_artifact(inputs, "scope_spec")
        spec = json.loads(spec_artifact.content)

        kf_map = {kf["id"]: kf["fact"] for kf in spec["key_facts"]}
        artifacts: list[Artifact] = []

        for q in spec["questions"]:
            # Resolve evidence refs
            evidence_refs = []
            for ref in q["ground_truth"]["evidence"]:
                try:
                    evidence_refs.append(spec_utils.resolve_phase_ref(ref, spec))
                except ValueError:
                    pass

            # Map key fact IDs to text
            key_fact_texts = [kf_map[fid] for fid in q["ground_truth"]["key_facts"] if fid in kf_map]

            question_data = {
                "question_id": q["id"],
                "scope_id": spec["scope_id"],
                "checkpoint_after": q["checkpoint_after"],
                "question_type": q["type"],
                "prompt": q["prompt"],
                "ground_truth": {
                    "canonical_answer": q["ground_truth"]["canonical_answer"],
                    "required_evidence_refs": evidence_refs,
                    "key_facts": key_fact_texts,
                },
            }

            input_ids = [spec_artifact.artifact_id]
            # Also track dependency on signal episodes referenced in evidence
            for ref in evidence_refs:
                for a in inputs:
                    if a.artifact_type == "signal_episode" and a.label == ref:
                        input_ids.append(a.artifact_id)

            artifacts.append(Artifact(
                label=f"q_{q['id']}",
                artifact_type="question",
                content=json.dumps(question_data, indent=2),
                input_ids=input_ids,
                metadata={
                    "question_id": q["id"],
                    "question_type": q["type"],
                    "checkpoint_after": q["checkpoint_after"],
                    "scope_id": spec["scope_id"],
                },
            ))

        return artifacts


# ---------------------------------------------------------------------------
# Level 2: audit_key_facts
# ---------------------------------------------------------------------------


class AuditKeyFacts(Transform):
    """Audit key fact coverage across signal episodes."""

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        return [(inputs, {})]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        spec_artifact = _find_artifact(inputs, "scope_spec")
        spec = json.loads(spec_artifact.content)

        signal_episodes = [a for a in inputs if a.artifact_type == "signal_episode"]

        coverage: dict = {}
        total_targets = 0
        total_found = 0

        for kf in spec["key_facts"]:
            # Collect target episode IDs
            target_ids = []
            all_refs = []
            if kf["first_appears"]:
                all_refs.append(kf["first_appears"])
            all_refs.extend(kf["reinforced_in"])

            for ref in all_refs:
                try:
                    eid = spec_utils.resolve_phase_ref(ref, spec)
                    target_ids.append(eid)
                except ValueError:
                    pass

            # Check which episodes contain evidence of the fact.
            # Uses keyword-based matching: each key fact has associated
            # indicator terms that appear in terse metric-heavy log entries.
            found_in = []
            indicators = _fact_indicators(kf)
            for eid in target_ids:
                for ep in signal_episodes:
                    if ep.label == eid:
                        text_lower = ep.content.lower()
                        matched = sum(1 for ind in indicators if ind in text_lower)
                        if matched >= max(1, len(indicators) * 0.3):
                            found_in.append(eid)
                        break

            coverage[kf["id"]] = {
                "target_episodes": target_ids,
                "found_in": found_in,
            }
            total_targets += len(target_ids)
            total_found += len(found_in)

        audit_report = {
            "key_fact_coverage": coverage,
            "hit_rate": total_found / total_targets if total_targets > 0 else 1.0,
            "total_targets": total_targets,
            "total_found": total_found,
        }

        return [Artifact(
            label="key-fact-audit",
            artifact_type="audit",
            content=json.dumps(audit_report, indent=2),
            input_ids=[spec_artifact.artifact_id] + [a.artifact_id for a in signal_episodes],
            metadata={
                "hit_rate": audit_report["hit_rate"],
                "total_targets": total_targets,
                "total_found": total_found,
            },
        )]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact_indicators(kf: dict) -> list[str]:
    """Return indicator terms for a key fact that match terse log entries.

    Key facts like "geo-lookup API latency increasing" won't appear verbatim
    in metric-heavy logs. Instead, we look for domain-relevant terms that
    indicate the fact's presence (e.g., "geo_lookup", "p99", "exhaustion").
    """
    fact_lower = kf["fact"].lower()
    kf_id = kf.get("id", "").lower()

    # Build indicator list from fact text + id
    indicators: list[str] = []

    # Extract meaningful words (>3 chars, not stopwords)
    stopwords = {"the", "and", "for", "that", "this", "with", "from", "not", "are", "was", "has", "its"}
    for word in fact_lower.replace("-", " ").replace("_", " ").split():
        if len(word) > 3 and word not in stopwords:
            indicators.append(word)

    # Add common metric-log variants
    if "geo" in fact_lower or "geo" in kf_id:
        indicators.extend(["geo_lookup", "geo-lookup", "geo"])
    if "latency" in fact_lower:
        indicators.extend(["p99", "latency", "p95"])
    if "pool" in fact_lower or "exhaust" in fact_lower:
        indicators.extend(["exhaustion", "pool", "waiting"])
    if "retry" in fact_lower or "retries" in fact_lower or "retry" in kf_id:
        indicators.extend(["retry", "retries", "retry_count", "retry_rate"])
    if "service-b" in fact_lower or "service_b" in kf_id:
        indicators.extend(["service-b", "service_b"])
    if "deploy" in fact_lower or "deploy" in kf_id:
        indicators.extend(["deploy", "rollback", "service-c", "service_c"])
    if "red_herring" in kf_id or "red herring" in fact_lower:
        indicators.extend(["rollback", "service-c"])

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for ind in indicators:
        if ind not in seen:
            seen.add(ind)
            unique.append(ind)

    return unique


def _find_artifact(inputs: list[Artifact], artifact_type: str) -> Artifact:
    """Find the first artifact of a given type in inputs."""
    for a in inputs:
        if a.artifact_type == artifact_type:
            return a
    raise ValueError(f"No artifact of type {artifact_type!r} found in inputs")


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def _expand_episode(
    client, config: dict, text: str, target_words: int,
    temperature: float, seed: int,
) -> str:
    """Expand a short episode via LLM. Up to 2 retries."""
    for _ in range(2):
        prompt = prompt_utils.build_expand_prompt(text, target_words)
        response = _logged_complete(
            client, config,
            messages=[
                {"role": "system", "content": prompt_utils.EXPAND_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            artifact_desc="episode-expand",
        )
        expanded = response.content.strip()
        if scoring.compute_word_count(expanded) >= target_words * 0.7:
            return expanded
        text = expanded
    return text
