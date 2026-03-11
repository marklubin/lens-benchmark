"""Scoring V2 — three-metric scoring pipeline.

Primary composite: 0.5 * fact_f1 + 0.3 * evidence_support + 0.2 * citation_validity

Metrics:
  fact_f1:            LLM-judged key fact presence (precision × recall → F1)
  evidence_support:   LLM-judged evidence quality backing the answer
  citation_validity:  Mechanical check — do cited refs resolve to real artifacts?
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from bench.broker import ModalBroker
from bench.dataset import QuestionData
from bench.schemas import Diagnostics, EventType, ScoreRecord
from bench.state import EventWriter, StateStore

logger = logging.getLogger(__name__)


def _strip_think(text: str) -> str:
    """Strip thinking preamble from judge model output.

    Handles both <think>...</think> XML tags and plain-text
    "Thinking Process:" preamble that Qwen3.5 sometimes produces.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip plain-text thinking preamble — take only the last line
    # which should contain the actual verdict (YES/NO or a score)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        # If the last line looks like a verdict, use it; otherwise return full text
        last = lines[-1].upper()
        if last in ("YES", "NO", "YES.", "NO.") or re.match(r"^\d+\.?\d*$", lines[-1]):
            return lines[-1]
    return text


def _extract_yes_no(text: str) -> str | None:
    """Extract YES or NO from judge response. Returns None if ambiguous."""
    text = text.strip().upper()
    if text.startswith("YES"):
        return "YES"
    if text.startswith("NO"):
        return "NO"
    # Search anywhere in the text for a clear verdict
    if re.search(r"\bYES\b", text) and not re.search(r"\bNO\b", text):
        return "YES"
    if re.search(r"\bNO\b", text) and not re.search(r"\bYES\b", text):
        return "NO"
    return None

FACT_CHECK_PROMPT = """\
You are a fact-checking judge. Determine whether the following fact is \
clearly stated or implied in the given answer. Accept paraphrases and \
equivalent formulations — the answer does not need to use the exact same words.

Fact to check:
{fact}

Answer to evaluate:
{answer}

/no_think
Respond with exactly one word: YES or NO."""

EVIDENCE_SUPPORT_PROMPT = """\
You are an evidence quality judge. Evaluate how well the cited evidence \
supports the claims in the answer.

Question: {question}

Answer: {answer}

Evidence passages:
{evidence}

Rate the evidence support on a scale from 0.0 to 1.0:
- 1.0: Evidence directly and completely supports all claims
- 0.7: Evidence supports most claims with minor gaps
- 0.4: Evidence partially supports claims
- 0.1: Evidence barely supports claims
- 0.0: No evidence supports the claims

/no_think
Respond with a single decimal number between 0.0 and 1.0."""


class ScorerV2:
    """Three-metric scorer for V2 benchmark answers."""

    def __init__(
        self,
        broker: ModalBroker,
        *,
        judge_model: str = "Qwen/Qwen3.5-35B-A3B",
        scorer_version: str = "v2.0",
    ) -> None:
        self._broker = broker
        self._judge_model = judge_model
        self._scorer_version = scorer_version

    def score_answer(
        self,
        question: QuestionData,
        answer: dict[str, Any],
        release: Any | None = None,
    ) -> ScoreRecord:
        """Score a single answer against ground truth.

        Args:
            question: The question with ground truth.
            answer: The saved answer dict (answer_text, cited_refs, etc.).
            release: Optional Synix Release for citation validation.

        Returns:
            A fully populated ScoreRecord.
        """
        answer_text = answer.get("answer_text", "")
        cited_refs = answer.get("cited_refs", [])

        # 1. Fact F1
        fact_p, fact_r, fact_f1 = self._fact_score(question, answer_text)

        # 2. Citation validity (mechanical)
        citation_val = self._citation_validity(cited_refs, release)

        # 3. Evidence support (LLM-judged)
        evidence_sup = self._evidence_support(question, answer_text, cited_refs, release)

        # Composite
        primary = 0.5 * fact_f1 + 0.3 * evidence_sup + 0.2 * citation_val

        return ScoreRecord(
            score_id=str(uuid.uuid4()),
            study_id=answer.get("study_id", ""),
            run_id=answer.get("run_id", ""),
            question_id=question.question_id,
            scope_id=question.scope_id,
            policy_id=answer.get("policy_id", ""),
            checkpoint_id=answer.get("checkpoint_id", ""),
            bank_manifest_id=answer.get("bank_manifest_id", ""),
            fact_precision=fact_p,
            fact_recall=fact_r,
            fact_f1=fact_f1,
            evidence_support=evidence_sup,
            citation_validity=citation_val,
            primary_score=round(primary, 4),
            diagnostics=Diagnostics(
                tool_count=answer.get("tool_calls_made", 0),
                prompt_tokens=answer.get("prompt_tokens", 0),
                completion_tokens=answer.get("completion_tokens", 0),
                wall_time_ms=answer.get("wall_time_ms", 0.0),
            ),
            scored_at=datetime.now(timezone.utc),
            scorer_version=self._scorer_version,
            judge_model=self._judge_model,
        )

    def _fact_score(
        self,
        question: QuestionData,
        answer_text: str,
    ) -> tuple[float, float, float]:
        """Score fact presence using LLM judge.

        For each key fact in ground truth, ask the LLM if it's present
        in the answer. Returns (precision, recall, f1).
        """
        key_facts = question.ground_truth.get("key_facts", [])
        if not key_facts:
            return (1.0, 1.0, 1.0)

        if not answer_text.strip():
            return (0.0, 0.0, 0.0)

        facts_found = 0
        for fact in key_facts:
            prompt = FACT_CHECK_PROMPT.format(fact=fact, answer=answer_text)
            try:
                response = self._broker.chat_completion(
                    model=self._judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content or ""
                verdict = _extract_yes_no(_strip_think(raw))
                if verdict == "YES":
                    facts_found += 1
                elif verdict is None:
                    logger.warning("Ambiguous judge response for fact %r: %r", fact[:60], raw[:100])
            except Exception as exc:
                logger.error("Fact check LLM call failed: %s", exc)

        recall = facts_found / len(key_facts) if key_facts else 0.0
        # For V2 simplicity: precision = recall (we don't count spurious facts)
        precision = recall
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return (round(precision, 4), round(recall, 4), round(f1, 4))

    def _citation_validity(
        self,
        cited_refs: list[str],
        release: Any | None,
    ) -> float:
        """Check what fraction of cited refs resolve to real artifacts.

        Mechanical — no LLM call needed.
        """
        if not cited_refs:
            # No citations: score 0 (citations are expected)
            return 0.0

        if release is None:
            # Can't validate without release — give benefit of doubt
            return 1.0

        valid = 0
        for ref in cited_refs:
            resolved = self._resolve_ref(ref, release)
            if resolved:
                valid += 1

        return round(valid / len(cited_refs), 4) if cited_refs else 0.0

    @staticmethod
    def _resolve_ref(ref: str, release: Any) -> bool:
        """Try to resolve a citation ref to a real artifact.

        The model may cite in shortened forms (signal_003, chunk-signal_003-abc123)
        so we try the exact ref first, then common prefix expansions.
        """
        candidates = [ref]
        # Model often drops the t-text- prefix
        if ref.startswith("signal_") or ref.startswith("distractor_"):
            candidates.append(f"t-text-{ref}")
        if ref.startswith("chunk-"):
            candidates.append(ref.replace("chunk-", "chunks-t-text-", 1))
        # signal_003-abcdef12 → chunks-t-text-signal_003-abcdef12
        if re.match(r"^(signal|distractor)_.*-[a-f0-9]+$", ref):
            candidates.append(f"chunks-t-text-{ref}")

        for candidate in candidates:
            try:
                art = release.artifact(candidate)
                if art is not None:
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _resolve_artifact(ref: str, release: Any) -> Any:
        """Try to resolve a citation ref to an artifact object.

        Same logic as _resolve_ref but returns the artifact instead of bool.
        """
        candidates = [ref]
        if ref.startswith("signal_") or ref.startswith("distractor_"):
            candidates.append(f"t-text-{ref}")
        if ref.startswith("chunk-"):
            candidates.append(ref.replace("chunk-", "chunks-t-text-", 1))
        if re.match(r"^(signal|distractor)_.*-[a-f0-9]+$", ref):
            candidates.append(f"chunks-t-text-{ref}")

        for candidate in candidates:
            try:
                art = release.artifact(candidate)
                if art is not None:
                    return art
            except Exception:
                continue
        return None

    def _evidence_support(
        self,
        question: QuestionData,
        answer_text: str,
        cited_refs: list[str],
        release: Any | None,
    ) -> float:
        """LLM-judged evidence support score.

        Collects evidence text from cited refs and asks the LLM
        how well the evidence supports the answer.
        """
        if not answer_text.strip():
            return 0.0

        # Collect evidence text — try resolving shortened labels
        evidence_texts = []
        if release is not None and cited_refs:
            for ref in cited_refs[:10]:  # Cap evidence collection
                art = self._resolve_artifact(ref, release)
                if art is not None:
                    evidence_texts.append(f"[{ref}]: {art.content[:500]}")

        if not evidence_texts:
            # No evidence available — score based on answer alone
            # If no citations and no release, give a low default
            return 0.0

        evidence = "\n\n".join(evidence_texts)
        prompt = EVIDENCE_SUPPORT_PROMPT.format(
            question=question.prompt,
            answer=answer_text,
            evidence=evidence,
        )

        try:
            response = self._broker.chat_completion(
                model=self._judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            raw = response.choices[0].message.content or ""
            text = _strip_think(raw).strip()
            # Parse numeric score — find the last number (in case of preamble text)
            m = re.search(r"(\d+\.?\d*)", text)
            if m:
                score = float(m.group(1))
                return round(min(max(score, 0.0), 1.0), 4)
        except Exception as exc:
            logger.error("Evidence support LLM call failed: %s", exc)

        return 0.0

    def score_run(
        self,
        run_id: str,
        store: StateStore,
        questions: list[QuestionData],
        release_map: dict[str, Any] | None = None,
        *,
        study_id: str = "",
        policy_id: str = "",
        event_writer: EventWriter | None = None,
    ) -> list[ScoreRecord]:
        """Score all answers for a run.

        Args:
            run_id: The run to score.
            store: State store with saved answers.
            questions: All questions in the scope.
            release_map: Optional map of checkpoint_id → Release for citation validation.
            study_id: For score record metadata.
            policy_id: For score record metadata.
            event_writer: Optional event writer for scoring events.

        Returns:
            List of ScoreRecords, one per answered question.
        """
        if event_writer:
            event_writer.emit(EventType.scoring_started, run_id=run_id)

        scores: list[ScoreRecord] = []

        for question in questions:
            answer = store.get_answer(run_id, question.question_id)
            if answer is None:
                logger.warning("No answer for %s in run %s, skipping", question.question_id, run_id)
                continue

            # Enrich answer with metadata for ScoreRecord
            answer["run_id"] = run_id
            answer["study_id"] = study_id
            answer["policy_id"] = policy_id

            # Get release for citation validation
            checkpoint_id = answer.get("checkpoint_id")
            release = release_map.get(checkpoint_id) if release_map and checkpoint_id else None

            score = self.score_answer(question, answer, release)
            store.save_score(score)
            scores.append(score)

            logger.info(
                "Scored %s: fact_f1=%.3f, evidence=%.3f, citation=%.3f, primary=%.3f",
                question.question_id, score.fact_f1, score.evidence_support,
                score.citation_validity, score.primary_score,
            )

        if event_writer:
            event_writer.emit(
                EventType.scoring_completed,
                run_id=run_id,
                payload={
                    "questions_scored": len(scores),
                    "mean_primary": round(sum(s.primary_score for s in scores) / len(scores), 4) if scores else 0.0,
                },
            )

        return scores
