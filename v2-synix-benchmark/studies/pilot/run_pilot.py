"""Smoke pilot — gated E2E validation for V2 benchmark.

Gates:
  0. Imports + SDK validation
  1. Build one bank (S08-cp06, chunks only, no LLM)
  2. Build one bank with synthesis (S08-cp06, requires Modal LLM)
  3. Run one question under null policy (no tools)
  4. Run one question under base policy (memory_search)
  5. Score answers from gates 3-4
  6. Full S08 (4 checkpoints × 4 policies)
  7. Full pilot (S08 + S10)

Usage:
  uv run python studies/pilot/run_pilot.py --gate 1       # run through gate 1 only
  uv run python studies/pilot/run_pilot.py --gate 7       # full pilot
  uv run python studies/pilot/run_pilot.py --gate 6 --dry-run  # dry run (no LLM calls)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bench.bank import BankBuilder, SYNIX_LABEL_PREFIX
from bench.broker import ModalBroker
from bench.dataset import ScopeData, load_scope
from bench.policy import create_policy
from bench.runner import StudyRunner
from bench.runtime import BenchmarkRuntime
from bench.agent import AgentHarness
from bench.scorer import ScorerV2
from bench.schemas import (
    BankStatus,
    EventType,
    RunStatus,
    StudyManifest,
)
from bench.state import EventWriter, StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pilot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCOPES_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "scopes"
SCOPE_DIRS = {
    "corporate_acquisition_08": SCOPES_ROOT / "08_corporate_acquisition",
    "clinical_trial_10": SCOPES_ROOT / "10_clinical_trial",
}
WORK_DIR = Path(__file__).resolve().parent / "work"
STATE_DB = WORK_DIR / "pilot_state.db"

LLM_BASE_URL = "https://synix--lens-llm-llm-serve.modal.run/v1"
EMBED_BASE_URL = "https://synix--lens-embed-serve.modal.run"
AGENT_MODEL = "Qwen/Qwen3.5-35B-A3B"
JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"
EMBED_MODEL = "Xenova/gte-modernbert-base"
POLICY_IDS = ["null", "policy_base", "policy_core", "policy_summary"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_study(scope_ids: list[str]) -> StudyManifest:
    """Create the pilot study manifest."""
    return StudyManifest(
        study_id=f"pilot-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        benchmark_version="v2",
        scope_ids=scope_ids,
        policy_ids=POLICY_IDS,
        agent_model=AGENT_MODEL,
        judge_model=JUDGE_MODEL,
        embedding_model=EMBED_MODEL,
        prompt_set_version="v2.0",
        scoring_version="v2.0",
        code_sha="local",
        artifact_family_set_version="v2.0-pilot",
        created_at=datetime.now(timezone.utc),
        notes="Smoke pilot: 2 scopes × 4 policies, gated validation",
    )


def _make_broker(*, cache_dir: Path | None = None) -> ModalBroker:
    return ModalBroker(
        llm_base_url=LLM_BASE_URL,
        embed_base_url=EMBED_BASE_URL,
        llm_api_key="unused",
        cache_enabled=True,
        cache_dir=cache_dir or WORK_DIR / "cache",
        default_extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def _print_gate(gate: int, name: str, status: str):
    icon = "PASS" if status == "pass" else "FAIL" if status == "fail" else "SKIP"
    logger.info("Gate %d [%s]: %s", gate, icon, name)


def _report_summary(store: StateStore, study_id: str):
    """Print a summary of all runs and scores."""
    events = store.get_events(study_id=study_id)

    # Collect run completions
    run_events = [e for e in events if e.event_type == EventType.run_completed]
    score_events = [e for e in events if e.event_type == EventType.scoring_completed]
    bank_events = [e for e in events if e.event_type == EventType.bank_build_completed]

    logger.info("=" * 70)
    logger.info("PILOT SUMMARY")
    logger.info("=" * 70)
    logger.info("Study: %s", study_id)
    logger.info("Banks built: %d", len(bank_events))
    logger.info("Runs completed: %d", len(run_events))
    logger.info("Scoring rounds: %d", len(score_events))

    for se in score_events:
        payload = se.payload or {}
        logger.info(
            "  Scored run %s: %d questions, mean_primary=%.4f",
            se.run_id or "?",
            payload.get("questions_scored", 0),
            payload.get("mean_primary", 0.0),
        )

    # Cost summary from bank builds
    total_bank_time = sum(
        (e.payload or {}).get("total_time", 0.0) for e in bank_events
    )
    total_run_tokens = sum(
        (e.payload or {}).get("prompt_tokens", 0) + (e.payload or {}).get("completion_tokens", 0)
        for e in run_events
    )
    logger.info("Total bank build time: %.1fs", total_bank_time)
    logger.info("Total run tokens: %d", total_run_tokens)
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Gate Implementations
# ---------------------------------------------------------------------------


def gate_0() -> bool:
    """Verify imports and SDK compatibility."""
    logger.info("Gate 0: verifying imports and SDK...")
    try:
        import synix
        from synix.core.models import Pipeline, Source, SearchSurface, SynixSearch
        from synix.ext.chunk import Chunk
        from synix.ext.fold_synthesis import FoldSynthesis
        from synix.ext.group_synthesis import GroupSynthesis
        from synix.ext.reduce_synthesis import ReduceSynthesis

        assert synix.__version__ >= "0.20.0", f"synix {synix.__version__} < 0.20.0"
        logger.info("  synix %s OK", synix.__version__)
        logger.info("  All synthesis transforms importable")

        # Verify scope dirs exist
        for name, path in SCOPE_DIRS.items():
            assert path.exists(), f"Scope dir missing: {path}"
            logger.info("  Scope %s: %s", name, path)

        return True
    except Exception as e:
        logger.error("  FAILED: %s", e)
        return False


def gate_1() -> bool:
    """Build one bank (S08-cp06, chunks only, no LLM needed)."""
    logger.info("Gate 1: building S08-cp06 (chunks only, no LLM)...")
    import synix
    import tempfile

    scope = load_scope(SCOPE_DIRS["corporate_acquisition_08"])
    episodes = scope.episodes_up_to(6)
    logger.info("  %d episodes for cp06", len(episodes))

    with tempfile.TemporaryDirectory() as td:
        from synix.core.models import Pipeline, Source
        from synix.ext.chunk import Chunk

        ep_source = Source("episodes")
        chunks = Chunk("chunks", depends_on=[ep_source], chunk_size=1000, chunk_overlap=100)
        pipeline = Pipeline("gate1-test")
        pipeline.add(ep_source, chunks)

        project = synix.init(td, pipeline=pipeline)
        source = project.source("episodes")
        for ep in episodes:
            original = scope.scope_dir / "generated" / "episodes" / ep.filename
            source.add(original)

        result = project.build()
        logger.info("  Build: built=%d cached=%d time=%.1fs", result.built, result.cached, result.total_time)

        if result.built == 0:
            logger.error("  FAILED: built=0, no artifacts produced")
            return False

        project.release_to("bank")
        release = project.release("bank")

        # Count artifacts
        episode_count = 0
        chunk_count = 0
        for art in release.artifacts():
            if art.layer == "episodes":
                episode_count += 1
            elif art.layer == "chunks":
                chunk_count += 1

        logger.info("  Artifacts: %d episodes, %d chunks", episode_count, chunk_count)

        if episode_count != len(episodes):
            logger.error("  FAILED: expected %d episode artifacts, got %d", len(episodes), episode_count)
            return False

        # Verify episode labels have t-text- prefix
        sample_art = release.artifact(f"{SYNIX_LABEL_PREFIX}signal_001")
        logger.info("  Sample artifact label: %r, content=%d chars", sample_art.label, len(sample_art.content))

        logger.info("  Chunks per episode: ~%d (at chunk_size=1000)", chunk_count // max(episode_count, 1))
        return True


def gate_2(store: StateStore, broker: ModalBroker) -> bool:
    """Build S08-cp06 with all families (chunks + core_memory + summary). Requires Modal."""
    logger.info("Gate 2: building S08-cp06 with synthesis (requires Modal LLM)...")

    scope = load_scope(SCOPE_DIRS["corporate_acquisition_08"])
    study_id = "gate2-test"
    ew = EventWriter(store, study_id)

    builder = BankBuilder(store, broker, WORK_DIR / "banks")
    manifest = builder._build_checkpoint_bank(
        study_id=study_id,
        scope=scope,
        checkpoint=6,
        families=["chunks", "core_memory", "summary"],
        event_writer=ew,
    )

    logger.info("  Bank %s: status=%s built_at=%s", manifest.bank_manifest_id, manifest.status, manifest.built_at)

    if manifest.status != BankStatus.released:
        logger.error("  FAILED: bank status=%s, expected released", manifest.status)
        return False

    # Verify derived artifacts exist
    release = builder.open_release(manifest)

    try:
        core = release.artifact("core-memory")
        logger.info("  Core memory: %d chars", len(core.content))
        logger.info("  Core preview: %s", core.content[:200])
    except Exception as e:
        logger.error("  FAILED: core-memory artifact not found: %s", e)
        return False

    try:
        summary = release.artifact("summary")
        logger.info("  Summary: %d chars", len(summary.content))
        logger.info("  Summary preview: %s", summary.content[:200])
    except Exception as e:
        logger.error("  FAILED: summary artifact not found: %s", e)
        return False

    # Verify search works
    results = release.search("strategic options board meeting", mode="hybrid", limit=5)
    logger.info("  FTS results: %d", len(results))
    if results:
        logger.info("  Top hit: label=%r score=%.3f", results[0].label, results[0].score)

    return True


def gate_3(store: StateStore, broker: ModalBroker) -> str | None:
    """Run one question under null policy. Returns run_id or None on failure."""
    logger.info("Gate 3: answering one question under null policy...")

    scope = load_scope(SCOPE_DIRS["corporate_acquisition_08"])
    question = scope.questions[0]  # ca08_q01_longitudinal at cp06
    logger.info("  Question: %s — %s", question.question_id, question.prompt[:80])

    null_policy = create_policy("null", "v2.0")

    # Null policy: no release needed, no tools, no context
    runtime = BenchmarkRuntime(release=None, policy=null_policy)
    harness = AgentHarness(broker, runtime, model=AGENT_MODEL, max_turns=5, max_tool_calls=0)

    answer = harness.answer(question.prompt, question_id=question.question_id)
    logger.info("  Answer: %d chars, %d turns, %.0fms", len(answer.answer_text), len(answer.turns), answer.wall_time_ms)
    logger.info("  Preview: %s", answer.answer_text[:200])

    # Save to state store
    run_id = "gate3-null-test"
    store.save_answer(run_id, question.question_id, "cp06", {
        "answer_text": answer.answer_text,
        "cited_refs": answer.cited_refs,
        "tool_calls_made": answer.tool_calls_made,
        "prompt_tokens": answer.total_prompt_tokens,
        "completion_tokens": answer.total_completion_tokens,
        "wall_time_ms": answer.wall_time_ms,
    })

    logger.info("  Answer saved to state store under run_id=%s", run_id)
    return run_id


def gate_4(store: StateStore, broker: ModalBroker) -> str | None:
    """Run one question under base policy with memory_search. Returns run_id or None."""
    logger.info("Gate 4: answering one question under base policy (memory_search)...")

    # We need a release for search — check if gate 2 built one
    scope = load_scope(SCOPE_DIRS["corporate_acquisition_08"])
    question = scope.questions[0]

    builder = BankBuilder(store, broker, WORK_DIR / "banks")
    bank_id = f"bank-{scope.scope_id}-cp06-gate2-te"
    bank = store.get_bank(bank_id)
    if bank is None or bank.status != BankStatus.released:
        logger.error("  FAILED: gate 2 bank not found (run gate 2 first)")
        return None

    release = builder.open_release(bank)
    base_policy = create_policy("policy_base", "v2.0")
    runtime = BenchmarkRuntime(release, base_policy)
    harness = AgentHarness(broker, runtime, model=AGENT_MODEL, max_turns=5, max_tool_calls=5)

    answer = harness.answer(question.prompt, question_id=question.question_id)
    logger.info("  Answer: %d chars, %d tool calls, %d turns, %.0fms",
                len(answer.answer_text), answer.tool_calls_made, len(answer.turns), answer.wall_time_ms)
    logger.info("  Cited refs: %s", answer.cited_refs[:5])
    logger.info("  Preview: %s", answer.answer_text[:200])

    run_id = "gate4-base-test"
    store.save_answer(run_id, question.question_id, "cp06", {
        "answer_text": answer.answer_text,
        "cited_refs": answer.cited_refs,
        "tool_calls_made": answer.tool_calls_made,
        "prompt_tokens": answer.total_prompt_tokens,
        "completion_tokens": answer.total_completion_tokens,
        "wall_time_ms": answer.wall_time_ms,
    })

    logger.info("  Answer saved under run_id=%s", run_id)
    return run_id


def gate_5(store: StateStore, broker: ModalBroker) -> bool:
    """Score answers from gates 3 and 4."""
    logger.info("Gate 5: scoring answers from gates 3-4...")

    scope = load_scope(SCOPE_DIRS["corporate_acquisition_08"])
    question = scope.questions[0]
    scorer = ScorerV2(broker, judge_model=JUDGE_MODEL)

    for run_id, label in [("gate3-null-test", "null"), ("gate4-base-test", "base")]:
        answer = store.get_answer(run_id, question.question_id)
        if answer is None:
            logger.warning("  No answer for %s/%s, skipping", run_id, question.question_id)
            continue

        score = scorer.score_answer(question, answer, release=None)
        store.save_score(score)

        logger.info(
            "  %s: fact_f1=%.3f evidence=%.3f citation=%.3f primary=%.4f",
            label, score.fact_f1, score.evidence_support,
            score.citation_validity, score.primary_score,
        )

    return True


def gate_6(store: StateStore, broker: ModalBroker) -> bool:
    """Full S08: 4 checkpoints × 4 policies."""
    logger.info("Gate 6: full S08 run (4 checkpoints × 4 policies)...")

    scope_ids = ["corporate_acquisition_08"]
    study = _make_study(scope_ids)
    store.save_study(study)
    logger.info("  Study: %s", study.study_id)

    runner = StudyRunner(
        study=study,
        store=store,
        broker=broker,
        work_dir=WORK_DIR,
        scope_dirs={k: v for k, v in SCOPE_DIRS.items() if k in scope_ids},
        families=["chunks", "core_memory", "summary"],
        max_turns=10,
        max_tool_calls=20,
    )

    runs = runner.run_study()
    logger.info("  Completed %d runs", len(runs))

    # Build release_map for citation validation
    scope = load_scope(SCOPE_DIRS["corporate_acquisition_08"])
    builder = BankBuilder(store, broker, WORK_DIR / "banks")
    release_map: dict[str, object] = {}
    for cp in scope.checkpoints:
        cp_id = f"cp{cp:02d}"
        bank_id = f"bank-{scope.scope_id}-{cp_id}-{study.study_id[:8]}"
        bank = store.get_bank(bank_id)
        if bank and bank.status == BankStatus.released:
            release_map[cp_id] = builder.open_release(bank)

    # Score all runs
    scorer = ScorerV2(broker, judge_model=JUDGE_MODEL)
    ew = EventWriter(store, study.study_id)

    for run_manifest in runs:
        scores = scorer.score_run(
            run_manifest.run_id,
            store,
            scope.questions,
            release_map=release_map,
            study_id=study.study_id,
            policy_id=run_manifest.policy_id,
            event_writer=ew,
        )
        if scores:
            mean_primary = sum(s.primary_score for s in scores) / len(scores)
            logger.info(
                "  %s/%s: %d scored, mean_primary=%.4f",
                run_manifest.scope_id, run_manifest.policy_id,
                len(scores), mean_primary,
            )

    _report_summary(store, study.study_id)
    return True


def gate_7(store: StateStore, broker: ModalBroker) -> bool:
    """Full pilot: S08 + S10."""
    logger.info("Gate 7: full pilot (S08 + S10)...")

    scope_ids = list(SCOPE_DIRS.keys())
    study = _make_study(scope_ids)
    store.save_study(study)
    logger.info("  Study: %s", study.study_id)

    runner = StudyRunner(
        study=study,
        store=store,
        broker=broker,
        work_dir=WORK_DIR,
        scope_dirs=SCOPE_DIRS,
        families=["chunks", "core_memory", "summary"],
        max_turns=10,
        max_tool_calls=20,
    )

    runs = runner.run_study()
    logger.info("  Completed %d runs", len(runs))

    # Score all runs (with release_map for citation validation)
    scorer = ScorerV2(broker, judge_model=JUDGE_MODEL)
    ew = EventWriter(store, study.study_id)
    builder = BankBuilder(store, broker, WORK_DIR / "banks")

    for scope_id in scope_ids:
        scope = load_scope(SCOPE_DIRS[scope_id])

        # Build release_map for this scope
        release_map: dict[str, object] = {}
        for cp in scope.checkpoints:
            cp_id = f"cp{cp:02d}"
            bank_id = f"bank-{scope.scope_id}-{cp_id}-{study.study_id[:8]}"
            bank = store.get_bank(bank_id)
            if bank and bank.status == BankStatus.released:
                release_map[cp_id] = builder.open_release(bank)

        scope_runs = [r for r in runs if r.scope_id == scope_id]
        for run_manifest in scope_runs:
            scores = scorer.score_run(
                run_manifest.run_id,
                store,
                scope.questions,
                release_map=release_map,
                study_id=study.study_id,
                policy_id=run_manifest.policy_id,
                event_writer=ew,
            )
            if scores:
                mean_primary = sum(s.primary_score for s in scores) / len(scores)
                logger.info(
                    "  %s/%s: %d scored, mean_primary=%.4f",
                    run_manifest.scope_id, run_manifest.policy_id,
                    len(scores), mean_primary,
                )

    _report_summary(store, study.study_id)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="V2 smoke pilot — gated E2E validation")
    parser.add_argument("--gate", type=int, default=1, help="Run through this gate (0-7)")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM-dependent gates")
    parser.add_argument("--clean", action="store_true", help="Delete work dir before starting")
    args = parser.parse_args()

    max_gate = args.gate

    if args.clean and WORK_DIR.exists():
        import shutil
        shutil.rmtree(WORK_DIR)
        logger.info("Cleaned work directory: %s", WORK_DIR)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Gate 0: always runs
    if not gate_0():
        _print_gate(0, "SDK + imports", "fail")
        return 1
    _print_gate(0, "SDK + imports", "pass")
    if max_gate == 0:
        return 0

    # Gate 1: no LLM needed
    if not gate_1():
        _print_gate(1, "Bank build (chunks only)", "fail")
        return 1
    _print_gate(1, "Bank build (chunks only)", "pass")
    if max_gate == 1:
        return 0

    # Gates 2+ need broker and store
    if args.dry_run:
        logger.info("Dry run — skipping gates 2+")
        return 0

    store = StateStore(STATE_DB)
    broker = _make_broker()

    try:
        # Gate 2
        if max_gate >= 2:
            if not gate_2(store, broker):
                _print_gate(2, "Bank build (with synthesis)", "fail")
                return 1
            _print_gate(2, "Bank build (with synthesis)", "pass")

        # Gate 3
        if max_gate >= 3:
            run_id = gate_3(store, broker)
            if run_id is None:
                _print_gate(3, "Null policy answer", "fail")
                return 1
            _print_gate(3, "Null policy answer", "pass")

        # Gate 4
        if max_gate >= 4:
            run_id = gate_4(store, broker)
            if run_id is None:
                _print_gate(4, "Base policy answer", "fail")
                return 1
            _print_gate(4, "Base policy answer", "pass")

        # Gate 5
        if max_gate >= 5:
            if not gate_5(store, broker):
                _print_gate(5, "Scoring", "fail")
                return 1
            _print_gate(5, "Scoring", "pass")

        # Gate 6
        if max_gate >= 6:
            if not gate_6(store, broker):
                _print_gate(6, "Full S08", "fail")
                return 1
            _print_gate(6, "Full S08", "pass")

        # Gate 7
        if max_gate >= 7:
            if not gate_7(store, broker):
                _print_gate(7, "Full pilot (S08 + S10)", "fail")
                return 1
            _print_gate(7, "Full pilot (S08 + S10)", "pass")

    finally:
        store.close()

    logger.info("All gates through %d passed.", max_gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
