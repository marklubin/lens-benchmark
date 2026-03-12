"""Grid scope runner — runs one scope through all policies.

Designed for parallel execution: each scope gets its own state DB.
Launch 4 instances simultaneously, one per scope.

Usage:
  uv run python studies/grid/run_scope.py \
    --scope-id tutoring_jailbreak_07 \
    --scope-dir ../../datasets/scopes/07_tutoring_jailbreak \
    --temperature 0.3

  # With replicate for M>1:
  uv run python studies/grid/run_scope.py \
    --scope-id tutoring_jailbreak_07 \
    --scope-dir ../../datasets/scopes/07_tutoring_jailbreak \
    --temperature 0.3 \
    --replicate-id r02
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bench.bank import BankBuilder
from bench.broker import ModalBroker
from bench.dataset import load_scope
from bench.runner import StudyRunner
from bench.schemas import BankStatus, StudyManifest
from bench.scorer import ScorerV2
from bench.state import EventWriter, StateStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_BASE_URL = "https://synix--lens-llm-llm-serve.modal.run/v1"
EMBED_BASE_URL = "https://synix--lens-embed-serve.modal.run"
AGENT_MODEL = "Qwen/Qwen3.5-35B-A3B"
JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"
EMBED_MODEL = "Xenova/gte-modernbert-base"
DEFAULT_POLICY_IDS = ["null", "policy_base", "policy_core", "policy_summary"]
DEFAULT_FAMILIES = ["chunks", "core_memory", "summary"]

# Mapping from policy to required families (beyond chunks which is always present)
POLICY_FAMILY_REQUIREMENTS = {
    "null": [],
    "policy_base": [],
    "policy_core": ["core_memory"],
    "policy_summary": ["summary"],
    "policy_core_structured": ["core_structured"],
    "policy_core_maintained": ["core_maintained"],
    "policy_core_faceted": ["core_faceted"],
}


def main():
    parser = argparse.ArgumentParser(description="Grid scope runner")
    parser.add_argument("--scope-id", required=True, help="Scope ID (e.g. tutoring_jailbreak_07)")
    parser.add_argument("--scope-dir", required=True, help="Path to scope directory")
    parser.add_argument("--temperature", type=float, default=0.3, help="Agent temperature")
    parser.add_argument("--replicate-id", default="r01", help="Replicate ID for M>1")
    parser.add_argument("--clean", action="store_true", help="Delete work dir before starting")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip Qwen auto-scoring")
    parser.add_argument("--no-cache", action="store_true", help="Disable response cache (for M>1 variance)")
    parser.add_argument("--policies", nargs="+", default=None, help="Policy IDs to run (default: null policy_base policy_core policy_summary)")
    parser.add_argument("--work-prefix", default="work", help="Work directory prefix (default: work)")
    args = parser.parse_args()

    scope_dir = Path(args.scope_dir).resolve()
    work_dir = Path(__file__).resolve().parent / args.work_prefix / args.scope_id
    state_db = work_dir / "state.db"
    log_file = work_dir / "run.log"

    if args.clean and work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to both file and stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr),
        ],
    )
    logger = logging.getLogger(f"grid.{args.scope_id}")

    # Resolve policies and families
    policy_ids = args.policies or DEFAULT_POLICY_IDS
    families = set(["chunks"])  # always need chunks
    for pid in policy_ids:
        for fam in POLICY_FAMILY_REQUIREMENTS.get(pid, []):
            families.add(fam)
    families = sorted(families)

    logger.info("=" * 70)
    logger.info("GRID SCOPE RUNNER: %s", args.scope_id)
    logger.info("  scope_dir: %s", scope_dir)
    logger.info("  temperature: %.1f", args.temperature)
    logger.info("  replicate: %s", args.replicate_id)
    logger.info("  policies: %s", policy_ids)
    logger.info("  families: %s", families)
    logger.info("  work_dir: %s", work_dir)
    logger.info("=" * 70)

    # Load scope
    scope = load_scope(scope_dir)
    logger.info(
        "Loaded scope %s: %d episodes, %d questions, checkpoints=%s",
        scope.scope_id, len(scope.episodes), len(scope.questions), scope.checkpoints,
    )

    # Init store and broker
    store = StateStore(state_db)
    broker = ModalBroker(
        llm_base_url=LLM_BASE_URL,
        embed_base_url=EMBED_BASE_URL,
        llm_api_key="unused",
        cache_enabled=not args.no_cache,
        cache_dir=work_dir / "cache",
        default_extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    try:
        # Create study manifest
        study = StudyManifest(
            study_id=f"grid-{args.scope_id}-{args.replicate_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            benchmark_version="v2",
            scope_ids=[scope.scope_id],
            policy_ids=policy_ids,
            agent_model=AGENT_MODEL,
            judge_model=JUDGE_MODEL,
            embedding_model=EMBED_MODEL,
            prompt_set_version="v2.0",
            scoring_version="v2.0",
            code_sha="local",
            artifact_family_set_version="v2.0-grid",
            random_seed_policy=f"temp={args.temperature}",
            created_at=datetime.now(timezone.utc),
            notes=f"Grid run: {args.scope_id}, temp={args.temperature}, replicate={args.replicate_id}",
        )
        store.save_study(study)
        logger.info("Study: %s", study.study_id)

        # Run all policies
        runner = StudyRunner(
            study=study,
            store=store,
            broker=broker,
            work_dir=work_dir,
            scope_dirs={scope.scope_id: scope_dir},
            families=families,
            max_turns=10,
            max_tool_calls=20,
            temperature=args.temperature,
            replicate_id=args.replicate_id,
        )

        runs = runner.run_study()
        logger.info("Completed %d runs", len(runs))

        # Score all runs (Qwen auto-scoring as baseline)
        if not args.skip_scoring:
            logger.info("Auto-scoring with %s...", JUDGE_MODEL)
            scorer = ScorerV2(broker, judge_model=JUDGE_MODEL)
            ew = EventWriter(store, study.study_id)
            builder = BankBuilder(store, broker, work_dir / "banks")

            # Build release map
            release_map: dict[str, object] = {}
            for cp in scope.checkpoints:
                cp_id = f"cp{cp:02d}"
                bank_id = f"bank-{scope.scope_id}-{cp_id}-{study.study_id[:8]}"
                bank = store.get_bank(bank_id)
                if bank and bank.status == BankStatus.released:
                    release_map[cp_id] = builder.open_release(bank)

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
                        "  %s: %d scored, mean_primary=%.4f",
                        run_manifest.policy_id, len(scores), mean_primary,
                    )

        # Summary
        logger.info("=" * 70)
        logger.info("SCOPE %s COMPLETE", args.scope_id)
        logger.info("  Runs: %d", len(runs))
        logger.info("  State DB: %s", state_db)
        logger.info("=" * 70)

    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
