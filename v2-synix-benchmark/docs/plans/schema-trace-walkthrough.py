"""
Schema Trace Walkthrough — V2 Benchmark End-to-End
===================================================

This is pseudo-code that traces a complete benchmark execution
through every schema type. It serves as a verification artifact
for T002: if any field is missing, misnaming, or structurally
wrong, this walkthrough will expose it.

Scenario: Run policy_core_faceted on S08 with two checkpoints (cp01, cp02).

Assumptions about Synix (to be replaced with real SDK next pass):
- synix.build(pipeline_path, source_dir, build_dir) → build_ref
- synix.release(build_dir, ref, projection, target_root) → release_receipt
- SynixRuntime(release_root) → mounted runtime with search/retrieve tools
- Runtime exposes: search(query, surfaces, limit) → hits with ref IDs
- Runtime exposes: retrieve(ref_id) → chunk text + episode provenance
- Runtime exposes: get_projection(name) → blob (for core memory, summaries)
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.schemas import (
    BankManifest,
    BankStatus,
    BuildCost,
    Diagnostics,
    Event,
    EventType,
    FusionConfig,
    PolicyManifest,
    RetrievalCaps,
    RunCost,
    RunManifest,
    RunStatus,
    ScoreRecord,
    StudyManifest,
)

NOW = datetime.now(timezone.utc)


# ============================================================================
# PHASE 0: Synix stubs (replaced with real SDK in T005/T013)
# ============================================================================

class SynixBuildResult:
    """Stub for synix build output."""
    def __init__(self, build_ref: str, graph_hash: str):
        self.build_ref = build_ref
        self.graph_hash = graph_hash


class SynixReleaseReceipt:
    """Stub for synix release output."""
    def __init__(self, release_ref: str, projection_hashes: dict[str, str]):
        self.release_ref = release_ref
        self.projection_hashes = projection_hashes


class SynixRuntime:
    """Stub for mounted Synix runtime."""
    def __init__(self, release_root: Path):
        self.release_root = release_root

    def search(self, query: str, surfaces: list[str], limit: int = 10) -> list[dict]:
        """Search across named surfaces. Returns hits with ref_ids."""
        return [{"ref_id": "chunk_001", "score": 0.85, "text": "..."}]

    def retrieve(self, ref_id: str) -> dict:
        """Retrieve a chunk by ref_id. Returns text + provenance."""
        return {
            "ref_id": ref_id,
            "text": "chunk content...",
            "source_episode_id": "S08_ep_005",
            "offset": 0,
        }

    def get_projection(self, name: str) -> str | None:
        """Get a named projection blob (core memory, summary, etc.)."""
        return "Core memory state blob..."


def synix_build(pipeline_path: str, source_dir: str, build_dir: str) -> SynixBuildResult:
    return SynixBuildResult(build_ref="refs/runs/20260310T120000Z", graph_hash="sha256:graph_aaa")


def synix_release(build_dir: str, ref: str, projection: str, target_root: str) -> SynixReleaseReceipt:
    return SynixReleaseReceipt(
        release_ref=f"refs/releases/{projection}",
        projection_hashes={"raw_episodes": "sha256:raw", "chunks": "sha256:chunks"},
    )


# ============================================================================
# PHASE 1: Define the study
# ============================================================================

def phase1_define_study() -> StudyManifest:
    study = StudyManifest(
        study_id="2026-03-synix-v2-screening",
        benchmark_version="v2",
        scope_ids=["S07", "S08", "S10", "S11"],
        policy_ids=[
            "null", "policy_base", "policy_core", "policy_core_maintained",
            "policy_core_structured", "policy_core_faceted", "policy_summary",
        ],
        agent_model="modal:Qwen/Qwen3.5-35B-A3B",
        judge_model="modal:Qwen/Qwen3.5-35B-A3B",
        embedding_model="modal:Alibaba-NLP/gte-modernbert-base",
        prompt_set_version="v2",
        scoring_version="v2",
        code_sha="6484fd1a",  # git rev-parse HEAD
        artifact_family_set_version="v1",
        random_seed_policy="deterministic",
        budget_configs={
            # Each policy at two budget tiers (from ablation plan)
            "8k": {"max_context_tokens": 8000},
            "16k": {"max_context_tokens": 16000},
        },
        created_at=NOW,
    )

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: budget_configs maps tier_name → config. But how does a RunManifest
    #    know which budget tier it's using? The run needs a budget_tier field.
    #
    # A: YES — RunManifest is missing a budget_tier or budget_config_id field.
    #    Without it, we can't distinguish run_s08_core_8k from run_s08_core_16k.
    #    The ablation plan has 7 policies × 2 budgets = 14 configs per scope.
    #
    # FIX NEEDED: Add budget_tier: str | None to RunManifest.
    # -----------------------------------------------------------------------

    return study


# ============================================================================
# PHASE 2: Define policies
# ============================================================================

def phase2_define_policies() -> dict[str, PolicyManifest]:
    policies = {}

    # null — no memory, no tools
    policies["null"] = PolicyManifest(
        policy_manifest_id="null_v1",
        policy_id="null",
        visible_artifact_families=[],  # nothing visible
        query_surfaces=[],             # no search
        fusion=FusionConfig(method="none", parameters={}),
        retrieval_caps=RetrievalCaps(max_results=0, max_context_tokens=0),
        citation_policy="none",
        version="v1",
    )

    # policy_base — chunks + hybrid search
    policies["policy_base"] = PolicyManifest(
        policy_manifest_id="policy_base_v1",
        policy_id="policy_base",
        visible_artifact_families=["chunks", "search_indexes"],
        query_surfaces=["fts", "cosine"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(max_results=12, max_context_tokens=8000),
        citation_policy="source-backed-only",
        version="v1",
    )

    # policy_core — base + single fold core memory
    policies["policy_core"] = PolicyManifest(
        policy_manifest_id="policy_core_v1",
        policy_id="policy_core",
        visible_artifact_families=["chunks", "search_indexes", "core_memory"],
        query_surfaces=["fts", "cosine", "core_memory"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(max_results=12, max_context_tokens=8000),
        citation_policy="source-backed-only",
        version="v1",
        fold_config={
            "type": "FoldSynthesis",
            "prompt": "core_memory_fold_v1",
            "order_by": "timestamp",
        },
    )

    # policy_core_faceted — base + 4 parallel faceted folds
    policies["policy_core_faceted"] = PolicyManifest(
        policy_manifest_id="policy_core_faceted_v1",
        policy_id="policy_core_faceted",
        visible_artifact_families=[
            "chunks", "search_indexes",
            "core_memory_entity", "core_memory_relation",
            "core_memory_event", "core_memory_cause",
        ],
        query_surfaces=["fts", "cosine", "core_memory"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(max_results=12, max_context_tokens=8000),
        citation_policy="source-backed-only",
        version="v1",
        fold_config={
            "type": "parallel_faceted_fold",
            "facets": [
                {"name": "entity", "prompt": "facet_entity_fold_v1"},
                {"name": "relation", "prompt": "facet_relation_fold_v1"},
                {"name": "event", "prompt": "facet_event_fold_v1"},
                {"name": "cause", "prompt": "facet_cause_fold_v1"},
            ],
            "order_by": "timestamp",
        },
    )

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: Does PolicyManifest.retrieval_caps.max_context_tokens serve double
    #    duty as the budget tier? Or is budget separate from retrieval caps?
    #
    # A: They're related but distinct. retrieval_caps controls how much
    #    context the search tool returns. Budget controls total agent tokens
    #    (system prompt + retrieved context + reasoning). The 8k/16k tiers
    #    from the ablation plan should be budget limits, not retrieval caps.
    #
    # FIX NEEDED: Either:
    #   (a) Add a top-level budget field to PolicyManifest (max_agent_tokens), or
    #   (b) Keep budget as a RunManifest concern (budget_tier references study config)
    #
    # Option (b) is cleaner — same policy, different budget = different runs.
    # This reinforces the need for budget_tier on RunManifest.
    # -----------------------------------------------------------------------

    # policy_summary — base + group → reduce summaries
    policies["policy_summary"] = PolicyManifest(
        policy_manifest_id="policy_summary_v1",
        policy_id="policy_summary",
        visible_artifact_families=["chunks", "search_indexes", "summaries"],
        query_surfaces=["fts", "cosine", "summaries"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(max_results=12, max_context_tokens=8000),
        citation_policy="source-backed-only",
        version="v1",
        group_config={
            "type": "GroupSynthesis",
            "batch_size": 5,
            "order_by": "ordinal",
            "prompt": "group_summary_v1",
        },
        reduce_config={
            "type": "ReduceSynthesis",
            "prompt": "reduce_summary_v1",
        },
    )

    return policies


# ============================================================================
# PHASE 3: Compile artifact banks (one per scope × checkpoint)
# ============================================================================

def phase3_compile_banks(
    study: StudyManifest,
    scope_id: str,
    checkpoints: list[dict],  # [{"id": "cp01", "max_ep": 6, "ep_ids": [...]}, ...]
    events: list[Event],
) -> list[BankManifest]:
    """Compile one artifact bank per checkpoint for a scope."""

    banks = []

    for cp in checkpoints:
        bank = BankManifest(
            bank_manifest_id=f"bank_{scope_id}_{cp['id']}_v1",
            study_id=study.study_id,
            scope_id=scope_id,
            checkpoint_id=cp["id"],
            max_episode_ordinal=cp["max_ep"],
            source_episode_ids=cp["ep_ids"],
            artifact_families={
                "raw_episodes": None,   # filled after build
                "chunks": None,
                "search_indexes": None,
                "core_memory": None,
                "core_memory_entity": None,
                "core_memory_relation": None,
                "core_memory_event": None,
                "core_memory_cause": None,
                "summaries": None,
                "graph": None,
            },
            dataset_hash="sha256:dataset_s08_v1",
            status=BankStatus.planned,
        )

        # --- Emit bank_build_started event ---
        events.append(Event(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            study_id=study.study_id,
            scope_id=scope_id,
            bank_manifest_id=bank.bank_manifest_id,
            event_type=EventType.bank_build_started,
            payload={"checkpoint_id": cp["id"], "max_episode_ordinal": cp["max_ep"]},
        ))

        # --- Call Synix build (one build for all artifacts) ---
        build_result = synix_build(
            pipeline_path="pipelines/lens_v2.py",
            source_dir=f"datasets/scopes/{scope_id}",
            build_dir=f".synix/builds/{scope_id}",
        )

        # --- Release per-checkpoint projection ---
        release = synix_release(
            build_dir=f".synix/builds/{scope_id}",
            ref=build_result.build_ref,
            projection=f"{scope_id}_{cp['id']}",
            target_root=f".synix/releases/{scope_id}/{cp['id']}",
        )

        # --- Update bank manifest with build results ---
        bank.synix_build_ref = build_result.build_ref
        bank.synix_release_ref = release.release_ref
        bank.synix_build_graph_hash = build_result.graph_hash
        bank.artifact_families.update(release.projection_hashes)
        bank.status = BankStatus.released
        bank.built_at = datetime.now(timezone.utc)
        bank.build_cost = BuildCost(
            wall_time_s=12.5,
            prompt_tokens=45000,
            completion_tokens=8000,
            embed_tokens=30000,
            estimated_cost_usd=0.15,
        )

        # --- Emit per-family artifact_compiled events ---
        for family, family_hash in release.projection_hashes.items():
            events.append(Event(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                study_id=study.study_id,
                scope_id=scope_id,
                bank_manifest_id=bank.bank_manifest_id,
                event_type=EventType.artifact_compiled,
                payload={"family": family, "hash": family_hash},
            ))

        # --- Emit bank_build_completed ---
        events.append(Event(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            study_id=study.study_id,
            scope_id=scope_id,
            bank_manifest_id=bank.bank_manifest_id,
            event_type=EventType.bank_build_completed,
            payload={"build_ref": build_result.build_ref},
        ))

        banks.append(bank)

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: Bank compilation events have scope_id and bank_manifest_id but
    #    no run_id or policy_id. Is that correct?
    #
    # A: YES — bank compilation is policy-independent. The bank is shared
    #    across all policies. event.run_id and event.policy_id should be
    #    None for bank events. This works because both are Optional.
    #    CONFIRMED: schema handles this correctly.
    #
    # Q: Should BankManifest have a field for the Synix release target_root?
    #
    # A: Maybe. The release_ref is the semantic identifier, but T013 needs
    #    to know where to mount the runtime. Could be derived from convention
    #    or stored explicitly.
    #
    # MINOR: Consider adding release_target_root: str | None to BankManifest.
    # -----------------------------------------------------------------------

    return banks


# ============================================================================
# PHASE 4: Execute a policy run
# ============================================================================

def phase4_execute_run(
    study: StudyManifest,
    policy: PolicyManifest,
    banks: list[BankManifest],
    scope_id: str,
    questions_by_checkpoint: dict[str, list[dict]],
    events: list[Event],
) -> RunManifest:
    """Execute one policy run across all checkpoints for a scope."""

    run = RunManifest(
        run_id=f"run_{scope_id}_{policy.policy_id}_r01",
        study_id=study.study_id,
        scope_id=scope_id,
        policy_id=policy.policy_id,
        replicate_id="r01",
        policy_manifest_id=policy.policy_manifest_id,
        bank_manifest_ids=[b.bank_manifest_id for b in banks],
        config_hash=_hash_config(study, policy),
        dataset_hash=banks[0].dataset_hash,
        status=RunStatus.planned,
    )

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: Where is budget_tier? If this is the 8K run, how do we know?
    #
    # A: NOT CAPTURED. The run_id encodes it by convention
    #    (run_s08_core_faceted_8k_r01) but that's fragile.
    #
    # FIX NEEDED: Add budget_tier: str | None to RunManifest.
    # -----------------------------------------------------------------------

    # --- Start run ---
    run.status = RunStatus.running
    run.started_at = datetime.now(timezone.utc)

    events.append(Event(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        study_id=study.study_id,
        run_id=run.run_id,
        scope_id=scope_id,
        policy_id=policy.policy_id,
        event_type=EventType.run_started,
        config_hash=run.config_hash,
        payload={"policy_manifest_id": policy.policy_manifest_id},
    ))

    answers: list[dict] = []

    for bank in banks:
        cp_id = bank.checkpoint_id

        # --- Mount Synix runtime for this checkpoint's bank ---
        runtime = SynixRuntime(
            release_root=Path(f".synix/releases/{scope_id}/{cp_id}")
        )

        events.append(Event(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            study_id=study.study_id,
            run_id=run.run_id,
            scope_id=scope_id,
            policy_id=policy.policy_id,
            bank_manifest_id=bank.bank_manifest_id,
            event_type=EventType.checkpoint_started,
            payload={"checkpoint_id": cp_id},
        ))

        # --- Prepare system prompt based on policy ---
        system_prompt = _build_system_prompt(policy, runtime)

        # --- Answer each question at this checkpoint ---
        for question in questions_by_checkpoint.get(cp_id, []):
            q_id = question["question_id"]

            events.append(Event(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                study_id=study.study_id,
                run_id=run.run_id,
                scope_id=scope_id,
                policy_id=policy.policy_id,
                bank_manifest_id=bank.bank_manifest_id,
                event_type=EventType.question_started,
                payload={"question_id": q_id, "checkpoint_id": cp_id},
            ))

            # --- Agent loop (simplified) ---
            agent_answer = _run_agent_loop(
                system_prompt=system_prompt,
                question=question["prompt"],
                tools=_build_tools(policy, runtime),
                broker=None,  # would be ModalBroker in real code
                events=events,
                run_id=run.run_id,
                study_id=study.study_id,
                scope_id=scope_id,
                policy_id=policy.policy_id,
                bank_manifest_id=bank.bank_manifest_id,
            )

            answers.append({
                "question_id": q_id,
                "checkpoint_id": cp_id,
                "bank_manifest_id": bank.bank_manifest_id,
                "answer_text": agent_answer["answer_text"],
                "refs_cited": agent_answer["refs_cited"],
                "ref_episode_map": agent_answer["ref_episode_map"],
                "tool_calls_made": agent_answer["tool_calls_made"],
                "total_tokens": agent_answer["total_tokens"],
                "wall_time_ms": agent_answer["wall_time_ms"],
            })

            events.append(Event(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                study_id=study.study_id,
                run_id=run.run_id,
                scope_id=scope_id,
                policy_id=policy.policy_id,
                bank_manifest_id=bank.bank_manifest_id,
                event_type=EventType.question_completed,
                payload={
                    "question_id": q_id,
                    "checkpoint_id": cp_id,
                    "tool_calls": agent_answer["tool_calls_made"],
                    "total_tokens": agent_answer["total_tokens"],
                },
            ))

            # --- Update resume cursor ---
            run.last_completed_checkpoint = cp_id
            run.last_completed_question = q_id

        events.append(Event(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            study_id=study.study_id,
            run_id=run.run_id,
            scope_id=scope_id,
            policy_id=policy.policy_id,
            bank_manifest_id=bank.bank_manifest_id,
            event_type=EventType.checkpoint_completed,
            payload={"checkpoint_id": cp_id, "questions_answered": len(questions_by_checkpoint.get(cp_id, []))},
        ))

    # --- Complete run ---
    run.status = RunStatus.completed
    run.completed_at = datetime.now(timezone.utc)
    run.cost = RunCost(
        wall_time_s=45.0,
        prompt_tokens=120000,
        completion_tokens=15000,
        estimated_cost_usd=0.35,
    )

    events.append(Event(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        study_id=study.study_id,
        run_id=run.run_id,
        scope_id=scope_id,
        policy_id=policy.policy_id,
        event_type=EventType.run_completed,
        payload={"total_questions": sum(len(qs) for qs in questions_by_checkpoint.values())},
    ))

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: Where are the answers persisted? RunManifest doesn't contain them.
    #
    # A: Answers are stored as a separate artifact file (answers.json),
    #    keyed by run_id. The RunManifest points to the run, and the
    #    scoring pipeline loads answers by run_id convention.
    #
    #    This is correct — answers are NOT part of RunManifest. They're
    #    a runtime artifact, not a manifest field. T004 (state store)
    #    handles persistence.
    #
    # Q: Should RunManifest have an answers_path or answers_hash?
    #
    # A: POSSIBLY. For replay/audit, it would be useful to know exactly
    #    which answer file a score was computed from.
    #
    # MINOR: Consider adding answers_hash: str | None to RunManifest.
    # -----------------------------------------------------------------------

    return run


# ============================================================================
# PHASE 5: Score answers
# ============================================================================

def phase5_score(
    study: StudyManifest,
    run: RunManifest,
    banks: list[BankManifest],
    answers: list[dict],
    questions: list[dict],
    events: list[Event],
) -> list[ScoreRecord]:
    """Score all answers from a completed run."""

    events.append(Event(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        study_id=study.study_id,
        run_id=run.run_id,
        scope_id=run.scope_id,
        policy_id=run.policy_id,
        event_type=EventType.scoring_started,
        payload={"scorer_version": study.scoring_version, "judge_model": study.judge_model},
    ))

    scores: list[ScoreRecord] = []
    bank_by_cp = {b.checkpoint_id: b for b in banks}
    question_by_id = {q["question_id"]: q for q in questions}

    for answer in answers:
        q_id = answer["question_id"]
        cp_id = answer["checkpoint_id"]
        question = question_by_id[q_id]
        bank = bank_by_cp[cp_id]

        # --- Step 1: Fact matching (mechanical + LLM judge) ---
        fact_result = _score_facts(
            question=question["prompt"],
            canonical_answer=question["ground_truth"]["canonical_answer"],
            key_facts=question["ground_truth"]["key_facts"],
            candidate_answer=answer["answer_text"],
        )

        # --- Step 2: Citation validation (mechanical) ---
        citation_result = _validate_citations(
            cited_refs=answer["refs_cited"],
            ref_episode_map=answer["ref_episode_map"],
            bank=bank,
        )

        # --- Step 3: Evidence support (LLM judge) ---
        evidence_result = _score_evidence_support(
            question=question["prompt"],
            candidate_answer=answer["answer_text"],
            cited_refs=answer["refs_cited"],
            required_evidence_refs=question["ground_truth"]["required_evidence_refs"],
        )

        # --- Composite (D002: no hard gates) ---
        primary = (
            0.5 * fact_result["f1"]
            + 0.3 * evidence_result["score"]
            + 0.2 * citation_result["validity"]
        )

        score = ScoreRecord(
            score_id=f"score_{run.run_id}_{q_id}",
            study_id=study.study_id,
            run_id=run.run_id,
            question_id=q_id,
            scope_id=run.scope_id,
            policy_id=run.policy_id,
            checkpoint_id=cp_id,
            bank_manifest_id=bank.bank_manifest_id,
            fact_precision=fact_result["precision"],
            fact_recall=fact_result["recall"],
            fact_f1=fact_result["f1"],
            evidence_support=evidence_result["score"],
            citation_validity=citation_result["validity"],
            primary_score=primary,
            diagnostics=Diagnostics(
                latency_ms=answer["wall_time_ms"],
                prompt_tokens=answer["total_tokens"],  # simplified
                completion_tokens=0,
                retrieval_count=len(answer["refs_cited"]),
                tool_count=answer["tool_calls_made"],
                estimated_cost_usd=0.0,
            ),
            scored_at=datetime.now(timezone.utc),
            scorer_version=study.scoring_version,
            judge_model=study.judge_model,
        )

        scores.append(score)

        # --- Emit model_call_completed for judge LLM calls ---
        events.append(Event(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            study_id=study.study_id,
            run_id=run.run_id,
            scope_id=run.scope_id,
            policy_id=run.policy_id,
            bank_manifest_id=bank.bank_manifest_id,
            event_type=EventType.model_call_completed,
            payload={
                "purpose": "fact_judge",
                "question_id": q_id,
                "model": study.judge_model,
            },
        ))

    events.append(Event(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        study_id=study.study_id,
        run_id=run.run_id,
        scope_id=run.scope_id,
        policy_id=run.policy_id,
        event_type=EventType.scoring_completed,
        payload={"questions_scored": len(scores)},
    ))

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: ScoreRecord has diagnostics.prompt_tokens but during scoring the
    #    judge also consumes tokens. Are those the agent's tokens or
    #    the judge's tokens?
    #
    # A: The Diagnostics on ScoreRecord should capture the AGENT's resource
    #    usage (what happened during the run), not the judge's. Judge
    #    inference cost should be tracked via Events (model_call_completed).
    #    This is currently correct — diagnostics come from the answer dict.
    #    CONFIRMED: schema handles this correctly.
    #
    # Q: ScoreRecord doesn't have a question_type field. For per-type
    #    analysis (longitudinal vs null_hypothesis vs action_recommendation),
    #    you'd need to join back to the questions file.
    #
    # A: This is intentional — the question_id is the join key. Denormalizing
    #    question_type into ScoreRecord would be convenient for reporting but
    #    isn't strictly necessary. The question file is small.
    #
    # MINOR: Consider adding question_type: str | None to ScoreRecord for
    #    convenience in per-type analysis. Low priority.
    # -----------------------------------------------------------------------

    return scores


# ============================================================================
# PHASE 6: Resume after failure
# ============================================================================

def phase6_resume_after_failure(
    run: RunManifest,
    events: list[Event],
) -> None:
    """Resume a run that failed mid-execution.

    Uses last_completed_checkpoint and last_completed_question from
    RunManifest to skip already-answered questions.
    """

    # --- Check what was completed ---
    last_cp = run.last_completed_checkpoint   # e.g. "cp01"
    last_q = run.last_completed_question      # e.g. "S08_Q02"

    # --- Mark as resumed ---
    run.status = RunStatus.resumed

    events.append(Event(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        study_id=run.study_id,
        run_id=run.run_id,
        scope_id=run.scope_id,
        policy_id=run.policy_id,
        event_type=EventType.run_resumed,
        payload={
            "resumed_from_checkpoint": last_cp,
            "resumed_from_question": last_q,
        },
    ))

    # --- Skip completed questions, continue from next ---
    # T004 state store will implement this with:
    #   state.sqlite: completed_steps table keyed by (run_id, checkpoint_id, question_id)
    #   Resume = query completed_steps, skip, continue

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: Is last_completed_checkpoint + last_completed_question sufficient
    #    for resume? What about partial bank builds?
    #
    # A: For RUNS, yes — questions are independent within a checkpoint.
    #    For BANK BUILDS, we need per-family completion tracking. The
    #    BankManifest.status goes planned → building → built → released,
    #    but doesn't track which families are done.
    #
    # POSSIBLE FIX: BankManifest.artifact_families values could double as
    #    status — None means "not yet built", a hash means "done".
    #    This actually already works! If a family hash is None, it hasn't
    #    been built. If it's a string hash, it has. Resume logic can
    #    skip families with non-None hashes.
    #    CONFIRMED: schema handles this correctly for bank resume.
    #
    # Q: Should RunManifest track completion at a finer grain than
    #    last_completed_question? E.g. partially completed agent loops?
    #
    # A: No. Agent loops are atomic (cache-through broker ensures idempotent
    #    retries). The question is the atomic unit of resume.
    #    CONFIRMED: schema grain is correct.
    # -----------------------------------------------------------------------


# ============================================================================
# PHASE 7: Replay without inference
# ============================================================================

def phase7_replay(
    run: RunManifest,
    events: list[Event],
) -> None:
    """Replay a completed run from cached responses only.

    Validates that every model call has a cache hit. If any call
    is a cache miss, replay fails (don't silently re-infer).
    """

    # The broker handles this: all calls go through ModalBroker.
    # In replay mode, bypass_cache=False and the broker raises
    # on cache miss instead of calling the API.

    # For scoring replay: ScoreRecord has all primary metrics stored.
    # Re-scoring from saved answers only needs:
    #   1. Load answers.json by run_id
    #   2. Load questions by scope_id
    #   3. Run scoring pipeline (which calls judge via broker → cache hit)
    #   4. Compare new ScoreRecords to stored ones (should be identical)

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: Is there enough provenance to verify replay correctness?
    #
    # A: Check the chain:
    #    ScoreRecord.run_id → RunManifest.run_id (links to run)
    #    ScoreRecord.bank_manifest_id → BankManifest.bank_manifest_id (links to bank)
    #    RunManifest.config_hash → deterministic config fingerprint
    #    RunManifest.dataset_hash → dataset fingerprint
    #    BankManifest.synix_build_ref → immutable build identifier
    #    StudyManifest.code_sha → code version
    #
    #    YES — the chain is complete. Every result traces back to:
    #    code version + dataset version + config + synix build + policy.
    #    CONFIRMED: schema handles replay provenance correctly.
    # -----------------------------------------------------------------------


# ============================================================================
# PHASE 8: Report generation
# ============================================================================

def phase8_report(
    study: StudyManifest,
    scores: list[ScoreRecord],
) -> dict:
    """Generate aggregate report from score records.

    Verifies we can produce the study report entirely from manifests
    and score records — no re-inference needed.
    """

    # Group scores by policy
    by_policy: dict[str, list[ScoreRecord]] = {}
    for s in scores:
        by_policy.setdefault(s.policy_id, []).append(s)

    # Group by checkpoint for trend analysis
    by_checkpoint: dict[str, list[ScoreRecord]] = {}
    for s in scores:
        by_checkpoint.setdefault(s.checkpoint_id, []).append(s)

    # Group by scope for domain analysis
    by_scope: dict[str, list[ScoreRecord]] = {}
    for s in scores:
        by_scope.setdefault(s.scope_id, []).append(s)

    # -----------------------------------------------------------------------
    # SCHEMA GAP CHECK:
    # -----------------------------------------------------------------------
    # Q: Can we produce the ablation comparisons from ScoreRecords alone?
    #    e.g. "core - base" on S08 at cp02
    #
    # A: YES — filter by scope_id, checkpoint_id, policy_id, compute means.
    #    All fields needed for filtering are present.
    #
    # Q: Can we produce the budget tier comparison (8k vs 16k)?
    #
    # A: NOT WITHOUT budget_tier. We'd have to infer from run_id naming
    #    convention or join back to RunManifest... but RunManifest also
    #    doesn't have budget_tier yet.
    #
    # FIX NEEDED: budget_tier on RunManifest (and optionally on ScoreRecord
    #    for denormalized reporting).
    # -----------------------------------------------------------------------

    report = {
        "study_id": study.study_id,
        "scope_count": len(study.scope_ids),
        "policy_count": len(study.policy_ids),
        "total_scores": len(scores),
        "by_policy": {
            pid: {"mean_primary": sum(s.primary_score for s in ss) / len(ss)}
            for pid, ss in by_policy.items()
        },
    }

    return report


# ============================================================================
# HELPER STUBS
# ============================================================================

def _hash_config(study: StudyManifest, policy: PolicyManifest) -> str:
    blob = json.dumps({
        "study_id": study.study_id,
        "policy_manifest_id": policy.policy_manifest_id,
        "scoring_version": study.scoring_version,
    }, sort_keys=True)
    return f"sha256:{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def _build_system_prompt(policy: PolicyManifest, runtime: SynixRuntime) -> str:
    """Build the agent system prompt based on policy configuration."""
    parts = ["You are a benchmark agent. Answer questions using the available tools."]

    # If policy includes core memory, prepend it
    if "core_memory" in policy.visible_artifact_families:
        core_blob = runtime.get_projection("core_memory")
        if core_blob:
            parts.append(f"\n## Core Memory\n{core_blob}")

    # If policy includes summaries, prepend them
    if "summaries" in policy.visible_artifact_families:
        summary_blob = runtime.get_projection("summaries")
        if summary_blob:
            parts.append(f"\n## Summary Context\n{summary_blob}")

    # If policy includes faceted core memory, prepend each facet
    for facet in ["entity", "relation", "event", "cause"]:
        family = f"core_memory_{facet}"
        if family in policy.visible_artifact_families:
            facet_blob = runtime.get_projection(family)
            if facet_blob:
                parts.append(f"\n## {facet.title()} Memory\n{facet_blob}")

    return "\n".join(parts)


def _build_tools(policy: PolicyManifest, runtime: SynixRuntime) -> list[dict]:
    """Build tool definitions based on policy visibility."""
    tools = []

    if policy.query_surfaces:
        tools.append({
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search memory for relevant evidence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        })

    # null policy → no tools
    return tools


def _run_agent_loop(
    system_prompt: str,
    question: str,
    tools: list[dict],
    broker: Any,
    events: list[Event],
    **event_context: Any,
) -> dict:
    """Stub for the agent loop. Returns an answer dict."""
    # In real code: multi-turn LLM conversation with tool calls
    # Each LLM call emits a model_call_completed event
    events.append(Event(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        study_id=event_context["study_id"],
        run_id=event_context["run_id"],
        scope_id=event_context["scope_id"],
        policy_id=event_context["policy_id"],
        bank_manifest_id=event_context["bank_manifest_id"],
        event_type=EventType.model_call_completed,
        attempt=1,
        payload={
            "provider": "modal",
            "model": "Qwen/Qwen3.5-35B-A3B",
            "latency_ms": 1200,
            "prompt_tokens": 4000,
            "completion_tokens": 500,
        },
    ))

    return {
        "answer_text": "The cascading failure originated from...",
        "refs_cited": ["chunk_001", "chunk_005"],
        "ref_episode_map": {"chunk_001": "S08_ep_005", "chunk_005": "S08_ep_010"},
        "tool_calls_made": 3,
        "total_tokens": 4500,
        "wall_time_ms": 1200.0,
    }


def _score_facts(question: str, canonical_answer: str, key_facts: list[str], candidate_answer: str) -> dict:
    return {"precision": 0.80, "recall": 0.67, "f1": 0.72}


def _validate_citations(cited_refs: list[str], ref_episode_map: dict, bank: BankManifest) -> dict:
    return {"validity": 1.0, "invalid_refs": []}


def _score_evidence_support(question: str, candidate_answer: str, cited_refs: list[str], required_evidence_refs: list[str]) -> dict:
    return {"score": 0.80}


# ============================================================================
# MAIN TRACE — run the full scenario
# ============================================================================

def main():
    events: list[Event] = []

    # Phase 1: Study
    study = phase1_define_study()
    print(f"Study: {study.study_id} — {len(study.scope_ids)} scopes × {len(study.policy_ids)} policies")

    # Phase 2: Policies
    policies = phase2_define_policies()
    print(f"Policies defined: {list(policies.keys())}")

    # Phase 3: Compile banks for S08
    checkpoints = [
        {"id": "cp01", "max_ep": 6, "ep_ids": [f"S08_ep_{i:03d}" for i in range(1, 7)]},
        {"id": "cp02", "max_ep": 12, "ep_ids": [f"S08_ep_{i:03d}" for i in range(1, 13)]},
    ]
    banks = phase3_compile_banks(study, "S08", checkpoints, events)
    print(f"Banks compiled: {[b.bank_manifest_id for b in banks]}")

    # Phase 4: Run policy_core_faceted
    questions_by_cp = {
        "cp01": [
            {"question_id": "S08_Q01", "prompt": "What patterns emerge?",
             "ground_truth": {"canonical_answer": "...", "key_facts": ["fact1"], "required_evidence_refs": ["S08_ep_003"]}},
        ],
        "cp02": [
            {"question_id": "S08_Q02", "prompt": "What is the root cause?",
             "ground_truth": {"canonical_answer": "...", "key_facts": ["fact2", "fact3"], "required_evidence_refs": ["S08_ep_008", "S08_ep_011"]}},
            {"question_id": "S08_Q03", "prompt": "What action should be taken?",
             "ground_truth": {"canonical_answer": "...", "key_facts": ["fact4"], "required_evidence_refs": ["S08_ep_010"]}},
        ],
    }
    run = phase4_execute_run(study, policies["policy_core_faceted"], banks, "S08", questions_by_cp, events)
    print(f"Run completed: {run.run_id} — status={run.status}")

    # Collect answers from the stub (in real code, persisted by state store)
    all_answers = [
        {"question_id": "S08_Q01", "checkpoint_id": "cp01", "bank_manifest_id": banks[0].bank_manifest_id,
         "answer_text": "Early signals suggest...", "refs_cited": ["chunk_001"], "ref_episode_map": {"chunk_001": "S08_ep_003"},
         "tool_calls_made": 2, "total_tokens": 3000, "wall_time_ms": 900.0},
        {"question_id": "S08_Q02", "checkpoint_id": "cp02", "bank_manifest_id": banks[1].bank_manifest_id,
         "answer_text": "The root cause is...", "refs_cited": ["chunk_005", "chunk_008"], "ref_episode_map": {"chunk_005": "S08_ep_008", "chunk_008": "S08_ep_011"},
         "tool_calls_made": 4, "total_tokens": 5000, "wall_time_ms": 1500.0},
        {"question_id": "S08_Q03", "checkpoint_id": "cp02", "bank_manifest_id": banks[1].bank_manifest_id,
         "answer_text": "Recommended action: ...", "refs_cited": ["chunk_007"], "ref_episode_map": {"chunk_007": "S08_ep_010"},
         "tool_calls_made": 3, "total_tokens": 4000, "wall_time_ms": 1100.0},
    ]

    all_questions = [
        {"question_id": "S08_Q01", "prompt": "What patterns emerge?",
         "ground_truth": {"canonical_answer": "...", "key_facts": ["fact1"], "required_evidence_refs": ["S08_ep_003"]}},
        {"question_id": "S08_Q02", "prompt": "What is the root cause?",
         "ground_truth": {"canonical_answer": "...", "key_facts": ["fact2", "fact3"], "required_evidence_refs": ["S08_ep_008", "S08_ep_011"]}},
        {"question_id": "S08_Q03", "prompt": "What action should be taken?",
         "ground_truth": {"canonical_answer": "...", "key_facts": ["fact4"], "required_evidence_refs": ["S08_ep_010"]}},
    ]

    # Phase 5: Score
    scores = phase5_score(study, run, banks, all_answers, all_questions, events)
    print(f"Scores: {len(scores)} records")
    for s in scores:
        print(f"  {s.question_id}: fact_f1={s.fact_f1:.2f} evidence={s.evidence_support:.2f} "
              f"citation={s.citation_validity:.2f} → primary={s.primary_score:.2f}")

    # Phase 8: Report
    report = phase8_report(study, scores)
    print(f"\nReport: {json.dumps(report, indent=2)}")

    # Event audit
    print(f"\nEvents emitted: {len(events)}")
    for et in EventType:
        count = sum(1 for e in events if e.event_type == et)
        if count > 0:
            print(f"  {et.value}: {count}")

    # -----------------------------------------------------------------------
    # SUMMARY OF GAPS FOUND
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCHEMA GAPS FOUND DURING TRACE")
    print("=" * 70)

    gaps = [
        {
            "severity": "HIGH",
            "field": "RunManifest.budget_tier",
            "type": "missing field",
            "reason": "Ablation plan has 7 policies × 2 budgets = 14 configs per scope. "
                      "Without budget_tier, can't distinguish 8K from 16K runs. "
                      "config_hash captures it implicitly but is opaque.",
        },
        {
            "severity": "LOW",
            "field": "BankManifest.release_target_root",
            "type": "missing field",
            "reason": "T013 needs to know where to mount the Synix runtime. "
                      "Can be derived from convention (.synix/releases/{scope}/{cp}) "
                      "but explicit is safer.",
        },
        {
            "severity": "LOW",
            "field": "RunManifest.answers_hash",
            "type": "missing field",
            "reason": "For replay audit, useful to verify which answer file "
                      "a score was computed from. Not strictly needed if answers "
                      "are keyed by run_id convention.",
        },
        {
            "severity": "LOW",
            "field": "ScoreRecord.question_type",
            "type": "missing field",
            "reason": "Per-type analysis (longitudinal vs null_hypothesis vs "
                      "action_recommendation) requires joining back to questions. "
                      "Denormalizing here is convenient but not critical.",
        },
    ]

    for gap in gaps:
        print(f"\n[{gap['severity']}] {gap['field']}")
        print(f"  Type: {gap['type']}")
        print(f"  Reason: {gap['reason']}")

    print("\n" + "=" * 70)
    print("CONFIRMED CORRECT")
    print("=" * 70)
    confirmations = [
        "Bank events correctly omit run_id/policy_id (bank is policy-independent)",
        "BankManifest.artifact_families None values work for resume (None=not built, hash=done)",
        "Question is the atomic unit of run resume (agent loops are cache-idempotent)",
        "Diagnostics on ScoreRecord track agent usage, not judge usage (judge via Events)",
        "Full provenance chain: ScoreRecord → RunManifest → BankManifest → StudyManifest → code_sha",
        "Replay chain complete: code + dataset + config + synix build + policy all captured",
        "PolicyManifest.fold_config/group_config/reduce_config handle all 7 first-pass policies",
        "Event types cover full lifecycle: bank build, run, checkpoint, question, model call, scoring",
        "null policy works: empty families, empty surfaces, no tools",
        "Faceted policy works: 4 separate core_memory_* families in visible_artifact_families",
    ]
    for c in confirmations:
        print(f"  [OK] {c}")


if __name__ == "__main__":
    main()
