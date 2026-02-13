from __future__ import annotations

from lens.core.errors import EvidenceError, ValidationError
from lens.core.models import Hit, Insight
from lens.runner.anticheat import EpisodeVault


class OutputValidator:
    """Validates adapter outputs against schema requirements and evidence rules."""

    def __init__(self, vault: EpisodeVault, max_evidence_episodes: int = 8) -> None:
        self.vault = vault
        self.max_evidence_episodes = max_evidence_episodes

    def validate_insights(self, insights: list[Insight]) -> list[str]:
        """Validate a list of insights. Returns list of error messages."""
        errors: list[str] = []

        for i, insight in enumerate(insights):
            prefix = f"insight[{i}]"

            # Schema checks
            if not insight.text.strip():
                errors.append(f"{prefix}: empty text")

            if not 0.0 <= insight.confidence <= 1.0:
                errors.append(f"{prefix}: confidence {insight.confidence} not in [0, 1]")

            if not insight.falsifier.strip():
                errors.append(f"{prefix}: empty falsifier")

            # Evidence checks
            if len(insight.evidence) < 1:
                errors.append(f"{prefix}: no evidence refs")

            episode_ids = set()
            for j, ref in enumerate(insight.evidence):
                ref_prefix = f"{prefix}.evidence[{j}]"

                if not self.vault.has(ref.episode_id):
                    errors.append(f"{ref_prefix}: unknown episode_id {ref.episode_id!r}")
                    continue

                episode_ids.add(ref.episode_id)

                # Exact substring match
                if not self.vault.verify_quote(ref.episode_id, ref.quote):
                    errors.append(
                        f"{ref_prefix}: quote not found in episode {ref.episode_id!r}: "
                        f"{ref.quote[:60]!r}..."
                    )

            # Evidence episode count cap
            if len(episode_ids) > self.max_evidence_episodes:
                errors.append(
                    f"{prefix}: evidence spans {len(episode_ids)} episodes "
                    f"(max {self.max_evidence_episodes})"
                )

        return errors

    def validate_hits(self, hits: list[Hit]) -> list[str]:
        """Validate search hits. Returns list of error messages."""
        errors: list[str] = []

        for i, hit in enumerate(hits):
            prefix = f"hit[{i}]"

            if not hit.id.strip():
                errors.append(f"{prefix}: empty id")

            if not hit.kind.strip():
                errors.append(f"{prefix}: empty kind")

            if not hit.text.strip():
                errors.append(f"{prefix}: empty text")

        return errors

    def validate_and_raise(self, insights: list[Insight]) -> None:
        """Validate insights, raising on first evidence mismatch."""
        for insight in insights:
            for ref in insight.evidence:
                if not self.vault.has(ref.episode_id):
                    raise ValidationError(f"Unknown episode_id: {ref.episode_id!r}")
                if not self.vault.verify_quote(ref.episode_id, ref.quote):
                    raise EvidenceError(ref.episode_id, ref.quote)
