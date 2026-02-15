from __future__ import annotations

from lens.runner.anticheat import EpisodeVault


class OutputValidator:
    """Validates agent output ref_ids against the episode vault."""

    def __init__(self, vault: EpisodeVault) -> None:
        self.vault = vault

    def validate_refs(self, ref_ids: list[str]) -> tuple[list[str], list[str]]:
        """Validate a list of ref_ids.

        Returns (valid_ids, invalid_ids).
        """
        valid: list[str] = []
        invalid: list[str] = []
        for ref_id in ref_ids:
            if self.vault.has(ref_id):
                valid.append(ref_id)
            else:
                invalid.append(ref_id)
        return valid, invalid
