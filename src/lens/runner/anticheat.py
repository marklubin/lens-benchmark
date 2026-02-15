from __future__ import annotations

from lens.core.errors import AntiCheatError


class EpisodeVault:
    """Stores episode text as ingested. Used runner-side for evidence validation.

    Adapters cannot access the vault. This prevents query-time access to raw
    episode text — adapters can only use whatever they stored during ingest().
    """

    def __init__(self) -> None:
        self._episodes: dict[str, str] = {}  # episode_id -> text

    def store(self, episode_id: str, text: str) -> None:
        """Store episode text at ingest time."""
        self._episodes[episode_id] = text

    def get(self, episode_id: str) -> str:
        """Retrieve episode text for validation. Runner-side only."""
        if episode_id not in self._episodes:
            msg = f"Episode {episode_id!r} not found in vault"
            raise AntiCheatError(msg)
        return self._episodes[episode_id]

    def has(self, episode_id: str) -> bool:
        """Check if an episode exists in the vault."""
        return episode_id in self._episodes

    def verify_quote(self, episode_id: str, quote: str) -> bool:
        """Verify that a quote is an exact substring of the episode text."""
        text = self.get(episode_id)
        return quote in text

    def clear(self) -> None:
        """Clear all stored episodes."""
        self._episodes.clear()

    def clear_scope(self, episode_ids: list[str]) -> None:
        """Clear specific episodes (used when resetting a scope)."""
        for eid in episode_ids:
            self._episodes.pop(eid, None)

    @property
    def episode_count(self) -> int:
        return len(self._episodes)
