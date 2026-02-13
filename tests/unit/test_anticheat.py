from __future__ import annotations

import pytest

from lens.core.errors import AntiCheatError
from lens.runner.anticheat import EpisodeVault


class TestEpisodeVault:
    def test_store_and_get(self):
        vault = EpisodeVault()
        vault.store("ep_001", "This is episode text")
        assert vault.get("ep_001") == "This is episode text"

    def test_get_missing_raises(self):
        vault = EpisodeVault()
        with pytest.raises(AntiCheatError, match="not found"):
            vault.get("nonexistent")

    def test_has(self):
        vault = EpisodeVault()
        vault.store("ep_001", "text")
        assert vault.has("ep_001")
        assert not vault.has("ep_002")

    def test_verify_quote_found(self):
        vault = EpisodeVault()
        vault.store("ep_001", "The evidence_fragment is here in the text")
        assert vault.verify_quote("ep_001", "evidence_fragment")

    def test_verify_quote_not_found(self):
        vault = EpisodeVault()
        vault.store("ep_001", "Some episode text")
        assert not vault.verify_quote("ep_001", "nonexistent quote")

    def test_clear(self):
        vault = EpisodeVault()
        vault.store("ep_001", "text1")
        vault.store("ep_002", "text2")
        vault.clear()
        assert vault.episode_count == 0

    def test_clear_persona(self):
        vault = EpisodeVault()
        vault.store("ep_001", "text1")
        vault.store("ep_002", "text2")
        vault.store("ep_003", "text3")
        vault.clear_persona(["ep_001", "ep_002"])
        assert vault.episode_count == 1
        assert vault.has("ep_003")
