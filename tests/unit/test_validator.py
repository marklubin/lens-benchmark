from __future__ import annotations

import pytest

from lens.runner.anticheat import EpisodeVault
from lens.runner.validator import OutputValidator


@pytest.fixture
def vault() -> EpisodeVault:
    v = EpisodeVault()
    v.store("ep_001", "This contains evidence_fragment_1 and other text")
    v.store("ep_002", "This contains evidence_fragment_2 in the middle")
    v.store("ep_003", "This contains evidence_fragment_3 at the end")
    return v


@pytest.fixture
def validator(vault: EpisodeVault) -> OutputValidator:
    return OutputValidator(vault)


class TestOutputValidator:
    def test_validate_refs_all_valid(self, validator):
        valid, invalid = validator.validate_refs(["ep_001", "ep_002", "ep_003"])
        assert valid == ["ep_001", "ep_002", "ep_003"]
        assert invalid == []

    def test_validate_refs_some_invalid(self, validator):
        valid, invalid = validator.validate_refs(["ep_001", "ep_999", "ep_002"])
        assert valid == ["ep_001", "ep_002"]
        assert invalid == ["ep_999"]

    def test_validate_refs_all_invalid(self, validator):
        valid, invalid = validator.validate_refs(["ep_998", "ep_999"])
        assert valid == []
        assert invalid == ["ep_998", "ep_999"]

    def test_validate_refs_empty(self, validator):
        valid, invalid = validator.validate_refs([])
        assert valid == []
        assert invalid == []
