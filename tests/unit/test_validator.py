from __future__ import annotations

import pytest

from lens.core.errors import EvidenceError
from lens.core.models import EvidenceRef, Insight
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
    def test_valid_insights(self, validator):
        insights = [
            Insight(
                text="valid insight",
                confidence=0.8,
                evidence=[
                    EvidenceRef(episode_id="ep_001", quote="evidence_fragment_1"),
                    EvidenceRef(episode_id="ep_002", quote="evidence_fragment_2"),
                    EvidenceRef(episode_id="ep_003", quote="evidence_fragment_3"),
                ],
                falsifier="test falsifier",
            )
        ]
        errors = validator.validate_insights(insights)
        assert errors == []

    def test_invalid_quote(self, validator):
        insights = [
            Insight(
                text="insight with bad quote",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="ep_001", quote="nonexistent_text"),
                ],
                falsifier="test",
            )
        ]
        errors = validator.validate_insights(insights)
        assert any("quote not found" in e for e in errors)

    def test_unknown_episode(self, validator):
        insights = [
            Insight(
                text="insight with unknown episode",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="ep_999", quote="text"),
                ],
                falsifier="test",
            )
        ]
        errors = validator.validate_insights(insights)
        assert any("unknown episode_id" in e for e in errors)

    def test_empty_text(self, validator):
        insights = [
            Insight(
                text="",
                confidence=0.5,
                evidence=[],
                falsifier="test",
            )
        ]
        errors = validator.validate_insights(insights)
        assert any("empty text" in e for e in errors)

    def test_empty_falsifier(self, validator):
        insights = [
            Insight(
                text="insight",
                confidence=0.5,
                evidence=[],
                falsifier="",
            )
        ]
        errors = validator.validate_insights(insights)
        assert any("empty falsifier" in e for e in errors)

    def test_validate_and_raise(self, validator):
        insights = [
            Insight(
                text="test",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="ep_001", quote="nonexistent"),
                ],
                falsifier="test",
            )
        ]
        with pytest.raises(EvidenceError):
            validator.validate_and_raise(insights)
