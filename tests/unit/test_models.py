from __future__ import annotations

from datetime import datetime

import pytest

from lens.core.models import (
    CheckpointResult,
    Episode,
    EvidenceFragment,
    EvidenceRef,
    Hit,
    Insight,
    MetricResult,
    PersonaResult,
    RunResult,
    ScoreCard,
    TruthPattern,
)


class TestEvidenceRef:
    def test_round_trip(self):
        ref = EvidenceRef(episode_id="ep_001", quote="some text")
        d = ref.to_dict()
        restored = EvidenceRef.from_dict(d)
        assert restored == ref

    def test_frozen(self):
        ref = EvidenceRef(episode_id="ep_001", quote="text")
        with pytest.raises(AttributeError):
            ref.episode_id = "changed"


class TestInsight:
    def test_valid_confidence(self):
        insight = Insight(
            text="test",
            confidence=0.5,
            evidence=[],
            falsifier="falsifier",
        )
        assert insight.confidence == 0.5

    def test_invalid_confidence_high(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            Insight(text="test", confidence=1.5, evidence=[], falsifier="f")

    def test_invalid_confidence_low(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            Insight(text="test", confidence=-0.1, evidence=[], falsifier="f")

    def test_round_trip(self):
        insight = Insight(
            text="pattern found",
            confidence=0.8,
            evidence=[
                EvidenceRef(episode_id="ep_001", quote="quote1"),
                EvidenceRef(episode_id="ep_002", quote="quote2"),
            ],
            falsifier="if contradicted",
        )
        d = insight.to_dict()
        restored = Insight.from_dict(d)
        assert restored.text == insight.text
        assert restored.confidence == insight.confidence
        assert len(restored.evidence) == 2


class TestHit:
    def test_round_trip(self):
        hit = Hit(id="h1", kind="episode", text="content", score=0.95)
        d = hit.to_dict()
        restored = Hit.from_dict(d)
        assert restored == hit


class TestEpisode:
    def test_round_trip(self):
        ep = Episode(
            episode_id="ep_001",
            persona_id="p1",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            text="episode text",
            meta={"key": "value"},
        )
        d = ep.to_dict()
        restored = Episode.from_dict(d)
        assert restored.episode_id == ep.episode_id
        assert restored.timestamp == ep.timestamp
        assert restored.meta == ep.meta

    def test_from_dict_string_timestamp(self):
        d = {
            "episode_id": "ep_001",
            "persona_id": "p1",
            "timestamp": "2024-01-15T10:00:00",
            "text": "text",
        }
        ep = Episode.from_dict(d)
        assert ep.timestamp == datetime(2024, 1, 15, 10, 0, 0)


class TestTruthPattern:
    def test_round_trip(self):
        tp = TruthPattern(
            pattern_id="tp_01",
            persona_id="p1",
            canonical_insight="insight text",
            insight_category="trend",
            evidence_episode_ids=["ep_001", "ep_002"],
            evidence_fragments=[
                EvidenceFragment(episode_id="ep_001", fragment="frag1"),
            ],
            min_episodes_required=2,
            first_signal_episode=1,
            difficulty="easy",
            expected_confidence=0.9,
        )
        d = tp.to_dict()
        restored = TruthPattern.from_dict(d)
        assert restored.pattern_id == tp.pattern_id
        assert restored.supersedes is None


class TestScoreCard:
    def test_round_trip(self):
        sc = ScoreCard(
            run_id="abc123",
            adapter="null",
            dataset_version="0.1.0",
            budget_preset="standard",
            metrics=[
                MetricResult(name="ev", tier=1, value=0.95),
            ],
            composite_score=0.5,
        )
        d = sc.to_dict()
        restored = ScoreCard.from_dict(d)
        assert restored.run_id == sc.run_id
        assert len(restored.metrics) == 1
        assert restored.composite_score == 0.5
