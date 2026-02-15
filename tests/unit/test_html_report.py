from __future__ import annotations

from lens.core.models import MetricResult, ScoreCard
from lens.report.html_report import generate_html_report


class TestHTMLReport:
    def _make_scorecard(self) -> ScoreCard:
        return ScoreCard(
            run_id="abc123",
            adapter="sqlite",
            dataset_version="1.0",
            budget_preset="standard",
            metrics=[
                MetricResult(name="evidence_grounding", tier=1, value=0.85, details={"matched": 17, "total": 20}),
                MetricResult(name="budget_compliance", tier=1, value=1.0, details={}),
                MetricResult(name="reasoning_quality", tier=2, value=0.45, details={"avg_score": 0.45}),
                MetricResult(name="temporal_accuracy", tier=3, value=0.2, details={"correct": 2, "total": 10}),
            ],
            composite_score=0.625,
        )

    def test_generates_valid_html(self):
        sc = self._make_scorecard()
        html = generate_html_report(sc)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_run_metadata(self):
        sc = self._make_scorecard()
        html = generate_html_report(sc)
        assert "abc123" in html
        assert "sqlite" in html
        assert "1.0" in html
        assert "standard" in html

    def test_contains_composite_score(self):
        sc = self._make_scorecard()
        html = generate_html_report(sc)
        assert "0.6250" in html

    def test_contains_all_metrics(self):
        sc = self._make_scorecard()
        html = generate_html_report(sc)
        assert "evidence_grounding" in html
        assert "budget_compliance" in html
        assert "reasoning_quality" in html
        assert "temporal_accuracy" in html

    def test_contains_metric_scores(self):
        sc = self._make_scorecard()
        html = generate_html_report(sc)
        assert "0.8500" in html
        assert "1.0000" in html
        assert "0.4500" in html

    def test_contains_details(self):
        sc = self._make_scorecard()
        html = generate_html_report(sc)
        assert "matched" in html
        assert "17" in html

    def test_score_colors(self):
        sc = self._make_scorecard()
        html = generate_html_report(sc)
        # Green for high scores (>=70%)
        assert "#22c55e" in html
        # Yellow for medium scores (40-70%)
        assert "#eab308" in html
        # Red for low scores (<40%)
        assert "#ef4444" in html

    def test_html_escaping(self):
        sc = ScoreCard(
            run_id="<script>alert('xss')</script>",
            adapter="test",
            dataset_version="1.0",
            budget_preset="fast",
            metrics=[],
            composite_score=0.0,
        )
        html = generate_html_report(sc)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_empty_metrics(self):
        sc = ScoreCard(
            run_id="empty",
            adapter="null",
            dataset_version="1.0",
            budget_preset="fast",
            metrics=[],
            composite_score=0.0,
        )
        html = generate_html_report(sc)
        assert "<!DOCTYPE html>" in html
        assert "empty" in html
