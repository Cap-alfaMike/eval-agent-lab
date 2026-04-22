"""Unit tests for run comparison (A/B testing)."""

import json
import pytest
from pathlib import Path

from eval_agent_lab.evals.comparison import (
    MetricDelta,
    RunComparison,
    compare_runs,
    ComparisonError,
)


def _make_report(report_id: str, model: str, metrics: dict, items: int = 10,
                 success: int = 8) -> dict:
    """Helper to build a minimal report dict."""
    return {
        "report_id": report_id,
        "model": model,
        "dataset_name": "test_dataset",
        "total_items": items,
        "successful_items": success,
        "failed_items": items - success,
        "aggregate_metrics": metrics,
    }


@pytest.mark.unit
class TestCompareRuns:
    def test_basic_comparison(self, tmp_path):
        report_a = _make_report("run_a", "gpt-4o-mini", {
            "avg_exact_match": 0.5,
            "avg_composite": 0.72,
        })
        report_b = _make_report("run_b", "gpt-4o", {
            "avg_exact_match": 0.6,
            "avg_composite": 0.81,
        })
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(report_a), encoding="utf-8")
        path_b.write_text(json.dumps(report_b), encoding="utf-8")

        result = compare_runs(path_a, path_b)
        assert result.run_a_id == "run_a"
        assert result.run_b_id == "run_b"
        assert result.improved_count > 0
        assert len(result.metric_deltas) > 0

    def test_regression_detected(self, tmp_path):
        report_a = _make_report("a", "model", {"avg_composite": 0.9}, success=9)
        report_b = _make_report("b", "model", {"avg_composite": 0.5}, success=5)
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(report_a), encoding="utf-8")
        path_b.write_text(json.dumps(report_b), encoding="utf-8")

        result = compare_runs(path_a, path_b)
        assert result.regressed_count > 0
        assert "BETTER" in result.summary or "EQUIVALENT" in result.summary

    def test_unchanged_within_threshold(self, tmp_path):
        report_a = _make_report("a", "model", {"avg_composite": 0.7000})
        report_b = _make_report("b", "model", {"avg_composite": 0.7001})
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(report_a), encoding="utf-8")
        path_b.write_text(json.dumps(report_b), encoding="utf-8")

        result = compare_runs(path_a, path_b, threshold=0.005)
        # avg_composite delta is 0.0001, below threshold
        composite_delta = next(d for d in result.metric_deltas if d.metric == "avg_composite")
        assert composite_delta.direction == "unchanged"

    def test_missing_file_raises(self):
        with pytest.raises(ComparisonError, match="not found"):
            compare_runs("nonexistent_a.json", "nonexistent_b.json")

    def test_metrics_union(self, tmp_path):
        """Metrics only in one run get 0.0 for the other."""
        report_a = _make_report("a", "model", {"metric_x": 0.5})
        report_b = _make_report("b", "model", {"metric_y": 0.8})
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(report_a), encoding="utf-8")
        path_b.write_text(json.dumps(report_b), encoding="utf-8")

        result = compare_runs(path_a, path_b)
        metric_names = {d.metric for d in result.metric_deltas}
        assert "metric_x" in metric_names
        assert "metric_y" in metric_names

    def test_summary_text(self, tmp_path):
        report_a = _make_report("a", "model", {"m1": 0.5, "m2": 0.5})
        report_b = _make_report("b", "model", {"m1": 0.9, "m2": 0.9})
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(report_a), encoding="utf-8")
        path_b.write_text(json.dumps(report_b), encoding="utf-8")

        result = compare_runs(path_a, path_b)
        assert "Run B is BETTER" in result.summary
