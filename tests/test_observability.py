"""Unit tests for observability: cost tracking and trace logging."""

import pytest
from eval_agent_lab.observability import CostTracker


@pytest.mark.unit
class TestCostTracker:
    def test_record_cost(self):
        tracker = CostTracker()
        record = tracker.record("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500,
                               latency_ms=200.0)
        assert record["total_tokens"] == 1500
        assert record["estimated_cost_usd"] > 0

    def test_multiple_records(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 1000, 500, 200.0)
        tracker.record("gpt-4o-mini", 2000, 1000, 300.0)
        assert tracker.total_tokens == 4500
        assert tracker.total_cost > 0

    def test_summary(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 1000, 500, 200.0)
        summary = tracker.summary()
        assert summary["total_requests"] == 1
        assert summary["total_tokens"] == 1500
        assert "total_cost_usd" in summary

    def test_empty_summary(self):
        tracker = CostTracker()
        summary = tracker.summary()
        assert summary["total_requests"] == 0
        assert summary["total_tokens"] == 0
