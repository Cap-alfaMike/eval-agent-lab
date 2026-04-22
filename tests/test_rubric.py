"""Unit tests for the rubric system."""

import json
import pytest

from eval_agent_lab.evals.rubric import (
    MetricWeight,
    RubricConfig,
    RubricValidationError,
)


@pytest.mark.unit
class TestMetricWeight:
    def test_valid_weight(self):
        mw = MetricWeight(name="exact_match", weight=2.0)
        assert mw.name == "exact_match"
        assert mw.weight == 2.0

    def test_zero_weight(self):
        mw = MetricWeight(name="step_efficiency", weight=0.0)
        assert mw.weight == 0.0

    def test_negative_weight_raises(self):
        with pytest.raises(Exception):
            MetricWeight(name="bad", weight=-1.0)


@pytest.mark.unit
class TestRubricConfig:
    def test_balanced_factory(self):
        rubric = RubricConfig.balanced()
        assert rubric.name == "balanced"
        assert rubric.default_weight == 1.0
        assert len(rubric.metrics) == 0

    def test_accuracy_focused_factory(self):
        rubric = RubricConfig.accuracy_focused()
        assert rubric.name == "accuracy_focused"
        assert len(rubric.metrics) == 8
        # exact_match should have highest weight
        em = next(m for m in rubric.metrics if m.name == "exact_match")
        assert em.weight == 3.0

    def test_agent_focused_factory(self):
        rubric = RubricConfig.agent_focused()
        assert rubric.name == "agent_focused"
        # tool_selection_accuracy should have highest weight
        ts = next(m for m in rubric.metrics if m.name == "tool_selection_accuracy")
        assert ts.weight == 3.0

    def test_weight_for_explicit(self):
        rubric = RubricConfig(
            metrics=[MetricWeight(name="exact_match", weight=5.0)],
            default_weight=1.0,
        )
        assert rubric.weight_for("exact_match") == 5.0

    def test_weight_for_fallback(self):
        rubric = RubricConfig(
            metrics=[MetricWeight(name="exact_match", weight=5.0)],
            default_weight=1.0,
        )
        assert rubric.weight_for("unknown_metric") == 1.0

    def test_get_weight_map(self):
        rubric = RubricConfig(
            metrics=[
                MetricWeight(name="a", weight=1.0),
                MetricWeight(name="b", weight=2.0),
            ],
        )
        wm = rubric.get_weight_map()
        assert wm == {"a": 1.0, "b": 2.0}

    def test_metric_names(self):
        rubric = RubricConfig(
            metrics=[
                MetricWeight(name="a", weight=1.0),
                MetricWeight(name="b", weight=2.0),
            ],
        )
        assert rubric.metric_names == ["a", "b"]


@pytest.mark.unit
class TestRubricStrictMode:
    def test_strict_valid(self):
        """Weights sum to 1.0 in strict mode."""
        rubric = RubricConfig(
            strict=True,
            default_weight=0.0,
            judge_weight=0.0,
            metrics=[
                MetricWeight(name="a", weight=0.6),
                MetricWeight(name="b", weight=0.4),
            ],
        )
        assert rubric.strict is True

    def test_strict_invalid_sum(self):
        """Weights don't sum to 1.0 should raise."""
        with pytest.raises(RubricValidationError, match="must sum to 1.0"):
            RubricConfig(
                strict=True,
                default_weight=0.0,
                judge_weight=0.0,
                metrics=[
                    MetricWeight(name="a", weight=0.5),
                    MetricWeight(name="b", weight=0.3),
                ],
            )

    def test_strict_nonzero_default_raises(self):
        """Strict mode requires default_weight=0."""
        with pytest.raises(RubricValidationError, match="default_weight must be 0.0"):
            RubricConfig(
                strict=True,
                default_weight=1.0,
                metrics=[MetricWeight(name="a", weight=1.0)],
            )


@pytest.mark.unit
class TestRubricSerialization:
    def test_to_and_from_json(self, tmp_path):
        rubric = RubricConfig(
            name="test_rubric",
            description="Test",
            metrics=[MetricWeight(name="exact_match", weight=2.0)],
            default_weight=1.0,
            pass_threshold=0.6,
        )
        path = tmp_path / "rubric.json"
        rubric.to_json(path)

        loaded = RubricConfig.from_json(path)
        assert loaded.name == "test_rubric"
        assert loaded.pass_threshold == 0.6
        assert len(loaded.metrics) == 1
        assert loaded.metrics[0].name == "exact_match"

    def test_from_json_missing_file(self):
        with pytest.raises(RubricValidationError, match="not found"):
            RubricConfig.from_json("nonexistent.json")

    def test_from_json_invalid(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(RubricValidationError, match="Invalid JSON"):
            RubricConfig.from_json(path)
