"""Unit tests for the evaluation engine and metrics."""

import pytest
from eval_agent_lab.evals.metrics import (
    ExactMatchMetric,
    ContainsAnswerMetric,
    LevenshteinMetric,
    SemanticSimilarityMetric,
    HallucinationDetector,
    ToolSelectionAccuracy,
    ReasoningConsistencyMetric,
    StepEfficiencyMetric,
    get_default_metrics,
)
from eval_agent_lab.evals import EvaluationEngine


@pytest.mark.unit
class TestExactMatch:
    @pytest.mark.asyncio
    async def test_exact_match_true(self):
        m = ExactMatchMetric()
        result = await m.compute("Hello World", "hello world")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_false(self):
        m = ExactMatchMetric()
        result = await m.compute("Hello World", "Goodbye")
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_normalization(self):
        m = ExactMatchMetric()
        result = await m.compute("  Hello   World  ", "hello world")
        assert result.score == 1.0


@pytest.mark.unit
class TestContainsAnswer:
    @pytest.mark.asyncio
    async def test_contains_true(self):
        m = ContainsAnswerMetric()
        result = await m.compute("Python is a programming language", "python")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_contains_false(self):
        m = ContainsAnswerMetric()
        result = await m.compute("JavaScript is great", "python")
        assert result.score == 0.0


@pytest.mark.unit
class TestLevenshtein:
    @pytest.mark.asyncio
    async def test_identical_strings(self):
        m = LevenshteinMetric()
        result = await m.compute("hello", "hello")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_completely_different(self):
        m = LevenshteinMetric()
        result = await m.compute("abc", "xyz")
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_partial_similarity(self):
        m = LevenshteinMetric()
        result = await m.compute("kitten", "sitting")
        assert 0.0 < result.score < 1.0


@pytest.mark.unit
class TestSemanticSimilarity:
    @pytest.mark.asyncio
    async def test_fallback_word_overlap(self):
        m = SemanticSimilarityMetric()
        result = await m.compute("machine learning is great", "machine learning")
        assert result.score > 0.0


@pytest.mark.unit
class TestHallucinationDetector:
    @pytest.mark.asyncio
    async def test_grounded_response(self):
        m = HallucinationDetector()
        context = "Python is a programming language created by Guido van Rossum."
        result = await m.compute(
            "Python was created by Guido van Rossum.", context, context=context
        )
        assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_hallucinated_response(self):
        m = HallucinationDetector()
        context = "Python is a programming language."
        result = await m.compute(
            "Elephants discovered quantum teleportation. "
            "Mars colonists invented neural warp drives. "
            "Underwater cities use blockchain for oxygen trading.",
            context,
            context=context,
        )
        assert result.score < 1.0


@pytest.mark.unit
class TestToolSelectionAccuracy:
    @pytest.mark.asyncio
    async def test_perfect_selection(self):
        m = ToolSelectionAccuracy()
        result = await m.compute("", "", expected_tools=["search"], actual_tools=["search"])
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_wrong_selection(self):
        m = ToolSelectionAccuracy()
        result = await m.compute(
            "", "", expected_tools=["search"], actual_tools=["calculator"]
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_partial_selection(self):
        m = ToolSelectionAccuracy()
        result = await m.compute(
            "", "", expected_tools=["search", "calculator"], actual_tools=["search"]
        )
        assert 0.0 < result.score < 1.0


@pytest.mark.unit
class TestReasoningConsistency:
    @pytest.mark.asyncio
    async def test_with_steps(self):
        m = ReasoningConsistencyMetric()
        steps = [
            {"step_type": "think", "content": "analyzing"},
            {"step_type": "think", "content": "planning"},
            {"step_type": "think", "content": "deciding"},
            {"step_type": "respond", "content": "answer"},
        ]
        result = await m.compute("", "", steps=steps)
        assert result.score > 0.5


@pytest.mark.unit
class TestStepEfficiency:
    @pytest.mark.asyncio
    async def test_efficient(self):
        m = StepEfficiencyMetric()
        result = await m.compute("", "", total_steps=2, max_steps=10)
        assert result.score > 0.8

    @pytest.mark.asyncio
    async def test_inefficient(self):
        m = StepEfficiencyMetric()
        result = await m.compute("", "", total_steps=10, max_steps=10)
        assert result.score < 0.2


@pytest.mark.unit
class TestEvaluationEngine:
    @pytest.mark.asyncio
    async def test_evaluate_item(self):
        engine = EvaluationEngine()
        result = await engine.evaluate_item(
            input_text="What is Python?",
            expected_output="A programming language",
            actual_output="Python is a programming language",
            item_id="test_001",
        )
        assert result.item_id == "test_001"
        assert len(result.metrics) > 0
        assert result.composite_score >= 0.0

    @pytest.mark.asyncio
    async def test_evaluate_batch(self):
        engine = EvaluationEngine()
        items = [
            {"input": "Q1", "expected_output": "A1", "actual_output": "A1"},
            {"input": "Q2", "expected_output": "A2", "actual_output": "wrong"},
        ]
        report = await engine.evaluate_batch(items)
        assert report.total_items == 2
        assert len(report.results) == 2


@pytest.mark.unit
class TestDefaultMetrics:
    def test_get_default_metrics(self):
        metrics = get_default_metrics()
        assert len(metrics) == 8
        names = {m.name for m in metrics}
        assert "exact_match" in names
        assert "semantic_similarity" in names
        assert "hallucination_score" in names
