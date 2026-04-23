"""Evaluation metrics: deterministic, semantic, and LLM-as-judge."""

from __future__ import annotations

import abc
import re
from typing import Any

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Result from a single metric computation."""

    metric_name: str
    score: float  # 0.0 - 1.0 normalized
    raw_score: Any = None
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class BaseMetric(abc.ABC):
    """Abstract base class for evaluation metrics."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult: ...


# ---------------------------------------------------------------------------
# Deterministic Metrics
# ---------------------------------------------------------------------------


class ExactMatchMetric(BaseMetric):
    """Exact string match (case-insensitive, whitespace-normalized)."""

    @property
    def name(self) -> str:
        return "exact_match"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        pred_norm = " ".join(prediction.strip().lower().split())
        ref_norm = " ".join(reference.strip().lower().split())
        match = pred_norm == ref_norm
        return MetricResult(
            metric_name=self.name,
            score=1.0 if match else 0.0,
            raw_score=match,
            details={"prediction": pred_norm, "reference": ref_norm},
        )


class AcceptableOutputMetric(BaseMetric):
    """Match prediction against a list of acceptable outputs.

    Returns 1.0 if the prediction matches *any* acceptable output
    (case-insensitive, whitespace-normalized).  Falls back to the
    primary ``expected_output`` when no acceptable_outputs are provided.
    """

    @property
    def name(self) -> str:
        return "acceptable_output_match"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        acceptable: list[str] = kwargs.get("acceptable_outputs", [])
        candidates = list(acceptable) if acceptable else [reference]
        pred_norm = " ".join(prediction.strip().lower().split())

        for candidate in candidates:
            cand_norm = " ".join(candidate.strip().lower().split())
            if pred_norm == cand_norm or cand_norm in pred_norm:
                return MetricResult(
                    metric_name=self.name,
                    score=1.0,
                    raw_score=True,
                    details={"matched": candidate},
                )

        return MetricResult(
            metric_name=self.name,
            score=0.0,
            raw_score=False,
            details={"candidates": candidates},
        )


class ContainsAnswerMetric(BaseMetric):
    """Check if the reference answer is contained in the prediction."""

    @property
    def name(self) -> str:
        return "contains_answer"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        pred_lower = prediction.lower()
        ref_lower = reference.lower()
        contains = ref_lower in pred_lower
        return MetricResult(
            metric_name=self.name, score=1.0 if contains else 0.0, raw_score=contains
        )


class ContainsExpectedMetric(BaseMetric):
    """Check if the prediction contains all expected keywords/phrases.

    Uses the ``expected_contains`` field from the dataset item for
    lightweight soft matching — critical for LLM outputs that may
    rephrase but should still include key terms.
    """

    @property
    def name(self) -> str:
        return "contains_expected"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        expected: list[str] = kwargs.get("expected_contains", [])
        if not expected:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"note": "No expected_contains specified"},
            )

        pred_lower = prediction.lower()
        found = [kw for kw in expected if kw.lower() in pred_lower]
        missing = [kw for kw in expected if kw.lower() not in pred_lower]
        score = len(found) / len(expected)

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            raw_score=len(found),
            details={"found": found, "missing": missing},
        )


class LevenshteinMetric(BaseMetric):
    """Normalized Levenshtein distance similarity."""

    @property
    def name(self) -> str:
        return "levenshtein_similarity"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        s1, s2 = prediction.lower(), reference.lower()
        if not s1 and not s2:
            return MetricResult(metric_name=self.name, score=1.0)
        if not s1 or not s2:
            return MetricResult(metric_name=self.name, score=0.0)

        rows = len(s1) + 1
        cols = len(s2) + 1
        dist = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            dist[i][0] = i
        for j in range(cols):
            dist[0][j] = j
        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j - 1] + 1, dist[i - 1][j - 1] + cost)
        max_len = max(len(s1), len(s2))
        similarity = 1 - dist[rows - 1][cols - 1] / max_len
        return MetricResult(
            metric_name=self.name, score=round(similarity, 4), raw_score=dist[rows - 1][cols - 1]
        )


# ---------------------------------------------------------------------------
# Semantic Metrics
# ---------------------------------------------------------------------------


class SemanticSimilarityMetric(BaseMetric):
    """Embedding-based semantic similarity using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                return None
        return self._model

    @property
    def name(self) -> str:
        return "semantic_similarity"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        model = self._get_model()
        if model is None:
            # Fallback: word overlap
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            if not pred_words or not ref_words:
                return MetricResult(
                    metric_name=self.name, score=0.0, details={"fallback": "word_overlap"}
                )
            overlap = len(pred_words & ref_words) / len(pred_words | ref_words)
            return MetricResult(
                metric_name=self.name, score=round(overlap, 4), details={"fallback": "word_overlap"}
            )

        import numpy as np

        embs = model.encode([prediction, reference])
        cos_sim = float(
            np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8)
        )
        # Normalize to 0-1
        score = max(0.0, min(1.0, (cos_sim + 1) / 2))
        return MetricResult(metric_name=self.name, score=round(score, 4), raw_score=cos_sim)


# ---------------------------------------------------------------------------
# Hallucination Detection
# ---------------------------------------------------------------------------


class HallucinationDetector(BaseMetric):
    """Simple hallucination detection via factual overlap analysis."""

    @property
    def name(self) -> str:
        return "hallucination_score"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        context = kwargs.get("context", reference)
        pred_sentences = [s.strip() for s in re.split(r"[.!?]", prediction) if s.strip()]
        context_lower = context.lower()

        if not pred_sentences:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"supported": 0, "total": 0},
            )

        supported = 0
        unsupported_claims = []
        for sent in pred_sentences:
            words = sent.lower().split()
            if len(words) < 3:
                supported += 1
                continue
            overlap = sum(1 for w in words if w in context_lower)
            ratio = overlap / len(words)
            if ratio >= 0.3:
                supported += 1
            else:
                unsupported_claims.append(sent)

        score = supported / len(pred_sentences)
        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            details={
                "supported": supported,
                "total": len(pred_sentences),
                "unsupported_claims": unsupported_claims[:5],
            },
        )


# ---------------------------------------------------------------------------
# Agent-Specific Metrics
# ---------------------------------------------------------------------------


class ToolSelectionAccuracy(BaseMetric):
    """Evaluate if the agent selected the correct tools."""

    @property
    def name(self) -> str:
        return "tool_selection_accuracy"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        expected_tools: list[str] = kwargs.get("expected_tools", [])
        actual_tools: list[str] = kwargs.get("actual_tools", [])

        if not expected_tools:
            return MetricResult(
                metric_name=self.name, score=1.0, details={"note": "No expected tools specified"}
            )

        expected_set = set(expected_tools)
        actual_set = set(actual_tools)

        if not expected_set:
            return MetricResult(metric_name=self.name, score=1.0)

        correct = len(expected_set & actual_set)
        precision = correct / len(actual_set) if actual_set else 0.0
        recall = correct / len(expected_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            metric_name=self.name,
            score=round(f1, 4),
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "expected": expected_tools,
                "actual": actual_tools,
            },
        )


class ToolStrategyComplianceMetric(BaseMetric):
    """Evaluate whether the agent followed the prescribed tool strategy.

    - ``must_use``: all expected tools MUST be used → penalize missing
    - ``optional``: tool usage is acceptable but not required → no penalty
    - ``forbidden``: NO tools should be used → penalize any tool call
    """

    @property
    def name(self) -> str:
        return "tool_strategy_compliance"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        strategy: str = kwargs.get("tool_strategy", "optional")
        expected_tools: list[str] = kwargs.get("expected_tools", [])
        actual_tools: list[str] = kwargs.get("actual_tools", [])

        if strategy == "forbidden":
            # Any tool use is wrong
            if actual_tools:
                return MetricResult(
                    metric_name=self.name,
                    score=0.0,
                    details={
                        "strategy": strategy,
                        "violation": f"Used tools when forbidden: {actual_tools}",
                    },
                )
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"strategy": strategy, "compliance": "No tools used (correct)"},
            )

        if strategy == "must_use":
            if not expected_tools:
                return MetricResult(
                    metric_name=self.name,
                    score=1.0,
                    details={"strategy": strategy, "note": "No expected tools defined"},
                )
            expected_set = set(expected_tools)
            actual_set = set(actual_tools)
            recall = len(expected_set & actual_set) / len(expected_set)
            return MetricResult(
                metric_name=self.name,
                score=round(recall, 4),
                details={
                    "strategy": strategy,
                    "expected": expected_tools,
                    "actual": actual_tools,
                    "missing": sorted(expected_set - actual_set),
                },
            )

        # strategy == "optional" → always compliant
        return MetricResult(
            metric_name=self.name,
            score=1.0,
            details={"strategy": strategy, "compliance": "Tools are optional"},
        )


class ReasoningConsistencyMetric(BaseMetric):
    """Evaluate reasoning consistency from agent steps."""

    @property
    def name(self) -> str:
        return "reasoning_consistency"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        steps: list[dict[str, Any]] = kwargs.get("steps", [])
        if not steps:
            return MetricResult(
                metric_name=self.name, score=0.5, details={"note": "No steps provided"}
            )

        # Heuristics for reasoning quality
        scores = []
        # 1. Did the agent produce thinking steps?
        think_steps = [s for s in steps if s.get("step_type") == "think"]
        scores.append(min(len(think_steps) / 3, 1.0))

        # 2. Were there errors?
        error_steps = [s for s in steps if s.get("step_type") == "error"]
        scores.append(1.0 - min(len(error_steps) / 3, 1.0))

        # 3. Did the agent reach a conclusion?
        respond_steps = [s for s in steps if s.get("step_type") == "respond"]
        scores.append(1.0 if respond_steps else 0.0)

        avg_score = sum(scores) / len(scores)
        return MetricResult(
            metric_name=self.name,
            score=round(avg_score, 4),
            details={
                "think_steps": len(think_steps),
                "error_steps": len(error_steps),
                "has_conclusion": bool(respond_steps),
            },
        )


class StepEfficiencyMetric(BaseMetric):
    """Evaluate the efficiency of agent execution.

    Scores degrade linearly as ``total_steps`` approaches ``max_steps``.
    When ``penalize_overuse`` is True, an additional penalty is applied
    for exceeding the expected step budget.
    """

    @property
    def name(self) -> str:
        return "step_efficiency"

    async def compute(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        total_steps: int = kwargs.get("total_steps", 0)
        max_steps: int = kwargs.get("max_steps", 10)
        penalize_overuse: bool = kwargs.get("penalize_overuse", False)

        if total_steps == 0:
            return MetricResult(metric_name=self.name, score=0.0)

        # Base efficiency: optimal is 1-3 steps, degrades linearly
        efficiency = max(0.0, 1.0 - (total_steps - 1) / max_steps)

        # Overuse penalty: halve the score if steps exceed max_steps
        overuse_penalty = False
        if penalize_overuse and total_steps > max_steps:
            efficiency *= 0.5
            overuse_penalty = True

        return MetricResult(
            metric_name=self.name,
            score=round(efficiency, 4),
            details={
                "total_steps": total_steps,
                "max_steps": max_steps,
                "penalize_overuse": penalize_overuse,
                "overuse_penalty_applied": overuse_penalty,
            },
        )


# ---------------------------------------------------------------------------
# Metric Registry
# ---------------------------------------------------------------------------


def get_default_metrics() -> list[BaseMetric]:
    """Return all default metrics.

    Includes the three evaluation axes:
      - **Correctness**: exact_match, acceptable_output_match,
        contains_answer, contains_expected, levenshtein, semantic
      - **Skill Adherence**: tool_selection_accuracy,
        tool_strategy_compliance, reasoning_consistency
      - **Execution Efficiency**: step_efficiency
    """
    return [
        # Correctness
        ExactMatchMetric(),
        AcceptableOutputMetric(),
        ContainsAnswerMetric(),
        ContainsExpectedMetric(),
        LevenshteinMetric(),
        SemanticSimilarityMetric(),
        HallucinationDetector(),
        # Skill Adherence
        ToolSelectionAccuracy(),
        ToolStrategyComplianceMetric(),
        ReasoningConsistencyMetric(),
        # Execution Efficiency
        StepEfficiencyMetric(),
    ]
