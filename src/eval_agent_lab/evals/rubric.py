"""Rubric system: formal, weighted evaluation rubric definitions.

A rubric controls how individual metric scores are combined into a
composite score. Rubrics can be loaded from JSON files, passed via
CLI, or constructed programmatically.

Design decisions:
  - MetricWeight.weight is a raw weight, NOT a probability.  The
    RubricConfig normalizes on access so users can write intuitive
    numbers (e.g. 3, 2, 1) without manually summing to 1.0.
  - A ``strict`` mode flag optionally enforces that weights
    sum to exactly 1.0 (for reproducibility audits).
  - Unknown metrics (present in results but absent from the rubric)
    get ``default_weight`` so the system never silently drops data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from eval_agent_lab.exceptions import EvaluationError


class RubricValidationError(EvaluationError):
    """Rubric failed validation."""


class MetricWeight(BaseModel):
    """A single metric name → weight pair inside a rubric."""

    name: str = Field(..., description="Metric name (must match a registered metric)")
    weight: float = Field(..., ge=0.0, description="Relative weight (>=0)")


class RubricConfig(BaseModel):
    """Formal rubric that defines how metrics are weighted.

    Attributes:
        name: Human-readable rubric name.
        description: Optional description of what this rubric targets.
        metrics: Explicit list of metric weights.
        default_weight: Weight assigned to metrics NOT listed explicitly.
        judge_weight: Weight assigned to the LLM-as-judge score (0 to skip).
        strict: When True, ``metrics`` weights must sum to exactly 1.0
                and ``default_weight`` must be 0.
        pass_threshold: Composite score >= this value counts as "passed".
    """

    name: str = "default"
    description: str = ""
    metrics: List[MetricWeight] = Field(default_factory=list)
    default_weight: float = Field(default=1.0, ge=0.0)
    judge_weight: float = Field(default=2.0, ge=0.0)
    strict: bool = False
    pass_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_rubric(self) -> "RubricConfig":
        """Enforce strict-mode constraints when enabled."""
        if not self.strict:
            return self

        if self.default_weight != 0.0:
            raise RubricValidationError(
                "In strict mode, default_weight must be 0.0 "
                "(all metrics must be listed explicitly)."
            )

        total = sum(mw.weight for mw in self.metrics)
        if self.judge_weight > 0:
            total += self.judge_weight

        if abs(total - 1.0) > 1e-6:
            raise RubricValidationError(
                f"In strict mode, metric weights (including judge_weight) "
                f"must sum to 1.0, got {total:.6f}."
            )

        return self

    # ------------------------------------------------------------------
    # Weight resolution helpers
    # ------------------------------------------------------------------

    def get_weight_map(self) -> Dict[str, float]:
        """Return {metric_name: weight} for explicitly listed metrics."""
        return {mw.name: mw.weight for mw in self.metrics}

    def weight_for(self, metric_name: str) -> float:
        """Return the weight for a given metric name, falling back to default."""
        weight_map = self.get_weight_map()
        return weight_map.get(metric_name, self.default_weight)

    @property
    def metric_names(self) -> List[str]:
        """Return the list of explicitly configured metric names."""
        return [mw.name for mw in self.metrics]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "RubricConfig":
        """Load a rubric from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise RubricValidationError(f"Rubric file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise RubricValidationError(
                f"Invalid JSON in rubric file {path}: {exc}"
            ) from exc

        try:
            return cls(**data)
        except Exception as exc:
            raise RubricValidationError(
                f"Failed to parse rubric from {path}: {exc}"
            ) from exc

    def to_json(self, path: str | Path) -> None:
        """Persist rubric to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def balanced(cls) -> "RubricConfig":
        """Return a balanced rubric where every metric has weight 1.0."""
        return cls(name="balanced", description="Equal-weight rubric")

    @classmethod
    def accuracy_focused(cls) -> "RubricConfig":
        """Return a rubric that emphasizes correctness metrics."""
        return cls(
            name="accuracy_focused",
            description="Prioritizes exact match, contains answer, and semantic similarity",
            metrics=[
                MetricWeight(name="exact_match", weight=3.0),
                MetricWeight(name="contains_answer", weight=2.5),
                MetricWeight(name="semantic_similarity", weight=2.0),
                MetricWeight(name="levenshtein_similarity", weight=1.0),
                MetricWeight(name="hallucination_score", weight=1.5),
                MetricWeight(name="tool_selection_accuracy", weight=1.0),
                MetricWeight(name="reasoning_consistency", weight=0.5),
                MetricWeight(name="step_efficiency", weight=0.5),
            ],
            judge_weight=2.0,
        )

    @classmethod
    def agent_focused(cls) -> "RubricConfig":
        """Return a rubric that emphasizes agent behaviour metrics."""
        return cls(
            name="agent_focused",
            description="Prioritizes tool use, reasoning, and step efficiency",
            metrics=[
                MetricWeight(name="exact_match", weight=1.0),
                MetricWeight(name="contains_answer", weight=1.0),
                MetricWeight(name="semantic_similarity", weight=1.0),
                MetricWeight(name="levenshtein_similarity", weight=0.5),
                MetricWeight(name="hallucination_score", weight=1.5),
                MetricWeight(name="tool_selection_accuracy", weight=3.0),
                MetricWeight(name="reasoning_consistency", weight=2.5),
                MetricWeight(name="step_efficiency", weight=2.0),
            ],
            judge_weight=1.0,
        )
