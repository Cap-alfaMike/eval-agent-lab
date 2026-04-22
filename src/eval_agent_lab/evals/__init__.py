"""Evaluation engine: orchestrates metric computation and LLM-as-judge."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from eval_agent_lab.agents import AgentTrace
from eval_agent_lab.evals.metrics import BaseMetric, MetricResult, get_default_metrics
from eval_agent_lab.evals.rubric import RubricConfig
from eval_agent_lab.llm import BaseLLMProvider, LLMMessage
from eval_agent_lab.llm.prompts import render_judge_prompt


class EvaluationResult(BaseModel):
    """Result of evaluating a single item."""

    item_id: str = ""
    input_text: str = ""
    expected_output: str = ""
    actual_output: str = ""
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    judge_score: float | None = None
    judge_reasoning: str | None = None
    composite_score: float = 0.0
    metric_breakdown: dict[str, Any] = Field(default_factory=dict)
    rubric_name: str = ""
    evaluation_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Aggregated evaluation report across all items."""

    report_id: str = ""
    dataset_name: str = ""
    model: str = ""
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    results: list[EvaluationResult] = Field(default_factory=list)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "model": self.model,
            "total_items": self.total_items,
            "success_rate": round(self.successful_items / max(self.total_items, 1), 4),
            "aggregate_metrics": self.aggregate_metrics,
            "total_time_ms": self.total_time_ms,
        }


class EvaluationEngine:
    """Core evaluation engine that orchestrates metrics and LLM-as-judge."""

    def __init__(
        self,
        metrics: list[BaseMetric] | None = None,
        judge_llm: BaseLLMProvider | None = None,
        use_judge: bool = False,
        metric_weights: dict[str, float] | None = None,
        rubric: RubricConfig | None = None,
    ):
        self.metrics = metrics or get_default_metrics()
        self.judge_llm = judge_llm
        self.use_judge = use_judge and judge_llm is not None
        # Rubric takes precedence over raw metric_weights for composite scoring
        self.rubric = rubric or RubricConfig.balanced()
        # Legacy support: if caller passed metric_weights but no rubric,
        # bake them into the rubric so behaviour is identical.
        if metric_weights and rubric is None:
            from eval_agent_lab.evals.rubric import MetricWeight

            self.rubric = RubricConfig(
                name="custom",
                metrics=[MetricWeight(name=n, weight=w) for n, w in metric_weights.items()],
                default_weight=1.0,
                judge_weight=metric_weights.get("judge", 2.0),
            )

    async def evaluate_item(
        self,
        input_text: str,
        expected_output: str,
        actual_output: str,
        item_id: str = "",
        trace: AgentTrace | None = None,
        expected_tools: list[str] | None = None,
        context: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single prediction against a reference."""
        start = time.perf_counter()
        result = EvaluationResult(
            item_id=item_id,
            input_text=input_text,
            expected_output=expected_output,
            actual_output=actual_output,
        )

        # Build extra kwargs for agent-specific metrics
        extra_kwargs: dict[str, Any] = {}
        if trace:
            extra_kwargs["steps"] = [s.model_dump() for s in trace.steps]
            extra_kwargs["total_steps"] = trace.total_steps
            extra_kwargs["max_steps"] = 10
            extra_kwargs["actual_tools"] = trace.tools_used
        if expected_tools:
            extra_kwargs["expected_tools"] = expected_tools
        if context:
            extra_kwargs["context"] = context

        # Run all metrics
        for metric in self.metrics:
            try:
                mr = await metric.compute(actual_output, expected_output, **extra_kwargs)
                result.metrics[metric.name] = mr
            except Exception as exc:
                result.metrics[metric.name] = MetricResult(
                    metric_name=metric.name, score=0.0, error=str(exc)
                )

        # LLM-as-judge
        if self.use_judge and self.judge_llm:
            try:
                judge_result = await self._run_judge(input_text, expected_output, actual_output)
                result.judge_score = judge_result.get("score", 0) / 5.0
                result.judge_reasoning = judge_result.get("reasoning", "")
            except Exception as exc:
                result.metadata["judge_error"] = str(exc)

        # Composite score (rubric-aware)
        result.rubric_name = self.rubric.name
        composite_data = self._compute_composite(result)
        result.composite_score = composite_data["composite_score"]
        result.metric_breakdown = composite_data["metric_breakdown"]
        result.evaluation_time_ms = round((time.perf_counter() - start) * 1000, 2)
        return result

    async def _run_judge(self, question: str, expected: str, response: str) -> dict[str, Any]:
        """Run LLM-as-judge evaluation."""
        assert self.judge_llm is not None
        prompt = render_judge_prompt(question, expected, response)
        llm_response = await self.judge_llm.generate([LLMMessage(role="user", content=prompt)])
        try:
            result: dict[str, Any] = json.loads(llm_response.content)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from response
            content = llm_response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed: dict[str, Any] = json.loads(content[start:end])
                return parsed
            return {"score": 3, "reasoning": content}

    def _compute_composite(self, result: EvaluationResult) -> dict[str, Any]:
        """Compute weighted composite score from all metrics using the rubric.

        Returns a structured dict with composite_score and per-metric breakdown.
        """
        breakdown: dict[str, Any] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for name, mr in result.metrics.items():
            if mr.error:
                breakdown[name] = {
                    "score": 0.0,
                    "weight": 0.0,
                    "weighted": 0.0,
                    "error": mr.error,
                }
                continue
            w = self.rubric.weight_for(name)
            ws = mr.score * w
            breakdown[name] = {
                "score": round(mr.score, 4),
                "weight": round(w, 4),
                "weighted": round(ws, 4),
            }
            weighted_sum += ws
            total_weight += w

        if result.judge_score is not None:
            jw = self.rubric.judge_weight
            jws = result.judge_score * jw
            breakdown["judge"] = {
                "score": round(result.judge_score, 4),
                "weight": round(jw, 4),
                "weighted": round(jws, 4),
            }
            weighted_sum += jws
            total_weight += jw

        composite = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

        return {
            "composite_score": composite,
            "metric_breakdown": breakdown,
        }

    async def evaluate_batch(self, items: list[dict[str, Any]]) -> EvaluationReport:
        """Evaluate a batch of items and produce an aggregated report."""
        import uuid

        report = EvaluationReport(report_id=str(uuid.uuid4()))
        start = time.perf_counter()

        for idx, item in enumerate(items):
            result = await self.evaluate_item(
                input_text=item.get("input", ""),
                expected_output=item.get("expected_output", ""),
                actual_output=item.get("actual_output", ""),
                item_id=item.get("id", str(idx)),
                expected_tools=item.get("expected_tools"),
                context=item.get("context"),
            )
            report.results.append(result)
            report.total_items += 1
            if result.composite_score > 0:
                report.successful_items += 1
            else:
                report.failed_items += 1

        # Aggregate metrics
        metric_scores: dict[str, list[float]] = {}
        for result in report.results:
            for name, mr in result.metrics.items():
                if mr.error is None:
                    metric_scores.setdefault(name, []).append(mr.score)

        for name, scores in metric_scores.items():
            report.aggregate_metrics[f"avg_{name}"] = round(sum(scores) / len(scores), 4)

        composite_scores = [r.composite_score for r in report.results]
        if composite_scores:
            report.aggregate_metrics["avg_composite"] = round(
                sum(composite_scores) / len(composite_scores), 4
            )

        report.total_time_ms = round((time.perf_counter() - start) * 1000, 2)
        return report
