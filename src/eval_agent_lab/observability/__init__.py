"""Structured logging and observability using structlog."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from eval_agent_lab.agents import AgentTrace
from eval_agent_lab.config import ObservabilityConfig


def setup_logging(config: ObservabilityConfig) -> None:
    """Configure structured logging for the application."""
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "eval_agent_lab") -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class CostTracker:
    """Track estimated costs for LLM API calls."""

    # Pricing per 1K tokens (approximate, as of 2024)
    PRICING = {
        "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    }

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               latency_ms: float) -> Dict[str, Any]:
        pricing = self.PRICING.get(model, {"prompt": 0.001, "completion": 0.002})
        cost = (prompt_tokens / 1000 * pricing["prompt"] +
                completion_tokens / 1000 * pricing["completion"])

        record = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
            "estimated_cost_usd": round(cost, 6),
            "timestamp": time.time(),
        }
        self._records.append(record)
        return record

    @property
    def total_cost(self) -> float:
        return sum(r["estimated_cost_usd"] for r in self._records)

    @property
    def total_tokens(self) -> int:
        return sum(r["total_tokens"] for r in self._records)

    def summary(self) -> Dict[str, Any]:
        if not self._records:
            return {"total_requests": 0, "total_tokens": 0, "total_cost_usd": 0.0}
        return {
            "total_requests": len(self._records),
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in self._records),
            "total_completion_tokens": sum(r["completion_tokens"] for r in self._records),
            "total_cost_usd": round(self.total_cost, 6),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in self._records) / len(self._records), 2),
        }


class TraceLogger:
    """Log agent execution traces for debugging and analysis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self._logger = get_logger("trace")
        self._output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def log_trace(self, trace: AgentTrace) -> None:
        """Log a complete agent trace."""
        self._logger.info(
            "agent_trace",
            trace_id=trace.trace_id,
            task=trace.task[:100],
            success=trace.success,
            total_steps=trace.total_steps,
            tools_used=trace.tools_used,
            total_latency_ms=trace.total_latency_ms,
        )

        if self._output_dir:
            trace_path = self._output_dir / f"trace_{trace.trace_id}.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace.model_dump(), f, indent=2, default=str)

    def log_step(self, trace_id: str, step_number: int, step_type: str,
                 content: str, **kwargs: Any) -> None:
        """Log an individual agent step."""
        self._logger.debug(
            "agent_step",
            trace_id=trace_id,
            step_number=step_number,
            step_type=step_type,
            content=content[:200],
            **kwargs,
        )
