"""Custom exception hierarchy for EvalAgentLab."""

from __future__ import annotations


class EvalAgentLabError(Exception):
    """Base exception for all EvalAgentLab errors."""


# --- LLM Errors ---


class LLMError(EvalAgentLabError):
    """Base error for LLM-related failures."""


class LLMConnectionError(LLMError):
    """Failed to connect to LLM provider."""


class LLMRateLimitError(LLMError):
    """Rate limit exceeded on LLM provider."""


class LLMResponseError(LLMError):
    """Invalid or malformed LLM response."""


# --- Tool Errors ---


class ToolError(EvalAgentLabError):
    """Base error for tool-related failures."""


class ToolNotFoundError(ToolError):
    """Requested tool is not registered."""


class ToolExecutionError(ToolError):
    """Tool execution failed."""


class ToolValidationError(ToolError):
    """Tool input/output validation failed."""


# --- Agent Errors ---


class AgentError(EvalAgentLabError):
    """Base error for agent-related failures."""


class AgentMaxStepsError(AgentError):
    """Agent exceeded the maximum number of reasoning steps."""


class AgentToolSelectionError(AgentError):
    """Agent selected an invalid tool or parameters."""


# --- Evaluation Errors ---


class EvaluationError(EvalAgentLabError):
    """Base error for evaluation-related failures."""


class DatasetValidationError(EvaluationError):
    """Dataset failed schema validation."""


class MetricComputationError(EvaluationError):
    """A metric computation failed."""


# --- Pipeline Errors ---


class PipelineError(EvalAgentLabError):
    """Base error for pipeline execution failures."""


class PipelineTimeoutError(PipelineError):
    """Pipeline or item execution timed out."""
