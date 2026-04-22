"""FastAPI service for EvalAgentLab."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eval_agent_lab import __version__
from eval_agent_lab.datasets import DatasetLoader
from eval_agent_lab.evals import EvaluationEngine
from eval_agent_lab.evals.metrics import get_default_metrics
from eval_agent_lab.mcp import ToolRegistry
from eval_agent_lab.mcp.tools import register_default_tools

app = FastAPI(
    title="EvalAgentLab API",
    description="LLM Evaluation & Agent Reliability Framework API",
    version=__version__,
)

# Global state
_registry = ToolRegistry()
register_default_tools(_registry)
_eval_engine = EvaluationEngine(metrics=get_default_metrics())


# --- Request/Response Models ---


class ToolInvokeRequest(BaseModel):
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvalItemRequest(BaseModel):
    input_text: str
    expected_output: str
    actual_output: str
    expected_tools: list[str] = Field(default_factory=list)
    context: str | None = None


class EvalBatchRequest(BaseModel):
    items: list[EvalItemRequest]


class DatasetValidateRequest(BaseModel):
    data: dict[str, Any]


# --- Endpoints ---


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "EvalAgentLab", "version": __version__, "status": "running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    """List all registered MCP tools."""
    return [t.model_dump() for t in _registry.list_tools()]


@app.post("/tools/invoke")
async def invoke_tool(request: ToolInvokeRequest) -> dict[str, Any]:
    """Invoke a registered tool."""
    try:
        result = await _registry.invoke(request.tool_name, request.params)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/eval/item")
async def evaluate_item(request: EvalItemRequest) -> dict[str, Any]:
    """Evaluate a single prediction."""
    result = await _eval_engine.evaluate_item(
        input_text=request.input_text,
        expected_output=request.expected_output,
        actual_output=request.actual_output,
        expected_tools=request.expected_tools,
        context=request.context,
    )
    return result.model_dump()


@app.post("/eval/batch")
async def evaluate_batch(request: EvalBatchRequest) -> dict[str, Any]:
    """Evaluate a batch of predictions."""
    items = [item.model_dump() for item in request.items]
    report = await _eval_engine.evaluate_batch(items)
    return report.model_dump()


@app.post("/dataset/validate")
async def validate_dataset(request: DatasetValidateRequest) -> dict[str, Any]:
    """Validate a dataset."""
    try:
        ds = DatasetLoader.from_dict(request.data)
        warnings = DatasetLoader.validate(ds)
        return {"valid": True, "name": ds.name, "size": ds.size, "warnings": warnings}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}
