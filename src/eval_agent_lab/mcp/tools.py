"""Built-in MCP tools: search, calculator, and vector retrieval."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from eval_agent_lab.mcp import (
    BaseTool,
    ParameterType,
    ToolDefinition,
    ToolParameter,
)


class SearchTool(BaseTool):
    """Simulated web search tool for agent evaluation."""

    def __init__(self, knowledge_base: dict[str, str] | None = None) -> None:
        self._kb: dict[str, str] = knowledge_base or {
            "python": "Python is a high-level programming language created by Guido van Rossum.",
            "machine learning": "Machine learning enables systems to learn from data.",
            "transformer": (
                "The Transformer architecture was introduced in "
                "'Attention Is All You Need'."
            ),
            "mcp": "The Model Context Protocol is an open standard for AI tool interoperability.",
            "evaluation": "LLM evaluation includes accuracy, reasoning, and safety metrics.",
            "climate change": "Climate change refers to long-term shifts in global temperatures.",
            "quantum computing": "Quantum computing uses superposition and entanglement.",
        }

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search",
            description="Search for information on a given topic.",
            parameters=[
                ToolParameter(name="query", type=ParameterType.STRING, description="Search query"),
                ToolParameter(name="max_results", type=ParameterType.INTEGER,
                            description="Max results", required=False, default=3),
            ],
            returns="List of search results",
            category="information_retrieval",
        )

    async def execute(self, params: dict[str, Any]) -> Any:
        query = params["query"].lower()
        max_results = params.get("max_results", 3)
        results: list[dict[str, Any]] = []
        for topic, content in self._kb.items():
            query_words = set(query.split())
            topic_words = set(topic.split())
            content_words = set(content.lower().split())
            score = len(query_words & topic_words) * 3.0 + len(query_words & content_words) * 0.5
            if score > 0:
                results.append({"topic": topic, "content": content,
                              "relevance_score": round(min(score / 5.0, 1.0), 3)})
        results.sort(key=lambda x: float(x.get("relevance_score", 0)), reverse=True)
        return results[:max_results]


class CalculatorTool(BaseTool):
    """Safe mathematical expression evaluator."""

    SAFE_FUNCTIONS = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "pow": pow, "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
    }

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Evaluate a mathematical expression safely.",
            parameters=[
                ToolParameter(name="expression", type=ParameterType.STRING,
                            description="Math expression to evaluate"),
            ],
            returns="Numeric result",
            category="computation",
        )

    async def execute(self, params: dict[str, Any]) -> Any:
        expression = params["expression"]
        if re.search(r"(import|exec|eval|open|__)", expression):
            raise ValueError(f"Unsafe expression: {expression}")
        try:
            result = eval(expression, {"__builtins__": {}}, self.SAFE_FUNCTIONS)
            return {"result": round(result, 10) if isinstance(result, float) else result,
                    "expression": expression}
        except Exception as exc:
            raise ValueError(f"Failed to evaluate '{expression}': {exc}") from exc


class VectorRetrievalTool(BaseTool):
    """In-memory vector similarity search for document retrieval."""

    def __init__(self, documents: list[dict[str, Any]] | None = None, embedding_dim: int = 64):
        self._dim = embedding_dim
        self._documents: list[dict[str, Any]] = documents or [
            {
                "id": "doc_1", "title": "Neural Networks",
                "content": "Neural networks are computing systems "
                "inspired by biological neural networks.",
            },
            {
                "id": "doc_2", "title": "Transformer Architecture",
                "content": "The transformer relies on "
                "self-attention mechanisms.",
            },
            {
                "id": "doc_3", "title": "RLHF",
                "content": "RLHF trains language models using "
                "human preferences.",
            },
            {
                "id": "doc_4", "title": "Prompt Engineering",
                "content": "Effective prompts include clear "
                "instructions and context.",
            },
            {
                "id": "doc_5", "title": "LLM Evaluation",
                "content": "Evaluating LLMs requires "
                "multi-dimensional assessment.",
            },
        ]
        self._embeddings: np.ndarray | None = None
        self._build_index()

    def _text_to_embedding(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text.lower().strip()) % (2**31))
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _build_index(self) -> None:
        embeddings = [
            self._text_to_embedding(f"{d['title']} {d['content']}")
            for d in self._documents
        ]
        self._embeddings = np.stack(embeddings) if embeddings else np.empty((0, self._dim))

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vector_retrieval",
            description="Retrieve documents using semantic similarity search.",
            parameters=[
                ToolParameter(name="query", type=ParameterType.STRING, description="Search query"),
                ToolParameter(
                    name="top_k", type=ParameterType.INTEGER,
                    description="Number of results",
                    required=False, default=3,
                ),
            ],
            returns="List of documents with similarity scores",
            category="information_retrieval",
        )

    async def execute(self, params: dict[str, Any]) -> Any:
        query_emb = self._text_to_embedding(params["query"])
        top_k = params.get("top_k", 3)
        assert self._embeddings is not None
        similarities = np.dot(self._embeddings, query_emb)
        scored = []
        for idx, score in enumerate(similarities):
            doc = self._documents[idx].copy()
            doc["similarity_score"] = round(float(score), 4)
            scored.append(doc)
        scored.sort(key=lambda x: float(x["similarity_score"]), reverse=True)
        return scored[:top_k]


def register_default_tools(registry: Any) -> None:
    """Register all built-in tools into a registry."""
    registry.register(SearchTool())
    registry.register(CalculatorTool())
    registry.register(VectorRetrievalTool())
