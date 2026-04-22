"""MCP-inspired tool system: base interfaces, registry, and execution abstraction."""

from __future__ import annotations

import abc
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from eval_agent_lab.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)


# ---------------------------------------------------------------------------
# Tool Schema Definitions (JSON Schema-based, MCP-style)
# ---------------------------------------------------------------------------

class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Single parameter definition for a tool, JSON Schema-compatible."""

    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class ToolDefinition(BaseModel):
    """Complete tool definition following MCP-style JSON schema conventions.

    This is the wire format used for tool registration, discovery, and
    for injecting tool descriptions into LLM system prompts.
    """

    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(..., description="Human-readable description")
    parameters: List[ToolParameter] = Field(default_factory=list)
    returns: str = Field(default="string", description="Return type description")
    category: str = Field(default="general", description="Tool category for grouping")
    version: str = Field(default="1.0.0")

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function-calling schema format."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.type.value,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ---------------------------------------------------------------------------
# Tool Execution Result
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    """Standardized result from tool execution."""

    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base Tool Interface
# ---------------------------------------------------------------------------

class BaseTool(abc.ABC):
    """Abstract base class for all MCP-inspired tools.

    Every tool must:
    1. Define its schema via `definition()`
    2. Validate inputs via `validate_input()`
    3. Execute via `execute()`
    """

    @abc.abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the JSON-schema-based tool definition."""
        ...

    def validate_input(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize input parameters.

        Override for custom validation logic. Default implementation checks
        required parameters and applies defaults.
        """
        defn = self.definition()
        validated: Dict[str, Any] = {}

        for param in defn.parameters:
            if param.name in params:
                validated[param.name] = params[param.name]
            elif param.default is not None:
                validated[param.name] = param.default
            elif param.required:
                raise ToolValidationError(
                    f"Missing required parameter '{param.name}' for tool '{defn.name}'"
                )

            # Enum validation
            if param.name in validated and param.enum:
                if validated[param.name] not in param.enum:
                    raise ToolValidationError(
                        f"Parameter '{param.name}' must be one of {param.enum}, "
                        f"got '{validated[param.name]}'"
                    )

        return validated

    @abc.abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Any:
        """Execute the tool with validated parameters."""
        ...

    async def safe_execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute with error handling, timing, and structured result."""
        defn = self.definition()
        start = time.perf_counter()

        try:
            validated = self.validate_input(params)
            output = await self.execute(validated)
            elapsed = (time.perf_counter() - start) * 1000

            return ToolResult(
                tool_name=defn.name,
                success=True,
                output=output,
                execution_time_ms=round(elapsed, 2),
            )
        except ToolValidationError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=defn.name,
                success=False,
                error=f"Validation error: {exc}",
                execution_time_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=defn.name,
                success=False,
                error=f"Execution error: {type(exc).__name__}: {exc}",
                execution_time_ms=round(elapsed, 2),
            )


# ---------------------------------------------------------------------------
# Tool Registry (MCP-style dynamic registration & discovery)
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for tool discovery, registration, and invocation.

    Inspired by MCP's tool server pattern: tools are registered once and
    made available to agents via structured definitions.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        defn = tool.definition()
        if defn.name in self._tools:
            raise ValueError(f"Tool '{defn.name}' is already registered")
        self._tools[defn.name] = tool
        self._categories.setdefault(defn.category, []).append(defn.name)

    def register_class(self, tool_cls: Type[BaseTool]) -> None:
        """Instantiate and register a tool class."""
        self.register(tool_cls())

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found in registry")
        defn = self._tools[name].definition()
        self._tools.pop(name)
        if defn.category in self._categories:
            self._categories[defn.category] = [
                n for n in self._categories[defn.category] if n != name
            ]

    def get(self, name: str) -> BaseTool:
        """Get a registered tool by name."""
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool '{name}' not found. Available: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> List[ToolDefinition]:
        """Return definitions for all registered tools."""
        return [tool.definition() for tool in self._tools.values()]

    def list_by_category(self, category: str) -> List[ToolDefinition]:
        """Return tool definitions filtered by category."""
        names = self._categories.get(category, [])
        return [self._tools[n].definition() for n in names if n in self._tools]

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Return all tool definitions in OpenAI function-calling format."""
        return [tool.definition().to_openai_schema() for tool in self._tools.values()]

    async def invoke(self, name: str, params: Dict[str, Any]) -> ToolResult:
        """Invoke a tool by name with given parameters."""
        tool = self.get(name)
        return await tool.safe_execute(params)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
