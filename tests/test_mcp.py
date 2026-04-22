"""Unit tests for the MCP tool system."""

import pytest

from eval_agent_lab.exceptions import ToolNotFoundError
from eval_agent_lab.mcp import (
    ParameterType,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)
from eval_agent_lab.mcp.tools import (
    CalculatorTool,
    SearchTool,
    VectorRetrievalTool,
    register_default_tools,
)

# --- Tool Definition Tests ---


@pytest.mark.unit
class TestToolDefinition:
    def test_openai_schema_conversion(self):
        defn = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(name="query", type=ParameterType.STRING, description="Search query"),
                ToolParameter(
                    name="count",
                    type=ParameterType.INTEGER,
                    description="Count",
                    required=False,
                    default=5,
                ),
            ],
        )
        schema = defn.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]
        assert "count" not in schema["function"]["parameters"]["required"]


# --- Tool Registry Tests ---


@pytest.mark.unit
class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = SearchTool()
        registry.register(tool)
        assert "search" in registry
        assert len(registry) == 1
        retrieved = registry.get("search")
        assert retrieved is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(SearchTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(SearchTool())

    def test_get_missing_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent")

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.unregister("search")
        assert "search" not in registry

    def test_list_tools(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert names == {"search", "calculator", "vector_retrieval"}

    def test_openai_schemas(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        schemas = registry.get_openai_schemas()
        assert len(schemas) == 3
        assert all(s["type"] == "function" for s in schemas)


# --- Search Tool Tests ---


@pytest.mark.unit
class TestSearchTool:
    @pytest.fixture
    def tool(self):
        return SearchTool()

    def test_definition(self, tool):
        defn = tool.definition()
        assert defn.name == "search"
        assert len(defn.parameters) == 2

    @pytest.mark.asyncio
    async def test_search_with_results(self, tool):
        result = await tool.safe_execute({"query": "python programming"})
        assert result.success
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_search_no_results(self, tool):
        result = await tool.safe_execute({"query": "xyznonexistent123"})
        assert result.success
        assert len(result.output) == 0


# --- Calculator Tool Tests ---


@pytest.mark.unit
class TestCalculatorTool:
    @pytest.fixture
    def tool(self):
        return CalculatorTool()

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, tool):
        result = await tool.safe_execute({"expression": "2 + 3 * 4"})
        assert result.success
        assert result.output["result"] == 14

    @pytest.mark.asyncio
    async def test_math_functions(self, tool):
        result = await tool.safe_execute({"expression": "sqrt(144)"})
        assert result.success
        assert result.output["result"] == 12.0

    @pytest.mark.asyncio
    async def test_unsafe_expression_rejected(self, tool):
        result = await tool.safe_execute({"expression": "import os"})
        assert not result.success
        assert "error" in result.error.lower() or "unsafe" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_expression(self, tool):
        result = await tool.safe_execute({"expression": "1 / 0"})
        assert not result.success


# --- Vector Retrieval Tool Tests ---


@pytest.mark.unit
class TestVectorRetrievalTool:
    @pytest.fixture
    def tool(self):
        return VectorRetrievalTool()

    @pytest.mark.asyncio
    async def test_retrieval_returns_results(self, tool):
        result = await tool.safe_execute({"query": "neural networks", "top_k": 2})
        assert result.success
        assert len(result.output) <= 2

    @pytest.mark.asyncio
    async def test_retrieval_has_scores(self, tool):
        result = await tool.safe_execute({"query": "transformers"})
        assert result.success
        for doc in result.output:
            assert "similarity_score" in doc


# --- Tool Invocation via Registry ---


@pytest.mark.unit
class TestToolInvocation:
    @pytest.mark.asyncio
    async def test_invoke_through_registry(self):
        registry = ToolRegistry()
        register_default_tools(registry)
        result = await registry.invoke("calculator", {"expression": "2 + 2"})
        assert result.success
        assert result.output["result"] == 4

    @pytest.mark.asyncio
    async def test_invoke_missing_tool(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            await registry.invoke("nonexistent", {})
