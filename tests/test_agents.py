"""Unit tests for the agent system."""

import pytest

from eval_agent_lab.agents import (
    AgentStep,
    AgentTrace,
    ShortTermMemory,
    StepType,
)


@pytest.mark.unit
class TestShortTermMemory:
    def test_add_and_search(self):
        mem = ShortTermMemory()
        mem.add("key1", "value1")
        mem.add("key2", "value2")
        results = mem.search("key1")
        assert results == ["value1"]

    def test_max_entries(self):
        mem = ShortTermMemory(max_entries=3)
        for i in range(5):
            mem.add(f"key_{i}", f"value_{i}")
        recent = mem.get_recent(10)
        assert len(recent) == 3

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add("key", "value")
        mem.clear()
        assert len(mem.get_recent(10)) == 0

    def test_context_string(self):
        mem = ShortTermMemory()
        mem.add("tool_result:search", "found info")
        ctx = mem.to_context_string()
        assert "tool_result:search" in ctx


@pytest.mark.unit
class TestAgentTrace:
    def test_empty_trace(self):
        trace = AgentTrace(task="test task")
        assert trace.task == "test task"
        assert len(trace.steps) == 0
        assert not trace.success

    def test_trace_with_steps(self):
        trace = AgentTrace(task="test")
        trace.steps.append(
            AgentStep(
                step_number=1,
                step_type=StepType.THINK,
                content="Analyzing the task",
                latency_ms=100.0,
            )
        )
        trace.steps.append(
            AgentStep(
                step_number=2,
                step_type=StepType.RESPOND,
                content="Final answer",
            )
        )
        trace.total_steps = 2
        trace.success = True
        assert trace.total_steps == 2
        assert trace.success


@pytest.mark.unit
class TestAgentStep:
    def test_think_step(self):
        step = AgentStep(
            step_number=1,
            step_type=StepType.THINK,
            content="I need to search for information",
        )
        assert step.step_type == StepType.THINK

    def test_act_step_with_tool(self):
        step = AgentStep(
            step_number=2,
            step_type=StepType.ACT,
            tool_name="search",
            tool_input={"query": "python"},
        )
        assert step.tool_name == "search"
