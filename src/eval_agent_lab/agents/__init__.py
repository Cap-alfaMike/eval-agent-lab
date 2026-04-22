"""Agent system: base agent, ReAct agent, and execution lifecycle."""

from __future__ import annotations

import abc
import json
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from eval_agent_lab.llm import BaseLLMProvider, LLMMessage
from eval_agent_lab.mcp import ToolRegistry


class StepType(str, Enum):
    THINK = "think"
    DECIDE = "decide"
    ACT = "act"
    OBSERVE = "observe"
    RESPOND = "respond"
    ERROR = "error"


class AgentStep(BaseModel):
    """A single step in the agent's execution trace."""
    step_number: int
    step_type: StepType
    content: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    timestamp: float = Field(default_factory=time.time)
    latency_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    """Complete execution trace for an agent run."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    total_steps: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    tools_used: list[str] = Field(default_factory=list)
    error: str | None = None


class ShortTermMemory:
    """Short-term memory for agent context within a single task."""

    def __init__(self, max_entries: int = 50):
        self._entries: list[dict[str, Any]] = []
        self._max = max_entries

    def add(self, key: str, value: Any) -> None:
        self._entries.append({"key": key, "value": value, "ts": time.time()})
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]

    def search(self, key: str) -> list[Any]:
        return [e["value"] for e in self._entries if e["key"] == key]

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        return self._entries[-n:]

    def clear(self) -> None:
        self._entries.clear()

    def to_context_string(self) -> str:
        if not self._entries:
            return ""
        lines = [f"[Memory] {e['key']}: {e['value']}" for e in self._entries[-10:]]
        return "\n".join(lines)


class BaseAgent(abc.ABC):
    """Abstract base class for all agents."""

    def __init__(self, llm: BaseLLMProvider, tools: ToolRegistry,
                 max_steps: int = 10, name: str = "Agent"):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.name = name
        self.memory = ShortTermMemory()

    @abc.abstractmethod
    async def run(self, task: str, context: str | None = None) -> AgentTrace:
        """Execute the agent on a given task and return the full trace."""
        ...


class ReActAgent(BaseAgent):
    """ReAct-style agent: Think → Decide → Act → Observe → Repeat.

    Implements the Claude-style reasoning loop with tool use,
    multi-step reasoning, and short-term memory.
    """

    def __init__(self, llm: BaseLLMProvider, tools: ToolRegistry,
                 max_steps: int = 10, name: str = "ReActAgent"):
        super().__init__(llm, tools, max_steps, name)

    async def run(self, task: str, context: str | None = None) -> AgentTrace:
        trace = AgentTrace(task=task)
        messages: list[LLMMessage] = []
        step_num = 0

        # Build system prompt with tool definitions
        tool_defs = self.tools.list_tools()
        tool_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in tool_defs
        )
        system_msg = (
            "You are an AI assistant. Use the think→act→observe loop.\n"
            "Available tools:\n" + tool_desc + "\n\n"
            "To use a tool, respond with JSON: "
            '{"tool": "<name>", "params": {<params>}}\n'
            "When you have the final answer, respond with: "
            '{"answer": "<your answer>"}'
        )
        messages.append(LLMMessage(role="system", content=system_msg))

        # Add context and task
        user_content = f"Task: {task}"
        if context:
            user_content = f"Context: {context}\n\n{user_content}"
        memory_ctx = self.memory.to_context_string()
        if memory_ctx:
            user_content = f"{memory_ctx}\n\n{user_content}"
        messages.append(LLMMessage(role="user", content=user_content))

        try:
            while step_num < self.max_steps:
                step_num += 1

                # THINK + DECIDE: Ask LLM for next action
                start = time.perf_counter()
                response = await self.llm.generate(messages)
                latency = (time.perf_counter() - start) * 1000

                think_step = AgentStep(
                    step_number=step_num,
                    step_type=StepType.THINK,
                    content=response.content,
                    latency_ms=round(latency, 2),
                    token_usage=response.usage,
                )
                trace.steps.append(think_step)

                # Parse response
                action = self._parse_action(response.content)

                if action is None:
                    # No parseable action — treat as final answer
                    trace.final_answer = response.content
                    trace.success = True
                    break

                if "answer" in action:
                    # Agent decided to give final answer
                    trace.final_answer = action["answer"]
                    trace.success = True
                    trace.steps.append(AgentStep(
                        step_number=step_num,
                        step_type=StepType.RESPOND,
                        content=action["answer"],
                    ))
                    break

                if "tool" in action:
                    tool_name = action["tool"]
                    tool_params = action.get("params", {})

                    # ACT: Execute tool
                    act_step = AgentStep(
                        step_number=step_num,
                        step_type=StepType.ACT,
                        tool_name=tool_name,
                        tool_input=tool_params,
                    )

                    tool_result = await self.tools.invoke(tool_name, tool_params)
                    act_step.tool_result = tool_result.model_dump()
                    act_step.latency_ms = tool_result.execution_time_ms
                    trace.steps.append(act_step)

                    if tool_name not in trace.tools_used:
                        trace.tools_used.append(tool_name)

                    # OBSERVE: Feed result back
                    obs_content = (
                        json.dumps(tool_result.output)
                        if tool_result.success
                        else f"Error: {tool_result.error}"
                    )
                    messages.append(LLMMessage(role="assistant", content=response.content))
                    messages.append(LLMMessage(role="user", content=f"Observation: {obs_content}"))

                    trace.steps.append(AgentStep(
                        step_number=step_num,
                        step_type=StepType.OBSERVE,
                        content=obs_content,
                        tool_name=tool_name,
                    ))

                    # Store in memory
                    self.memory.add(f"tool_result:{tool_name}", obs_content[:500])

            if not trace.success:
                trace.error = f"Max steps ({self.max_steps}) exceeded"

        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
            trace.steps.append(AgentStep(
                step_number=step_num,
                step_type=StepType.ERROR,
                content=str(exc),
            ))

        # Finalize trace
        trace.total_steps = len(trace.steps)
        trace.total_latency_ms = sum(s.latency_ms for s in trace.steps)
        trace.total_tokens = sum(
            sum(s.token_usage.values()) for s in trace.steps
        )
        return trace

    @staticmethod
    def _parse_action(content: str) -> dict[str, Any] | None:
        """Extract JSON action from LLM response."""
        # Try to find JSON in the response
        content = content.strip()

        # Look for JSON blocks
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = content.find(start_char)
            if start == -1:
                continue
            # Find matching end
            depth = 0
            for i, ch in enumerate(content[start:], start):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                if depth == 0:
                    try:
                        parsed: dict[str, Any] = json.loads(content[start:i + 1])
                        return parsed
                    except json.JSONDecodeError:
                        continue

        return None
