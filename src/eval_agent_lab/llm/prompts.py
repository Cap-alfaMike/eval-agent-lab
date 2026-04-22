"""Prompt template management using Jinja2."""

from __future__ import annotations

from typing import Any

from jinja2 import BaseLoader, Environment

_ENV = Environment(loader=BaseLoader(), autoescape=False)

# ---------------------------------------------------------------------------
# Core prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = _ENV.from_string("""\
You are an advanced AI assistant with access to tools. Follow this process:

1. THINK: Analyze the user's request carefully
2. PLAN: Determine the best approach and which tools (if any) to use
3. ACT: Execute your plan using available tools
4. OBSERVE: Review the results
5. RESPOND: Provide a clear, accurate answer

## Available Tools
{% for tool in tools %}
### {{ tool.name }}
{{ tool.description }}
Parameters:
{% for param in tool.parameters %}
- {{ param.name }} ({{ param.type }}
  {%- if not param.required %}, optional{% endif %}): {{ param.description }}
{% endfor %}
{% endfor %}

## Rules
- Use tools when the question requires external information or computation
- Always verify tool results before including in your response
- If a tool fails, try an alternative approach
- Be concise and accurate in your final response
""")

REACT_PROMPT = _ENV.from_string("""\
Thought: I need to {{ task_description }}
{% for step in history %}
Action: {{ step.action }}
Action Input: {{ step.action_input }}
Observation: {{ step.observation }}
Thought: {{ step.thought }}
{% endfor %}
""")

JUDGE_PROMPT = _ENV.from_string("""\
You are an expert evaluator. Score the following AI response on a scale of 1-5.

## Evaluation Criteria
- **Accuracy**: Is the response factually correct?
- **Completeness**: Does it fully address the question?
- **Reasoning**: Is the reasoning sound and well-structured?
- **Relevance**: Is the response relevant to the question?

## Input
**Question:** {{ question }}
**Expected Answer:** {{ expected_answer }}
**AI Response:** {{ response }}

## Instructions
Provide your evaluation as JSON:
{
  "score": <1-5>,
  "reasoning": "<explanation>",
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "relevance": <1-5>
}
""")


def render_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Render the system prompt with available tools."""
    return SYSTEM_PROMPT.render(tools=tools)


def render_judge_prompt(question: str, expected_answer: str, response: str) -> str:
    """Render the LLM-as-judge evaluation prompt."""
    return JUDGE_PROMPT.render(
        question=question,
        expected_answer=expected_answer,
        response=response,
    )
