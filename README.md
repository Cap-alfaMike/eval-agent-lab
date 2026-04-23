<div align="center">

# EvalAgentLab

### Evaluation Infrastructure for LLMs and Agentic Systems

[![CI](https://github.com/Cap-alfaMike/eval-agent-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Cap-alfaMike/eval-agent-lab/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-117%20passed-brightgreen.svg)](#testing)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-yellow.svg)](https://huggingface.co/datasets/Cap-alfaMike/eval-agent-lab-benchmark)

**A production-grade evaluation platform for LLM outputs, agent execution traces, and tool-augmented workflows.**

> EvalAgentLab is designed to evaluate not only what models answer, but **how they arrive at the answer**.

[Quick Start](#-quick-start) · [Evaluation Axes](#-evaluation-axes) · [Architecture](#-architecture) · [Core Capabilities](#-core-capabilities) · [Rubric System](#-rubric-system) · [A/B Comparison](#-ab-run-comparison)

</div>

---

## Why EvalAgentLab?

Most LLM evaluation stops at task accuracy: _"did the model get the right answer?"_

That is not enough for production agentic systems, where failures come from **wrong tool calls, hallucinated reasoning, inefficient step counts, and inconsistent behavior across runs**. Existing eval setups break down because:

- **They ignore the execution trace.** A correct answer reached via flawed reasoning is a false positive.
- **They treat all metrics equally.** Exact match and hallucination score have different weights depending on the use case.
- **They lack experiment comparison.** Without A/B deltas, you cannot tell if a model swap improved or regressed your system.

EvalAgentLab addresses these gaps with:

| Problem | EvalAgentLab Solution |
|---------|----------------------|
| Flat accuracy-only evals | **11 metrics** spanning accuracy, semantics, hallucination, tool use, strategy compliance, and reasoning |
| No weighting or rubric control | **Formal rubric system** with JSON-configurable weights and strict validation |
| No trace-level evaluation | **Agent trace recording** with per-step reasoning and tool call audit |
| No experiment comparison | **A/B run comparison** with per-metric deltas and regression detection |
| Fragmented tooling | **Unified CLI** for evaluation, comparison, dataset export, and deployment |
| No skill adherence testing | **Tool strategy compliance** (must_use / optional / forbidden) evaluation |
| No efficiency scoring | **Step efficiency** with overuse penalization and max-step constraints |

**Who this is for:** ML engineers, AI platform teams, and researchers who need reproducible, production-aligned evaluation workflows — not just leaderboard scores.

---

## Core Capabilities

| Category | Capabilities |
|----------|-------------|
| **Agent System** | ReAct-style reasoning loop (think/decide/act/observe), short-term memory, multi-step execution |
| **MCP Tools** | JSON schema-based tool definitions, dynamic registry, 3 built-in tools (search, calculator, vector retrieval) |
| **Evaluation** | 11 metrics across 3 axes: correctness, skill adherence, execution efficiency |
| **Rubric System** | Formal weighted rubrics for metric scoring, JSON-configurable, strict validation mode, 3 presets |
| **Composite Scoring** | Weighted composite scores with per-metric breakdown, rubric-aware pass/fail classification |
| **LLM-as-Judge** | Configurable LLM-based evaluation with structured scoring |
| **Datasets** | JSON-based datasets with behavioural expectations, validation, filtering, and Hugging Face Hub export |
| **Streaming** | Incremental token streaming for OpenAI provider with transparent aggregation |
| **A/B Comparison** | Compare two evaluation runs with per-metric deltas, direction classification, and rich tables |
| **Pipeline** | End-to-end: dataset -> inference -> evaluation -> rich terminal reports |
| **Observability** | Structured logging (structlog), cost tracking, agent trace persistence |
| **Deployment** | FastAPI service, multi-stage Dockerfile, GitHub Actions CI/CD |
| **LLM Providers** | OpenAI (GPT-4o, etc.) with streaming, HuggingFace Transformers, disk caching |

---

## Rubric System

Rubrics define **how metrics are weighted** when computing composite scores. This gives you full control over what "good" means for your specific evaluation scenario.

**Example rubric** (`rubrics/agent_focused.json`):

```json
{
  "name": "agent_focused",
  "description": "Prioritizes tool use, reasoning, and step efficiency",
  "metrics": [
    { "name": "exact_match",            "weight": 1.0 },
    { "name": "semantic_similarity",    "weight": 1.0 },
    { "name": "tool_selection_accuracy","weight": 3.0 },
    { "name": "reasoning_consistency",  "weight": 2.5 },
    { "name": "hallucination_score",    "weight": 1.5 },
    { "name": "step_efficiency",        "weight": 2.0 }
  ],
  "pass_threshold": 0.60,
  "judge_weight": 1.0
}
```

**Usage:**

```bash
eval-agent-lab run datasets/core_evaluation_suite.json --rubric rubrics/agent_focused.json
```

Built-in presets: `balanced` (equal weights), `accuracy_focused` (correctness-heavy), `agent_focused` (tool/reasoning-heavy). Strict mode enforces weights sum to 1.0 for reproducibility audits.

---

## A/B Run Comparison

Compare any two evaluation reports to detect **improvements, regressions, and unchanged metrics** across model swaps, prompt changes, or rubric adjustments.

```bash
eval-agent-lab compare outputs/report_gpt4o_mini.json outputs/report_gpt4o.json
```

**Example output:**

```
== Run Comparison ==

  Run A: run_20240415_mini (gpt-4o-mini)
  Run B: run_20240415_full (gpt-4o)

┌──────────────────────────┬──────────┬──────────┬──────────┬─────────────┐
│ Metric                   │    Run A │    Run B │    Delta │ Direction   │
├──────────────────────────┼──────────┼──────────┼──────────┼─────────────┤
│ success_rate             │   0.7000 │   0.9000 │  +0.2000 │ improved    │
│ avg_composite            │   0.7200 │   0.8100 │  +0.0900 │ improved    │
│ avg_exact_match          │   0.5000 │   0.6500 │  +0.1500 │ improved    │
│ avg_hallucination_score  │   0.8500 │   0.8600 │  +0.0100 │ unchanged   │
│ avg_tool_selection_acc   │   0.7500 │   0.7200 │  -0.0300 │ regressed   │
└──────────────────────────┴──────────┴──────────┴──────────┴─────────────┘

  Run B is BETTER overall: 3 improved, 1 regressed, 1 unchanged across 5 metrics.
```

---

## Evaluation Axes

EvalAgentLab evaluates LLM and agent behavior across **three core dimensions**:

### 1. Correctness
Did the system produce the right answer?

- Exact match and acceptable output matching
- Semantic similarity (embedding-based)
- Keyword/phrase containment (`expected_contains`)
- Hallucination detection

### 2. Skill Adherence
Did the agent follow the correct capability pathway?

- Tool selection accuracy (F1 score)
- **Tool strategy compliance** (`must_use` / `optional` / `forbidden`)
- Reasoning consistency (step-type analysis)
- Expected reasoning step validation

### 3. Execution Efficiency
Did the agent solve the task efficiently?

- Step count vs expected (`max_steps`)
- Avoidance of redundant tool calls
- Penalization of unnecessary actions (`penalize_overuse`)

This moves evaluation beyond output correctness into **behavioral reliability and decision quality**.

---

## Dataset Design

Each dataset item encodes not only the expected output, but also the **expected behavior** of the agent.

Key fields include:

| Field | Purpose |
|------|--------|
| `expected_tools` | Defines which tools should be used |
| `tool_strategy` | Enforces required, optional, or forbidden tool usage |
| `acceptable_outputs` | Handles natural variation in LLM responses |
| `max_steps` | Constrains execution efficiency |
| `penalize_overuse` | Penalizes redundant or excessive actions |
| `expected_contains` | Soft matching via keywords/phrases |
| `expected_reasoning` | Captures expected reasoning steps |

This enables evaluation of **skill adherence and execution efficiency**, not just correctness.

---

## Benchmark Datasets

EvalAgentLab ships with curated evaluation datasets published on **[Hugging Face Hub](https://huggingface.co/datasets/Cap-alfaMike/eval-agent-lab-benchmark)** for community access and reproducibility.

| Dataset | Items | Categories | Purpose |
|---------|-------|------------|---------|
| `core_evaluation_suite.json` | 15 | knowledge, reasoning, computation, tool_use, multi_step, hallucination | General LLM + agent evaluation |
| `tool_selection_benchmark.json` | 5 | computation, search, retrieval, multi_tool | Tool selection accuracy testing |

**Use directly from Hugging Face:**

```python
from datasets import load_dataset

ds = load_dataset("Cap-alfaMike/eval-agent-lab-benchmark")
```

**Or push your own datasets to HF Hub:**

```bash
eval-agent-lab push-dataset datasets/core_evaluation_suite.json \
  --repo your-username/your-benchmark
```

This generates a dataset card, exports as JSONL, and uploads to your HF namespace.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI / FastAPI                        │
├─────────────────────────────────────────────────────────┤
│                   Pipeline Engine                       │
│         dataset → inference → evaluation → report       │
├──────────┬──────────┬───────────┬───────────────────────┤
│  Agent   │   LLM    │   Eval    │    Observability      │
│  System  │  Layer   │  Engine   │    & LLMOps           │
├──────────┼──────────┼───────────┼───────────────────────┤
│ ReAct    │ OpenAI   │ Metrics   │ Structured Logging    │
│ Memory   │ HF       │ Rubric    │ Cost Tracking         │
│ Trace    │ Stream   │ A/B Comp  │ Trace Persistence     │
├──────────┴──────────┴───────────┴───────────────────────┤
│              MCP Tool Registry & Execution              │
│         search │ calculator │ vector_retrieval           │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
Dataset (JSON)
    │
    ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  Loader  │───▶│  Agent/LLM   │───▶│  Evaluation  │
│  +Valid  │    │  Inference   │    │  + Rubric    │
└──────────┘    └──────┬───────┘    └──────┬───────┘
                       │                    │
                  ┌────▼────┐          ┌────▼────┐
                  │  Tool   │          │ Report  │
                  │ Registry│          │ (JSON)  │
                  └─────────┘          └─────────┘
```

### Agent Execution Lifecycle

```
THINK  →  Analyze task, review context and memory
  │
DECIDE →  Select tool or formulate direct answer
  │
ACT    →  Execute tool via MCP registry
  │
OBSERVE → Process tool result, update memory
  │
REPEAT →  Loop until answer found or max steps reached
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Cap-alfaMike/eval-agent-lab.git
cd eval-agent-lab

# Install in development mode
pip install -e ".[dev]"
```

### Run the Demo (No API Key Required)

```bash
eval-agent-lab demo
```

This demonstrates the tool system, evaluation metrics, and rubric system without any external API calls.

### Validate a Dataset

```bash
eval-agent-lab validate datasets/core_evaluation_suite.json
```

### Run a Full Evaluation Pipeline

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run agent evaluation
eval-agent-lab run datasets/core_evaluation_suite.json --mode agent --model gpt-4o-mini

# Run with a custom rubric
eval-agent-lab run datasets/core_evaluation_suite.json --rubric rubrics/accuracy_focused.json

# Run with streaming enabled
eval-agent-lab run datasets/core_evaluation_suite.json --stream

# With LLM-as-judge enabled
eval-agent-lab run datasets/core_evaluation_suite.json --judge
```

### Run via API

```bash
# Start the API server
uvicorn eval_agent_lab.api:app --reload

# Invoke a tool
curl -X POST http://localhost:8000/tools/invoke \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "calculator", "params": {"expression": "sqrt(144)"}}'

# Evaluate a prediction
curl -X POST http://localhost:8000/eval/item \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "What is Python?",
    "expected_output": "A programming language",
    "actual_output": "Python is a programming language created by Guido van Rossum"
  }'
```

### Run with Docker

```bash
# Build and run CLI mode
docker build --target cli -t eval-agent-lab-cli .
docker run eval-agent-lab-cli demo

# Build and run API mode
docker build --target api -t eval-agent-lab-api .
docker run -p 8000:8000 eval-agent-lab-api
```

## Repository Structure

```
eval-agent-lab/
├── src/eval_agent_lab/
│   ├── __init__.py              # Package root, version
│   ├── config.py                # Pydantic configuration management
│   ├── exceptions.py            # Structured exception hierarchy
│   ├── cli.py                   # Typer CLI (run, validate, demo, compare, push-dataset)
│   ├── main.py                  # Entry point
│   ├── agents/
│   │   └── __init__.py          # BaseAgent, ReActAgent, AgentTrace, Memory
│   ├── mcp/
│   │   ├── __init__.py          # BaseTool, ToolRegistry, ToolDefinition
│   │   └── tools.py             # SearchTool, CalculatorTool, VectorRetrievalTool
│   ├── llm/
│   │   ├── __init__.py          # BaseLLMProvider, LLMMessage, LLMResponse, StreamChunk
│   │   ├── openai_provider.py   # OpenAI/Azure provider with streaming
│   │   ├── huggingface_provider.py  # Local HuggingFace inference
│   │   └── prompts.py           # Jinja2 prompt templates
│   ├── evals/
│   │   ├── __init__.py          # EvaluationEngine, EvaluationReport
│   │   ├── metrics.py           # 8 evaluation metrics
│   │   ├── rubric.py            # RubricConfig, MetricWeight, presets
│   │   └── comparison.py        # Run comparison (A/B testing)
│   ├── datasets/
│   │   ├── __init__.py          # DatasetItem, DatasetLoader, validation
│   │   └── export_hf.py         # Hugging Face Hub export
│   ├── pipelines/
│   │   └── __init__.py          # Pipeline engine (dataset->eval->report)
│   ├── observability/
│   │   └── __init__.py          # Logging, CostTracker, TraceLogger
│   └── api/
│       └── __init__.py          # FastAPI service
├── datasets/
│   ├── core_evaluation_suite.json
│   └── tool_selection_benchmark.json
├── rubrics/
│   ├── accuracy_focused.json
│   └── agent_focused.json
├── tests/                       # 117 unit tests
│   ├── test_mcp.py
│   ├── test_evals.py
│   ├── test_datasets.py
│   ├── test_config.py
│   ├── test_agents.py
│   ├── test_observability.py
│   ├── test_rubric.py
│   ├── test_comparison.py
│   ├── test_streaming.py
│   └── test_export_hf.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `config` | Centralized Pydantic configuration with environment variable resolution |
| `exceptions` | Typed exception hierarchy for all error domains |
| `mcp` | MCP-inspired tool abstraction: schema definitions, registry, execution |
| `agents` | Agent lifecycle: ReAct loop, trace recording, short-term memory |
| `llm` | Provider abstraction with caching, streaming, batching, and usage statistics |
| `evals` | Metric computation engine, LLM-as-judge, rubric-weighted composite scoring |
| `evals.rubric` | Formal rubric definitions with JSON persistence and presets |
| `evals.comparison` | A/B testing: compare two runs with delta analysis |
| `datasets` | Dataset loading, validation, filtering, and HF Hub export |
| `pipelines` | End-to-end orchestration with rubric-aware progress tracking |
| `observability` | Structured logging, cost estimation, trace persistence |
| `api` | FastAPI REST interface for all core capabilities |
| `cli` | Typer CLI: run, validate, compare, push-dataset, demo, list-tools |

## Testing

```bash
# Run all unit tests (117 tests)
pytest tests/ -m unit -v

# Run with coverage
pytest tests/ -m unit --cov=src/eval_agent_lab --cov-report=term-missing

# Run specific test module
pytest tests/test_rubric.py -v
```

## Roadmap

- [x] **v0.1** — Rubric-based evaluation with weighted composite scoring
- [x] **v0.1** — LLM streaming support (OpenAI provider)
- [x] **v0.1** — A/B evaluation comparison mode
- [x] **v0.1** — Hugging Face Hub dataset export
- [ ] **v0.2** — Async parallel pipeline execution
- [ ] **v0.2** — Custom metric plugin system
- [ ] **v0.3** — Persistent agent memory (vector store backed)
- [ ] **v0.3** — OpenTelemetry integration for distributed tracing
- [ ] **v0.4** — Web dashboard for evaluation visualization
- [ ] **v0.5** — Multi-agent evaluation (agent collaboration scoring)
- [ ] **v0.5** — Safety & alignment evaluation suite
- [ ] **v1.0** — Kubernetes deployment with Helm charts
- [ ] **v1.0** — Plugin marketplace for community tools and metrics

## Cloud Deployment

| Cloud | Compute | Storage | Logging |
|-------|---------|---------|---------|
| **AWS** | ECS/Fargate | S3 (reports + traces) | CloudWatch (structlog JSON) |
| **GCP** | Cloud Run | GCS | Cloud Logging (native JSON) |
| **Azure** | Container Apps | Blob Storage | Application Insights |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

**An evaluation platform for correctness, skill adherence, and execution efficiency in LLM-based agents.**

</div>
