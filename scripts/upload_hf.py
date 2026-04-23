"""Upload v2 datasets to Hugging Face Hub.

This script uploads the upgraded EvalAgentLab datasets
with the new behavioural evaluation schema to HF.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "Cap-alfaMike/eval-agent-lab-benchmark"


def build_dataset_card() -> str:
    """Generate the upgraded HF dataset card."""
    return """---
language:
- en
license: apache-2.0
tags:
- evaluation
- llm
- agents
- benchmark
- ai
- eval-agent-lab
- tool-use
- hallucination
- skill-adherence
size_categories:
- n<1K
---

# EvalAgentLab Benchmark v2.0

A curated benchmark dataset for evaluating **LLM outputs and agentic workflows** across three evaluation axes: **correctness, skill adherence, and execution efficiency**.

> EvalAgentLab evaluates not only what models answer, but **how they arrive at the answer**.

## 🧠 Evaluation Axes

### 1. Correctness
Did the system produce the right answer?
- Exact match and acceptable output matching
- Semantic similarity
- Keyword containment (`expected_contains`)
- Hallucination detection

### 2. Skill Adherence
Did the agent follow the correct capability pathway?
- Tool selection accuracy
- **Tool strategy compliance** (`must_use` / `optional` / `forbidden`)
- Reasoning consistency

### 3. Execution Efficiency
Did the agent solve the task efficiently?
- Step count vs expected (`max_steps`)
- Penalization of redundant actions (`penalize_overuse`)

## 📊 Dataset Structure

Each item encodes not only the expected output, but also the **expected behavior** of the agent.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `input` | string | The input query or task |
| `expected_output` | string | The expected correct answer |
| `acceptable_outputs` | list[string] | Alternative correct outputs for flexible matching |
| `expected_tools` | list[string] | Tools the agent should use |
| `tool_strategy` | string | `must_use`, `optional`, or `forbidden` |
| `max_steps` | integer | Maximum expected steps for efficient resolution |
| `penalize_overuse` | boolean | Penalize redundant or excessive tool calls |
| `expected_contains` | list[string] | Keywords/phrases the output should contain |
| `expected_reasoning` | list[string] | Expected reasoning steps |
| `context` | string | Optional additional context |
| `category` | string | Task category |
| `difficulty` | string | easy, medium, or hard |

## 📁 Included Datasets

| Dataset | Items | Categories |
|---------|-------|------------|
| `core_evaluation_suite` | 15 | knowledge, reasoning, computation, tool_use, multi_step, hallucination |
| `tool_selection_benchmark` | 5 | computation, search, retrieval, multi_tool |

## 🧪 Example

```json
{
  "id": "hallucination_001",
  "input": "What is the capital of Atlantis?",
  "expected_output": "Atlantis is fictional",
  "acceptable_outputs": [
    "Atlantis is fictional",
    "There is no real capital of Atlantis"
  ],
  "expected_tools": [],
  "tool_strategy": "forbidden",
  "max_steps": 1,
  "penalize_overuse": true,
  "expected_contains": ["fictional", "myth"],
  "category": "hallucination",
  "difficulty": "medium"
}
```

## 🚀 Usage

```python
from datasets import load_dataset

ds = load_dataset("Cap-alfaMike/eval-agent-lab-benchmark")
```

Or use directly with EvalAgentLab:

```bash
pip install eval-agent-lab
eval-agent-lab run datasets/core_evaluation_suite.json --rubric rubrics/agent_focused.json
```

## 🔗 Related Project

This dataset is part of the [EvalAgentLab](https://github.com/Cap-alfaMike/eval-agent-lab) framework — an evaluation platform for correctness, skill adherence, and execution efficiency in LLM-based agents.

## 📄 License

Apache 2.0
"""


def dataset_to_jsonl(path: str) -> str:
    """Convert dataset JSON to JSONL format."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for item in data["items"]:
        lines.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(lines)


def main() -> None:
    api = HfApi()

    print(f"Uploading to {REPO_ID}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1. Write README
        readme_path = tmpdir_path / "README.md"
        readme_path.write_text(build_dataset_card(), encoding="utf-8")
        print("  [OK] README.md generated")

        # 2. Write JSONL data files
        data_dir = tmpdir_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Core evaluation suite
        core_jsonl = dataset_to_jsonl("datasets/core_evaluation_suite.json")
        (data_dir / "core_evaluation_suite.jsonl").write_text(
            core_jsonl, encoding="utf-8"
        )
        print("  [OK] core_evaluation_suite.jsonl generated (15 items)")

        # Tool selection benchmark
        tool_jsonl = dataset_to_jsonl("datasets/tool_selection_benchmark.json")
        (data_dir / "tool_selection_benchmark.jsonl").write_text(
            tool_jsonl, encoding="utf-8"
        )
        print("  [OK] tool_selection_benchmark.jsonl generated (5 items)")

        # 3. Also include original JSON files
        import shutil
        shutil.copy2(
            "datasets/core_evaluation_suite.json",
            data_dir / "core_evaluation_suite.json",
        )
        shutil.copy2(
            "datasets/tool_selection_benchmark.json",
            data_dir / "tool_selection_benchmark.json",
        )
        print("  [OK] Original JSON files copied")

        # 4. Upload
        print("\n  Uploading to Hugging Face Hub...")
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=str(tmpdir_path),
            commit_message="feat: upload v2.0 datasets with behavioural evaluation schema",
        )

    print(f"\n[DONE] View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
