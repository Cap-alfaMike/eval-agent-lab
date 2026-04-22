"""Export evaluation datasets to Hugging Face Hub.

Publishes an EvalAgentLab dataset as a HF Dataset repository with
a proper README card, metadata, and JSONL data file.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from eval_agent_lab.datasets import Dataset, DatasetLoader


def _build_dataset_card(dataset: Dataset, repo_id: str) -> str:
    """Generate a Hugging Face dataset card (README.md)."""
    categories = sorted({item.category for item in dataset.items})
    difficulties = sorted({item.difficulty for item in dataset.items})

    return f"""---
language:
- en
license: apache-2.0
tags:
- evaluation
- llm
- agents
- eval-agent-lab
size_categories:
- n<1K
---

# {dataset.name}

{dataset.description or "An evaluation dataset created with EvalAgentLab."}

## Dataset Details

| Property | Value |
|----------|-------|
| **Name** | {dataset.name} |
| **Version** | {dataset.version} |
| **Items** | {dataset.size} |
| **Categories** | {", ".join(categories)} |
| **Difficulties** | {", ".join(difficulties)} |
| **Source** | [EvalAgentLab](https://github.com/Cap-alfaMike/eval-agent-lab) |

## Schema

Each item in the dataset contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `input` | string | The input query or task |
| `expected_output` | string | The expected correct answer |
| `expected_tools` | list[string] | Tools the agent should use |
| `context` | string (optional) | Additional context |
| `category` | string | Task category |
| `difficulty` | string | easy, medium, or hard |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
```

## License

Apache 2.0
"""


def _dataset_to_jsonl(dataset: Dataset) -> str:
    """Convert dataset items to JSONL format."""
    lines = []
    for item in dataset.items:
        record = {
            "id": item.id,
            "input": item.input,
            "expected_output": item.expected_output,
            "expected_tools": item.expected_tools,
            "context": item.context or "",
            "category": item.category,
            "difficulty": item.difficulty,
            "metadata": item.metadata,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


def push_dataset_to_hf(
    dataset_path: str,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Push an EvalAgentLab dataset to Hugging Face Hub.

    Args:
        dataset_path: Path to the dataset JSON file.
        repo_id: HF Hub repo id (e.g. ``username/eval-agent-lab-benchmark``).
        token: HF API token. Falls back to ``HF_TOKEN`` env var / cached login.
        private: Whether to create a private repository.

    Returns:
        The URL of the published dataset.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for dataset export. "
            "Install with: pip install huggingface_hub"
        ) from exc

    # Load and validate
    dataset = DatasetLoader.from_json(dataset_path)

    api = HfApi(token=token)

    # Create or ensure repo exists
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write README.md (dataset card)
        readme_path = tmpdir_path / "README.md"
        readme_path.write_text(_build_dataset_card(dataset, repo_id), encoding="utf-8")

        # Write data file as JSONL
        data_path = tmpdir_path / "data" / "eval.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(_dataset_to_jsonl(dataset), encoding="utf-8")

        # Also include original JSON for full-fidelity round-trip
        original_path = tmpdir_path / "data" / "dataset.json"
        with open(dataset_path, encoding="utf-8") as src:
            original_path.write_text(src.read(), encoding="utf-8")

        # Upload folder
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(tmpdir_path),
            commit_message=f"Upload {dataset.name} v{dataset.version} via EvalAgentLab",
        )

    url = f"https://huggingface.co/datasets/{repo_id}"
    return url
