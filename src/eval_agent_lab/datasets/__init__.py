"""Dataset loading, validation, and management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from eval_agent_lab.exceptions import DatasetValidationError


class DatasetItem(BaseModel):
    """A single item in an evaluation dataset."""

    id: str = ""
    input: str
    expected_output: str
    expected_tools: list[str] = Field(default_factory=list)
    context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"

    @field_validator("input")
    @classmethod
    def input_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Input must not be empty")
        return v


class Dataset(BaseModel):
    """Evaluation dataset container."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    items: list[DatasetItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.items)

    def filter_by_category(self, category: str) -> Dataset:
        filtered = [item for item in self.items if item.category == category]
        return Dataset(
            name=f"{self.name}_{category}",
            description=self.description,
            items=filtered,
            metadata=self.metadata,
        )

    def filter_by_difficulty(self, difficulty: str) -> Dataset:
        filtered = [item for item in self.items if item.difficulty == difficulty]
        return Dataset(
            name=f"{self.name}_{difficulty}",
            description=self.description,
            items=filtered,
            metadata=self.metadata,
        )


class DatasetLoader:
    """Load and validate datasets from various sources."""

    @staticmethod
    def from_json(path: str | Path) -> Dataset:
        """Load a dataset from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise DatasetValidationError(f"Dataset file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"Invalid JSON in {path}: {exc}") from exc

        return DatasetLoader._parse_data(data, path.stem)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Dataset:
        """Load a dataset from a dictionary."""
        return DatasetLoader._parse_data(data, data.get("name", "unnamed"))

    @staticmethod
    def _parse_data(data: dict[str, Any], default_name: str) -> Dataset:
        """Parse raw data into a Dataset."""
        if "items" not in data:
            raise DatasetValidationError("Dataset must contain 'items' key")

        items = []
        for idx, item_data in enumerate(data["items"]):
            if isinstance(item_data, str):
                item_data = {"input": item_data, "expected_output": ""}
            if "id" not in item_data:
                item_data["id"] = f"item_{idx}"
            try:
                items.append(DatasetItem(**item_data))
            except Exception as exc:
                raise DatasetValidationError(f"Invalid item at index {idx}: {exc}") from exc

        return Dataset(
            name=data.get("name", default_name),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            items=items,
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def validate(dataset: Dataset) -> list[str]:
        """Validate a dataset and return list of warnings."""
        warnings = []
        ids_seen = set()
        for i, item in enumerate(dataset.items):
            if item.id in ids_seen:
                warnings.append(f"Duplicate ID '{item.id}' at index {i}")
            ids_seen.add(item.id)
            if not item.expected_output:
                warnings.append(f"Item '{item.id}' has no expected_output")
        return warnings
