"""Unit tests for the HF dataset export module."""

import json
import pytest
from pathlib import Path

from eval_agent_lab.datasets.export_hf import (
    _build_dataset_card,
    _dataset_to_jsonl,
)
from eval_agent_lab.datasets import DatasetLoader


@pytest.mark.unit
class TestDatasetCard:
    def test_card_contains_metadata(self, tmp_path):
        data = {
            "name": "test_ds",
            "description": "A test dataset",
            "version": "1.0.0",
            "items": [
                {"input": "Q1", "expected_output": "A1", "category": "knowledge"},
                {"input": "Q2", "expected_output": "A2", "category": "reasoning"},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        card = _build_dataset_card(ds, "user/test-repo")
        assert "test_ds" in card
        assert "A test dataset" in card
        assert "knowledge" in card
        assert "reasoning" in card
        assert "user/test-repo" in card
        assert "apache-2.0" in card


@pytest.mark.unit
class TestDatasetToJsonl:
    def test_jsonl_format(self):
        data = {
            "name": "test",
            "items": [
                {"input": "Q1", "expected_output": "A1"},
                {"input": "Q2", "expected_output": "A2"},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        jsonl = _dataset_to_jsonl(ds)
        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

        record = json.loads(lines[0])
        assert record["input"] == "Q1"
        assert record["expected_output"] == "A1"
        assert "category" in record
        assert "difficulty" in record

    def test_jsonl_preserves_fields(self):
        data = {
            "name": "test",
            "items": [
                {
                    "input": "Q1",
                    "expected_output": "A1",
                    "expected_tools": ["search"],
                    "context": "some context",
                    "category": "tool_use",
                    "difficulty": "hard",
                },
            ],
        }
        ds = DatasetLoader.from_dict(data)
        jsonl = _dataset_to_jsonl(ds)
        record = json.loads(jsonl.strip())
        assert record["expected_tools"] == ["search"]
        assert record["context"] == "some context"
        assert record["category"] == "tool_use"
        assert record["difficulty"] == "hard"
