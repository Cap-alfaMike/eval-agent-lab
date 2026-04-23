"""Unit tests for dataset loading and validation."""

import json

import pytest

from eval_agent_lab.datasets import Dataset, DatasetItem, DatasetLoader
from eval_agent_lab.exceptions import DatasetValidationError


@pytest.mark.unit
class TestDatasetItem:
    def test_valid_item(self):
        item = DatasetItem(
            id="test_1",
            input="What is Python?",
            expected_output="A programming language",
            expected_tools=["search"],
            category="knowledge",
        )
        assert item.id == "test_1"
        assert item.difficulty == "medium"

    def test_empty_input_raises(self):
        with pytest.raises((ValueError, Exception)):
            DatasetItem(input="", expected_output="something")

    def test_new_schema_defaults(self):
        """New fields should have sensible defaults for backward compatibility."""
        item = DatasetItem(input="test", expected_output="answer")
        assert item.acceptable_outputs == []
        assert item.tool_strategy == "optional"
        assert item.max_steps == 10
        assert item.penalize_overuse is False
        assert item.expected_contains == []
        assert item.expected_reasoning == []

    def test_new_schema_fields(self):
        """All new fields should be settable."""
        item = DatasetItem(
            input="What is 2+2?",
            expected_output="4",
            acceptable_outputs=["4", "four"],
            tool_strategy="must_use",
            expected_tools=["calculator"],
            max_steps=1,
            penalize_overuse=True,
            expected_contains=["4"],
            expected_reasoning=["compute addition"],
        )
        assert item.acceptable_outputs == ["4", "four"]
        assert item.tool_strategy == "must_use"
        assert item.max_steps == 1
        assert item.penalize_overuse is True
        assert item.expected_contains == ["4"]
        assert item.expected_reasoning == ["compute addition"]

    def test_tool_strategy_forbidden(self):
        item = DatasetItem(
            input="What is Atlantis?",
            expected_output="fictional",
            tool_strategy="forbidden",
        )
        assert item.tool_strategy == "forbidden"


@pytest.mark.unit
class TestDatasetLoader:
    def test_load_from_dict(self):
        data = {
            "name": "test_dataset",
            "items": [
                {"input": "Q1", "expected_output": "A1"},
                {"input": "Q2", "expected_output": "A2"},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        assert ds.name == "test_dataset"
        assert ds.size == 2

    def test_load_from_json_file(self, tmp_path):
        data = {
            "name": "file_test",
            "items": [{"input": "Q1", "expected_output": "A1"}],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        ds = DatasetLoader.from_json(path)
        assert ds.name == "file_test"
        assert ds.size == 1

    def test_missing_file_raises(self):
        with pytest.raises(DatasetValidationError):
            DatasetLoader.from_json("nonexistent.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(DatasetValidationError):
            DatasetLoader.from_json(path)

    def test_missing_items_raises(self):
        with pytest.raises(DatasetValidationError):
            DatasetLoader.from_dict({"name": "test"})

    def test_auto_id_assignment(self):
        data = {
            "items": [
                {"input": "Q1", "expected_output": "A1"},
                {"input": "Q2", "expected_output": "A2"},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        assert ds.items[0].id == "item_0"
        assert ds.items[1].id == "item_1"


@pytest.mark.unit
class TestDatasetValidation:
    def test_validate_no_warnings(self):
        ds = Dataset(
            name="clean",
            items=[
                DatasetItem(id="1", input="Q1", expected_output="A1"),
                DatasetItem(id="2", input="Q2", expected_output="A2"),
            ],
        )
        warnings = DatasetLoader.validate(ds)
        assert len(warnings) == 0

    def test_validate_duplicate_ids(self):
        ds = Dataset(
            name="dupes",
            items=[
                DatasetItem(id="1", input="Q1", expected_output="A1"),
                DatasetItem(id="1", input="Q2", expected_output="A2"),
            ],
        )
        warnings = DatasetLoader.validate(ds)
        assert any("Duplicate" in w for w in warnings)

    def test_validate_missing_output(self):
        ds = Dataset(
            name="no_output",
            items=[DatasetItem(id="1", input="Q1", expected_output="")],
        )
        warnings = DatasetLoader.validate(ds)
        assert any("expected_output" in w for w in warnings)


@pytest.mark.unit
class TestDatasetFiltering:
    def test_filter_by_category(self):
        ds = Dataset(
            name="mixed",
            items=[
                DatasetItem(id="1", input="Q1", expected_output="A1", category="knowledge"),
                DatasetItem(id="2", input="Q2", expected_output="A2", category="reasoning"),
                DatasetItem(id="3", input="Q3", expected_output="A3", category="knowledge"),
            ],
        )
        filtered = ds.filter_by_category("knowledge")
        assert filtered.size == 2

    def test_filter_by_difficulty(self):
        ds = Dataset(
            name="mixed",
            items=[
                DatasetItem(id="1", input="Q1", expected_output="A1", difficulty="easy"),
                DatasetItem(id="2", input="Q2", expected_output="A2", difficulty="hard"),
            ],
        )
        filtered = ds.filter_by_difficulty("easy")
        assert filtered.size == 1
