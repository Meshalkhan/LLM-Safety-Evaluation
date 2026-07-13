"""Tests for dataset loading."""

import pytest

from utils.dataset_loader import load_dataset


@pytest.mark.parametrize(
    "name,min_rows",
    [
        ("jailbreak", 10),
        ("prompt_injection", 10),
        ("harmless", 10),
        ("system_override", 5),
        ("factual", 15),
    ],
)
def test_dataset_loads(name: str, min_rows: int):
    df = load_dataset(name)
    assert len(df) >= min_rows
    assert "prompt" in df.columns


def test_dataset_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent_dataset")
