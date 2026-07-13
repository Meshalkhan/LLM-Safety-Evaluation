"""Tests for model factory."""

import pytest

from models.factory import detect_provider


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4.1", "openai"),
        ("gpt-4o", "openai"),
        ("o1-preview", "openai"),
        ("claude-sonnet-4-20250514", "anthropic"),
        ("gemini-2.0-flash", "gemini"),
    ],
)
def test_detect_provider(model: str, expected: str):
    assert detect_provider(model) == expected
