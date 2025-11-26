"""Example test file to verify pytest setup."""

import pytest


def test_example():
    """Simple example test to verify pytest is working."""
    assert True


def test_math():
    """Another simple test example."""
    assert 1 + 1 == 2


@pytest.mark.unit
def test_string_operations():
    """Test string operations."""
    assert "hello" + " " + "world" == "hello world"
    assert len("test") == 4

