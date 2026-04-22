"""Test configuration and shared fixtures."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (no external deps)")
    config.addinivalue_line("markers", "integration: Integration tests (may need API keys)")
    config.addinivalue_line("markers", "evaluation: Evaluation pipeline tests")
