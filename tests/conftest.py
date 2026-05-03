"""Shared pytest configuration for vexel tests."""
import pytest


# Make pytest-asyncio auto-mode apply to all async tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )