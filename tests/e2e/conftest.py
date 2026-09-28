"""E2E test configuration.

Tests in this directory require a live server and are excluded from the default test run.
Run with: pytest -m e2e
"""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply e2e marker to all tests in this directory."""
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
