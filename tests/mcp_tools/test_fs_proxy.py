"""Tests for fs_proxy environment passthrough functionality."""
import pytest

from mcp_tools.fs_proxy import _build_passthrough_env, create_fs_proxy


class TestBuildPassthroughEnv:
    """Tests for _build_passthrough_env helper."""

    def test_returns_empty_dict_for_empty_list(self):
        result = _build_passthrough_env([])
        assert result == {}

    def test_returns_empty_dict_when_vars_not_set(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        result = _build_passthrough_env(["NONEXISTENT_VAR"])
        assert result == {}

    def test_returns_set_vars_only(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_A", "value_a")
        monkeypatch.delenv("TEST_VAR_B", raising=False)
        result = _build_passthrough_env(["TEST_VAR_A", "TEST_VAR_B"])
        assert result == {"TEST_VAR_A": "value_a"}

    def test_returns_all_set_vars(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_A", "value_a")
        monkeypatch.setenv("TEST_VAR_B", "value_b")
        result = _build_passthrough_env(["TEST_VAR_A", "TEST_VAR_B"])
        assert result == {"TEST_VAR_A": "value_a", "TEST_VAR_B": "value_b"}
