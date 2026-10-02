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


class TestCreateFsProxy:
    """Tests for create_fs_proxy factory function."""

    def test_env_flows_to_create_proxy(self, mocker):
        """Verify env dict is passed through to create_proxy."""
        mock_create_proxy = mocker.patch("mcp_tools.fs_proxy.create_proxy")
        # create_proxy returns a mock that has the methods we call
        mock_proxy = mocker.MagicMock()
        mock_create_proxy.return_value = mock_proxy

        test_env = {"MY_VAR": "my_value"}
        create_fs_proxy(env=test_env)

        # Verify create_proxy was called with config containing our env
        mock_create_proxy.assert_called_once()
        config_arg = mock_create_proxy.call_args[0][0]
        assert config_arg["mcpServers"]["desktop-commander"]["env"] == test_env

    def test_no_env_key_when_none(self, mocker):
        """Verify env key is omitted from config when no env passed."""
        mock_create_proxy = mocker.patch("mcp_tools.fs_proxy.create_proxy")
        mock_proxy = mocker.MagicMock()
        mock_create_proxy.return_value = mock_proxy

        create_fs_proxy()

        mock_create_proxy.assert_called_once()
        config_arg = mock_create_proxy.call_args[0][0]
        assert "env" not in config_arg["mcpServers"]["desktop-commander"]
