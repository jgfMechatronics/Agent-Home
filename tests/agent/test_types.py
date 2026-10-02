"""
Tests for agent/types.py — Section 3.0

AgentConfig: Pydantic model for agent configuration
AgentDeps: Dataclass holding request-scoped agent state
"""
import json
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from conftest import SAMPLE_AGENT_CONFIG_DATA
from pydantic import ValidationError

from agent.types import AgentConfig, AgentDeps


@pytest.fixture
def valid_config_data() -> dict:
    """Complete valid config for use as baseline in tests (copy to avoid mutation)."""
    return SAMPLE_AGENT_CONFIG_DATA.copy()


# --- AgentConfig valid construction ---

def test_agentconfig_valid_construction(valid_config_data: dict):
    """AgentConfig should construct successfully with all required fields."""
    config = AgentConfig(**valid_config_data)
    
    assert config.model_name == valid_config_data["model_name"]
    assert config.tool_names == valid_config_data["tool_names"]
    assert config.soft_compaction_limit == valid_config_data["soft_compaction_limit"]
    assert config.is_deletable is False  # default


# --- AgentConfig required fields ---

@pytest.mark.parametrize("missing_field", [
    "model_name",
    "tool_names",
    "soft_compaction_limit",
])
def test_agentconfig_requires_field(valid_config_data: dict, missing_field: str):
    """AgentConfig should raise ValidationError when required field is missing."""
    del valid_config_data[missing_field]
    
    with pytest.raises(ValidationError) as exc_info:
        AgentConfig(**valid_config_data)
    
    # Verify the error mentions the missing field
    errors = exc_info.value.errors()
    assert any(missing_field in str(e["loc"]) for e in errors)


# --- AgentConfig type validation ---

@pytest.mark.parametrize("field,invalid_value,description", [
    ("model_name", "", "empty string — infer_model rejects"),
    ("model_name", "   ", "whitespace only — infer_model rejects"),
    ("model_name", ":claude-haiku-4-5", "empty provider — infer_model rejects"),
    ("model_name", "anthropic:", "empty model part — our check"),
    ("model_name", "anthropic:   ", "whitespace model part — our check"),
    ("tool_names", "not_a_list", "tool_names must be a list"),
    ("tool_names", [1, 2, 3], "tool_names must be list of strings"),
    ("soft_compaction_limit", 0, "soft_compaction_limit must be positive"),
    ("soft_compaction_limit", -100, "soft_compaction_limit must be positive"),
    ("soft_compaction_limit", "not_an_int", "soft_compaction_limit must be int"),
])
def test_agentconfig_validates_types(valid_config_data: dict, field: str, invalid_value, description: str):
    """AgentConfig should raise ValidationError for invalid field values."""
    valid_config_data[field] = invalid_value
    
    with pytest.raises(ValidationError):
        AgentConfig(**valid_config_data)


# --- AgentConfig retries validation ---

@pytest.mark.parametrize("invalid_retries", [-1, -100])
def test_agentconfig_retries_must_be_non_negative(valid_config_data: dict, invalid_retries: int):
    """Negative retry counts are nonsensical and should be rejected."""
    valid_config_data["retries"] = invalid_retries
    with pytest.raises(ValidationError):
        AgentConfig(**valid_config_data)


@pytest.mark.parametrize("valid_retries", [0, 1, 4])
def test_agentconfig_retries_non_negative_is_valid(valid_config_data: dict, valid_retries: int):
    """Zero and positive retry counts are all valid."""
    valid_config_data["retries"] = valid_retries
    config = AgentConfig(**valid_config_data)
    assert config.retries == valid_retries


# --- AgentConfig model_name validation ---

@pytest.mark.parametrize("model_name", [
    "anthropic:claude-haiku-4-5",
    "anthropic:claude-sonnet-4-20250514",
    "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together:THUDM/glm-4-9b-chat",
    "openai-chat:gpt-4o",
    "test",  # pydantic-ai magic string — instantiates TestModel
])
def test_agentconfig_accepts_valid_model_name(valid_config_data: dict, model_name: str):
    """Any model name pydantic-ai can resolve should be accepted."""
    valid_config_data["model_name"] = model_name
    config = AgentConfig(**valid_config_data)
    assert config.model_name == model_name


def test_agentconfig_rejects_unknown_provider(valid_config_data: dict):
    """Unknown providers should be rejected via provider registry lookup, surfacing as ValidationError."""
    valid_config_data["model_name"] = "badprovider:some-model"
    with pytest.raises(ValidationError):
        AgentConfig(**valid_config_data)


# --- AgentConfig defaults ---

def test_agentconfig_is_deletable_defaults_to_false(valid_config_data: dict):
    """is_deletable should default to False when not provided."""
    config = AgentConfig(**valid_config_data)
    assert config.is_deletable is False


def test_agentconfig_is_deletable_can_be_set_true(valid_config_data: dict):
    """is_deletable can be explicitly set to True."""
    valid_config_data["is_deletable"] = True
    config = AgentConfig(**valid_config_data)
    assert config.is_deletable is True


def test_agentconfig_thinking_enabled_defaults_to_false(valid_config_data: dict):
    """thinking_enabled should default to False when not provided."""
    config = AgentConfig(**valid_config_data)
    assert config.thinking_enabled is False


def test_agentconfig_thinking_enabled_can_be_set_true(valid_config_data: dict):
    """thinking_enabled can be explicitly set to True."""
    valid_config_data["thinking_enabled"] = True
    config = AgentConfig(**valid_config_data)
    assert config.thinking_enabled is True


# --- AgentConfig JSON round-trip ---

def test_agentconfig_json_roundtrip(valid_config_data: dict):
    """AgentConfig should round-trip through JSON correctly."""
    original = AgentConfig(**valid_config_data)
    
    # Serialize to JSON string
    json_str = original.model_dump_json()
    
    # Parse back
    restored = AgentConfig.model_validate_json(json_str)
    
    assert restored == original


def test_agentconfig_dict_roundtrip(valid_config_data: dict):
    """AgentConfig should round-trip through dict correctly."""
    original = AgentConfig(**valid_config_data)
    
    # To dict, then to JSON string, then back
    as_dict = original.model_dump()
    json_str = json.dumps(as_dict)
    parsed = json.loads(json_str)
    restored = AgentConfig.model_validate(parsed)
    
    assert restored == original


# --- AgentConfig extra fields ---

def test_agentconfig_rejects_extra_fields(valid_config_data: dict):
    """AgentConfig should reject unknown fields (extra='forbid')."""
    valid_config_data["unknown_field"] = "should_fail"
    
    with pytest.raises(ValidationError) as exc_info:
        AgentConfig(**valid_config_data)
    
    errors = exc_info.value.errors()
    assert any("extra" in str(e["type"]).lower() for e in errors)


# --- AgentDeps ---

@pytest.mark.parametrize("missing_field", ["session", "agent_record"])
def test_agentdeps_requires_field(missing_field: str):
    """AgentDeps should raise TypeError when a required constructor argument is missing."""
    all_fields = {
        "session": object(),
        "agent_record": MagicMock(),
    }
    del all_fields[missing_field]

    with pytest.raises(TypeError):
        AgentDeps(**all_fields)


class TestAgentDepsCommitChangesRefreshAgentRecord:
    """commit_changes_refresh_agent_record — commit+refresh ordering invariant."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_record = MagicMock()
        self.mock_session = AsyncMock()
        self.deps = AgentDeps(session=self.mock_session, agent_record=self.mock_record)

    async def test_commits_then_refreshes(self):
        """Refresh must follow commit — refreshing first would reload stale data."""
        await self.deps.commit_changes_refresh_agent_record()

        assert self.mock_session.mock_calls == [call.commit(), call.refresh(self.mock_record)]

    async def test_refresh_not_called_when_commit_raises(self):
        """A failed commit leaves the DB unchanged — refresh must not be called."""
        self.mock_session.commit.side_effect = RuntimeError("DB connection lost")

        with pytest.raises(RuntimeError, match="DB connection lost"):
            await self.deps.commit_changes_refresh_agent_record()

        self.mock_session.refresh.assert_not_called()


async def test_agentdeps_holds_expected_fields(session, agent_deps, agent_record):
    """AgentDeps properties should delegate to the underlying AgentRecord."""
    assert agent_deps.agent_id == agent_record.id
    assert agent_deps.session is session
    assert agent_deps.config is agent_record.agent_config
    assert agent_deps.name == agent_record.name
    assert agent_deps.system_instructions == agent_record.system_instructions
