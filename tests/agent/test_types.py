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
# --- Fixtures ---

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
    ("model_name", "", "model_name cannot be empty"),
    ("model_name", "   ", "model_name cannot be whitespace-only"),
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


@pytest.mark.xfail(reason="Need validate model name with get_model() once implemented.")
def test_todo_validate_model_name_w_get_model():
    """
    TODO, once get_model implemented validate that str corresponds to a valid AnthropicModel.
    Or, consider just storing model_name as an AnthropicModel and dealing with the DB integration.
    """
    pytest.fail()

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


async def test_agentdeps_holds_expected_fields(session, agent_record):
    """AgentDeps properties should delegate to the underlying AgentRecord."""
    deps = AgentDeps(session=session, agent_record=agent_record)

    assert deps.agent_id == agent_record.id
    assert deps.session is session
    assert deps.config is agent_record.agent_config
    assert deps.name == agent_record.name
    assert deps.system_instructions == agent_record.system_instructions
