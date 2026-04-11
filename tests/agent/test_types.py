"""
Tests for agent/types.py — Section 3.0

AgentConfig: Pydantic model for agent configuration
AgentDeps: Dataclass holding request-scoped agent state
"""
import json

import pytest
from conftest import SAMPLE_AGENT_CONFIG
from pydantic import ValidationError

from agent.types import AgentConfig, AgentDeps


# --- Fixtures ---

@pytest.fixture
def valid_config_data() -> dict:
    """Complete valid config for use as baseline in tests."""
    return SAMPLE_AGENT_CONFIG


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

@pytest.mark.parametrize("missing_field", ["session", "agent_id", "config"])
def test_agentdeps_requires_field(valid_config_data: dict, missing_field: str):
    """AgentDeps should raise TypeError when required field is missing."""
    config = AgentConfig(**valid_config_data)
    mock_session = object()
    
    all_fields = {
        "session": mock_session,
        "agent_id": "test-agent-id",
        "config": config,
    }
    del all_fields[missing_field]
    
    with pytest.raises(TypeError) as exc_info:
        AgentDeps(**all_fields)
    
    # Dataclass error message mentions the missing field
    assert missing_field in str(exc_info.value)


def test_agentdeps_holds_expected_fields(valid_config_data: dict):
    """AgentDeps should hold agent_id, session, and config."""
    config = AgentConfig(**valid_config_data)
    
    # Use a mock for session since we just need to verify field storage
    mock_session = object()  # Placeholder, not a real AsyncSession
    
    deps = AgentDeps(
        session=mock_session,
        agent_id="test-agent-id",
        config=config,
    )
    
    assert deps.agent_id == "test-agent-id"
    assert deps.session is mock_session
    assert deps.config is config
    assert deps.config.model_name == valid_config_data["model_name"]
