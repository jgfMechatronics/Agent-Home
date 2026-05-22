"""
Unit tests for API schemas — custom validation logic only.
Field definitions are self-documenting; only behavior we wrote belongs here.
"""
from datetime import datetime

import pytest
from pydantic import BaseModel, ValidationError

from agent.types import AgentConfig
from api.schemas import (
    AgentMetadataResponse,
    CoreMemoryResponse,
    CreateAgentRequest,
    HealthResponse,
    MemoryBlockResponse,
    MessageItem,
    MessageRequest,
    MessagesResponse,
)

# --- Shared fixtures for serialization tests ---

_NOW = datetime(2026, 1, 1)
_CONFIG = AgentConfig(model_name="claude-sonnet-4-20250514", tool_names=[], soft_compaction_limit=1000)
_BLOCK = MemoryBlockResponse(label="persona", description="desc", content="content", char_limit=1000, updated_at=_NOW)


@pytest.mark.parametrize("instance", [
    pytest.param(MessageRequest(message="hello"), id="MessageRequest"),
    pytest.param(CreateAgentRequest(name="test", system_instructions="sys", config=_CONFIG), id="CreateAgentRequest"),
    pytest.param(AgentMetadataResponse(id="abc", name="test", model="claude-sonnet-4-20250514", created_at=_NOW, updated_at=_NOW), id="AgentMetadataResponse"),
    pytest.param(_BLOCK, id="MemoryBlockResponse"),
    pytest.param(CoreMemoryResponse(blocks=[_BLOCK]), id="CoreMemoryResponse"),
    pytest.param(MessageItem(id="abc", type="ModelRequest", content="{}", timestamp=_NOW), id="MessageItem"),
    pytest.param(MessagesResponse(messages=[]), id="MessagesResponse"),
    pytest.param(HealthResponse(status="ok"), id="HealthResponse"),
])
def test_serializes_to_json(instance: BaseModel) -> None:
    """All API schemas must serialize to valid JSON without raising."""
    instance.model_dump_json()


@pytest.mark.parametrize("empty_msg", ["", " "])
def test_message_request_rejects_empty_string(empty_msg: str) -> None:
    """MessageRequest.message must not be empty."""
    with pytest.raises(ValidationError):
        MessageRequest(message=empty_msg)
