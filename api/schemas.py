"""
Pydantic request/response schemas for the API layer.
"""
from datetime import datetime

from pydantic import BaseModel, field_validator

from agent.types import AgentConfig


# --- Request Schemas ---

class MessageRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v


class CreateAgentRequest(BaseModel):
    name: str
    system_instructions: str
    config: AgentConfig


# --- Response Schemas ---

class AgentMetadataResponse(BaseModel):
    id: str
    name: str
    model: str
    created_at: datetime
    updated_at: datetime


class MemoryBlockResponse(BaseModel):
    label: str
    description: str
    content: str
    char_limit: int
    updated_at: datetime


class CoreMemoryResponse(BaseModel):
    blocks: list[MemoryBlockResponse]


# TODO: MessageItem returns raw type + content (serialized ModelMessage JSON) for now.
# We're punting display-layer parsing until we have hands-on experience with the format
# and know what the UI actually needs. Revisit when building the UI/CLI connection.
class MessageItem(BaseModel):
    id: str
    type: str       # 'ModelRequest' | 'ModelResponse' | 'Summary'
    content: str    # raw serialized ModelMessage JSON
    timestamp: datetime


class MessagesResponse(BaseModel):
    messages: list[MessageItem]


class HealthResponse(BaseModel):
    status: str
