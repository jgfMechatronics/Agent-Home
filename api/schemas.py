"""
Pydantic request/response schemas for the API layer.
TODO: CRITICAL these are just stubs. They need to be tested/implemented per the implementation_plan
"""
from datetime import datetime

from pydantic import BaseModel


# --- Request Schemas ---

class MessageRequest(BaseModel):
    message: str


class CreateAgentRequest(BaseModel):
    name: str
    model_name: str
    system_instructions: str


# --- Response Schemas ---

class MessageResponse(BaseModel):
    id: str
    name: str


class AgentMetadataResponse(BaseModel):
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


class MessagesResponse(BaseModel):
    messages: list


class HealthResponse(BaseModel):
    status: str
