"""
Prototype-quality route tests. These cover endpoints or behaviours that were
added on prototype branches and haven't been fully reviewed yet.
"""
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRecord


class TestGetMessages:
    """
    GET /agents/{agent_id}/messages — conversation history.
    TODO: This is OK for now but we will likely rework the endpoint after defining what is most useful for the frontend in terms of message format
    """

    @staticmethod
    def _make_message_record(
        id: str = "msg-1",
        seq_id: int = 0,
        type: str = "ModelResponse",
        content: str = '{"kind": "response", "parts": []}',
        timestamp: datetime | None = None,
    ) -> Mock:
        """Build a mock MessageRecord with the attributes the route accesses."""
        m = Mock()
        m.id = id
        m.seq_id = seq_id
        m.type = type
        m.content = content
        m.timestamp = timestamp or datetime(2026, 6, 9, 12, 0, 0)
        return m

    @pytest.fixture(autouse=True)
    def mock_message_loaders(self):
        """Patch load_messages for all TestGetMessages tests.

        Provides self.mock_load_messages for loader-routing assertions.
        """
        with patch("api.routes.load_messages", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = []
            self.mock_load_messages = mock_load
            yield

    async def test_default_loads_context_window_and_returns_messages(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """Without ?full=true: calls load_messages with context_window_start as start_seq_id."""
        self.mock_load_messages.return_value = [
            self._make_message_record(id="msg-1", seq_id=0, content='{"kind": "response", "parts": []}')
        ]

        response = await client.get(f"/agents/{agent_record.id}/messages")

        assert response.status_code == 200
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_seq_id=agent_record.context_window_start
        )

    async def test_full_true_returns_complete_history(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """With ?full=true: calls load_messages with start_seq_id=0 for full history."""
        self.mock_load_messages.return_value = [
            self._make_message_record(id="msg-1", seq_id=0),
            self._make_message_record(id="msg-2", seq_id=1),
        ]

        response = await client.get(f"/agents/{agent_record.id}/messages?full=true")

        assert response.status_code == 200
        assert len(response.json()["messages"]) == 2
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_seq_id=0
        )

    async def test_response_uses_message_item_format(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """Response items use MessageItem format: id, seq_id, type, content (raw JSON string), timestamp (ISO string)."""
        ts = datetime(2026, 6, 9, 12, 0, 0)
        raw_content = '{"kind": "response", "parts": [{"part_kind": "text", "content": "hello"}]}'
        self.mock_load_messages.return_value = [
            self._make_message_record(id="msg-42", seq_id=7, type="ModelResponse", content=raw_content, timestamp=ts)
        ]

        response = await client.get(f"/agents/{agent_record.id}/messages")

        assert response.status_code == 200
        messages = response.json()["messages"]
        assert len(messages) == 1
        item = messages[0]
        assert item["id"] == "msg-42"
        assert item["seq_id"] == 7
        assert item["type"] == "ModelResponse"
        assert item["content"] == raw_content  # raw JSON string — NOT parsed by the route
        assert item["timestamp"] == ts.isoformat()

    # 404 tested via parametrized test_get_endpoints_return_404_for_unknown_agent
