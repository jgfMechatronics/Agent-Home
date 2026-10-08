"""Tests for send_message tool — inter-agent communication.

Prototype-quality tests for the IAC implementation.
"""
import asyncio
import json

import pytest
import pytest_asyncio
from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, DeltaToolCalls, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.runner import COMPACTION_RESUME_NOTICE, run_stateful_agent, is_compaction_needed as _real_is_compaction_needed
from agent.types import AgentAppState, AgentDeps
from conftest import SAMPLE_AGENT_CONFIG, _make_mock_session, make_alternating_messages, mock_run_context, assert_ModelMessage_list_eq
from db.models import AgentRecord
from prototype.iac.send_message import (
    _format_inter_agent_message,
    _deliver_message,
    configure_iac,
    send_message,
)
import prototype.iac.send_message as iac_module

# TODO: move to conftest once on a proper PR branch
from tests.agent.test_runner import FunctionModelTestAgent, _PersistenceAndCancellationTestBase


class TestSendMessage:
    """send_message tool: inter-agent communication."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession):
        """Create sender agent and optionally a target for delivery tests."""
        sender = AgentRecord(
            name="sender-agent",
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="Sender",
        )
        session.add(sender)
        await session.flush()
        self.sender = sender
        self.session = session
        self.deps = AgentDeps(session=session, agent_record=sender)
        self.ctx = mock_run_context(self.deps)
        # Clear IAC module state before each test (some tests check unconfigured state)
        iac_module._agent_app_state_reg_IAC_ref = None
        iac_module._broadcast_hub_IAC_ref = None

    async def _create_target(self):
        """Helper: create target agent in DB."""
        target = AgentRecord(
            name="target-agent",
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="Target",
        )
        self.session.add(target)
        await self.session.flush()
        return target

    def _configure_iac(self, mocker):
        """Helper: configure IAC module (enables send_message)."""
        registry = {self.sender.id: mocker.MagicMock()}
        hub = mocker.MagicMock()
        configure_iac(registry, hub)
        return registry, hub

    async def test_target_not_found_raises_model_retry(self):
        """Raises ModelRetry when no agent with that name exists."""
        with pytest.raises(ModelRetry, match="ghost"):
            await send_message(self.ctx, target_name="ghost", content="hello")

    async def test_target_not_found_is_case_sensitive(self):
        """Name lookup is case-sensitive — mismatched case raises ModelRetry."""
        with pytest.raises(ModelRetry, match="Sender-Agent"):
            await send_message(self.ctx, target_name="Sender-Agent", content="hi")

    async def test_rejects_self_message(self):
        """Raises ModelRetry when agent tries to message itself."""
        with pytest.raises(ModelRetry, match="cannot send.*yourself|self"):
            await send_message(self.ctx, target_name="sender-agent", content="talking to myself")

    async def test_raises_model_retry_when_registry_not_configured(self):
        """Raises ModelRetry when deps lacks agent_app_state_reg."""
        await self._create_target()
        with pytest.raises(ModelRetry, match="not configured"):
            await send_message(self.ctx, target_name="target-agent", content="hello")

    @pytest.mark.parametrize("delivery_succeeds,expect_error", [
        pytest.param(True, False, id="success"),
        pytest.param(False, True, id="busy"),
    ])
    async def test_delivery_outcome(self, mocker, delivery_succeeds, expect_error):
        """Delivery success returns message; failure raises ModelRetry."""
        await self._create_target()
        self._configure_iac(mocker)

        async def mock_deliver(*args, **kwargs):
            kwargs["delivery_future"].set_result(delivery_succeeds)
        mocker.patch("prototype.iac.send_message._deliver_message", side_effect=mock_deliver)

        if expect_error:
            with pytest.raises(ModelRetry, match="target-agent"):
                await send_message(self.ctx, target_name="target-agent", content="hello")
        else:
            result = await send_message(self.ctx, target_name="target-agent", content="hello")
            assert "delivered" in result.lower() and "target-agent" in result


class TestDeliverMessage:
    """_deliver_message: background task for inter-agent delivery."""

    @pytest.fixture
    def mock_session(self, mocker):
        """Mock get_session context manager (patch at source — imports are deferred)."""
        mock_cm = mocker.AsyncMock()
        mock_cm.__aenter__.return_value = mocker.MagicMock()
        mock_cm.__aexit__.return_value = None
        mocker.patch("db.connection.get_session", return_value=mock_cm)
        return mock_cm

    async def test_lock_unavailable_signals_false(self, mocker, mock_session):
        """Future receives False when AgentLockedError."""
        from agent.types import AgentLockedError

        mock_factory_cm = mocker.AsyncMock()
        mock_factory_cm.__aenter__.side_effect = AgentLockedError("test-agent-id")

        mock_factory = mocker.MagicMock()
        mock_factory.build_agent_and_deps.return_value = mock_factory_cm
        mocker.patch("agent.factory.AgentFactory", return_value=mock_factory)

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        await _deliver_message(
            agent_id="test-agent-id",
            user_prompt="hello",
            engine=mocker.MagicMock(),
            agent_app_state_reg={"test-agent-id": mocker.MagicMock()},
            broadcast_hub=mocker.MagicMock(),
            delivery_future=future,
        )

        assert future.done() and future.result() is False

    async def test_lock_acquired_signals_true_and_uses_broadcast_runner(self, mocker, mock_session):
        """Lock acquired: future receives True and run_agent_with_broadcast is called."""
        mock_agent = mocker.MagicMock()
        mock_deps = mocker.MagicMock()
        mock_factory_cm = mocker.AsyncMock()
        mock_factory_cm.__aenter__.return_value = (mock_agent, mock_deps)
        mock_factory_cm.__aexit__.return_value = None

        async def mock_run(*args, **kwargs):
            return
            yield
        mock_broadcast_run = mocker.patch(
            "agent.broadcast_streaming.run_agent_with_broadcast", side_effect=mock_run
        )

        mock_factory = mocker.MagicMock()
        mock_factory.build_agent_and_deps.return_value = mock_factory_cm
        mocker.patch("agent.factory.AgentFactory", return_value=mock_factory)

        mock_app_state = mocker.MagicMock()
        mock_hub = mocker.MagicMock()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        await _deliver_message(
            agent_id="test-agent-id",
            user_prompt="hello",
            engine=mocker.MagicMock(),
            agent_app_state_reg={"test-agent-id": mock_app_state},
            broadcast_hub=mock_hub,
            delivery_future=future,
        )

        assert future.done() and future.result() is True
        mock_broadcast_run.assert_called_once_with(
            mock_agent, mock_deps, mock_app_state, "hello", mock_hub
        )


class _SenderTestAgent(FunctionModelTestAgent):
    """FunctionModelTestAgent subclass: emits a send_message tool call, then completes."""

    RECIPIENT_NAME = "recipient-agent"
    SEND_MSG_ARGS = json.dumps({"target_name": RECIPIENT_NAME, "content": "hello from sender"})
    SEND_MSG_CALL = DeltaToolCalls({
        0: DeltaToolCall(name="send_message", json_args=SEND_MSG_ARGS, tool_call_id="sm-tc-1")
    })

    def __init__(self):
        super().__init__()
        self.set_steps([[self.SEND_MSG_CALL], self.COMPLETION_TEXT])

    def _build(self) -> Agent:
        """Agent with send_message as its only tool."""
        agent = Agent(FunctionModel(stream_function=self._stream), deps_type=AgentDeps)
        agent.tool(send_message)
        return agent


class TestSendMessageContextIsolation(_PersistenceAndCancellationTestBase):
    """
    Integration: send_message spawns a background agent (B) that runs through
    compaction+resume (2 iterations). Verifies B's messages from both iterations
    are correctly persisted.

    Fails without the asyncio.create_task contextvar isolation fix: B's iter2
    messages are invisible to the runner (it watches the stale iter1 _RunMessages
    list while pydantic-ai writes to a fresh one).

    TODO: Move fixture dependencies to conftest once on a proper PR branch
    TODO: This test got very painful. Will need to rework once we are targeting the proper send messages impl,
    and really we probably want better integration test infrastructure in general.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, mocker):
        # Real DB records — only needed for flush (gives us real UUIDs) and
        # as registry keys. NOT used inside the agent-run path to avoid
        # ORM attribute access inside pydantic-ai's async context (which
        # uses anyio greenlets that are not SQLAlchemy greenlets, causing
        # MissingGreenlet on lazy-load of expired ORM objects).
        sender_record = AgentRecord(
            name="sender-agent",
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="You are the sender.",
        )
        recipient_record = AgentRecord(
            name=_SenderTestAgent.RECIPIENT_NAME,
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="You are the recipient.",
        )
        session.add_all([sender_record, recipient_record])
        await session.flush()
        # Refresh to load all attributes — prevents MissingGreenlet on later access
        await session.refresh(sender_record)
        await session.refresh(recipient_record)

        # Use real records directly (refresh prevents lazy load issues)
        self.sender_record = sender_record
        self.recipient_record = recipient_record
        sender_id = sender_record.id
        recipient_id = recipient_record.id

        # Shared registry — keyed by real recipient UUID
        self.app_state_reg = {recipient_id: AgentAppState()}
        # Mock hub for broadcast (we don't need real broadcast for these tests)
        self.mock_hub = mocker.MagicMock()
        # Configure IAC module with registry + hub
        configure_iac(self.app_state_reg, self.mock_hub)

        # Sender deps: mock session (DB access in send_message is bypassed via
        # mocked get_all_agents; session.bind still needed for engine extraction)
        self.sender_deps = AgentDeps(
            session=_make_mock_session(),
            agent_record=self.sender_record,
        )
        self.sender_app_state = AgentAppState()
        self.sender_agent = _SenderTestAgent()

        # Recipient deps: mock session + real record (refreshed to avoid lazy load)
        self.recipient_deps = AgentDeps(
            session=_make_mock_session(),
            agent_record=self.recipient_record,
        )
        # B runs two loop iterations: iter1 ends at ToolResultEvent (compaction fires),
        # iter2 runs steps 2–4 (two more tool calls + completion).  With the fix, all 9
        # expected messages are persisted correctly.  Without the fix, iter2's stale
        # _RunMessages list triggers re-persist of A's history via the ToolResultEvent
        # if-branch, producing 15 messages instead of 9.
        recipient_test_agent = FunctionModelTestAgent()
        # THREE_TOOL_CALL_STEPS: compaction fires after step 1 (iter1), then iter2 also has
        # tool calls. Critical: without the fix, iter2's ToolResultEvent fires with stale
        # L0 last_part=ToolCallPart, triggering re-persist of stale history via the if-branch.
        recipient_test_agent.set_steps(FunctionModelTestAgent.THREE_TOOL_CALL_STEPS)

        # Mock get_all_agents: return real refreshed records (no lazy load issues)
        mocker.patch(
            "prototype.iac.send_message.get_all_agents",
            new_callable=mocker.AsyncMock,
            return_value=[sender_record, recipient_record],
        )

        # Mock AgentFactory → yields (recipient's agent, recipient's deps)
        mock_cm = mocker.AsyncMock()
        mock_cm.__aenter__.return_value = (recipient_test_agent.agent, self.recipient_deps)
        mock_cm.__aexit__.return_value = None
        mock_factory = mocker.MagicMock()
        mock_factory.build_agent_and_deps.return_value = mock_cm
        mocker.patch("agent.factory.AgentFactory", return_value=mock_factory)

        # Mock get_session in _deliver_message (engine from session.bind isn't real in tests)
        mock_sess_cm = mocker.AsyncMock()
        mock_sess_cm.__aenter__.return_value = mocker.MagicMock()
        mock_sess_cm.__aexit__.return_value = None
        mocker.patch("db.connection.get_session", return_value=mock_sess_cm)

        # Use real is_compaction_needed so token values drive compaction, not call ordering.
        # _real_is_compaction_needed is captured at module import time, before _BaseRouteTest
        # patches agent.runner.is_compaction_needed, so it still points to the real function.
        self.mock_needs_compact.side_effect = _real_is_compaction_needed

        # Return a large token count only for B's first persist call; that value propagates
        # to last_total_tokens_value → real is_compaction_needed returns True → compaction fires.
        # All other calls (A's persists and B's iter2 persists) return None → no compaction.
        b_first_persist_done = [False]

        async def _persist_side_effect(deps, messages, toolsets):
            if deps._agent_record is self.recipient_record and not b_first_persist_done[0]:
                b_first_persist_done[0] = True
                return 999_999  # > soft_compaction_limit (10 000) → compaction fires
            return None

        self.mock_persist_messages.side_effect = _persist_side_effect

        # Simulate realistic message history: B has 5 pairs on iter1 (pre-compaction),
        # then 1 pair on iter2 (post-compaction trim). Without the fix, the stale
        # capture list still reflects the longer iter1 history, so the lower
        # new_message_idx from the short iter2 history causes old messages to be
        # re-persisted. Agent A gets its own distinct history — if the contextvar
        # bug causes A's messages to bleed into B's persist calls, the assertion fails.
        _A_HISTORY = "__a_history__"
        _B_ITER1 = "__b_iter1__"
        _B_ITER2 = "__b_iter2__"
        a_history = make_alternating_messages(4, "a's history should not bleed into b")
        b_history_long = make_alternating_messages(10, "b's history should not be persist")
        b_history_short = make_alternating_messages(2, "b's history should not be persist")
        b_load_call = [0]

        async def _load_side_effect(session, agent_id, start_seq_id=0, end_seq_id=None):
            if agent_id == sender_id:
                return _A_HISTORY
            if agent_id == recipient_id:
                b_load_call[0] += 1
                return _B_ITER1 if b_load_call[0] == 1 else _B_ITER2
            return []

        def _deserialize_side_effect(raw):
            if raw == _A_HISTORY:
                return a_history
            if raw == _B_ITER1:
                return b_history_long
            if raw == _B_ITER2:
                return b_history_short
            return []

        self.mock_load_messages.side_effect = _load_side_effect
        self.mock_deserialize_msgs.side_effect = _deserialize_side_effect

    async def test_recipient_messages_persisted_after_compaction(self):
        """
        A calls send_message → B spawned in background task.
        B runs 2 loop iterations using THREE_TOOL_CALL_STEPS (tool, tool, tool, done):
          iter1: step 1 (tool call) → compaction fires after ToolResultEvent
          iter2: steps 2–4 (two more tool calls + completion text)

        With the contextvar fix (context=contextvars.Context()):
          B's capture list is isolated per-iteration → all 9 messages persisted correctly.
        Without the fix (surgical _RunMessages only):
          iter2 reuses the stale iter1 capture list which ends in ToolCallPart;
          the ToolResultEvent if-branch fires with stale history and re-persists it → 15 msgs.

        Expected persisted messages for B (9 total):
          iter1  ① ModelRequest  [UserPromptPart(inter-agent msg)]        ← PartStartEvent elif
          iter1  ② ModelResponse [DUMMY_TOOL_CALL_PART]                   ┐ ToolResultEvent
          iter1  ③ ModelRequest  [DUMMY_TOOL_RETURN_PART]                 ┘  step 1 atomic persist
          iter2  ④ ModelRequest  [UserPromptPart(COMPACTION_RESUME_NOTICE)] ← PartStartEvent elif
          iter2  ⑤ ModelResponse [DUMMY_TOOL_CALL_PART]                   ┐ ToolResultEvent
          iter2  ⑥ ModelRequest  [DUMMY_TOOL_RETURN_PART]                 ┘  step 2 atomic persist
          iter2  ⑦ ModelResponse [DUMMY_TOOL_CALL_PART]                   ┐ ToolResultEvent
          iter2  ⑧ ModelRequest  [DUMMY_TOOL_RETURN_PART]                 ┘  step 3 atomic persist
          iter2  ⑨ ModelResponse [TextPart(COMPLETION_TEXT)]              ← AgentRunResultEvent elif
        """
        a_events = [event async for event in run_stateful_agent(
            self.sender_agent.agent,
            self.sender_deps,
            self.sender_app_state,
            "send a message",
        )]
        assert isinstance(a_events[-1], AgentRunResultEvent), "Sender should complete normally"

        # Await B's background task
        from prototype.iac.send_message import background_tasks
        await asyncio.gather(*list(background_tasks))

        # Flatten all messages persisted by B across all persist_messages calls.
        # Use _agent_record identity (not .id) to avoid ORM lazy-load outside async context.
        b_persisted = [
            msg
            for call in self.mock_persist_messages.call_args_list
            if call.kwargs["deps"]._agent_record is self.recipient_record
            for msg in call.kwargs["messages"]
        ]

        expected = [
            # iter1: initial inter-agent message + step 1 tool call/return
            ModelRequest(parts=[UserPromptPart(content=_format_inter_agent_message(
                "sender-agent", "hello from sender"
            ))]),
            ModelResponse(parts=[FunctionModelTestAgent.DUMMY_TOOL_CALL_PART]),
            ModelRequest(parts=[FunctionModelTestAgent.DUMMY_TOOL_RETURN_PART]),
            # iter2: compaction resume notice + steps 2 and 3 tool call/return pairs
            ModelRequest(parts=[UserPromptPart(content=COMPACTION_RESUME_NOTICE)]),
            ModelResponse(parts=[FunctionModelTestAgent.DUMMY_TOOL_CALL_PART]),
            ModelRequest(parts=[FunctionModelTestAgent.DUMMY_TOOL_RETURN_PART]),
            ModelResponse(parts=[FunctionModelTestAgent.DUMMY_TOOL_CALL_PART]),
            ModelRequest(parts=[FunctionModelTestAgent.DUMMY_TOOL_RETURN_PART]),
            # iter2: step 4 completion
            ModelResponse(parts=[TextPart(content=FunctionModelTestAgent.COMPLETION_TEXT)]),
        ]
        assert_ModelMessage_list_eq(b_persisted, expected)

        # A's history must not contain any of B's messages — bleed-through would
        # add extra messages and/or corrupt A's expected content.
        a_persisted = [
            msg
            for call in self.mock_persist_messages.call_args_list
            if call.kwargs["deps"]._agent_record is self.sender_record
            for msg in call.kwargs["messages"]
        ]
        a_expected = [
            ModelRequest(parts=[UserPromptPart(content="send a message")]),
            ModelResponse(parts=[ToolCallPart(
                tool_name="send_message",
                args=_SenderTestAgent.SEND_MSG_ARGS,
                tool_call_id="sm-tc-1",
            )]),
            ModelRequest(parts=[ToolReturnPart(
                tool_name="send_message",
                content="Message delivered to 'recipient-agent'.",
                tool_call_id="sm-tc-1",
            )]),
            ModelResponse(parts=[TextPart(content=FunctionModelTestAgent.COMPLETION_TEXT)]),
        ]
        assert_ModelMessage_list_eq(a_persisted, a_expected)


class TestFormatInterAgentMessage:
    """_format_inter_agent_message: pure helper for origin marker formatting."""

    def test_format_structure(self):
        """Message has header with sender, newline, then content."""
        result = _format_inter_agent_message("alice", "hello world")
        assert result.startswith("[INTER AGENT MESSAGE. If you want to reply, use the 'send_message' tool. From: alice]")
        header, _, body = result.partition("\n")
        assert body == "hello world"
