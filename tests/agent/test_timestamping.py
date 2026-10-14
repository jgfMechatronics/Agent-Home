"""Tests for agent/timestamping.py — wall-clock stamp injection."""

import logging
from zoneinfo import ZoneInfo

import pytest
from pydantic_ai.messages import ToolReturn

import agent.timestamping
from agent.timestamping import TZ_ENV_VAR, ToolReturnTimestamper, _resolve_tz, add_timestamp_prefix

from conftest import FROZEN_NOW

# Stamp assertions are deterministic — this module opts into the frozen clock.
pytestmark = pytest.mark.usefixtures("frozen_stamp_clock")


# Frozen timestamp from conftest.frozen_stamp_clock (2026-10-10 12:00:00 UTC)
FROZEN_STAMP = "⏰ [Oct 10, 2026 12:00:00 UTC]"


class TestAddTimestampPrefix:
    def test_stamp_value_frozen_clock(self):
        """With the suite-wide frozen clock (conftest), the stamp is fully deterministic."""
        assert add_timestamp_prefix("hello") == f"{FROZEN_STAMP}\nhello"

    def test_empty_string_passthrough(self):
        """Empty strings (empty user prompts, empty tool returns) are not stamped."""
        assert add_timestamp_prefix("") == ""


class TestToolReturnTimestamper:
    """Unit tests for the capability hook's gating. Real-hook-path behavior
    (production capabilities list, spill ordering) lives in test_tool_output_limits.py."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Create a fresh ToolReturnTimestamper for each test."""
        self.cap = ToolReturnTimestamper()

    async def run_hook(self, result):
        """Helper to execute the hook with minimal boilerplate."""
        return await self.cap.after_tool_execute(None, call=None, tool_def=None, args=None, result=result)

    @pytest.mark.parametrize(
        "result",
        [None, 42, 3.14, {"key": "value"}, ["a", "b"], {"nested": [1, 2]}],
        ids=["none", "int", "float", "dict", "list", "nested-dict"],
    )
    async def test_non_string_results_pass_through_untouched(self, result):
        """Non-string results (structured data, multimodal parts) are never
        stringified or copied — the provider serialization layer owns them."""
        out = await self.run_hook(result)
        assert out is result

    async def test_string_result_stamped(self):
        """String results get the timestamp header (frozen clock → deterministic)."""
        out = await self.run_hook("tool output")
        assert out == f"{FROZEN_STAMP}\ntool output"

    async def test_empty_string_result_not_stamped(self):
        """Empty tool returns stay empty rather than becoming a bare stamp."""
        out = await self.run_hook("")
        assert out == ""

    async def test_tool_return_envelope_string_stamped(self):
        """Spilled results arrive wrapped in ToolReturn (handle metadata envelope);
        the stamp goes on return_value, the envelope and metadata survive intact."""
        wrapped = ToolReturn(return_value="stand-in text", metadata={"handle": "run/call.0"})
        out = await self.run_hook(wrapped)
        assert isinstance(out, ToolReturn)
        assert out.return_value == f"{FROZEN_STAMP}\nstand-in text"
        assert out.metadata == wrapped.metadata, "envelope metadata must survive unstamped"

    async def test_tool_return_envelope_non_string_passthrough(self):
        """A ToolReturn with a non-string return_value passes through untouched."""
        wrapped = ToolReturn(return_value={"spilled": "dict"}, metadata=None)
        out = await self.run_hook(wrapped)
        assert out is wrapped


class TestResolveTz:
    def test_defaults_to_utc_when_unset(self, monkeypatch):
        monkeypatch.delenv(TZ_ENV_VAR, raising=False)
        assert _resolve_tz() == ZoneInfo("UTC")

    def test_env_var_iana_name(self, monkeypatch):
        monkeypatch.setenv(TZ_ENV_VAR, "America/New_York")
        assert _resolve_tz() == ZoneInfo("America/New_York")

    def test_invalid_zone_falls_back_to_utc_with_error_log(self, monkeypatch, caplog):
        """Genuinely unparseable names are rejected loudly and fall back to UTC.

        Note: legacy fixed-offset zones like 'EST' are NOT invalid to zoneinfo —
        they resolve (no DST, wrong half the year) and are the documented footgun;
        IANA names are recommended, not enforced — 'EST' is the operator's mistake.
        """
        monkeypatch.setenv(TZ_ENV_VAR, "Not/AZone")
        with caplog.at_level(logging.ERROR):
            resolved = _resolve_tz()
        assert resolved == ZoneInfo("UTC")
        assert "AGENT_HOME_TZ" in caplog.text and "Not/AZone" in caplog.text

    def test_stamp_zone_comes_from_tz_info(self, monkeypatch):
        """add_timestamp_prefix reads _tz_info at call time — patching the
        zone seam changes the stamp's zone. Frozen instant (FROZEN_NOW),
        expressed in the patched zone via astimezone math."""
        tokyo = ZoneInfo("Asia/Tokyo")
        monkeypatch.setattr(agent.timestamping, "_tz_info", tokyo)
        expected_ts = FROZEN_NOW.astimezone(tokyo).strftime("%b %d, %Y %H:%M:%S %Z")
        assert add_timestamp_prefix("hello") == f"⏰ [{expected_ts}]\nhello"
