"""Tests for agent/timestamping.py — wall-clock stamp injection."""

import logging
import re
from zoneinfo import ZoneInfo

import pytest

import agent.timestamping
from agent.timestamping import TZ_ENV_VAR, _resolve_tz, stamp_user_message

# Shape of the stamp prefix: [YYYY-MM-DD HH:MM:SS ZONE]
STAMP_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [A-Z]{2,5}\] ")


@pytest.fixture(autouse=True)
def _fresh_tz_cache(monkeypatch):
    """TZ resolution is cached per-process; tests each start with a clear cache."""
    monkeypatch.setattr(agent.timestamping, "_tz_cache", None)


class TestStampUserMessage:
    def test_stamp_shape(self):
        """Stamped message starts with a bracketed timestamp prefix."""
        stamped = stamp_user_message("hello")
        assert STAMP_RE.match(stamped), f"Bad stamp shape: {stamped!r}"
        assert stamped.endswith("hello")

    def test_stamp_value_frozen_clock(self):
        """With the suite-wide frozen clock (conftest), the stamp is fully deterministic."""
        assert stamp_user_message("hello") == "[2026-10-10 12:00:00 UTC] hello"

    def test_empty_message_passthrough(self):
        """Empty prompts (e.g. tool-continuation triggers) are not stamped."""
        assert stamp_user_message("") == ""


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
        the design just requires IANA names in practice.
        """
        monkeypatch.setenv(TZ_ENV_VAR, "Not/AZone")
        with caplog.at_level(logging.ERROR):
            resolved = _resolve_tz()
        assert resolved == ZoneInfo("UTC")
        assert "AGENT_HOME_TZ" in caplog.text and "Not/AZone" in caplog.text

    def test_resolution_is_cached(self, monkeypatch):
        """Second call does not re-read the env var (cheap after first resolution)."""
        monkeypatch.setenv(TZ_ENV_VAR, "America/New_York")
        first = _resolve_tz()
        monkeypatch.setenv(TZ_ENV_VAR, "UTC")  # would change the answer if re-read
        assert _resolve_tz() is first
