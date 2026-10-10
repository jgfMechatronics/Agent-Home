"""Wall-clock timestamp injection for agent context.

Agents have no clock — only dates arrive in metadata, which leads to fabricated
times in memory entries (e.g. recording an event as "~4am" from vibes alone).
Phase 1a stamps every inbound user message with the current wall-clock time so
agents always have a temporal anchor. See
"E-LLM Docs/Agent Home Docs/AgentTimestamps-Design.md" for the phased design.
"""

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# Must be an IANA zone name (e.g. "America/New_York"), NOT a fixed offset —
# fixed offsets (TZ=EST) are silently wrong half the year on DST transitions.
TZ_ENV_VAR = "AGENT_HOME_TZ"

_STAMP_FORMAT = "[%Y-%m-%d %H:%M:%S %Z]"

_tz_cache: ZoneInfo | None = None


def _resolve_tz() -> ZoneInfo:
    """Resolve the display timezone from AGENT_HOME_TZ, defaulting to UTC.

    Resolution is cached per-process; the env var is read once. An invalid zone
    logs an error and falls back to UTC — the stamp is self-describing (zone
    abbreviation included), so this never produces silently-wrong times.
    """
    global _tz_cache
    if _tz_cache is None:
        name = os.environ.get(TZ_ENV_VAR, "UTC")
        try:
            _tz_cache = ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.error(
                "Invalid %s=%r — falling back to UTC. Use an IANA zone name "
                "(e.g. 'America/New_York'), not a fixed offset like 'EST'.",
                TZ_ENV_VAR,
                name,
            )
            _tz_cache = ZoneInfo("UTC")
        if name == "UTC" and TZ_ENV_VAR not in os.environ:
            logger.info("%s not set — timestamps default to UTC", TZ_ENV_VAR)
    return _tz_cache


def _now() -> datetime:
    """Current time in the display timezone. Seam for deterministic tests."""
    return datetime.now(_resolve_tz())


def stamp_user_message(message: str) -> str:
    """Prefix a user message with the current wall-clock timestamp."""
    if not message:
        return message
    return f"{_now():{_STAMP_FORMAT}} {message}"
