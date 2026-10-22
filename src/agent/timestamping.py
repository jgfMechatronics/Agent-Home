"""Wall-clock timestamp injection for agent context.

Agents have no clock — only dates arrive in metadata, which leads to fabricated
times in memory entries (e.g. recording an event as "~4am" from vibes alone).
Phase 1a stamps every inbound user message; phase 1b stamps string tool returns
via a capability, so agents have a temporal anchor mid-run (run duration =
first-to-last stamp). See
"E-LLM Docs/Agent Home Docs/AgentTimestamps-Design.md" for the phased design.
"""

import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.tools import ToolDefinition

from agent.deps import AgentDeps

if TYPE_CHECKING:
    from pydantic_ai.tools import ValidatedToolArgs

logger = logging.getLogger(__name__)

# Should be an IANA zone name (e.g. "America/New_York"). Legacy fixed-offset
# zones (TZ=EST) resolve without error but are wrong half the year on DST
# transitions — recommended against, not rejected.
TZ_ENV_VAR = "AGENT_HOME_TZ"


def _resolve_tz() -> ZoneInfo:
    """Resolve the display timezone from AGENT_HOME_TZ, defaulting to UTC.

    Pure: reads the env var at call time and always returns a valid zone — an
    unparseable name logs an error and falls back to UTC (the fallback is
    self-describing, zone abbreviation included, so it never produces
    silently-wrong times). Legacy fixed-offset zones like 'EST' resolve and
    are used as-is — wrong half the year, but the operator's choice.
    Module-level `_tz_info` below captures the resolution once at import;
    the env var must therefore be set before this module is first imported
    (same pattern as api.app's AGENT_HOME_DB_PATH).
    """
    name = os.environ.get(TZ_ENV_VAR, None)
    if name is None:
        logger.info("%s not set — timestamps default to UTC", TZ_ENV_VAR)
        name = "UTC"

    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.error(
            "Invalid %s=%r — falling back to UTC. Prefer an IANA zone name "
            "(e.g. 'America/New_York'); legacy fixed-offset zones like 'EST' "
            "resolve but are wrong half the year.",
            TZ_ENV_VAR,
            name,
        )
        return ZoneInfo("UTC")


_tz_info: ZoneInfo = _resolve_tz()


def add_timestamp_prefix(text: str) -> str:
    """Prefix a non-empty string with the wall-clock timestamp header.

    Shared by both injection sites (user messages, string tool returns) so the
    marker and format mean the same thing everywhere. The ⏰ mark is applied at
    this common level deliberately: user messages (high frequency, unambiguous
    source) train its interpretation, so tool returns (rarer) inherit a learned
    token rather than needing explanation. Month-name dates are unambiguous to
    any reader; 24h time avoids the AM/PM ambiguity surface.
    """
    if not text:
        return text
    return f"⏰ [{datetime.now(_tz_info):%b %d, %Y %H:%M:%S %Z}]\n{text}"


@dataclass
class ToolReturnTimestamper(AbstractCapability[AgentDeps]):
    """Stamps string tool results with the timestamp header; other types pass through.

    Must run AFTER ToolOutputLimits' reduction: capabilities fire after_tool_execute in
    REVERSED list order (last capability sees the result first), so this sits between
    CompactionWarner and ToolOutputLimits — spill runs first, the stamp then lands on
    the stand-in/preview. Spilled results arrive wrapped in a ToolReturn envelope
    (carrying the handle metadata); the stamp goes on its return_value. Non-string
    results (dicts, lists, multimodal content parts) pass through untouched: only
    strings are already-final text, and stringifying anything else would risk breaking
    the provider-layer serialization that owns those types.
    """

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDeps],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: "ValidatedToolArgs",
        result: Any,
    ) -> Any:
        if isinstance(result, ToolReturn):
            if isinstance(result.return_value, str):
                return replace(result, return_value=add_timestamp_prefix(result.return_value))
            return result
        if isinstance(result, str):
            return add_timestamp_prefix(result)
        return result
