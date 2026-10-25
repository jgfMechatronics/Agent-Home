"""
AgentDeps and build_deps — agent dependency bundle and lifecycle management.

Separated from types.py because this module contains active infrastructure
(session lifecycle, lock management, factory guard) rather than passive domain types.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import object_session

from agent.types import AgentAppState, AgentConfig, AgentLockedError, AgentNotFoundError
from db.connection import ReadOnlySession
from db.models import AgentRecord

LOCK_TIMEOUT_SECONDS: int = 60

_deps_token = object()
"""Module-level sentinel proving AgentDeps was constructed via build_deps.
Not importable by convention — only this module should use it."""


class AgentDeps:
    """
    Dependency bundle for agent operations. Outside tests, should only be constructed by build_deps.
    This enforces the connection between AgentDeps and the lock that build_deps holds,
    making it so deps proves caller holds the per-agent lock.

    Construction is guarded: callers must either pass the private _deps_token (from
    build_deps) or explicitly pass bypass_factory_guard=True (for bootstrap cases like
    create_agent_record, and tests).

    _agent_record is private by convention — access via properties.

    The record is detached from any session between write operations. Property reads
    work on the detached record (simple columns, no lazy loading needed). Property
    setters raise RuntimeError unless called inside a write_scope block.

    Session lifecycle is fully internal:
    - write_scope(): re-attaches record, yields session for DB work, commits + refreshes
      + detaches on exit. All record mutations must happen inside a write_scope.
    - read_scope(): yields a read-only session for DB reads (blocks, messages, etc.).
      Record properties do NOT require a read_scope — the detached record is always
      readable (the agent lock guarantees no concurrent writes).

    TODO: Add a test that programmatically iterates all property setters on this class
    and verifies each one calls _require_write_scope (not a hardcoded list).
    """

    def __init__(
        self,
        rw_engine: AsyncEngine,
        ro_engine: AsyncEngine,
        agent_record: "AgentRecord",
        *,
        _token: object | None = None,
        bypass_factory_guard: bool = False,
    ) -> None:
        if not bypass_factory_guard and _token is not _deps_token:
            raise RuntimeError(
                "AgentDeps must be constructed via build_deps. "
                "If you have a legitimate reason to construct directly "
                "(bootstrap, tests), pass bypass_factory_guard=True."
            )
        self._rw_engine = rw_engine
        self._ro_engine = ro_engine
        self._agent_record = agent_record

    def _require_write_scope(self) -> None:
        """Raise if the agent record is not attached to a session (i.e., not in a write_scope)."""
        if object_session(self._agent_record) is None:
            raise RuntimeError(
                "Cannot mutate agent record outside a write scope. "
                "Use 'async with deps.write_scope():' to open a write session."
            )

    @asynccontextmanager
    async def write_scope(self) -> AsyncIterator[AsyncSession]:
        """Open a write session — re-attach record, yield session for DB work, commit on exit.

        Inside the scope, property setters are enabled (record is session-attached).
        On exit: commits, refreshes the record (absorbing post-commit attribute expiry),
        and detaches it again so it remains readable outside the scope.
        On exception: rolls back and reloads the record from the DB (discarding
        uncommitted mutations) so it stays readable; falls back to expiring it
        (reads fail loudly until the next write_scope) if the reload fails.

        Usage:
            async with deps.write_scope() as session:
                deps.compiled_system_prompt = new_value
                await session.execute(...)  # other DB work if needed
        """
        async with AsyncSession(self._rw_engine) as session:
            session.add(self._agent_record)
            try:
                yield session
                await session.commit()
                await session.refresh(self._agent_record)
            except BaseException:
                # The record may carry uncommitted mutations — discard them by reloading
                # from the DB, so the detached record stays readable with known-good state.
                # This matters for routine validation errors (e.g. ContentExceedsLimitError
                # on the ModelRetry path): they are normal operation and must not poison
                # subsequent property reads with DetachedInstanceError.
                # If the reload itself fails (e.g. dead connection), expire instead so
                # reads fail loudly rather than serving uncertain data; the next
                # write_scope re-attaches and self-heals with fresh DB values.
                try:
                    await session.rollback()
                    await session.refresh(self._agent_record)
                except Exception:
                    session.expire(self._agent_record)
                raise
            finally:
                session.expunge(self._agent_record)

    @asynccontextmanager
    async def read_scope(self) -> AsyncIterator[ReadOnlySession]:
        """Yield a read-only session for DB reads (blocks, messages, etc.).

        Not needed for record property reads — the detached record is always readable.
        """
        async with AsyncSession(self._ro_engine) as session:
            yield ReadOnlySession(session)

    @property
    def agent_id(self) -> str:
        return self._agent_record.id

    @property
    def name(self) -> str:
        return self._agent_record.name

    @property
    def config(self) -> AgentConfig:
        return self._agent_record.agent_config

    @config.setter
    def config(self, value: AgentConfig) -> None:
        self._require_write_scope()
        self._agent_record.agent_config = value

    @property
    def system_instructions(self) -> str:
        return self._agent_record.system_instructions or ""

    @system_instructions.setter
    def system_instructions(self, value: str) -> None:
        self._require_write_scope()
        self._agent_record.system_instructions = value

    @property
    def compiled_system_prompt(self) -> str:
        return self._agent_record.compiled_system_prompt

    @compiled_system_prompt.setter
    def compiled_system_prompt(self, value: str) -> None:
        self._require_write_scope()
        self._agent_record.compiled_system_prompt = value

    @property
    def sys_prompt_compiled_at(self) -> datetime | None:
        return self._agent_record.sys_prompt_compiled_at

    @sys_prompt_compiled_at.setter
    def sys_prompt_compiled_at(self, value: datetime) -> None:
        # Setters take non-None values intentionally — callers always provide a concrete timestamp
        self._require_write_scope()
        self._agent_record.sys_prompt_compiled_at = value

    @property
    def context_window_start(self) -> int:
        return self._agent_record.context_window_start

    @context_window_start.setter
    def context_window_start(self, value: int) -> None:
        self._require_write_scope()
        self._agent_record.context_window_start = value

    @property
    def compaction_warning_fired(self) -> bool:
        return self._agent_record.compaction_warning_fired

    @compaction_warning_fired.setter
    def compaction_warning_fired(self, value: bool) -> None:
        self._require_write_scope()
        self._agent_record.compaction_warning_fired = value


@asynccontextmanager
async def build_deps(
    agent_id: str,
    agent_app_state: AgentAppState,
    rw_engine: AsyncEngine,
    ro_engine: AsyncEngine,
    timeout: float = LOCK_TIMEOUT_SECONDS,
) -> AsyncIterator[AgentDeps]:
    """Async context manager that acquires the agent lock and yields AgentDeps.

    Lock-then-fetch: acquires lock BEFORE fetching from DB to prevent stale state.
    Releases lock on exit (normal or exception) via try/finally.
    """
    try:
        await asyncio.wait_for(agent_app_state.lock.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        raise AgentLockedError(f"Agent {agent_id!r} did not become available within {timeout}s")

    try:
        async with AsyncSession(ro_engine) as session:
            agent_record = await session.get(AgentRecord, agent_id)
            if agent_record is None:
                raise AgentNotFoundError(f"Agent {agent_id!r} not found")
            session.expunge(agent_record)

        deps = AgentDeps(rw_engine, ro_engine, agent_record, _token=_deps_token)
        yield deps
    finally:
        # Clear cancel_requested on exit to prevent a stale signal (e.g. a cancel that arrived
        # during teardown) from leaking into the next run. Tradeoff: if a second client is waiting
        # on the lock and sent a cancel for their queued run, that cancel is silently lost here.
        # Acceptable for now — our "queue" is just lock waiters with no ordering guarantees and no
        # run attribution. When we add a real queue, cancels should be associated with a specific
        # run so this ambiguity goes away.
        agent_app_state.cancel_requested.clear()
        agent_app_state.lock.release()
