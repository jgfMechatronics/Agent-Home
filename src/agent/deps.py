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
from agent.crud import get_agent_record
from db.connection import ReadOnlyEngine, ReadOnlySession, get_session
from db.models import AgentRecord

LOCK_TIMEOUT_SECONDS: int = 60

_deps_token = object()
"""Module-level sentinel proving AgentDeps was constructed via build_deps.
Not importable by convention — only this module should use it."""


class AgentDeps:
    """
    Dependency bundle for stateful agents. Instances built by build_deps hold the agent's
    AgentAppState.lock for their lifetime, so at most one such deps exists per agent at a time.
    This is an in-process lock, not a DB write lock: code holding its own session can still
    write the agent's rows.

    Construction is guarded: callers must either pass the private _deps_token (from
    build_deps) or explicitly pass bypass_factory_guard=True (for bootstrap cases like
    the create_agent route, and tests).

    _agent_record intended to be accessed via properties.

    The record is detached from any session between write operations. Property reads
    work on the detached record (simple columns, no lazy loading needed). Property
    setters raise RuntimeError unless called inside a write_scope block.

    Session lifecycle is fully internal:
    - write_scope(): re-attaches record, yields session for DB work, commits + refreshes
      + detaches on exit. All record mutations must happen inside a write_scope.
      Mutations to _agent_record must be done through setters inside write_scope.
    - read_scope(): yields a read-only session for DB reads (blocks, messages, etc.).
      Record properties do NOT require a read_scope — the detached record is always
      readable (the agent lock guarantees no concurrent writes).

    TODO: Add a test that programmatically iterates all property setters on this class
    and verifies each one calls _require_write_scope (not a hardcoded list).
    (see test_agentdeps_setter_requires_write_scope, needs update)
    """

    def __init__(
        self,
        rw_engine: AsyncEngine,
        ro_engine: ReadOnlyEngine,
        agent_record: AgentRecord,
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
                await session.execute(...)  # other DB work if needed.
                                            # Can discard "session" if only using deps setters
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
        async with get_session(self._ro_engine) as session:
            yield session

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
    ro_engine: ReadOnlyEngine,
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
        async with get_session(ro_engine) as session:
            agent_record = await get_agent_record(session, agent_id)
        if agent_record is None:
            raise AgentNotFoundError(f"Agent {agent_id!r} not found")

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


"""
TODO (follow-up PR, before removing ParallelToolCallTrimmer): AgentDeps is NOT task-safe.

PROBLEM
pydantic-ai runs multiple tool calls from one ModelResponse concurrently (separate asyncio
tasks), and all of them share one AgentDeps. Today ParallelToolCallTrimmer guarantees one tool
call per response, so this is latent. Once parallel calls are allowed, the current design breaks:

  (a) Crash on concurrent write_scopes: write_scope re-attaches the single shared _agent_record
      via session.add(). A second concurrent write_scope raises "Object is already attached to
      session X".
  (b) Expired-attribute window: write sessions use expire_on_commit=True. Between
      `await session.commit()` and the end of `await session.refresh(...)` (and likewise after
      rollback on the exception path), the shared record is attached with expired attributes.
      Another task reading any property (even deps.agent_id) in that window triggers a lazy load
      outside the greenlet -> MissingGreenlet.
  (c) Uncommitted visibility: properties read the shared record, so other tasks see in-scope
      mutations before commit, including values that are later rolled back.
  (d) _require_write_scope checks the wrong thing: it asks "is the record attached?", not "does
      the CURRENT TASK own the write scope?". A setter called from task B while task A holds a
      write_scope passes the guard, and A's commit/rollback silently absorbs B's mutation.
  (e) Nesting: a nested write_scope in the same task is currently only caught incidentally via
      (a). Any locking fix must keep it loud rather than deadlocking.

Related, but NOT fixed by anything below: read-modify-write lost updates in memory_replace /
memory_insert (see TODO in agent/tools.py).

PROPOSED DESIGN: frozen snapshot for reads, transaction handle for writes, per-deps write lock.
Drop the property setters entirely (only ~8 write sites in src, all already inside write_scope).
Keep the getters (~47 read sites), reading from an immutable snapshot.

    class AgentSnapshot(BaseModel):
        model_config = ConfigDict(frozen=True, from_attributes=True)
        id: str
        name: str
        agent_config: AgentConfig
        system_instructions: str
        compiled_system_prompt: str
        sys_prompt_compiled_at: datetime | None
        context_window_start: int
        compaction_warning_fired: bool

    @asynccontextmanager
    async def write_scope(self) -> AsyncIterator[WriteTxn]:
        if self._write_owner is asyncio.current_task():
            raise RuntimeError("nested write_scope")
        async with asyncio.timeout(WRITE_LOCK_TIMEOUT):   # deadlock -> loud error, not a hang
            await self._write_lock.acquire()
        try:
            self._write_owner = asyncio.current_task()
            async with AsyncSession(self._rw_engine, expire_on_commit=False) as session:
                record = await session.get(AgentRecord, self.agent_id)  # fresh, one PK SELECT
                txn = WriteTxn(session, record)
                yield txn
                await session.commit()
                self._snapshot = AgentSnapshot.model_validate(record)  # swap only after commit
                txn.close()  # txn.record raises after close, so the ORM object can't escape
        finally:
            self._write_owner = None
            self._write_lock.release()

Call sites mutate the scope's record directly:

    async with deps.write_scope() as txn:
        txn.record.context_window_start = new_start
        txn.record.compaction_warning_fired = False
        await compile_system_prompt(txn)  # reads txn.record -> sees in-scope writes

How this addresses each problem:
  (a) Each scope gets its own fresh record from session.get(); the snapshot is never attached.
  (b) The snapshot is plain frozen data and never expires. expire_on_commit=False avoids the
      need for refresh (onupdate=utcnow is Python-side, so values are already populated at flush).
  (c) The snapshot only changes after a successful commit.
  (d) There are no setters to guard; the only mutable record is the one the scope hands out.
  (e) The owner-task check raises on nesting; asyncio.Lock alone would deadlock silently.
  Exception path: drop the scope record, leave the snapshot untouched. The current
  rollback/refresh/expire-fallback logic disappears, along with the
  "commit succeeded but refresh raised" ambiguity.

Design notes / decisions:
  - Why a lock at all, if each scope gets its own record: without it, two concurrent scopes each
    UPDATE only their dirty columns (the DB ends up correct), but the last snapshot swap wins and
    drops the other scope's change. The lock also orders the snapshot swaps.
  - Why asyncio.Lock and not SQLite's write lock (e.g. a BEGIN IMMEDIATE event recipe): SQLite's
    lock is DB-wide (it would serialize all agents' write scopes for their whole duration, not
    just the commit), waiters retry inside busy_timeout with no FIFO ordering, the recipe changes
    begin behavior engine-wide, and it doesn't cover the post-commit snapshot swap. The per-deps
    (= per-agent) asyncio lock is exactly the granularity we want.
  - Rejected: mutexing all deps access. Any lock scope spanning an await on pydantic-ai work
    (the runner iterating the run while tools need deps) deadlocks; making it fine-grained enough
    to avoid that makes every access an await. Reads on immutable data need no lock.
  - Deadlock rule: a write_scope body must only do DB work. Never await another task, or slow
    I/O, while holding it. (persist_messages currently calls _extract_tool_definitions -- an MCP
    list_tools() round trip -- inside its scope; move it out as part of the persist-extraction
    refactor.) The acquire timeout turns any violation into a loud error.
  - Heads-up: with pysqlite's default (legacy) transaction control, BEGIN is only emitted before
    the first INSERT/UPDATE/DELETE. Reads inside a write_scope that precede the first write run
    outside the DB transaction. Fine given the agent lock + this write lock, but don't rely on
    write_scope for DB-level read-modify-write isolation.
  - Removes the last long-lived detached ORM object, so block_crud's "ORM objects never outlive
    their session" rule loses its exception. Same direction as the MemoryBlockRecord -> DTO
    follow-up and DTO-based test fixtures.
  - To verify when implementing: AgentConfigType (TypeDecorator) round-trips correctly through
    AgentSnapshot.model_validate(record).

Migration cost: the 8 setter write sites (compaction.py, crud.py, compaction_warner.py,
system_prompt_compilation.py), compile_system_prompt's signature (deps, session) -> (txn), deps.py
itself (net simpler), plus tests that set fields through deps inside write_scope (test_compaction,
test_crud, test_deps, test_system_prompt_compilation). test_agentdeps_setter_requires_write_scope
becomes moot.
"""
