# Section 1: DB Layer — Test Design Notes
*Updated after Opus review*

---

## Resolved design decisions

**Sync vs Async:** Async throughout (Option B). Tests with sync fixtures don't reflect actual runtime behavior.
Fixture pattern (Opus suggestion):
```python
@pytest_asyncio.fixture
async def session(engine):
    async with engine.connect() as conn:
        async with conn.begin() as trans:
            async_session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint")
            yield async_session
            await trans.rollback()
```

**Test structure:** Mirrored — `tests/db/test_models.py`, `tests/db/test_session.py`

**DRY for "stores all fields":** Helper function, not parametrization.
```python
async def assert_round_trips(session, record, expected_fields: dict):
    session.add(record)
    await session.flush()
    queried = await session.get(type(record), record.id)
    for field, expected in expected_fields.items():
        assert getattr(queried, field) == expected, f"{field} mismatch"
```
DRY on the pattern, explicit on the fields. Readable failure messages.

**session.py tests:** Separate file (`tests/db/test_session.py`). Some will be smoke tests — that's fine.

**Position defaulting scope:** Service-level (requires querying existing blocks) → Section 2 territory, not Section 1.

**Message timestamp:** Section 5 territory — model test just verifies the column stores what you put in.

---

## OPEN QUESTIONS — need James clarification before writing

### Q1: Column name — `AgentConfig` vs `settings`?
Schema table (line 380) calls it `AgentConfig`. Plan behavior (line 539) says "`tool_names` in `settings`". Are these the same column? What's the actual column name in the DB, and the Python attribute name on AgentRecord?

### Q2: Is `model` a top-level column or inside AgentConfig JSON?
Schema (line 379) shows `model` as its own TEXT column. But plan behavior (line 538) says "AgentConfig JSON contains required keys: `tool_names`, `soft_limit`, `model`". Code snippets use `agent_config.model_name`. Contradictions:
- Is `model` redundantly stored (top-level AND in JSON)?
- What's the Python attribute name — `model` or `model_name`?

---

## Test file plan (once naming is resolved)

```
tests/
├── __init__.py
├── conftest.py          # engine fixture (session-scoped), session fixture (function-scoped + rollback)
└── db/
    ├── __init__.py
    ├── test_models.py   # AgentRecord, MemoryBlockRecord, MessageRecord
    └── test_session.py  # engine creation, session factory
```

### test_models.py coverage

**AgentRecord:**
- Stores and retrieves all fields
- AgentConfig/settings JSON contains required keys (tool_names, soft_limit, model)
- tool_names is a JSON array of strings
- context_window_start defaults to NULL on creation
- compiled_at is NULL on creation

**MemoryBlockRecord:**
- Stores and retrieves all fields (label, description, content, char_limit, position, agent_id)
- FK to AgentRecord (agent_id)

**MessageRecord:**
- Stores and retrieves all fields (agent_id, type, content, input_tokens)
- FK to AgentRecord
- input_tokens is nullable

### test_session.py coverage

(Smoke tests mostly — connection lifecycle, basic round-trip)

---

## Dependencies needed in pyproject.toml

- sqlalchemy[asyncio]
- aiosqlite (SQLite async driver)
- pytest
- pytest-asyncio
- fastapi, uvicorn (for API layer later)
- pydantic-ai (for agent layer later)
