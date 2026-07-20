# Gap 4: Context Reconstruction Design

**Created:** July 20, 2026  
**Status:** Design complete, ready for implementation

## Problem

Reconstruct the exact context an LLM saw at any historical point. Enables debugging, analysis, and eventually auto-recall.

## Solution: Content-Addressable Snapshots

### New Tables

```python
class SystemPromptSnapshot(Base):
    """Content-addressable store for compiled system prompts."""
    __tablename__ = "system_prompt_snapshots"
    
    id: Mapped[str] = mapped_column(primary_key=True)  # SHA256 hash of content
    content: Mapped[str]  # The compiled system prompt
    created_at: Mapped[datetime]


class ToolSchemaSnapshot(Base):
    """Content-addressable store for tool schema arrays."""
    __tablename__ = "tool_schema_snapshots"
    
    id: Mapped[str] = mapped_column(primary_key=True)  # SHA256 hash of content
    content: Mapped[str]  # JSON array of tool schemas
    created_at: Mapped[datetime]
```

### MessageRecord Additions

```python
# New fields on MessageRecord:
system_prompt_hash: Mapped[str] = mapped_column(ForeignKey("system_prompt_snapshots.id"))
tool_schema_hash: Mapped[str] = mapped_column(ForeignKey("tool_schema_snapshots.id"))
context_window_start_msg_id: Mapped[str | None]  # UUID of first in-context message
```

## Key Design Decisions

### 1. Content-addressable storage (hash = identity)
- Same content across N messages = stored once, referenced N times
- No agent_id on snapshots — the hash IS the identity
- If two agents have identical prompts, they share the same snapshot row
- Provenance lives in the references (MessageRecord), not the content

### 2. Hash fields are NOT nullable
- Both `system_prompt_hash` and `tool_schema_hash` are required
- Protects against silent failures where we forget to store hashes
- Letta import will compute hashes during import (see below)

### 3. Upsert pattern for concurrent safety
```python
stmt = insert(SystemPromptSnapshot).values(...).on_conflict_do_nothing(index_elements=['id'])
```
- Only ignore PK conflict (duplicate hash from race condition)
- Other failures should still be loud

### 4. Hash consistency
- Always hash UTF-8 encoded bytes
- Whitespace-sensitive — any change creates unique entry
- Identical byte content must always produce identical hash

### 5. No cascade delete
- Snapshot rows should never be deleted
- No `ON DELETE CASCADE` — would break reconstruction
- Orphaned snapshots are acceptable (storage is cheap, correctness is not)

### 6. is_run_start flag: OUT OF SCOPE
- Originally considered for boundary-based reconstruction
- Content-addressable approach makes it unnecessary
- Can add later if concrete need emerges (YAGNI)

### 7. Letta import computes hashes
- During Letta conversation history import, extract system prompt and tools from stored request
- Compute hashes, upsert snapshots, store references on MessageRecord
- Post-import, all records look the same regardless of origin
- Reconstructor has one code path, no None handling
- **PUNTED:** Will implement when we do Letta import work

## Storage Analysis

**Raw sizes (from testing):**
- System prompt: ~115 KB compiled
- Tool schemas: ~38 KB
- Combined: ~152 KB per message

**Without dedup:** 50k messages × 152 KB = 7.28 GB per agent

**With content-addressable dedup:** ~50-100 unique versions × 152 KB = ~15 MB + tiny hash references

SQLite doesn't compress like Postgres TOAST. Content-addressable storage is essential for sustainable growth.

## Implementation Order (Top-Down)

1. **Reconstructor** — Write first with mock/stubbed data. This is what consumes the data, so it defines the interface.
2. **DB models** — Add new tables and MessageRecord fields
3. **Persistence logic** — Add hashing and upsert logic to runner.py / persist_messages

## Reconstruction Algorithm

Given a MessageRecord:
1. Look up `system_prompt_hash` → get compiled system prompt from SystemPromptSnapshot
2. Look up `tool_schema_hash` → get tool schemas from ToolSchemaSnapshot
3. Look up `context_window_start_msg_id` → get UUID, find that message's seq_id
4. Query messages from context_window_start seq_id through target message seq_id
5. Assemble: system prompt + tools + message history = full context

## Open Items

- [ ] Reconstructor implementation
- [ ] DB model changes
- [ ] Persistence logic in runner.py
- [ ] Letta import integration (punted)
