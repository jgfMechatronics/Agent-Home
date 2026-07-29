# Compaction Implementation Design (Gaps 9+10)

**Created:** July 25, 2026  
**Status:** Warning injection design settled; compaction mechanics TBD

## Compaction Warning (Gap 9)

### Mechanism

Custom capability using `before_model_request` hook.

### Behavior

- Fires ONCE when context tokens cross warning threshold
- Does NOT fire repeatedly while above threshold
- Resets only when compaction runs (puts context below threshold)
- Custom message telling agent to consolidate memories NOW

### State Tracking

- `compaction_warning_fired: bool` on `AgentRecord` (database)
- Must survive turn boundaries (not just instance-level)
- Reset to `False` when compaction runs

**Why database, not instance:**
- Turn 1: warning fires, flag = True
- Turn ends
- Turn 2: new run, new capability instance — but flag persists in DB
- Agent doesn't get warned again until compaction resets it

### Implementation Shape

```python
@dataclass
class CompactionWarner(AbstractCapability[AgentDeps]):
    warning_threshold_tokens: int
    warning_message: str

    async def before_model_request(
        self, ctx: RunContext[AgentDeps], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        agent_record = ctx.deps.agent_app_state.agent_record
        
        if agent_record.compaction_warning_fired:
            return request_context  # already warned this cycle
        
        estimated_tokens = _estimate_tokens(request_context.messages)
        if estimated_tokens >= self.warning_threshold_tokens:
            agent_record.compaction_warning_fired = True
            # persist to DB...
            
            # Inject warning as user message
            # JF Note: This part may require a bit more fleshing out, we might want to enqueue.
            # I havent seen this method but it might also be fine.
            request_context.messages = [
                *request_context.messages,
                ModelRequest(parts=[UserPromptPart(content=self.warning_message)])
            ]
        
        return request_context
```

### Hook Timing

`before_model_request` fires before EVERY model request in a turn:
- Before initial model request
- Before each subsequent request (after tool returns)

This allows mid-turn warning injection — the warning can fire after the agent has accumulated context from tool results.

### Deps Access

The capability receives `ctx: RunContext[AgentDeps]`, so we have full access to:
- `ctx.deps.agent_app_state.agent_record` — for the warning flag
- Database session for persistence

### Evaluated Alternative: pydantic-ai LimitWarner

Rejected because:
- Not in our pydantic-ai version (experimental `pydantic-ai-harness`)
- Fires repeatedly while above threshold (we need fire-once)
- No custom message support
- No state persistence across turns

Rolling our own is ~25 lines and gives exactly the behavior we need.

### Open Questions

- Exact warning message wording
- Warning threshold value (e.g., 75% of compaction limit)
- Token estimation approach (character heuristic vs tiktoken)

---

Behaviors to test:
- Injects warning once threshold crossed
	- Does NOT inject warning before threshold crossed
- Warning can be injected mid turn, not only at the start or end
- Only injects warning once when above threshold. Resets upon compaction.
	- After reset, warning can fire again, but still only once (IE once per compaction cycle)
- Warning appears to agent where expected
	- I don't actually know where in the agent's run/context it will appear. 
	- This test can be written after the impl, I mostly just want it so that we can inspect and define whatever the behavior is.
- Agent run continues after warning injected
- Warning text as below:
"""
You are nearing compaction. Oldest messages at the beginning of your context will soon be evicted.
Please scan your context and look for anything that should be moved to an appropriate core memory block.
Start your scan from the OLDEST messages. Compaction starts with the oldest messages and works forward until the token ct target is reached.
You do NOT need to re-save anything you have already saved in this context, or anything already captured in core memory.
Remember, anything compacted away and not archived or saved to core-memory will be GONE, as if it never happened from your perspective!
Take your time with the memory consolidation! Your active task will still be waiting for you when you're done. Preserving memories is more important than resuming the task quickly.
Once you are finished with consolidation, you can resume your previous task. The compaction will be seamless from your perspective
If your memory is already up to date with your active context, you can ignore this message.
"""
