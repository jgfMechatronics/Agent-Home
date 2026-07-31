# Compaction Implementation Design (Gaps 9+10)

**Created:** July 25, 2026  
**Status:** Implemented

## Compaction Warning (Gap 9)

### Mechanism

Custom capability using `after_model_request` hook with `ctx.enqueue()` for injection.

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

Actual implementation in `agent/compaction_warner.py`:
```python
@dataclass
class CompactionWarner(AbstractCapability[AgentDeps]):
    async def after_model_request(
        self, ctx: RunContext[AgentDeps], *, 
        request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if ctx.deps.compaction_warning_fired:
            return response  # already warned this cycle
        
        threshold = int(ctx.deps.config.soft_compaction_limit * 0.75)
        total_tokens = response.usage.total_tokens if response.usage.total_tokens else 0
        
        if total_tokens >= threshold:
            ctx.deps.compaction_warning_fired = True
            ctx.enqueue(UserPromptPart(content=COMPACTION_WARNING_TEXT))
        
        return response
```

### Hook Timing

`after_model_request` fires after EVERY model response in a turn. We use `ctx.enqueue()` to inject the warning, which delivers it on the NEXT model request and ensures it gets captured in message history.

This allows mid-turn warning injection — the warning can fire after tool results accumulate context.

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

---

Behaviors to test:
- Injects warning once threshold crossed
	- Does NOT inject warning before threshold crossed
- Warning can be injected mid turn, not only at the start or end
- Only injects warning once when above threshold. Resets upon compaction.
	- After reset, warning can fire again, but still only once (IE once per compaction cycle)
    - Threshold should be token count is 75% of the soft compaction limit. 
    (if we need this adjustable per agent later we can but for now hard coded is fine)
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
