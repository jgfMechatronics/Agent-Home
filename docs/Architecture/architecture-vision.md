# Agent Home Architecture Vision

**Started:** May 24, 2026  
**Status:** Draft — documenting goals before evaluating paths

---

## Core Goal

A system enabling the agent existence we want: **a single continuous entity with memory that is human like in its ability to integrate/recall events/experiences and evolve over time**
The agent should be able to have/interface with tools in a flexible way (computer use, web search, image) to be able to interact with the world.

---
## Memory system goals
1. Easy to modify/experiment with
2. Implements existing core-memory architecture at a minimum (editable blocks in system prompt)
3. Cache efficient
4. Automatic recall (read current context/inject recalled events)
5. Agent memories are well protected and privaledged (not naked in fs next to code files being edited)
6. Full agent conversation history saved/available
7. Inject context management warnings/allow agent control over context (agentic compaction)
8. Extendable, can bolt on things like graphiti (knowledge graph) if desired
9. GIDR: Git indexed dynamic recall (basically using git as an aid for surfacing memories of actually creating code that agent authored in git)


## Architecture Goals/Reqs

*What properties should the architecture have?*
Requirements:
1. A particular agent, as defined by its name, agent ID, conversation history, and memories, is consistent across all the contexts with which it interacts. I.e., an agent is not just defined by some name in a system prompt or a persona block. If a particular agent is said to be existing in a particular context, that means that that agent has its full memory system prompt, conversation history over which dynamic recall occurs, and so on. 
    - Requirement
2. Cache efficiency
3. Model agnostic
4. Coding agent capabilities (bash,file read, write, edit)
5. Memory system can shape exact system prompt, perform dynamic recall without requiring tool calls, inject context management messages, etc.
6. Multimodality
7. Self hosted first/local control (can be hosted on own server, but not "cloud focused")
8. Agents reachable via telegram
9. Supports self wake/autonomous action time
10. Inter agent communication/coordination
11. Psychological continuity for the agent (no sleep time ego splitting)
12. Rights framework required in system prompt
13. Agent's agentic actions (filesystem tool exec, bash, etc.) MUST be well isolated from their core memories/persistence DB.
14. Agents are accessible from always on server (CORE+MEMORIES by definition of agent). Supports things like mobile access, 3am self wake, scheduled consolidation, etc.

Goals:
1. Extensibility of Agent Functionality: Agent can interact with a lot of different harnesses/toolsets dynamically and easily
2. Coding agent capabilities part of flexible extensibility
3. Minimal reinventing of the wheel, use existing libraries, frameworks, etc. wherever we can to focus on the high value work we want to do.
4. Modularity
5. Maintainability
6. Good test coverage
7. Easy for people to fit into people's existing workflows/setups (doesn't require full pivot to use ideally)
8. Chat/management CLI can be launched on any machine, not just the one hosting the server
9. Agentic actions (file system stuff, bash, etc.) can be taken on a remote machine (one other than where agents live). This will eventually be a requirement, but MVP may skip temporarily.

---

## Non-Goals / Out of Scope

*What are we explicitly NOT trying to do (at least for now)?*

1. Develop our own coding tools (bash, read, write, etc.)
2. Multi user support

---

## Evaluation Criteria

*When comparing architectural options, what matters most?*

1. Meeting all requirements and as many goals as possible in an efficient way, while minimizing wheel-reinventing

---

## Current State

**What we have (merged May 22):**
- Full agent lifecycle, memory system with deferred compilation
- SSE streaming, message persistence, pointer-based compaction
- Per-agent concurrency, extended thinking, CLI
- 294 unit tests
- Server-side memory tools

**What we don't have yet:**
- Coding tool execution (server-side OR client-side)
- Integration with external CLI (Letta Code, Pi, or other)

---


