# Plan/Notes for Agent Home TUI exploratory prototype

## Requirements/Goals
### Requirements
1. Real time streaming display of agent activity (typewriter style)
    - Some buffering is totally fine. Just want that nice smooth typewriter effect + not having to wait for large text blocks to generate and send to start seeing what agent is up to.
    - This will require handling partial text updates, not just display new msgs one at a time
2. Display's agent activity even when not initiated by TUI
    - IE an inter agent message, group chat, self wake, etc. kicking off agent activity should be reflected in the TUI
    - Streaming in this state w/ typewriter would be highly preferable, but displaying in chunks would be OK as long as not more than a msg or so behind
    - Can simulate for testing with a direct API hit with TUI running
3. Display tool calls
    - Tool name
    - Args
    - Result (some truncation/hiding on a case by case basis)
    - File writes display output
4. file edits display git style diff (colored)
    - This is going to be hard. If the tool result doesn't give us what we need (diff) we are in trouble
5. Approve/Deny tools which require approval
6. Manage Tool permissions mode
    - Unclear how we will want to handle this, possibly by MCP server or particular bundle of MCP tools
    - Likely will want to manage permissions by environment, which is closely coupled to MCP tools, because the environment of concern is that which the MCP tools are running in.
7. Esc to halt active agent exectuion REGARDLESS of source of activity (TUI/External)
    - This will be tricky and require hammering out a fair bit of details.
8. When launching without arg specifying agent name/id to connect to immediately, launches landing page
    - Agent CRUD
    - Overall server status. Agent list/status summary (details of summary to be fleshed out)
    - View/select agent to connect to
        - Once connection established, that TUI window/process is matched to that particular agent
9. Once connected to agent, two display modes:
    - CC style focused interface for agentic coding. Chat, message history
    - Keyboard shortcut to toggle extended UI:
        - Display connected MCP servers
        - Display context stats/usage breakdown
        - Possibly some basic memory system info
            - Full memory system management per agent will likely be better served as its own TUI/GUI
        - Likely will want extended UI elements as an extra panel. Unclear if vertical to side of chat or horizontal above
10. Image support, paste in text box and send to model
    - This will be hard
11. Resizing friendly. Resizes without elements getting duplicated, busted, etc.
12. Streaming non-blocking
    - User can type + send messages to agents that get queued up for insertion *somewhere* in the agent's stream.

### Goals
1. Claude code/Letta code type styling of main CLI
    - Ref a screenshot of the claude code UI
    - Little animations while agent is running
    - "Sonnet is absolutely right, Opus is machinating" cute little rotating activity messages
2. Custom user styling/theming (NOT for prototype)
3. Reusable widgents such as:
    - "AgentCard" widget showing name + status + token count — used on landing page AND in a sidebar
      "MessageBubble" for conversation — same rendering whether it's user/assistant/tool
      "ToolCallDisplay" for showing what tool was invoked + args + result
      "ApprovalPrompt" — consistent y/n/edit interface
    - Aids stylistic consistency/avoids duplication
    - NOT required for prototype, but if it falls out naturally/ is easier, go ahead.

### Not pursuing currently (but could change our minds)
1. SSH connectivity

### Open questions
- Disconnection handling? How does TUI handle losing connection to the server
- How do reqs Esc (req 7) and  non-blocking stream (req 12) interact? Does esc clear msg queue too?
- How will we track message history and stay in sync with the server while streaming and such?
    - If we only rely on integrating streaming after an initial get of message history, we open ourselves up to desyncing from the server.
    - could consider always getting message history after a stream completes to make sure that we stay in sync, but that could be a lot of extra activity and might result in some UI chunkiness at the transition.

Thoughts from Opus on staying in sync with server:
```
**Source of truth = server.** Streaming is real-time *notification*, not the data model.

Pattern:
- On connect: fetch message history from server, render it
- While connected: SSE stream appends/updates the live view
- On stream complete (or periodically on idle): reconcile with server — fetch latest, diff against local, fix any drift
- On scroll-up past live buffer: fetch from server (lazy load historical messages)

The "chunkiness" concern is valid if we do full re-render on reconcile. Better: reconcile in background, only update UI if there's actual drift. Most of the time there won't be.

This also gives us natural reconnection handling: on reconnect, fetch history, diff, resume. The UI might jump a bit if messages happened while disconnected, but that's honest — things *did* happen.

The alternative (treating stream as authoritative) is fragile. Network blips become data loss.
```

### Notes
- These Reqs/goals will inevitably require expanding the agent home core API surface. The discovery of what new API routes we need is an important part of this prototyping process.
- For the prototype, we can selectively punt goals/requirements as needed. What we don't want to do though is miss critical challenges with any of the goals/requirements (reqs especially), IE if there is a critical difficulty with any of the goals/reqs and we miss it by punting, that will be a problem.