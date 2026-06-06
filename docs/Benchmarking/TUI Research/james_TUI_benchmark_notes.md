ACP is a promising emerging standard for Agent-Editor interaction. Might be overkill for a TUI, or might be just what we want. Unknown levels of adoption.
From the ACP website:
https://agentclientprotocol.com/get-started/introduction
```
ACP is suitable for both local and remote scenarios:

    Local agents run as sub-processes of the code editor, communicating via JSON-RPC over stdio.
    Remote agents can be hosted in the cloud or on separate infrastructure, communicating over HTTP or WebSocket

Full support for remote agents is a work in progress. We are actively collaborating with agentic platforms to ensure the protocol addresses the specific requirements of cloud-hosted and remote deployment scenarios.
```
Questionable. It seems to use JSON-RPC, which we could probably use.

https://github.com/iowarp/gact-tui
- Seems to be intended specifically as an interoperable thin client TUI.
- It has multiple backend adapters so we could pick the one closest to our desires or write our own
- 0 stars on github BUT if it solves our problem and works I don't care (few stars makes it more of a wildcard not necessarily bad)
- Doesn't seem to support ACP
- Go and typescript

https://github.com/batrachianai/toad
- Python :)
- 3.2k stars
- Planned UI for MCP servers
- Supports "multiple agents" and "sessions"
    - ONe or some combo of these could be leveraged to provide our per agent connection
- ACP Supposedly works, but limited detail about the kinds of agent connections available (and also not clear if ACP even supports HTTP yet, maybe the website is wrong? LOL)
- The readme doesn't describe the interconnect as well as gact-tui
    - The release announcement is a bit more detailed: https://willmcgugan.github.io/toad-released/
-

https://github.com/forge-agents/forge
- typescript
- ACP
- 20 stars
- "Single conversation history across all agents" its doing more than we want from our TUI here, we want convo history to live on the server.
