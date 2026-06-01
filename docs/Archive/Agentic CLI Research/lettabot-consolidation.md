# LettaBot → Letta Code Consolidation: Research Findings

**Date:** 2026-05-23  
**Researcher:** Sonnet

## Verdict: PARTIALLY TRUE — Migration in progress, not complete

The consolidation claim is real but more nuanced than assumed. Key facts:

### LettaBot repo status
- **Archived** on GitHub: repo banner reads "Archived - has been replaced by Letta Code channels/schedules!"
- However: the code is recent, has active release history, and still works as a standalone deployment
- Still supports: Telegram, Slack, Discord, **WhatsApp** (via Baileys), Signal, Bluesky
- WhatsApp uses Baileys (WhatsApp Web protocol) — no Business API required, free to use

### Letta Code Channels (the replacement)
- **Beta** as of current docs
- Currently supports: Telegram, Slack, Discord only — **no WhatsApp or Signal yet**
- Architecture: channels integrate via Letta server WebSocket connection
- Messages buffer during brief disconnects — no drops on reconnect
- The intent is clearly for Letta Code to absorb all of LettaBot's channel functionality over time

### Implication for us
- If we need WhatsApp: LettaBot is still the only option (Letta Code channels doesn't have it yet)
- If we can wait: Letta Code channels will eventually absorb WhatsApp
- The "two for the price of one" argument holds directionally — but WhatsApp integration would currently require LettaBot running separately alongside Letta Code

### Letta Code architecture (relevant to CLI harness decision)
- Client connects to Letta server via WebSocket
- `LETTA_BASE_URL` env var points to external server (our server)
- `/connect` for custom LLM API keys; `/model` to swap models
- Memory-first, persistent agents — already the architecture we want
- `letta --agent <agent_id>` to connect to specific agent

## Sources
- https://github.com/letta-ai/lettabot (archived repo)
- https://docs.letta.com/letta-code/channels/ (beta channels docs)
- https://www.letta.com/blog/our-next-phase (Letta's direction)
- https://github.com/letta-ai/letta-code/ (Letta Code repo)
