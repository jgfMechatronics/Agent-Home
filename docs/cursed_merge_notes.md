# Cursed Merge Notes

Reference for cherry-picking from prototype branches post-squash.
Each entry: squash merge commit on this branch → last commit on the source branch (tip to cherry-pick from).

## Squashed Branches

### Prototype/TUIRebasedOnMain
- Squash commit: `55c98a7` — "Squash merge of Prototype/TUIRebasedOnMain"
- Branch tip: `f177166` — "added missing test for slash commands"

### Development/MCPFilesystemToolsPrototype
- Squash commit: `d6170cb` — "Squash merge of Development/MCPFilesystemToolsPrototype. Fixed one conflict in pyproject.toml"
- Branch tip at squash: `35e568c` — "Skip test_mcp_connection — requires fs_proxy server running, meant for manual use"

#### Cherry-pick batch (Aug 10, 2026)
Six commits cherry-picked onto cursed_branch after squash. All applied cleanly (no conflicts).
- `3f6d167` — "Tests for attaching/detaching MCPToolsets"
- `ee29e34` — "AgentConfig new field for toolsets"
- `eaa96d7` — "Refactor: extract _construct_toolsets helper for conditional toolset building"
- `ee13c78` — "test for support of MCP schema snapshotting."
- `982a035` — "update _extract_tool_definitions to support MCP toolsets"
- `def81ff` — "moved tool schema capture spot for future support of agent attachable/detachable toolsets"
- Branch tip after pick: `def81ff` — cursed_branch tip: `0ddaba3`

#### Cherry-pick batch (Aug 11, 2026)
- `498c7fa` — "Use host.docker.internal for MCP filesystem toolset URL"
- Branch tip after pick: `498c7fa` — cursed_branch tip: `55f5684`
- `0bb8a34` — "Bind fs_proxy to 0.0.0.0 for cross-container MCP access"
- Branch tip after pick: `0bb8a34` — cursed_branch tip: `15fa71f`
- `924b846` — "Allowlist 14 tools in fs_proxy — cut DC junk from context"
- Branch tip after pick: `924b846` — cursed_branch tip: `3a9796b`

## Cherry-picks from main

### PR #21 — Add import_af.py CLI for .AF file ingestion
- `2585aa6` cherry-picked from main — cursed_branch tip: `6c5c24e`

### Prototype/InterAgentComms
- Squash commit: `79e1f40` — "Squash merge of Prototype/InterAgentComms"
- Branch tip: `b1bc4e8` — "Reject self-messages in send_message tool"
