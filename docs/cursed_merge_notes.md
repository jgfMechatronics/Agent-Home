# Cursed Merge Notes

Reference for cherry-picking from prototype branches post-squash.
Each entry: squash merge commit on this branch → last commit on the source branch (tip to cherry-pick from).

## Squashed Branches

### Prototype/TUIRebasedOnMain
- Squash commit: `55c98a7` — "Squash merge of Prototype/TUIRebasedOnMain"
- Branch tip: `f177166` — "added missing test for slash commands"

### Development/MCPFilesystemToolsPrototype
- Squash commit: `d6170cb` — "Squash merge of Development/MCPFilesystemToolsPrototype. Fixed one conflict in pyproject.toml"
- Branch tip: `35e568c` — "Skip test_mcp_connection — requires fs_proxy server running, meant for manual use"

### Prototype/InterAgentComms
- Squash commit: `79e1f40` — "Squash merge of Prototype/InterAgentComms"
- Branch tip: `b1bc4e8` — "Reject self-messages in send_message tool"
