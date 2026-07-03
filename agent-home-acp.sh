#!/bin/bash
# Wrapper script for agent-home-acp that activates the venv
# Toad calls this, it activates venv and runs the bridge

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
exec agent-home-acp "$@" 2>/tmp/acp-bridge.err
