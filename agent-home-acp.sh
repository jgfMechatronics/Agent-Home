#!/bin/bash
# Wrapper script for the ACP bridge. Runs the bridge module via uv.
# Nori/Toad point their run_command at this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec uv run python -m prototype.acp "$@"
