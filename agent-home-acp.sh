#!/bin/bash
# Wrapper script for the ACP bridge. Activates the venv and runs the bridge module.
# Nori/Toad point their run_command at this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv-host/bin/activate"
exec python -m acp "$@"
