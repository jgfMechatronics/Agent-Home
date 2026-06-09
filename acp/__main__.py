"""Entry point for `python -m acp`."""

import argparse
import asyncio
import logging
import os
import sys

from .bridge import main


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="ACP bridge: connects Toad TUI to Agent Home server"
    )
    parser.add_argument(
        "agent_id_pos",
        nargs="?",
        default=None,
        metavar="agent_id",
        help="Agent ID to connect to (positional; overridden by AGENT_HOME_AGENT_ID env var)",
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID to connect to (overridden by AGENT_HOME_AGENT_ID env var)",
        default=None,
    )
    parser.add_argument(
        "--server-url",
        help="Agent Home server URL (default: http://localhost:8000)",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to stderr",
    )

    args = parser.parse_args()

    # Env var takes priority, then --agent-id flag, then positional arg
    agent_id = os.environ.get("AGENT_HOME_AGENT_ID") or args.agent_id or args.agent_id_pos
    server_url = os.environ.get("AGENT_HOME_SERVER_URL") or args.server_url or "http://localhost:8000"

    if not agent_id:
        print("Error: agent_id required (positional arg, --agent-id flag, or AGENT_HOME_AGENT_ID env var)", file=sys.stderr)
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    
    asyncio.run(main(agent_id=agent_id, server_url=server_url))


if __name__ == "__main__":
    cli()
