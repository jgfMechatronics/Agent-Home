"""Entry point for `python -m acp`."""

import argparse
import asyncio
import logging
import os
import sys

from .bridge import env_flag, main


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

    parser.add_argument(
        "--watch-only",
        action="store_true",
        help="Observe only: never send prompts or cancels to the server (also AGENT_HOME_WATCH_ONLY=1)",
    )

    args = parser.parse_args()

    # Env var takes priority, then --agent-id flag, then positional arg
    agent_id = os.environ.get("AGENT_HOME_AGENT_ID") or args.agent_id or args.agent_id_pos
    server_url = os.environ.get("AGENT_HOME_SERVER_URL") or args.server_url or "http://localhost:8000"

    watch_only = args.watch_only or env_flag("AGENT_HOME_WATCH_ONLY")

    if not agent_id:
        print("Error: agent_id required (positional arg, --agent-id flag, or AGENT_HOME_AGENT_ID env var)", file=sys.stderr)
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    
    asyncio.run(main(agent_id=agent_id, server_url=server_url, watch_only=watch_only))


if __name__ == "__main__":
    cli()
