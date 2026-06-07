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
        "--agent-id",
        help="Agent ID to connect to (or set AGENT_HOME_AGENT_ID env var)",
        default=os.environ.get("AGENT_HOME_AGENT_ID"),
    )
    parser.add_argument(
        "--server-url",
        help="Agent Home server URL (default: http://localhost:8000)",
        default=os.environ.get("AGENT_HOME_SERVER_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to stderr",
    )
    
    args = parser.parse_args()
    
    if not args.agent_id:
        print("Error: --agent-id required (or set AGENT_HOME_AGENT_ID)", file=sys.stderr)
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    
    asyncio.run(main(agent_id=args.agent_id, server_url=args.server_url))


if __name__ == "__main__":
    cli()
