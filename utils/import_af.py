#!/usr/bin/env python3
"""
Import a Letta .AF file into Agent Home.

Usage:
    python import_af.py <path_to_af_file>

Example:
    python import_af.py ~/git/misc/Migration/haiku_export.af
"""
import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from af_ingestion import AFIngestionError, import_agent_file


AGENT_HOME_URL = "http://localhost:8000"


async def main(af_path: Path) -> None:
    """Import the .AF file and print the created agent ID."""
    if not af_path.exists():
        print(f"Error: File not found: {af_path}", file=sys.stderr)
        sys.exit(1)
    
    if not af_path.suffix == ".af":
        print(f"Warning: File does not have .af extension: {af_path}", file=sys.stderr)
    
    print(f"Importing {af_path.name} into Agent Home at {AGENT_HOME_URL}...")
    
    async with httpx.AsyncClient(base_url=AGENT_HOME_URL, timeout=30.0) as client:
        try:
            agent_id = await import_agent_file(af_path, client)
            print(f"Success! Created agent: {agent_id}")
        except AFIngestionError as e:
            print(f"Error parsing .AF file: {e}", file=sys.stderr)
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"API error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import a Letta .AF file into Agent Home")
    parser.add_argument("af_file", type=Path, help="Path to the .af file to import")
    args = parser.parse_args()
    
    asyncio.run(main(args.af_file))
