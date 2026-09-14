"""
Memory cleanup utility for Agent Home.

Provides workflows for agents to clean up their memory blocks:
1. run_cleanup_flow: Full flow with skill swap, prompt for labels, dump, edit, restore
2. put_blocks_from_files: Resume/recovery - put edited files back

Usage:
    python -m utils.memory_cleanup full --agent <name>
    python -m utils.memory_cleanup put --agent <name> --session-dir <path> --labels <label1> [label2 ...]

Environment variables (via .env):
    MEMORY_CLEANUP_WORKING_DIR: Base folder for cleanup sessions (used by 'full' command)
    MEMORY_CLEANUP_SKILL_PATH: Path to the cleanup skill file
"""
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---

WORKING_DIR = Path(os.environ.get("MEMORY_CLEANUP_WORKING_DIR", "./memory_cleanup_sessions"))
CLEANUP_SKILL_PATH = Path(os.environ.get("MEMORY_CLEANUP_SKILL_PATH", ""))
SERVER_URL = os.environ.get("AGENT_HOME_SERVER_URL", "http://localhost:8000")

FORBIDDEN_LABELS = {"active-skill"}



# --- Validation ---

class ValidationError(Exception):
    """Raised when label validation fails."""
    pass


def validate_labels(labels: list[str]) -> None:
    """Validate that no forbidden labels are in the list.
    
    Raises:
        ValidationError: If any forbidden label (e.g., 'active-skill') is included.
    """
    forbidden_found = FORBIDDEN_LABELS & set(labels)
    if forbidden_found:
        raise ValidationError(
            f"Cannot include protected labels in cleanup: {forbidden_found}"
        )


# --- File I/O Helpers ---

def create_session_dir(working_dir: Path, agent_name: str) -> Path:
    """Create a new session directory for an agent cleanup.
    
    Creates: <working_dir>/<date>-<agent-name>/
    
    Raises:
        FileExistsError: If session directory already exists (prevents overwriting).
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    session_dir = working_dir / f"{date_str}-{agent_name}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def dump_to_files(session_dir: Path, blocks: dict[str, str]) -> None:
    """Dump memory blocks to files in the session directory.
    
    Creates:
        <session_dir>/<label>.txt - Editable file
        <session_dir>/backups/<label>-backup.txt - Read-only backup
    
    Args:
        session_dir: Directory to write files to
        blocks: Dict mapping label -> content
    """
    backup_dir = session_dir / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    for label, content in blocks.items():
        # Write editable file
        edit_file = session_dir / f"{label}.txt"
        edit_file.write_text(content)
        
        # Write read-only backup
        backup_file = backup_dir / f"{label}-backup.txt"
        backup_file.write_text(content)
        backup_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def load_from_files(session_dir: Path, labels: list[str]) -> dict[str, str]:
    """Load memory block contents from files.
    
    Prompts user to retry if a file is missing.
    
    Args:
        session_dir: Directory to read files from
        labels: Labels to load
        
    Returns:
        Dict mapping label -> content
        
    Raises:
        FileNotFoundError: If a file is missing and user declines retry
    """
    blocks = {}
    
    for label in labels:
        file_path = session_dir / f"{label}.txt"
        
        while not file_path.exists():
            print(f"\nWarning: File not found: {file_path}")
            response = input("Retry after creating/restoring file? [y/N]: ").strip().lower()
            if response != 'y':
                raise FileNotFoundError(f"Missing file: {file_path}")
        
        blocks[label] = file_path.read_text()
    
    return blocks


def prompt_for_labels() -> list[str]:
    """Prompt user for memory block labels to clean up.
    
    Returns:
        List of label strings (comma-separated input, whitespace trimmed)
        
    Raises:
        ValueError: If no labels provided
    """
    print("\nEnter memory block labels to clean up (comma-separated):")
    raw = input("> ").strip()
    
    if not raw:
        raise ValueError("No labels provided")
    
    return [label.strip() for label in raw.split(",")]


# --- HTTP Helpers ---

def _get_agent_id(client: httpx.Client, agent_name: str) -> str:
    """Resolve agent name to ID via API."""
    response = client.get("/agents")
    response.raise_for_status()
    agents = response.json()
    
    for agent in agents:
        if agent["name"].lower() == agent_name.lower():
            return agent["id"]
    
    raise ValueError(f"Agent not found: {agent_name}")


def get_blocks(client: httpx.Client, agent_id: str, labels: list[str]) -> dict[str, str]:
    """Fetch memory block contents from server."""
    blocks = {}
    for label in labels:
        response = client.get(f"/agents/{agent_id}/memory/blocks/{label}")
        response.raise_for_status()
        blocks[label] = response.json()["content"]
    return blocks


def put_blocks(client: httpx.Client, agent_id: str, blocks: dict[str, str]) -> None:
    """Update memory block contents on server."""
    for label, content in blocks.items():
        response = client.put(
            f"/agents/{agent_id}/memory/blocks/{label}/content",
            json={"content": content}
        )
        response.raise_for_status()


def get_skill(client: httpx.Client, agent_id: str) -> str:
    """Get current active-skill content."""
    response = client.get(f"/agents/{agent_id}/memory/blocks/active-skill")
    response.raise_for_status()
    return response.json()["content"]


def put_skill(client: httpx.Client, agent_id: str, content: str) -> None:
    """Update active-skill content."""
    response = client.put(
        f"/agents/{agent_id}/memory/blocks/active-skill/content",
        json={"content": content}
    )
    response.raise_for_status()


def restore_skill(client: httpx.Client, agent_id: str, original_skill: str) -> None:
    """Restore original skill and recompile."""
    put_skill(client, agent_id, original_skill)
    recompile(client, agent_id)


def recompile(client: httpx.Client, agent_id: str) -> None:
    """Trigger system prompt recompilation."""
    response = client.post(f"/agents/{agent_id}/recompile_system_prompt")
    response.raise_for_status()


# --- Main Flows ---

def run_cleanup_flow(
    client: httpx.Client,
    agent_name: str,
    working_dir: Path,
    skill_path: Path,
) -> None:
    """Run the full cleanup flow.
    
    1. Swap in cleanup skill (so agent can see cleanup guidance)
    2. Prompt for labels to clean up
    3. Validate labels
    4. Dump blocks to files
    5. Pause for agent to edit
    6. Put updated blocks
    7. Restore original skill
    
    Args:
        client: HTTP client for API calls
        agent_name: Name of the agent
        working_dir: Base directory for cleanup sessions
        skill_path: Path to the cleanup skill file
    """
    agent_id = _get_agent_id(client, agent_name)
    
    # Save current skill and swap in cleanup skill
    original_skill = get_skill(client, agent_id)
    cleanup_skill = skill_path.read_text()
    put_skill(client, agent_id, cleanup_skill)
    recompile(client, agent_id)
    print("Swapped in cleanup skill, recompiled.")
    
    # Prompt for labels (agent can now see cleanup guidance)
    labels = prompt_for_labels()
    validate_labels(labels)
    
    # Dump blocks to files
    session_dir = create_session_dir(working_dir, agent_name)
    blocks = get_blocks(client, agent_id, labels)
    dump_to_files(session_dir, blocks)
    print(f"\nBlocks dumped to: {session_dir}")
    print(f"  Editable: {', '.join(f'{l}.txt' for l in labels)}")
    print(f"  Backups:  backups/<label>-backup.txt (read-only)")

    
    # Pause for agent edit
    print("\n--- Agent can now edit the files ---")
    input("Press Enter when editing is complete...")
    
    # Put updated blocks
    updated_blocks = load_from_files(session_dir, labels)
    put_blocks(client, agent_id, updated_blocks)
    print(f"Updated {len(updated_blocks)} blocks.")
    
    # Restore original skill
    restore_skill(client, agent_id, original_skill)
    print("Restored original skill, recompiled.")
    print("\nCleanup complete!")


def put_blocks_from_files(
    client: httpx.Client,
    agent_name: str,
    labels: list[str],
    session_dir: Path,
) -> None:
    """Put blocks from files (recovery/resume flow).
    
    Use this if the full flow was interrupted after dumping but before putting,
    or to re-apply edits.
    
    Args:
        client: HTTP client for API calls
        agent_name: Name of the agent
        labels: Memory block labels to update
        session_dir: Path to the session directory containing edited files
    """
    validate_labels(labels)
    agent_id = _get_agent_id(client, agent_name)
    
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    
    updated_blocks = load_from_files(session_dir, labels)
    put_blocks(client, agent_id, updated_blocks)
    print(f"Updated {len(updated_blocks)} blocks from {session_dir}")


# --- CLI Entry Point ---

def main() -> None:
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Memory cleanup utility for Agent Home agents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Full cleanup flow
    full_parser = subparsers.add_parser(
        "full",
        help="Run full cleanup flow (swap skill, dump, edit, restore)"
    )
    full_parser.add_argument("--agent", required=True, help="Agent name")
    
    # Put blocks from files (recovery)
    put_parser = subparsers.add_parser(
        "put",
        help="Put blocks from files (recovery/resume)"
    )
    put_parser.add_argument("--agent", required=True, help="Agent name")
    put_parser.add_argument("--session-dir", required=True, type=Path, help="Session directory path")
    put_parser.add_argument("--labels", required=True, nargs="+", help="Labels to update")
    
    args = parser.parse_args()
    
    with httpx.Client(base_url=SERVER_URL) as client:
        if args.command == "full":
            run_cleanup_flow(client, args.agent, WORKING_DIR, CLEANUP_SKILL_PATH)
        elif args.command == "put":
            put_blocks_from_files(client, args.agent, args.labels, args.session_dir)


if __name__ == "__main__":
    main()
