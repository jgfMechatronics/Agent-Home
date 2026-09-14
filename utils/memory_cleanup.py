"""
Memory cleanup utility for Agent Home.

Provides workflows for agents to clean up their memory blocks:
1. run_cleanup_flow: Full flow with skill swap, dump, edit, restore
2. put_blocks_from_files: Resume/recovery - just put edited files back

Usage:
    python -m utils.memory_cleanup full <agent_name> <label1> <label2> ...
    python -m utils.memory_cleanup put <agent_name> <label1> <label2> ...

Environment variables (via .env):
    MEMORY_CLEANUP_WORKING_DIR: Base folder for cleanup sessions
    MEMORY_CLEANUP_SKILL_PATH: Path to the cleanup skill file
"""
import os
import stat
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

def get_session_dir(agent_name: str) -> Path:
    """Get or create the session directory for an agent cleanup.
    
    Creates: <WORKING_DIR>/<date>-<agent-name>/
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    session_dir = WORKING_DIR / f"{date_str}-{agent_name}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def dump_to_files(
    session_dir: Path,
    blocks: dict[str, str],
    create_backups: bool = True
) -> None:
    """Dump memory blocks to files in the session directory.
    
    Creates:
        <session_dir>/<label>.txt - Editable file
        <session_dir>/backups/<label>-backup.txt - Read-only backup (if create_backups=True)
    
    Args:
        session_dir: Directory to write files to
        blocks: Dict mapping label -> content
        create_backups: Whether to create read-only backups
    """
    if create_backups:
        backup_dir = session_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
    
    for label, content in blocks.items():
        # Write editable file
        edit_file = session_dir / f"{label}.txt"
        edit_file.write_text(content)
        
        # Write read-only backup
        if create_backups:
            backup_file = backup_dir / f"{label}-backup.txt"
            backup_file.write_text(content)
            # Set read-only (remove write permissions)
            backup_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def load_from_files(
    session_dir: Path,
    labels: list[str],
    prompt_on_missing: bool = True
) -> dict[str, str]:
    """Load memory block contents from files.
    
    Args:
        session_dir: Directory to read files from
        labels: Labels to load
        prompt_on_missing: If True, prompt user to retry when file is missing
        
    Returns:
        Dict mapping label -> content
        
    Raises:
        FileNotFoundError: If a file is missing and user doesn't retry
    """
    blocks = {}
    
    for label in labels:
        file_path = session_dir / f"{label}.txt"
        
        while not file_path.exists():
            if not prompt_on_missing:
                raise FileNotFoundError(f"Missing file: {file_path}")
            
            print(f"\nWarning: File not found: {file_path}")
            response = input("Retry after creating/restoring file? [y/N]: ").strip().lower()
            if response != 'y':
                raise FileNotFoundError(f"Missing file: {file_path}")
        
        blocks[label] = file_path.read_text()
    
    return blocks


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

def run_cleanup_flow(client: httpx.Client, agent_name: str, labels: list[str]) -> None:
    """Run the full cleanup flow.
    
    1. Validate labels
    2. Swap in cleanup skill
    3. Dump blocks to files
    4. Pause for agent to edit
    5. Put updated blocks
    6. Restore original skill
    
    Args:
        client: HTTP client for API calls
        agent_name: Name of the agent
        labels: Memory block labels to clean up
    """
    validate_labels(labels)
    agent_id = _get_agent_id(client, agent_name)
    
    # Save current skill and swap in cleanup skill
    original_skill = get_skill(client, agent_id)
    cleanup_skill = CLEANUP_SKILL_PATH.read_text()
    put_skill(client, agent_id, cleanup_skill)
    recompile(client, agent_id)
    print(f"Swapped in cleanup skill, recompiled.")
    
    # Dump blocks to files
    session_dir = get_session_dir(agent_name)
    blocks = get_blocks(client, agent_id, labels)
    dump_to_files(session_dir, blocks, create_backups=True)
    print(f"\nBlocks dumped to: {session_dir}")
    print(f"  Editable: {', '.join(f'{l}.txt' for l in labels)}")
    print(f"  Backups:  backups/<label>-backup.txt (read-only)")

    
    # Pause for agent edit
    print("\n--- Agent can now edit the files ---")
    input("Press Enter when editing is complete...")
    
    # Put updated blocks
    updated_blocks = load_from_files(session_dir, labels, prompt_on_missing=True)
    put_blocks(client, agent_id, updated_blocks)
    print(f"Updated {len(updated_blocks)} blocks.")
    
    # Restore original skill
    restore_skill(client, agent_id, original_skill)
    print("Restored original skill, recompiled.")
    print("\nCleanup complete!")


def put_blocks_from_files(client: httpx.Client, agent_name: str, labels: list[str]) -> None:
    """Put blocks from files (recovery/resume flow).
    
    Use this if the full flow was interrupted after dumping but before putting,
    or to re-apply edits.
    
    Args:
        client: HTTP client for API calls
        agent_name: Name of the agent
        labels: Memory block labels to update
    """
    validate_labels(labels)
    agent_id = _get_agent_id(client, agent_name)
    
    session_dir = get_session_dir(agent_name)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    
    updated_blocks = load_from_files(session_dir, labels, prompt_on_missing=True)
    put_blocks(client, agent_id, updated_blocks)
    print(f"Updated {len(updated_blocks)} blocks from {session_dir}")


# --- CLI Entry Point ---

def main() -> None:
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nError: Not enough arguments")
        print("Usage: python -m utils.memory_cleanup <full|put> <agent_name> <label1> [label2 ...]")
        sys.exit(1)
    
    command = sys.argv[1]
    agent_name = sys.argv[2]
    labels = sys.argv[3:]
    
    try:
        with httpx.Client(base_url=SERVER_URL) as client:
            if command == "full":
                run_cleanup_flow(client, agent_name, labels)
            elif command == "put":
                put_blocks_from_files(client, agent_name, labels)
            else:
                print(f"Unknown command: {command}")
                print("Use 'full' for complete flow or 'put' to just update blocks from files")
                sys.exit(1)
    except ValidationError as e:
        print(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
