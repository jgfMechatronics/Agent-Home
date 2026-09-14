"""Tests for memory cleanup utility."""
import stat
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from agent.types import AgentDeps, BlockSettings
from memory.block_crud import create_block
from utils.memory_cleanup import (
    ValidationError,
    dump_to_files,
    load_from_files,
    run_cleanup_flow,
    validate_labels,
)


class TestValidateLabels:
    """Tests for validate_labels function."""
    
    def test_valid_labels_pass(self):
        """Normal labels should not raise."""
        validate_labels(["persona", "working-memory", "ephemera"])
    
    def test_active_skill_rejected(self):
        """active-skill label should raise ValidationError."""
        with pytest.raises(ValidationError, match="active-skill"):
            validate_labels(["persona", "active-skill", "ephemera"])
    
    def test_only_active_skill_rejected(self):
        """Single active-skill label should raise."""
        with pytest.raises(ValidationError, match="active-skill"):
            validate_labels(["active-skill"])
    
    def test_empty_list_valid(self):
        """Empty list is valid (no forbidden labels)."""
        validate_labels([])


class TestDumpToFiles:
    """Tests for dump_to_files function."""
    
    def test_creates_editable_files(self, tmp_path: Path):
        """Should create .txt files for each block."""
        blocks = {"persona": "I am Opus", "ephemera": "Recent events"}
        
        dump_to_files(tmp_path, blocks, create_backups=False)
        
        assert (tmp_path / "persona.txt").read_text() == "I am Opus"
        assert (tmp_path / "ephemera.txt").read_text() == "Recent events"
    
    def test_creates_backup_directory(self, tmp_path: Path):
        """Should create backups/ subdirectory when backups enabled."""
        blocks = {"persona": "content"}
        
        dump_to_files(tmp_path, blocks, create_backups=True)
        
        assert (tmp_path / "backups").is_dir()
    
    def test_creates_readonly_backups(self, tmp_path: Path):
        """Backup files should be read-only."""
        blocks = {"persona": "content"}
        
        dump_to_files(tmp_path, blocks, create_backups=True)
        
        backup_file = tmp_path / "backups" / "persona-backup.txt"
        assert backup_file.exists()
        assert backup_file.read_text() == "content"
        
        # Check read-only (no write bits)
        mode = backup_file.stat().st_mode
        assert not (mode & stat.S_IWUSR)  # No owner write
        assert not (mode & stat.S_IWGRP)  # No group write
        assert not (mode & stat.S_IWOTH)  # No other write
    
    def test_no_backups_when_disabled(self, tmp_path: Path):
        """Should not create backups directory when disabled."""
        blocks = {"persona": "content"}
        
        dump_to_files(tmp_path, blocks, create_backups=False)
        
        assert not (tmp_path / "backups").exists()


class TestLoadFromFiles:
    """Tests for load_from_files function."""
    
    def test_loads_existing_files(self, tmp_path: Path):
        """Should load content from existing files."""
        (tmp_path / "persona.txt").write_text("I am Opus")
        (tmp_path / "ephemera.txt").write_text("Recent events")
        
        result = load_from_files(tmp_path, ["persona", "ephemera"], prompt_on_missing=False)
        
        assert result == {"persona": "I am Opus", "ephemera": "Recent events"}
    
    def test_raises_on_missing_file_no_prompt(self, tmp_path: Path):
        """Should raise FileNotFoundError when file missing and no prompt."""
        (tmp_path / "persona.txt").write_text("exists")
        
        with pytest.raises(FileNotFoundError, match="ephemera.txt"):
            load_from_files(tmp_path, ["persona", "ephemera"], prompt_on_missing=False)
    
    def test_prompts_on_missing_file(self, tmp_path: Path):
        """Should prompt user when file missing and prompt enabled."""
        (tmp_path / "persona.txt").write_text("exists")
        
        # User says no to retry
        with patch("builtins.input", return_value="n"):
            with patch("builtins.print"):  # Suppress warning output
                with pytest.raises(FileNotFoundError):
                    load_from_files(tmp_path, ["persona", "ephemera"], prompt_on_missing=True)
    
    def test_retry_succeeds_when_file_created(self, tmp_path: Path):
        """Should succeed if file is created between retries."""
        (tmp_path / "persona.txt").write_text("exists")
        
        call_count = 0
        def create_file_on_first_call(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Create the file before returning 'y'
                (tmp_path / "ephemera.txt").write_text("created")
                return "y"
            return "n"
        
        with patch("builtins.input", side_effect=create_file_on_first_call):
            with patch("builtins.print"):
                result = load_from_files(tmp_path, ["persona", "ephemera"], prompt_on_missing=True)
        
        assert result == {"persona": "exists", "ephemera": "created"}



# --- Integration Test ---

class TestRunCleanupFlowIntegration:
    """Integration test for full cleanup flow against real app.
    
    Key insight: TestClient only runs lifespan when used as context manager.
    Without context manager, it skips lifespan — we rely on override_db_session
    fixture (autouse) to inject the test session into routes.
    """
    
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session, app, agent_with_blocks):
        """Set up test with agent_with_blocks fixture plus skill block."""
        self.agent = agent_with_blocks["agent"]
        self.agent_id = self.agent.id
        self.agent_name = self.agent.name
        
        # Grab block info before any commits (avoids ORM expiry issues)
        self.cleanup_labels = [b.label for b in agent_with_blocks["blocks"]]
        
        # Add skill block for cleanup flow
        self.skill_label = "active-skill"
        self.skill_content = "Original skill content"
        deps = AgentDeps(session=session, agent_record=self.agent)
        await create_block(deps, BlockSettings(label=self.skill_label), self.skill_content)
        await session.commit()
        
        # TestClient WITHOUT context manager = no lifespan run
        # override_db_session fixture handles routing to test session
        # base_url required for TrustedHostMiddleware validation
        self.test_client = TestClient(app, base_url="http://localhost")
    
    @pytest.fixture
    def cleanup_skill_file(self, tmp_path):
        """Create a temporary cleanup skill file."""
        skill_file = tmp_path / "cleanup_skill.txt"
        skill_file.write_text("You are in cleanup mode. Edit your memory blocks.")
        return skill_file

    def test_full_cleanup_flow(self, tmp_path, cleanup_skill_file):
        """Full flow: swap skill, dump blocks, simulate edit, put, restore."""
        
        # Mock input() to simulate user pressing Enter after "editing"
        def mock_input_and_edit(prompt):
            # Simulate agent editing the files
            date_str = datetime.now().strftime('%Y-%m-%d')
            session_dir = tmp_path / f"{date_str}-{self.agent_name}"
            for label in self.cleanup_labels:
                (session_dir / f"{label}.txt").write_text(f"Edited {label}")
            return ""  # Simulate pressing Enter
        
        with patch("builtins.input", side_effect=mock_input_and_edit):
            with patch("builtins.print"):  # Suppress output
                run_cleanup_flow(
                    self.test_client,
                    self.agent_name,
                    self.cleanup_labels,
                    working_dir=tmp_path,
                    skill_path=cleanup_skill_file,
                )
        
        # Verify blocks were updated
        for label in self.cleanup_labels:
            resp = self.test_client.get(f"/agents/{self.agent_id}/memory/blocks/{label}")
            assert resp.json()["content"] == f"Edited {label}"
        
        # Verify skill was restored to original
        resp = self.test_client.get(
            f"/agents/{self.agent_id}/memory/blocks/{self.skill_label}"
        )
        assert resp.json()["content"] == self.skill_content
