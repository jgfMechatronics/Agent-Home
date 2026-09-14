"""Tests for memory cleanup utility."""
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

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
# Note: Full integration test of run_cleanup_flow with TestClient is complex due to
# async DB fixtures. The HTTP helpers are thin wrappers, so we test them via live testing
# against a real server instead. Unit tests above cover the file I/O logic thoroughly.
