"""Type safety tests — runs pyright as part of the test suite."""
import subprocess


def test_type_safety():
    """Verify codebase passes pyright type checking."""
    result = subprocess.run(["uv", "run", "pyright"], capture_output=True)
    assert result.returncode == 0, result.stdout.decode()
