"""Shared deployment constants used across server and standalone tools."""

# Lockfile written by the integrity checker to signal failures.
# The server reads this at startup and refuses to start if it exists.
# Both parties must agree on this name — change in one place only.
INTEGRITY_LOCKFILE_NAME = "INTEGRITY_CHECK_FAILED"
