#!/usr/bin/env python3
"""Dump raw message content for specific seq_ids from the test database."""

import sqlite3
import json
import sys

DB_PATH = "db.sqlite.good"
AGENT_ID = "b618245d-b6f9-4928-821e-9bf9ae8fe0a1"

# Default seq_ids if none provided on command line
DEFAULT_SEQ_IDS = [108, 109, 110, 111, 112, 113]


def dump_messages(seq_ids: list[int], output_file: str = "flagged_messages_dump.txt"):
    conn = sqlite3.connect(DB_PATH+"?mode=ro")
    cursor = conn.cursor()

    output_lines = []
    for seq_id in seq_ids:
        cursor.execute(
            "SELECT seq_id, type, content, timestamp FROM message WHERE agent_id = ? AND seq_id = ?",
            (AGENT_ID, seq_id),
        )
        row = cursor.fetchone()
        if row:
            seq, msg_type, content, ts = row
            output_lines.append("=" * 80)
            output_lines.append(f"seq_id: {seq}  |  type: {msg_type}  |  timestamp: {ts}")
            output_lines.append("=" * 80)
            # Pretty print the JSON content
            try:
                parsed = json.loads(content)
                output_lines.append(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                output_lines.append(content)
            output_lines.append("")
        else:
            output_lines.append(f"seq_id {seq_id}: NOT FOUND")
            output_lines.append("")

    conn.close()

    output = "\n".join(output_lines)
    with open(output_file, "w") as f:
        f.write(output)

    print(f"Dumped {len(seq_ids)} messages to {output_file}")


if __name__ == "__main__":
    # Parse seq_ids from command line args, or use defaults
    if len(sys.argv) > 1:
        seq_ids = [int(x) for x in sys.argv[1:]]
    else:
        seq_ids = DEFAULT_SEQ_IDS

    dump_messages(seq_ids)
