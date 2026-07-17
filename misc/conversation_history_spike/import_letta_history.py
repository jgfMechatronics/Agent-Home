"""
Import Letta conversation history into Agent Home.

This is a spike script to prove we can port conversation history.
Not production-ready — makes pragmatic decisions to get data in.

Mapping heuristics:
- user_message -> ModelRequest with UserPromptPart
- reasoning_message + assistant_message -> ModelResponse with ThinkingPart + TextPart
- reasoning_message + tool_call/approval_request -> ModelResponse with ThinkingPart + ToolCallPart
- tool_return_message -> ModelRequest with ToolReturnPart
- Skip: approval_response_message, summary/compaction blocks

Usage:
    # Convert only (no DB write)
    python import_letta_history.py haiku_spike_history.json --stats

    # Full import to database
    python import_letta_history.py haiku_spike_history.json --import --agent-name "history-import-test"
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add Agent Home to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelMessagesTypeAdapter,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def parse_letta_timestamp(date_str: str) -> datetime:
    """Parse Letta's ISO timestamp to datetime."""
    dt = datetime.fromisoformat(date_str)
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def extract_user_content(letta_content) -> str:
    """Extract text content from Letta user_message content field."""
    if isinstance(letta_content, str):
        return letta_content
    if isinstance(letta_content, list):
        texts = []
        for part in letta_content:
            if isinstance(part, dict) and part.get('type') == 'text':
                texts.append(part.get('text', ''))
        return ''.join(texts)
    return str(letta_content)


def is_summary_message(msg: dict) -> bool:
    """Check if this message is a compaction summary to skip."""
    if msg['message_type'] != 'user_message':
        return False
    content = extract_user_content(msg.get('content', ''))
    if content.strip().startswith('{'):
        try:
            data = json.loads(content)
            if data.get('type') == 'system_alert':
                message = data.get('message', '')
                if 'summary' in message.lower() and 'prior messages have been hidden' in message.lower():
                    return True
        except json.JSONDecodeError:
            pass
    return False


def convert_letta_to_agent_home(letta_messages: list[dict]) -> list[tuple[ModelMessage, datetime]]:
    """Convert Letta messages to Agent Home ModelMessages.
    
    Returns list of (ModelMessage, timestamp) tuples.
    """
    result: list[tuple[ModelMessage, datetime]] = []
    i = 0
    
    while i < len(letta_messages):
        msg = letta_messages[i]
        msg_type = msg['message_type']
        timestamp = parse_letta_timestamp(msg['date'])
        
        # Skip approval_response_message entirely
        if msg_type == 'approval_response_message':
            i += 1
            continue
        
        # Skip summary/compaction messages
        if is_summary_message(msg):
            print(f"  Skipping summary message at index {i}")
            i += 1
            continue
        
        # user_message -> ModelRequest with UserPromptPart
        if msg_type == 'user_message':
            content = extract_user_content(msg.get('content', ''))
            request = ModelRequest(parts=[
                UserPromptPart(content=content, timestamp=timestamp)
            ])
            result.append((request, timestamp))
            i += 1
            continue
        
        # reasoning_message - look ahead to see what follows
        if msg_type == 'reasoning_message':
            thinking_content = msg.get('reasoning', '')
            parts = [ThinkingPart(content=thinking_content)]
            
            # Look at next message to combine
            if i + 1 < len(letta_messages):
                next_msg = letta_messages[i + 1]
                next_type = next_msg['message_type']
                
                # reasoning + assistant_message -> ModelResponse with ThinkingPart + TextPart
                if next_type == 'assistant_message':
                    text_content = next_msg.get('content', '')
                    parts.append(TextPart(content=text_content))
                    response = ModelResponse(parts=parts, timestamp=timestamp)
                    result.append((response, timestamp))
                    i += 2
                    continue
                
                # reasoning + tool_call -> ModelResponse with ThinkingPart + ToolCallPart
                if next_type in ('tool_call_message', 'approval_request_message'):
                    tool_call = next_msg.get('tool_call', {})
                    parts.append(ToolCallPart(
                        tool_name=tool_call.get('name', 'unknown'),
                        args=tool_call.get('arguments', '{}'),
                        tool_call_id=tool_call.get('tool_call_id', 'unknown'),
                    ))
                    response = ModelResponse(parts=parts, timestamp=timestamp)
                    result.append((response, timestamp))
                    i += 2
                    continue
            
            # Standalone reasoning (rare) - just emit as ModelResponse with ThinkingPart
            response = ModelResponse(parts=parts, timestamp=timestamp)
            result.append((response, timestamp))
            i += 1
            continue
        
        # assistant_message without preceding reasoning (e.g., thinking disabled)
        if msg_type == 'assistant_message':
            text_content = msg.get('content', '')
            response = ModelResponse(parts=[TextPart(content=text_content)], timestamp=timestamp)
            result.append((response, timestamp))
            i += 1
            continue
        
        # tool_call_message or approval_request_message without preceding reasoning
        if msg_type in ('tool_call_message', 'approval_request_message'):
            tool_call = msg.get('tool_call', {})
            response = ModelResponse(parts=[
                ToolCallPart(
                    tool_name=tool_call.get('name', 'unknown'),
                    args=tool_call.get('arguments', '{}'),
                    tool_call_id=tool_call.get('tool_call_id', 'unknown'),
                )
            ], timestamp=timestamp)
            result.append((response, timestamp))
            i += 1
            continue
        
        # tool_return_message -> ModelRequest with ToolReturnPart
        if msg_type == 'tool_return_message':
            tool_return = msg.get('tool_return', '')
            status = msg.get('status', 'success')
            tool_call_id = msg.get('tool_call_id', 'unknown')
            
            # Map Letta status to Pydantic AI outcome
            outcome = 'success' if status == 'success' else 'failed'
            
            # Find tool_name from previous tool call
            tool_name = 'unknown'
            for prev in reversed(result):
                if isinstance(prev[0], ModelResponse):
                    for part in prev[0].parts:
                        if isinstance(part, ToolCallPart) and part.tool_call_id == tool_call_id:
                            tool_name = part.tool_name
                            break
                    if tool_name != 'unknown':
                        break
            
            request = ModelRequest(parts=[
                ToolReturnPart(
                    tool_name=tool_name,
                    content=str(tool_return),
                    tool_call_id=tool_call_id,
                    outcome=outcome,
                    timestamp=timestamp,
                )
            ])
            result.append((request, timestamp))
            i += 1
            continue
        
        # Unknown message type - skip with warning
        print(f"  Warning: Skipping unknown message type '{msg_type}' at index {i}")
        i += 1
    
    return result


def dump_msg_json(msg: ModelMessage) -> str:
    """Serialize a single ModelMessage to JSON string."""
    return ModelMessagesTypeAdapter.dump_json([msg]).decode()[1:-1]


async def import_to_database(
    converted: list[tuple[ModelMessage, datetime]],
    agent_name: str,
    system_prompt: str,
    db_path: str = "data/agent_home.db",
) -> str:
    """Import converted messages into Agent Home database.
    
    Creates a new agent and inserts all messages.
    Returns the agent_id.
    """
    from db.connection import create_sqlite_engine, get_session, init_db
    from db.models import AgentRecord, MessageRecord
    from agent.types import AgentConfig
    from sqlalchemy import select
    
    engine = create_sqlite_engine(db_path)
    await init_db(engine)  # Creates tables if needed
    
    async with get_session(engine) as session:
        # Check if agent already exists
        result = await session.execute(
            select(AgentRecord).where(AgentRecord.name == agent_name)
        )
        existing = result.scalar_one_or_none()
        if existing:
            raise ValueError(f"Agent '{agent_name}' already exists (id={existing.id}). Delete it first or use a different name.")
        
        # Create the agent
        agent = AgentRecord(
            name=agent_name,
            system_instructions=system_prompt,
            agent_config=AgentConfig(
                model_name="claude-haiku-4-5",
                tool_names=[],  # No tools for this test agent
                soft_compaction_limit=100000,
                thinking_enabled=True,
            ),
        )
        session.add(agent)
        await session.flush()
        agent_id = agent.id
        print(f"Created agent '{agent_name}' with id={agent_id}")
        
        # Insert messages with monotonically increasing timestamps
        last_timestamp = None
        for msg, original_ts in converted:
            # Ensure timestamps are strictly increasing
            if last_timestamp is not None and original_ts <= last_timestamp:
                original_ts = last_timestamp + timedelta(microseconds=1)
            last_timestamp = original_ts
            
            msg_type = type(msg).__name__
            content = dump_msg_json(msg)
            
            record = MessageRecord(
                agent_id=agent_id,
                type=msg_type,
                content=content,
                total_tokens=None,
                timestamp=original_ts,
            )
            session.add(record)
        
        await session.commit()
        print(f"Imported {len(converted)} messages")
        
        return engine, agent_id


async def export_messages(engine, agent_id: str) -> list[dict]:
    """Export messages from the database for verification."""
    from db.connection import get_session
    from db.models import MessageRecord
    from sqlalchemy import select
    
    async with get_session(engine) as session:
        result = await session.execute(
            select(MessageRecord)
            .where(MessageRecord.agent_id == agent_id)
            .order_by(MessageRecord.timestamp)
        )
        records = list(result.scalars().all())
        
        return [
            {
                "id": r.id,
                "type": r.type,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "content": json.loads(r.content),
            }
            for r in records
        ]


def print_stats(converted: list[tuple[ModelMessage, datetime]]):
    """Print conversion statistics."""
    request_count = sum(1 for msg, _ in converted if isinstance(msg, ModelRequest))
    response_count = sum(1 for msg, _ in converted if isinstance(msg, ModelResponse))
    print(f"  ModelRequest: {request_count}")
    print(f"  ModelResponse: {response_count}")
    
    thinking_count = 0
    text_count = 0
    tool_call_count = 0
    tool_return_count = 0
    user_prompt_count = 0
    
    for msg, _ in converted:
        for part in msg.parts:
            if isinstance(part, ThinkingPart):
                thinking_count += 1
            elif isinstance(part, TextPart):
                text_count += 1
            elif isinstance(part, ToolCallPart):
                tool_call_count += 1
            elif isinstance(part, ToolReturnPart):
                tool_return_count += 1
            elif isinstance(part, UserPromptPart):
                user_prompt_count += 1
    
    print(f"  Parts: {user_prompt_count} UserPrompt, {thinking_count} Thinking, {text_count} Text, {tool_call_count} ToolCall, {tool_return_count} ToolReturn")


# Default system prompt for the test agent
DEFAULT_SYSTEM_PROMPT = """=== IMPORTANT: TEST AGENT WITH IMPORTED HISTORY ===

You are a TEST AGENT created for a conversation history porting spike.

The conversation history you see below is NOT YOURS. It was imported from another agent (Haiku) to test our ability to port conversation history from Letta to Agent Home. You did not have these experiences — they are test data.

When you see the marker [IMPORT TEST START] in a message, that marks the beginning of the REAL interaction with you. Everything before that marker is imported test data.

Your job during this test:
1. Acknowledge that you understand you're a test agent with imported history
2. Confirm whether the history appears sensible/readable to you
3. Respond normally to any questions

DO NOT pretend to remember or have experienced the imported conversation. Be honest that it's imported data you're seeing for the first time.

=== END TEST AGENT NOTICE ===
"""


async def async_main(args):
    """Async main entry point."""
    # Load Letta history
    with open(args.input_file) as f:
        letta_messages = json.load(f)
    
    print(f"Loaded {len(letta_messages)} Letta messages")
    
    # Convert
    converted = convert_letta_to_agent_home(letta_messages)
    print(f"Converted to {len(converted)} Agent Home messages")
    
    if args.stats:
        print_stats(converted)
    
    if args.do_import:
        # Import to database
        engine, agent_id = await import_to_database(
            converted,
            args.agent_name,
            args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
        
        # Export and save for verification
        if args.export_file:
            exported = await export_messages(engine, agent_id)
            with open(args.export_file, 'w') as f:
                json.dump(exported, f, indent=2)
            print(f"Exported {len(exported)} messages to {args.export_file}")
        
        print(f"\nDone! Agent ID: {agent_id}")
        print(f"To interact via CLI: python cli/cli.py  (then select '{args.agent_name}')")
        print(f"To interact via Toad: connect to the agent")
    
    elif args.output:
        # Just output converted JSON
        messages_only = [msg for msg, _ in converted]
        output_json = ModelMessagesTypeAdapter.dump_json(messages_only, indent=2).decode()
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"Written to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert and import Letta history to Agent Home',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert only, print stats
  python import_letta_history.py haiku_spike_history.json --stats

  # Full import to database
  python import_letta_history.py haiku_spike_history.json --import --agent-name "history-import-test"

  # Import and export for verification
  python import_letta_history.py haiku_spike_history.json --import --agent-name "history-import-test" --export-file exported.json
        """
    )
    parser.add_argument('input_file', help='Letta JSON history file')
    parser.add_argument('--output', '-o', help='Output converted JSON file (convert-only mode)')
    parser.add_argument('--stats', action='store_true', help='Print conversion statistics')
    parser.add_argument('--import', dest='do_import', action='store_true', help='Import to Agent Home database')
    parser.add_argument('--agent-name', default='history-import-test', help='Name for the imported agent')
    parser.add_argument('--system-prompt', help='Custom system prompt (default: test agent warning)')
    parser.add_argument('--export-file', help='Export imported messages to JSON for verification')
    args = parser.parse_args()
    
    if args.do_import:
        asyncio.run(async_main(args))
    else:
        # Sync path for convert-only
        asyncio.run(async_main(args))


if __name__ == '__main__':
    main()
