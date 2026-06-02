"""Quick test script to verify MCP connection and tool discovery.

Run the fs_proxy first in another terminal:
    uv run python -m mcp_tools.fs_proxy

Then run this test:
    uv run python -m mcp_tools.test_connection
"""
import asyncio

from pydantic_ai.mcp import MCPToolset


async def test_mcp_connection():
    """Test that we can connect to the MCP server and list tools."""
    server = MCPToolset("http://localhost:8080/mcp")
    
    print("Connecting to MCP filesystem server...")
    async with server:
        tools = await server.list_tools()
        print(f"\nDiscovered {len(tools)} tools:")
        for tool in tools:
            desc = tool.description or "(no description)"
            if len(desc) > 60:
                desc = desc[:60] + "..."
            print(f"  - {tool.name}: {desc}")
    
    print("\nMCP connection test passed!")


if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
