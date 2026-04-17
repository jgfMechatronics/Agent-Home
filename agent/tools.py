"""
Agent tool registry — Section 3.2

Maps tool name strings to callable tool functions for agent construction.
"""
from typing import Callable


def get_tools_for_agent(tool_names: list[str]) -> list[Callable]:
    """Return the list of tool callables for the given tool names.
    
    TODO: Section 3.2 — implement tool registry and lookup.
    """
    raise NotImplementedError
