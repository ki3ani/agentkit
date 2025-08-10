"""Tool registry and base classes for AgentKit."""

from .base import BaseTool, ToolResult, ToolError
from .registry import (
    ToolRegistry,
    get_global_registry,
    register_tool,
    get_tool,
    list_tools,
    has_tool,
)

# Import and auto-register built-in tools
from .builtin import EchoTool, CalculatorTool, TextCountTool

# Auto-register built-in tools
_builtin_tools = [EchoTool, CalculatorTool, TextCountTool]

for tool_cls in _builtin_tools:
    try:
        register_tool(tool_cls)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to auto-register tool {tool_cls.__name__}: {e}")

__all__ = [
    "BaseTool",
    "ToolResult", 
    "ToolError",
    "ToolRegistry",
    "get_global_registry",
    "register_tool",
    "get_tool",
    "list_tools",
    "has_tool",
    # Built-in tools
    "EchoTool",
    "CalculatorTool", 
    "TextCountTool",
]