"""Tool registry and base classes for AgentKit."""

from typing import Dict, Any, Callable

# Tool registry - will be populated in later prompts
TOOL_REGISTRY: Dict[str, Callable] = {}


class BaseTool:
    """Base class for AgentKit tools."""
    
    def __init__(self, name: str, description: str):
        """Initialize tool.
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
    
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution result
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Tool must implement execute method")


def register_tool(name: str, tool_func: Callable) -> None:
    """Register a tool in the global registry.
    
    Args:
        name: Tool name
        tool_func: Tool function to register
    """
    TOOL_REGISTRY[name] = tool_func


def get_tool(name: str) -> Callable:
    """Get a tool from the registry.
    
    Args:
        name: Tool name
        
    Returns:
        Tool function
        
    Raises:
        KeyError: If tool not found
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found in registry")
    return TOOL_REGISTRY[name]


def list_tools() -> Dict[str, Callable]:
    """List all registered tools.
    
    Returns:
        Dictionary of tool names to tool functions
    """
    return TOOL_REGISTRY.copy()