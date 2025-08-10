"""Tool registry for managing and discovering AgentKit tools."""

from typing import Dict, List, Type, Optional, Any
import inspect
from .base import BaseTool, ToolError
from ..core.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for managing AgentKit tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instances: Dict[str, BaseTool] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def register_tool(self, tool_cls: Type[BaseTool]) -> None:
        """Register a tool class in the registry.
        
        Args:
            tool_cls: Tool class to register (must inherit from BaseTool)
            
        Raises:
            ToolError: If tool is invalid or name conflicts exist
        """
        # Validate that it's a proper tool class
        if not inspect.isclass(tool_cls):
            raise ToolError(f"Expected tool class, got {type(tool_cls)}")
        
        if not issubclass(tool_cls, BaseTool):
            raise ToolError(f"Tool class {tool_cls.__name__} must inherit from BaseTool")
        
        # Create temporary instance to get name (tools must be instantiable)
        try:
            temp_instance = tool_cls()
            tool_name = temp_instance.name
        except Exception as e:
            raise ToolError(
                f"Failed to instantiate tool class {tool_cls.__name__}: {str(e)}",
                original_error=e
            )
        
        # Check for name conflicts
        if tool_name in self._tools:
            existing_cls = self._tools[tool_name]
            if existing_cls != tool_cls:
                raise ToolError(
                    f"Tool name conflict: '{tool_name}' is already registered "
                    f"by {existing_cls.__name__}"
                )
            else:
                # Same class, already registered
                self.logger.debug(f"Tool '{tool_name}' already registered")
                return
        
        # Register the tool
        self._tools[tool_name] = tool_cls
        self.logger.info(f"Registered tool: {tool_name} ({tool_cls.__name__})")
    
    def get_tool(self, name: str) -> BaseTool:
        """Get a tool instance by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance
            
        Raises:
            ToolError: If tool not found or instantiation fails
        """
        if name not in self._tools:
            available_tools = ", ".join(self._tools.keys())
            raise ToolError(
                f"Tool '{name}' not found. Available tools: {available_tools}"
            )
        
        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]
        
        # Create and cache new instance
        try:
            tool_cls = self._tools[name]
            instance = tool_cls()
            self._instances[name] = instance
            self.logger.debug(f"Created instance of tool '{name}'")
            return instance
            
        except Exception as e:
            raise ToolError(
                f"Failed to create instance of tool '{name}': {str(e)}",
                tool_name=name,
                original_error=e
            )
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool is registered
        """
        return name in self._tools
    
    def list_tools(self) -> List[str]:
        """Get list of registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Dictionary with tool information
            
        Raises:
            ToolError: If tool not found
        """
        tool = self.get_tool(name)
        return tool.get_tool_info()
    
    def get_all_tool_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered tools.
        
        Returns:
            Dictionary mapping tool names to tool information
        """
        result = {}
        for name in self._tools.keys():
            try:
                result[name] = self.get_tool_info(name)
            except Exception as e:
                self.logger.error(f"Failed to get info for tool '{name}': {str(e)}")
                result[name] = {
                    "name": name,
                    "description": f"Error loading tool: {str(e)}",
                    "parameters_schema": {}
                }
        return result
    
    def unregister_tool(self, name: str) -> None:
        """Unregister a tool by name.
        
        Args:
            name: Tool name to unregister
        """
        if name in self._tools:
            del self._tools[name]
            if name in self._instances:
                del self._instances[name]
            self.logger.info(f"Unregistered tool: {name}")
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._instances.clear()
        self.logger.info("Cleared all registered tools")
    
    def register_from_module(self, module: Any) -> int:
        """Register all BaseTool subclasses found in a module.
        
        Args:
            module: Python module to scan for tools
            
        Returns:
            Number of tools registered
        """
        registered_count = 0
        
        for name in dir(module):
            obj = getattr(module, name)
            
            # Check if it's a class that inherits from BaseTool
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseTool) and 
                obj != BaseTool):
                
                try:
                    self.register_tool(obj)
                    registered_count += 1
                except ToolError as e:
                    self.logger.warning(f"Failed to register tool {name}: {e.message}")
        
        if registered_count > 0:
            self.logger.info(f"Registered {registered_count} tools from module {module.__name__}")
        
        return registered_count
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool name is in registry."""
        return name in self._tools
    
    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools.keys())


# Global tool registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance.
    
    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


def register_tool(tool_cls: Type[BaseTool]) -> None:
    """Register a tool in the global registry.
    
    Args:
        tool_cls: Tool class to register
    """
    _global_registry.register_tool(tool_cls)


def get_tool(name: str) -> BaseTool:
    """Get a tool from the global registry.
    
    Args:
        name: Tool name
        
    Returns:
        Tool instance
    """
    return _global_registry.get_tool(name)


def list_tools() -> List[str]:
    """List all tools in the global registry.
    
    Returns:
        List of tool names
    """
    return _global_registry.list_tools()


def has_tool(name: str) -> bool:
    """Check if tool exists in global registry.
    
    Args:
        name: Tool name
        
    Returns:
        True if tool exists
    """
    return _global_registry.has_tool(name)