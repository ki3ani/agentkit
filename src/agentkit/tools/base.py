"""Base classes and interfaces for AgentKit tools."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import jsonschema
from jsonschema import validate, ValidationError

from ..core.logger import get_logger

logger = get_logger(__name__)


class ToolError(Exception):
    """Exception raised for tool-related errors."""
    
    def __init__(self, message: str, tool_name: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.original_error = original_error


class ToolResult:
    """Represents the result of a tool execution."""
    
    def __init__(
        self, 
        success: bool, 
        result: Any = None, 
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize tool result.
        
        Args:
            success: Whether the tool execution was successful
            result: The result data if successful
            error: Error message if unsuccessful
            metadata: Additional metadata about the execution
        """
        self.success = success
        self.result = result
        self.error = error
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the result."""
        if self.success:
            return f"Success: {self.result}"
        else:
            return f"Error: {self.error}"


class BaseTool(ABC):
    """Abstract base class for all AgentKit tools."""
    
    def __init__(self):
        """Initialize the tool."""
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier - must be unique across all tools."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Short human-readable description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema defining the parameters this tool accepts."""
        pass
    
    @abstractmethod
    def _execute(self, **kwargs: Any) -> Any:
        """Execute the tool logic - to be implemented by subclasses.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ToolError: If execution fails
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters against the tool's schema.
        
        Args:
            parameters: Parameters to validate
            
        Raises:
            ToolError: If parameters are invalid
        """
        try:
            validate(instance=parameters, schema=self.parameters_schema)
        except ValidationError as e:
            raise ToolError(
                f"Invalid parameters for tool '{self.name}': {e.message}",
                tool_name=self.name,
                original_error=e
            )
    
    def run(self, **kwargs: Any) -> ToolResult:
        """Run the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult containing success status and result/error
        """
        try:
            self.logger.info(f"Executing tool '{self.name}' with parameters: {kwargs}")
            
            # Validate parameters
            self.validate_parameters(kwargs)
            
            # Execute tool logic
            result = self._execute(**kwargs)
            
            self.logger.info(f"Tool '{self.name}' executed successfully")
            return ToolResult(success=True, result=result)
            
        except ToolError as e:
            self.logger.error(f"Tool '{self.name}' failed: {e.message}")
            return ToolResult(success=False, error=e.message)
            
        except Exception as e:
            error_msg = f"Unexpected error in tool '{self.name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return ToolResult(success=False, error=error_msg)
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the tool.
        
        Returns:
            Dictionary containing tool name, description, and schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters_schema": self.parameters_schema
        }
    
    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        """Developer representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"