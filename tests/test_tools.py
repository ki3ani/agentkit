"""Tests for AgentKit tool system."""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any

from agentkit.tools.base import BaseTool, ToolResult, ToolError
from agentkit.tools.registry import ToolRegistry, get_global_registry
from agentkit.tools.builtin import EchoTool, CalculatorTool, TextCountTool
from agentkit.core.tool_executor import ToolExecutor
from agentkit.core.schema import normalize_tools_config


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    @property
    def name(self) -> str:
        return "mock_tool"
    
    @property
    def description(self) -> str:
        return "A mock tool for testing"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Test parameter"
                }
            },
            "required": ["param1"],
            "additionalProperties": False
        }
    
    def _execute(self, **kwargs: Any) -> str:
        return f"Mock result: {kwargs['param1']}"


class TestBaseTool:
    """Test BaseTool abstract base class."""
    
    def test_abstract_instantiation(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()
    
    def test_tool_implementation(self):
        """Test that tools can be properly implemented."""
        tool = MockTool()
        
        assert tool.name == "mock_tool"
        assert tool.description == "A mock tool for testing"
        assert "param1" in tool.parameters_schema["properties"]
    
    def test_parameter_validation_success(self):
        """Test successful parameter validation."""
        tool = MockTool()
        
        # Should not raise an exception
        tool.validate_parameters({"param1": "test_value"})
    
    def test_parameter_validation_failure(self):
        """Test parameter validation failure."""
        tool = MockTool()
        
        # Missing required parameter
        with pytest.raises(ToolError) as exc_info:
            tool.validate_parameters({})
        assert "Invalid parameters" in str(exc_info.value)
        assert tool.name in str(exc_info.value)
    
    def test_run_success(self):
        """Test successful tool execution."""
        tool = MockTool()
        result = tool.run(param1="test_value")
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.result == "Mock result: test_value"
        assert result.error is None
    
    def test_run_parameter_error(self):
        """Test tool execution with parameter error."""
        tool = MockTool()
        result = tool.run()  # Missing required parameter
        
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Invalid parameters" in result.error
    
    def test_tool_info(self):
        """Test tool info generation."""
        tool = MockTool()
        info = tool.get_tool_info()
        
        assert info["name"] == "mock_tool"
        assert info["description"] == "A mock tool for testing"
        assert "properties" in info["parameters_schema"]


class TestToolRegistry:
    """Test ToolRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        
        assert len(registry) == 0
        assert list(registry) == []
    
    def test_tool_registration(self):
        """Test tool registration."""
        registry = ToolRegistry()
        registry.register_tool(MockTool)
        
        assert "mock_tool" in registry
        assert len(registry) == 1
    
    def test_tool_retrieval(self):
        """Test tool instance retrieval."""
        registry = ToolRegistry()
        registry.register_tool(MockTool)
        
        tool = registry.get_tool("mock_tool")
        assert isinstance(tool, MockTool)
        
        # Should return same instance on second call
        tool2 = registry.get_tool("mock_tool")
        assert tool is tool2
    
    def test_tool_not_found(self):
        """Test retrieval of non-existent tool."""
        registry = ToolRegistry()
        
        with pytest.raises(ToolError) as exc_info:
            registry.get_tool("nonexistent")
        assert "not found" in str(exc_info.value)
    
    def test_duplicate_registration(self):
        """Test registration of duplicate tool names."""
        registry = ToolRegistry()
        registry.register_tool(MockTool)
        
        # Registering same class again should not raise error
        registry.register_tool(MockTool)
        assert len(registry) == 1
    
    def test_tool_list(self):
        """Test listing registered tools."""
        registry = ToolRegistry()
        registry.register_tool(MockTool)
        
        tools = registry.list_tools()
        assert tools == ["mock_tool"]
    
    def test_tool_info_retrieval(self):
        """Test tool info retrieval."""
        registry = ToolRegistry()
        registry.register_tool(MockTool)
        
        info = registry.get_tool_info("mock_tool")
        assert info["name"] == "mock_tool"
        
        all_info = registry.get_all_tool_info()
        assert "mock_tool" in all_info
    
    def test_tool_unregistration(self):
        """Test tool unregistration."""
        registry = ToolRegistry()
        registry.register_tool(MockTool)
        
        assert "mock_tool" in registry
        registry.unregister_tool("mock_tool")
        assert "mock_tool" not in registry
    
    def test_registry_clear(self):
        """Test clearing registry."""
        registry = ToolRegistry()
        registry.register_tool(MockTool)
        
        assert len(registry) == 1
        registry.clear()
        assert len(registry) == 0


class TestBuiltinTools:
    """Test built-in tool implementations."""
    
    def test_echo_tool(self):
        """Test EchoTool functionality."""
        tool = EchoTool()
        
        assert tool.name == "echo"
        assert "echo" in tool.description.lower()
        
        result = tool.run(text="Hello, World!")
        assert result.success is True
        assert result.result == "Hello, World!"
    
    def test_echo_tool_empty_text(self):
        """Test EchoTool with empty text."""
        tool = EchoTool()
        
        result = tool.run(text="")
        assert result.success is False
        assert "Invalid parameters" in result.error
    
    def test_calculator_tool_basic(self):
        """Test CalculatorTool basic operations."""
        tool = CalculatorTool()
        
        assert tool.name == "calculator"
        
        # Test basic arithmetic
        test_cases = [
            ("2 + 2", 4),
            ("10 - 3", 7),
            ("5 * 6", 30),
            ("15 / 3", 5.0),
            ("10 % 3", 1),
            ("2^3", 8),  # Power operator
            ("(2 + 3) * 4", 20),
        ]
        
        for expression, expected in test_cases:
            result = tool.run(expression=expression)
            assert result.success is True, f"Failed for expression: {expression}"
            assert result.result == expected, f"Expected {expected}, got {result.result} for {expression}"
    
    def test_calculator_tool_errors(self):
        """Test CalculatorTool error handling."""
        tool = CalculatorTool()
        
        # Division by zero
        result = tool.run(expression="10 / 0")
        assert result.success is False
        assert "Division by zero" in result.error
        
        # Invalid expression
        result = tool.run(expression="invalid")
        assert result.success is False
        assert "Invalid parameters" in result.error
        
        # Unsafe characters
        result = tool.run(expression="import os")
        assert result.success is False
        assert "Invalid parameters" in result.error
    
    def test_text_count_tool(self):
        """Test TextCountTool functionality."""
        tool = TextCountTool()
        
        assert tool.name == "text_count"
        
        test_text = "Hello world\nThis is a test\nWith multiple lines"
        
        # Test all counts
        result = tool.run(text=test_text)
        assert result.success is True
        assert isinstance(result.result, dict)
        assert result.result["words"] == 9  # "Hello", "world", "This", "is", "a", "test", "With", "multiple", "lines"
        assert result.result["lines"] == 3
        assert result.result["characters"] == len(test_text)
        
        # Test specific count types
        result = tool.run(text=test_text, count_type="words")
        assert result.success is True
        assert result.result == 9
        
        result = tool.run(text=test_text, count_type="lines")
        assert result.success is True
        assert result.result == 3


class TestToolExecutor:
    """Test ToolExecutor functionality."""
    
    def test_executor_initialization(self):
        """Test tool executor initialization."""
        tools_config = [
            {"name": "echo", "parameters": {}},
            {"name": "calculator", "parameters": {}}
        ]
        
        executor = ToolExecutor(tools_config)
        assert len(executor.get_available_tool_names()) == 2
        assert "echo" in executor.get_available_tool_names()
    
    def test_tools_context_generation(self):
        """Test tools context generation for prompts."""
        tools_config = [{"name": "echo", "parameters": {}}]
        executor = ToolExecutor(tools_config)
        
        context = executor.get_tools_context()
        assert "Available tools:" in context
        assert "echo" in context
        assert "JSON" in context
        assert "tool_call" in context
    
    def test_tool_call_extraction(self):
        """Test tool call extraction from text."""
        executor = ToolExecutor([{"name": "echo", "parameters": {}}])
        
        # Valid tool call
        text = 'Please echo this: {"tool_call": {"name": "echo", "parameters": {"text": "hello"}}}'
        tool_call = executor.extract_tool_call(text)
        
        assert tool_call is not None
        assert tool_call["name"] == "echo"
        assert tool_call["parameters"]["text"] == "hello"
        
        # No tool call
        text = "Just regular text with no tool calls"
        tool_call = executor.extract_tool_call(text)
        assert tool_call is None
    
    def test_tool_execution(self):
        """Test tool execution via executor."""
        tools_config = [{"name": "echo", "parameters": {}}]
        executor = ToolExecutor(tools_config)
        
        tool_call = {
            "name": "echo",
            "parameters": {"text": "test message"}
        }
        
        result = executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["result"] == "test message"
    
    def test_unavailable_tool_execution(self):
        """Test execution of unavailable tool."""
        tools_config = [{"name": "echo", "parameters": {}}]
        executor = ToolExecutor(tools_config)
        
        tool_call = {
            "name": "unavailable_tool",
            "parameters": {}
        }
        
        result = executor.execute_tool(tool_call)
        assert result["success"] is False
        assert "not available" in result["error"]
    
    def test_agent_response_processing(self):
        """Test processing agent response with tool calls."""
        tools_config = [{"name": "echo", "parameters": {}}]
        executor = ToolExecutor(tools_config)
        
        response = 'I will echo your message: {"tool_call": {"name": "echo", "parameters": {"text": "Hello!"}}}'
        
        final_response, tool_results = executor.process_agent_response(response)
        
        assert len(tool_results) == 1
        assert tool_results[0]["tool_call"]["name"] == "echo"
        assert tool_results[0]["result"]["success"] is True
        assert "[Tool Result: Hello!]" in final_response


class TestSchemaExtensions:
    """Test schema extensions for tools."""
    
    def test_normalize_tools_config_strings(self):
        """Test normalizing string tool names."""
        tools_config = ["echo", "calculator"]
        normalized = normalize_tools_config(tools_config)
        
        assert len(normalized) == 2
        assert normalized[0] == {"name": "echo", "parameters": {}}
        assert normalized[1] == {"name": "calculator", "parameters": {}}
    
    def test_normalize_tools_config_objects(self):
        """Test normalizing object tool configurations."""
        tools_config = [
            {"name": "calculator", "parameters": {"expression": "2+2"}},
            {"name": "echo"}  # Missing parameters
        ]
        
        normalized = normalize_tools_config(tools_config)
        
        assert len(normalized) == 2
        assert normalized[0]["parameters"]["expression"] == "2+2"
        assert normalized[1]["parameters"] == {}
    
    def test_normalize_tools_config_mixed(self):
        """Test normalizing mixed tool configurations."""
        tools_config = [
            "echo",
            {"name": "calculator", "parameters": {"expression": "5*5"}},
            "text_count"
        ]
        
        normalized = normalize_tools_config(tools_config)
        
        assert len(normalized) == 3
        assert normalized[0] == {"name": "echo", "parameters": {}}
        assert normalized[1]["parameters"]["expression"] == "5*5"
        assert normalized[2] == {"name": "text_count", "parameters": {}}


class TestGlobalRegistry:
    """Test global registry functionality."""
    
    def test_global_registry_access(self):
        """Test accessing global registry."""
        registry = get_global_registry()
        assert isinstance(registry, ToolRegistry)
        
        # Should return same instance
        registry2 = get_global_registry()
        assert registry is registry2
    
    def test_builtin_tools_registered(self):
        """Test that built-in tools are auto-registered."""
        from agentkit.tools import list_tools, has_tool
        
        tools = list_tools()
        
        # Should have built-in tools
        assert "echo" in tools
        assert "calculator" in tools
        assert "text_count" in tools
        
        # Test tool existence
        assert has_tool("echo")
        assert has_tool("calculator")
        assert not has_tool("nonexistent_tool")


if __name__ == "__main__":
    pytest.main([__file__])