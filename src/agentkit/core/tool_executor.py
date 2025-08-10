"""Tool execution and integration for AgentKit agents."""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from ..tools import get_tool, has_tool, ToolError
from .logger import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    """Handles tool execution and integration with agent responses."""
    
    def __init__(self, available_tools: List[Dict[str, Any]]):
        """Initialize tool executor with available tools.
        
        Args:
            available_tools: List of tool configurations with names and parameters
        """
        self.available_tools = {tool["name"]: tool for tool in available_tools}
        self.logger = get_logger(self.__class__.__name__)
        
        # Validate that all tools exist
        for tool_name in self.available_tools.keys():
            if not has_tool(tool_name):
                self.logger.warning(f"Tool '{tool_name}' not found in registry")
    
    def get_tools_context(self) -> str:
        """Generate context about available tools for the agent prompt.
        
        Returns:
            Formatted string describing available tools
        """
        if not self.available_tools:
            return "No tools are available."
        
        context = "Available tools:\n"
        
        for tool_name, tool_config in self.available_tools.items():
            if not has_tool(tool_name):
                continue
                
            try:
                tool = get_tool(tool_name)
                context += f"- {tool_name}: {tool.description}\n"
                
                # Add parameter information
                schema = tool.parameters_schema
                if schema.get("properties"):
                    params_info = []
                    for param, param_schema in schema["properties"].items():
                        param_desc = param_schema.get("description", "")
                        param_type = param_schema.get("type", "any")
                        param_info = f"{param} ({param_type})"
                        if param_desc:
                            param_info += f": {param_desc}"
                        params_info.append(param_info)
                    
                    context += f"  Parameters: {', '.join(params_info)}\n"
                
            except Exception as e:
                self.logger.error(f"Error getting info for tool '{tool_name}': {e}")
                context += f"- {tool_name}: Error loading tool\n"
        
        context += "\nTo use a tool, respond with JSON in this format:\n"
        context += '{"tool_call": {"name": "tool_name", "parameters": {"param1": "value1"}}}\n'
        context += "After using a tool, I will provide the result and you can continue your response.\n"
        
        return context
    
    def extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from agent response text.
        
        Args:
            text: Agent response text
            
        Returns:
            Tool call dictionary if found, None otherwise
        """
        # Look for JSON tool call in the response using a more robust approach
        # Find the start of a potential tool_call JSON
        start_pattern = r'\{"tool_call"\s*:'
        match = re.search(start_pattern, text, re.IGNORECASE)
        
        if not match:
            return None
        
        start_pos = match.start()
        
        # Find the matching closing brace by counting braces
        brace_count = 0
        json_end = -1
        
        for i, char in enumerate(text[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end == -1:
            return None
        
        try:
            # Extract and parse the JSON
            tool_call_json = text[start_pos:json_end]
            parsed = json.loads(tool_call_json)
            
            if "tool_call" in parsed:
                tool_call = parsed["tool_call"]
                
                # Validate required fields
                if "name" not in tool_call:
                    self.logger.warning("Tool call missing 'name' field")
                    return None
                
                if "parameters" not in tool_call:
                    tool_call["parameters"] = {}
                
                return tool_call
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse tool call JSON: {e}")
        
        return None
    
    def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call.
        
        Args:
            tool_call: Tool call dictionary with name and parameters
            
        Returns:
            Dictionary with execution result
        """
        tool_name = tool_call["name"]
        parameters = tool_call.get("parameters", {})
        
        self.logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
        
        # Check if tool is available
        if tool_name not in self.available_tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not available for this agent",
                "available_tools": list(self.available_tools.keys())
            }
        
        if not has_tool(tool_name):
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found in registry"
            }
        
        try:
            # Get tool instance and execute
            tool = get_tool(tool_name)
            result = tool.run(**parameters)
            
            return {
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "metadata": result.metadata
            }
            
        except ToolError as e:
            self.logger.error(f"Tool error executing '{tool_name}': {e.message}")
            return {
                "success": False,
                "error": e.message
            }
        
        except Exception as e:
            self.logger.error(f"Unexpected error executing tool '{tool_name}': {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def process_agent_response(self, response: str, max_tool_calls: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """Process agent response and execute any tool calls.
        
        Args:
            response: Agent response text
            max_tool_calls: Maximum number of tool calls to process
            
        Returns:
            Tuple of (final_response, tool_results)
        """
        tool_results = []
        current_response = response
        
        for i in range(max_tool_calls):
            # Extract tool call from current response
            tool_call = self.extract_tool_call(current_response)
            
            if not tool_call:
                # No more tool calls found
                break
            
            # Execute the tool
            tool_result = self.execute_tool(tool_call)
            tool_results.append({
                "tool_call": tool_call,
                "result": tool_result
            })
            
            # Remove the tool call JSON from response and add result
            # Use the same approach as extract_tool_call to find and remove the JSON
            start_pattern = r'\{"tool_call"\s*:'
            match = re.search(start_pattern, current_response, re.IGNORECASE)
            
            if match:
                start_pos = match.start()
                brace_count = 0
                json_end = -1
                
                for i, char in enumerate(current_response[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end != -1:
                    # Remove the JSON from the response
                    current_response = current_response[:start_pos] + current_response[json_end:]
            
            # Add tool result to response
            if tool_result["success"]:
                result_text = f"\n[Tool Result: {tool_result['result']}]\n"
            else:
                result_text = f"\n[Tool Error: {tool_result['error']}]\n"
            
            current_response = current_response.strip() + result_text
            
            self.logger.info(f"Tool call {i+1} executed: {tool_call['name']}")
        
        return current_response.strip(), tool_results
    
    def get_available_tool_names(self) -> List[str]:
        """Get list of available tool names.
        
        Returns:
            List of tool names
        """
        return list(self.available_tools.keys())