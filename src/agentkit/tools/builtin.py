"""Built-in tools for AgentKit."""

import ast
import operator
import re
from typing import Dict, Any, Union
from .base import BaseTool, ToolError


class EchoTool(BaseTool):
    """Simple tool that echoes back the input text - useful for testing."""
    
    @property
    def name(self) -> str:
        return "echo"
    
    @property
    def description(self) -> str:
        return "Echoes back the provided text - useful for testing tool functionality"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to echo back",
                    "minLength": 1
                }
            },
            "required": ["text"],
            "additionalProperties": False
        }
    
    def _execute(self, **kwargs: Any) -> str:
        """Echo the input text back."""
        text = kwargs["text"]
        self.logger.debug(f"Echoing text: {text}")
        return text


class CalculatorTool(BaseTool):
    """Safe arithmetic calculator that evaluates mathematical expressions."""
    
    # Allowed operators for safe evaluation
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Performs safe arithmetic calculations from mathematical expressions"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 3.5')",
                    "minLength": 1,
                    "pattern": r"^[\d\s\+\-\*\/\(\)\.\%\^]+$"
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    
    def _execute(self, **kwargs: Any) -> Union[int, float]:
        """Safely evaluate a mathematical expression."""
        expression = kwargs["expression"].strip()
        self.logger.debug(f"Evaluating expression: {expression}")
        
        # Basic input validation
        if not expression:
            raise ToolError("Empty expression provided", tool_name=self.name)
        
        # Additional safety check - only allow basic math characters
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.\%\^]+$', expression):
            raise ToolError(
                "Expression contains invalid characters. Only numbers, +, -, *, /, %, ^, (, ), and spaces are allowed.",
                tool_name=self.name
            )
        
        # Replace ^ with ** for Python power operator
        expression = expression.replace('^', '**')
        
        try:
            # Parse the expression into an AST
            parsed = ast.parse(expression, mode='eval')
            
            # Evaluate safely using our restricted operator set
            result = self._safe_eval(parsed.body)
            
            self.logger.debug(f"Expression result: {result}")
            return result
            
        except SyntaxError as e:
            raise ToolError(f"Invalid mathematical expression: {str(e)}", tool_name=self.name)
        except ZeroDivisionError:
            raise ToolError("Division by zero", tool_name=self.name)
        except OverflowError:
            raise ToolError("Calculation resulted in overflow", tool_name=self.name)
        except Exception as e:
            raise ToolError(f"Calculation error: {str(e)}", tool_name=self.name)
    
    def _safe_eval(self, node: ast.AST) -> Union[int, float]:
        """Safely evaluate an AST node using only allowed operations.
        
        Args:
            node: AST node to evaluate
            
        Returns:
            Numeric result
            
        Raises:
            ToolError: If unsupported operations are used
        """
        if isinstance(node, ast.Constant):
            # Python 3.8+ uses ast.Constant for literals
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ToolError(f"Unsupported literal type: {type(node.value)}", tool_name=self.name)
        
        elif isinstance(node, ast.Num):
            # Fallback for older Python versions
            return node.n
        
        elif isinstance(node, ast.BinOp):
            # Binary operations (+, -, *, /, etc.)
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            op_type = type(node.op)
            
            if op_type not in self.ALLOWED_OPERATORS:
                raise ToolError(f"Unsupported operation: {op_type.__name__}", tool_name=self.name)
            
            return self.ALLOWED_OPERATORS[op_type](left, right)
        
        elif isinstance(node, ast.UnaryOp):
            # Unary operations (+, -)
            operand = self._safe_eval(node.operand)
            op_type = type(node.op)
            
            if op_type not in self.ALLOWED_OPERATORS:
                raise ToolError(f"Unsupported unary operation: {op_type.__name__}", tool_name=self.name)
            
            return self.ALLOWED_OPERATORS[op_type](operand)
        
        else:
            raise ToolError(f"Unsupported expression type: {type(node).__name__}", tool_name=self.name)


class TextCountTool(BaseTool):
    """Tool for counting characters, words, and lines in text."""
    
    @property
    def name(self) -> str:
        return "text_count"
    
    @property
    def description(self) -> str:
        return "Counts characters, words, and lines in provided text"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to analyze"
                },
                "count_type": {
                    "type": "string",
                    "enum": ["characters", "words", "lines", "all"],
                    "description": "Type of count to perform",
                    "default": "all"
                }
            },
            "required": ["text"],
            "additionalProperties": False
        }
    
    def _execute(self, **kwargs: Any) -> Union[int, Dict[str, int]]:
        """Count text statistics."""
        text = kwargs["text"]
        count_type = kwargs.get("count_type", "all")
        
        self.logger.debug(f"Counting {count_type} in text of length {len(text)}")
        
        # Calculate all counts
        char_count = len(text)
        word_count = len(text.split()) if text.strip() else 0
        line_count = len(text.splitlines()) if text else 0
        
        if count_type == "characters":
            return char_count
        elif count_type == "words":
            return word_count
        elif count_type == "lines":
            return line_count
        else:  # "all"
            return {
                "characters": char_count,
                "words": word_count,
                "lines": line_count
            }