"""YAML schema validation and parsing for AgentKit configuration files."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
import jsonschema
from jsonschema import validate, ValidationError
from rich.panel import Panel
from rich.console import Console

from .logger import get_logger

logger = get_logger(__name__)
console = Console()


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, details: str = None):
        """Initialize the exception.
        
        Args:
            message: Main error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details
    
    def display_error(self) -> None:
        """Display a rich error panel with the validation error."""
        error_text = f"[bold red]Configuration Validation Error[/bold red]\n\n"
        error_text += f"[red]{self.message}[/red]"
        
        if self.details:
            error_text += f"\n\n[yellow]Details:[/yellow]\n{self.details}"
        
        console.print(Panel.fit(
            error_text,
            title="❌ Invalid Agent Configuration",
            border_style="red"
        ))


# JSON Schema for AgentKit YAML configuration
AGENT_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "type": "object",
    "properties": {
        "agent": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Name of the agent"
                },
                "model": {
                    "type": "string",
                    "enum": [
                        "claude-3-opus",
                        "claude-3-sonnet", 
                        "claude-3-haiku"
                    ],
                    "description": "AI model to use for the agent"
                },
                "tools": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "default": [],
                    "description": "List of tools available to the agent"
                },
                "prompts": {
                    "type": "object",
                    "properties": {
                        "system": {
                            "type": "string",
                            "minLength": 1,
                            "description": "System prompt for the agent"
                        },
                        "task": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Task prompt for the agent"
                        }
                    },
                    "required": ["system", "task"],
                    "additionalProperties": False
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata for the agent",
                    "additionalProperties": True
                }
            },
            "required": ["name", "model", "prompts"],
            "additionalProperties": False
        }
    },
    "required": ["agent"],
    "additionalProperties": False
}


def load_and_validate_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate an agent configuration YAML file.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigValidationError: If the configuration is invalid
    """
    config_path = Path(path)
    
    # Check if file exists
    if not config_path.exists():
        raise ConfigValidationError(
            f"Configuration file not found: {config_path}",
            "Please ensure the file path is correct and the file exists."
        )
    
    # Load YAML file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(
            f"Invalid YAML syntax in {config_path}",
            f"YAML parsing error: {str(e)}"
        )
    except Exception as e:
        raise ConfigValidationError(
            f"Failed to read configuration file: {config_path}",
            f"Error: {str(e)}"
        )
    
    # Handle empty file
    if config_data is None:
        raise ConfigValidationError(
            f"Configuration file is empty: {config_path}",
            "The YAML file contains no data. Please add agent configuration."
        )
    
    # Validate against schema
    try:
        validate(instance=config_data, schema=AGENT_CONFIG_SCHEMA)
        logger.info(f"Successfully validated configuration: {config_path}")
        
        # Set default values for optional fields
        if "tools" not in config_data["agent"]:
            config_data["agent"]["tools"] = []
            
        return config_data
        
    except ValidationError as e:
        # Format the validation error for better user experience
        error_path = " -> ".join([str(p) for p in e.absolute_path]) if e.absolute_path else "root"
        
        # Create user-friendly error messages
        if "is not one of" in str(e.message):
            # Model validation error
            available_models = ", ".join(AGENT_CONFIG_SCHEMA["properties"]["agent"]["properties"]["model"]["enum"])
            details = f"Available models: {available_models}"
        elif "is a required property" in str(e.message):
            # Missing required field
            details = f"Missing required field at: {error_path}"
        else:
            # Generic validation error
            details = f"Validation error at {error_path}: {e.message}"
        
        raise ConfigValidationError(
            f"Schema validation failed for {config_path}",
            details
        )


def validate_config_dict(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a configuration dictionary against the schema.
    
    Args:
        config_data: Configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigValidationError: If the configuration is invalid
    """
    try:
        validate(instance=config_data, schema=AGENT_CONFIG_SCHEMA)
        
        # Set default values for optional fields
        if "tools" not in config_data["agent"]:
            config_data["agent"]["tools"] = []
            
        return config_data
        
    except ValidationError as e:
        error_path = " -> ".join([str(p) for p in e.absolute_path]) if e.absolute_path else "root"
        
        raise ConfigValidationError(
            "Schema validation failed for configuration data",
            f"Validation error at {error_path}: {e.message}"
        )


def get_available_models() -> list[str]:
    """Get list of available AI models.
    
    Returns:
        List of available model names
    """
    return AGENT_CONFIG_SCHEMA["properties"]["agent"]["properties"]["model"]["enum"]


def create_example_config() -> Dict[str, Any]:
    """Create an example valid configuration.
    
    Returns:
        Example configuration dictionary
    """
    return {
        "agent": {
            "name": "example-agent",
            "model": "claude-3-sonnet",
            "tools": ["web_search", "file_write"],
            "prompts": {
                "system": "You are a helpful AI assistant.",
                "task": "Help the user with their request."
            },
            "metadata": {
                "version": "1.0",
                "description": "Example agent configuration"
            }
        }
    }