"""Tests for YAML schema validation and parsing."""

import pytest
import tempfile
import os
from pathlib import Path
import yaml

from agentkit.core.schema import (
    load_and_validate_config,
    validate_config_dict,
    ConfigValidationError,
    get_available_models,
    create_example_config,
    AGENT_CONFIG_SCHEMA
)


class TestSchemaValidation:
    """Test schema validation functionality."""

    def test_valid_complete_config(self):
        """Test that a valid complete configuration passes validation."""
        valid_config = {
            "agent": {
                "name": "test-agent",
                "model": "claude-3-sonnet",
                "tools": ["web_search", "file_write"],
                "prompts": {
                    "system": "You are a helpful assistant",
                    "task": "Help the user"
                },
                "metadata": {
                    "version": "1.0",
                    "description": "Test agent"
                }
            }
        }
        
        result = validate_config_dict(valid_config)
        assert result == valid_config
        assert result["agent"]["name"] == "test-agent"
        assert result["agent"]["model"] == "claude-3-sonnet"
        assert "web_search" in result["agent"]["tools"]

    def test_valid_minimal_config(self):
        """Test that a minimal valid configuration passes validation."""
        minimal_config = {
            "agent": {
                "name": "minimal-agent",
                "model": "claude-3-haiku",
                "prompts": {
                    "system": "System prompt",
                    "task": "Task prompt"
                }
            }
        }
        
        result = validate_config_dict(minimal_config)
        assert result["agent"]["name"] == "minimal-agent"
        assert result["agent"]["model"] == "claude-3-haiku"
        assert result["agent"]["tools"] == []  # Default empty list

    def test_missing_required_field_agent(self):
        """Test that missing 'agent' field fails validation."""
        invalid_config = {
            "not_agent": {
                "name": "test"
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict(invalid_config)
        assert "required property" in str(exc_info.value.details).lower()

    def test_missing_required_field_name(self):
        """Test that missing 'name' field fails validation."""
        invalid_config = {
            "agent": {
                "model": "claude-3-sonnet",
                "prompts": {
                    "system": "System prompt",
                    "task": "Task prompt"
                }
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict(invalid_config)
        assert "required property" in str(exc_info.value.details).lower()

    def test_missing_required_field_model(self):
        """Test that missing 'model' field fails validation."""
        invalid_config = {
            "agent": {
                "name": "test-agent",
                "prompts": {
                    "system": "System prompt",
                    "task": "Task prompt"
                }
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict(invalid_config)
        assert "required property" in str(exc_info.value.details).lower()

    def test_missing_required_field_prompts(self):
        """Test that missing 'prompts' field fails validation."""
        invalid_config = {
            "agent": {
                "name": "test-agent",
                "model": "claude-3-sonnet"
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict(invalid_config)
        assert "required property" in str(exc_info.value.details).lower()

    def test_missing_required_prompt_fields(self):
        """Test that missing system or task prompts fail validation."""
        # Missing system prompt
        invalid_config_1 = {
            "agent": {
                "name": "test-agent",
                "model": "claude-3-sonnet",
                "prompts": {
                    "task": "Task prompt"
                }
            }
        }
        
        with pytest.raises(ConfigValidationError):
            validate_config_dict(invalid_config_1)
            
        # Missing task prompt
        invalid_config_2 = {
            "agent": {
                "name": "test-agent",
                "model": "claude-3-sonnet",
                "prompts": {
                    "system": "System prompt"
                }
            }
        }
        
        with pytest.raises(ConfigValidationError):
            validate_config_dict(invalid_config_2)

    def test_invalid_model_name(self):
        """Test that invalid model names fail validation."""
        invalid_config = {
            "agent": {
                "name": "test-agent",
                "model": "invalid-model",
                "prompts": {
                    "system": "System prompt",
                    "task": "Task prompt"
                }
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict(invalid_config)
        assert "is not one of" in str(exc_info.value.details) or "available models" in str(exc_info.value.details).lower()

    def test_valid_model_names(self):
        """Test that all valid model names pass validation."""
        valid_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        
        for model in valid_models:
            config = {
                "agent": {
                    "name": "test-agent",
                    "model": model,
                    "prompts": {
                        "system": "System prompt",
                        "task": "Task prompt"
                    }
                }
            }
            
            result = validate_config_dict(config)
            assert result["agent"]["model"] == model

    def test_tools_validation(self):
        """Test that tools field accepts array of strings."""
        valid_config = {
            "agent": {
                "name": "test-agent",
                "model": "claude-3-sonnet",
                "tools": ["web_search", "file_read", "api_call"],
                "prompts": {
                    "system": "System prompt",
                    "task": "Task prompt"
                }
            }
        }
        
        result = validate_config_dict(valid_config)
        assert result["agent"]["tools"] == ["web_search", "file_read", "api_call"]

    def test_empty_string_fields(self):
        """Test that empty string fields fail validation."""
        invalid_configs = [
            {
                "agent": {
                    "name": "",  # Empty name
                    "model": "claude-3-sonnet",
                    "prompts": {"system": "System", "task": "Task"}
                }
            },
            {
                "agent": {
                    "name": "test",
                    "model": "claude-3-sonnet",
                    "prompts": {"system": "", "task": "Task"}  # Empty system
                }
            },
            {
                "agent": {
                    "name": "test",
                    "model": "claude-3-sonnet",
                    "prompts": {"system": "System", "task": ""}  # Empty task
                }
            }
        ]
        
        for config in invalid_configs:
            with pytest.raises(ConfigValidationError):
                validate_config_dict(config)

    def test_additional_properties_not_allowed(self):
        """Test that additional properties in agent fail validation."""
        invalid_config = {
            "agent": {
                "name": "test-agent",
                "model": "claude-3-sonnet",
                "prompts": {"system": "System", "task": "Task"},
                "invalid_field": "should not be allowed"  # Additional property
            }
        }
        
        with pytest.raises(ConfigValidationError):
            validate_config_dict(invalid_config)


class TestFileLoading:
    """Test file loading and validation."""

    def create_temp_yaml_file(self, content: dict) -> str:
        """Helper to create temporary YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(content, f)
            return f.name

    def test_load_valid_yaml_file(self):
        """Test loading a valid YAML file."""
        valid_config = {
            "agent": {
                "name": "file-agent",
                "model": "claude-3-opus",
                "tools": ["web_search"],
                "prompts": {
                    "system": "You are helpful",
                    "task": "Help the user"
                }
            }
        }
        
        yaml_file = self.create_temp_yaml_file(valid_config)
        
        try:
            result = load_and_validate_config(yaml_file)
            assert result["agent"]["name"] == "file-agent"
            assert result["agent"]["model"] == "claude-3-opus"
        finally:
            os.unlink(yaml_file)

    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises appropriate error."""
        with pytest.raises(ConfigValidationError) as exc_info:
            load_and_validate_config("nonexistent-file.yaml")
        assert "not found" in str(exc_info.value.message).lower()

    def test_load_invalid_yaml_syntax(self):
        """Test that invalid YAML syntax raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:\n  - broken\n    - syntax")
            yaml_file = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                load_and_validate_config(yaml_file)
            assert "yaml" in str(exc_info.value.message).lower()
        finally:
            os.unlink(yaml_file)

    def test_load_empty_file(self):
        """Test that empty YAML file raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            yaml_file = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                load_and_validate_config(yaml_file)
            assert "empty" in str(exc_info.value.message).lower()
        finally:
            os.unlink(yaml_file)

    def test_load_invalid_config_from_file(self):
        """Test that loading invalid config from file raises validation error."""
        invalid_config = {
            "agent": {
                "name": "test",
                "model": "invalid-model-name",
                "prompts": {"system": "System", "task": "Task"}
            }
        }
        
        yaml_file = self.create_temp_yaml_file(invalid_config)
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                load_and_validate_config(yaml_file)
            assert "validation failed" in str(exc_info.value.message).lower()
        finally:
            os.unlink(yaml_file)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_available_models(self):
        """Test that get_available_models returns correct models."""
        models = get_available_models()
        expected_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        assert models == expected_models

    def test_create_example_config(self):
        """Test that create_example_config returns valid config."""
        example_config = create_example_config()
        
        # Should be able to validate without errors
        result = validate_config_dict(example_config)
        assert result["agent"]["name"] == "example-agent"
        assert result["agent"]["model"] in get_available_models()


class TestConfigValidationError:
    """Test ConfigValidationError exception class."""

    def test_config_validation_error_creation(self):
        """Test creating ConfigValidationError with message and details."""
        error = ConfigValidationError("Main message", "Additional details")
        assert error.message == "Main message"
        assert error.details == "Additional details"
        assert str(error) == "Main message"

    def test_config_validation_error_without_details(self):
        """Test creating ConfigValidationError without details."""
        error = ConfigValidationError("Main message only")
        assert error.message == "Main message only"
        assert error.details is None


if __name__ == "__main__":
    pytest.main([__file__])