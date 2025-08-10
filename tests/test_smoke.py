"""Smoke tests for AgentKit to verify basic functionality."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import os

from agentkit.cli import app
from agentkit import __version__
from agentkit.core.config import Config
from agentkit.core.logger import get_logger
from agentkit.tools import TOOL_REGISTRY, register_tool, get_tool, list_tools


class TestSmoke:
    """Basic smoke tests to verify AgentKit is working."""

    def test_version_import(self):
        """Test that version can be imported."""
        assert __version__
        assert isinstance(__version__, str)

    def test_config_initialization(self):
        """Test that Config can be initialized."""
        config = Config()
        assert config is not None

    def test_logger_creation(self):
        """Test that logger can be created."""
        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "test"

    def test_tool_registry_operations(self):
        """Test basic tool registry operations."""
        # Test initial state
        initial_tools = list_tools()
        assert isinstance(initial_tools, dict)
        
        # Test registering a tool
        def dummy_tool(param: str) -> str:
            return f"Result: {param}"
        
        register_tool("dummy", dummy_tool)
        
        # Test tool was registered
        assert "dummy" in TOOL_REGISTRY
        retrieved_tool = get_tool("dummy")
        assert retrieved_tool == dummy_tool
        
        # Test tool execution
        result = retrieved_tool("test")
        assert result == "Result: test"
        
        # Test getting non-existent tool raises error
        with pytest.raises(KeyError):
            get_tool("non_existent_tool")

    def test_cli_version_command(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


    def test_cli_run_command_placeholder(self):
        """Test CLI run command with mocked model provider."""
        from unittest.mock import Mock, patch
        runner = CliRunner()
        
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
agent:
  name: "test-agent"
  model: "claude-3-sonnet"
  tools: []
  prompts:
    system: "You are a helpful assistant"
    task: "Help the user"
""")
            yaml_path = f.name
        
        try:
            # Mock the model provider to return a test response
            mock_provider = Mock()
            mock_provider.generate.return_value = "Test response from mock model"
            
            with patch('agentkit.cli.get_model_provider', return_value=mock_provider):
                result = runner.invoke(app, [
                    "run", 
                    yaml_path, 
                    "--input", "test query",
                    "--format", "text"
                ])
                
            assert result.exit_code == 0
            assert "test-agent" in result.stdout
            assert "Test response from mock model" in result.stdout
            
            # Verify the mock was called correctly
            mock_provider.generate.assert_called_once()
            call_args = mock_provider.generate.call_args
            assert "You are a helpful assistant" in call_args[1]['system_prompt']
            assert "test query" in call_args[1]['task_prompt']
            
        finally:
            # Clean up temporary file
            os.unlink(yaml_path)

    def test_config_api_key_methods(self):
        """Test Config API key retrieval methods."""
        config = Config()
        
        # Test getting API keys (should return None if not set)
        anthropic_key = config.get_api_key("anthropic")
        openai_key = config.get_api_key("openai")
        mistral_key = config.get_api_key("mistral")
        
        # Should return None or actual key if set in environment
        assert anthropic_key is None or isinstance(anthropic_key, str)
        assert openai_key is None or isinstance(openai_key, str)
        assert mistral_key is None or isinstance(mistral_key, str)

    def test_config_aws_methods(self):
        """Test Config AWS configuration methods."""
        config = Config()
        aws_config = config.get_aws_config()
        
        assert isinstance(aws_config, dict)
        assert "region" in aws_config
        assert "access_key" in aws_config
        assert "secret_key" in aws_config
        assert "session_token" in aws_config
        
        # Should have default region
        assert aws_config["region"] in ["us-east-1", None] or isinstance(aws_config["region"], str)

    def test_project_structure_exists(self):
        """Test that all expected project structure exists."""
        src_dir = Path("src/agentkit")
        assert src_dir.exists()
        assert (src_dir / "__init__.py").exists()
        assert (src_dir / "__main__.py").exists()
        assert (src_dir / "cli.py").exists()
        assert (src_dir / "core").exists()
        assert (src_dir / "core" / "__init__.py").exists()
        assert (src_dir / "core" / "config.py").exists()
        assert (src_dir / "core" / "logger.py").exists()
        assert (src_dir / "tools").exists()
        assert (src_dir / "tools" / "__init__.py").exists()


if __name__ == "__main__":
    pytest.main([__file__])