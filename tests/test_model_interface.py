"""Tests for model interface and providers."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import json

from agentkit.core.model_interface import (
    ModelProvider,
    ClaudeProvider,
    ModelError,
    get_model_provider,
    get_supported_models,
)
from agentkit.core.config import Config


class MockAnthropicClient:
    """Mock Anthropic client for testing."""
    
    def __init__(self, response_text: str = "Test response", should_fail: bool = False):
        self.response_text = response_text
        self.should_fail = should_fail
        self.messages = Mock()
        self.messages.create = Mock(side_effect=self._create_message)
    
    def _create_message(self, **kwargs):
        """Mock message creation."""
        if self.should_fail:
            raise Exception("Mocked API failure")
        
        # Create mock response object
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = self.response_text
        mock_response.content = [mock_content]
        
        return mock_response


class TestModelProvider:
    """Test ModelProvider abstract base class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that ModelProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelProvider("test-model")


class TestClaudeProvider:
    """Test ClaudeProvider implementation."""
    
    def test_unsupported_model_raises_error(self):
        """Test that unsupported model names raise ModelError."""
        with pytest.raises(ModelError) as exc_info:
            ClaudeProvider("gpt-4")
        assert "Unsupported Claude model" in str(exc_info.value)
        assert "claude-3-opus" in str(exc_info.value)
    
    def test_supported_models_accepted(self):
        """Test that supported model names are accepted."""
        supported_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        
        for model in supported_models:
            with patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True), \
                 patch('agentkit.core.model_interface.anthropic.Anthropic') as mock_anthropic:
                mock_anthropic.return_value = MockAnthropicClient()
                
                with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
                    provider = ClaudeProvider(model)
                    assert provider.model_name == model
                    assert model in ClaudeProvider.MODEL_MAPPING
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', False)
    def test_missing_anthropic_sdk_raises_error(self):
        """Test that missing Anthropic SDK raises appropriate error."""
        with pytest.raises(ModelError) as exc_info:
            ClaudeProvider("claude-3-sonnet")
        assert "Anthropic SDK not available" in str(exc_info.value)
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    def test_missing_api_key_raises_error(self, mock_anthropic):
        """Test that missing API key raises appropriate error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ModelError) as exc_info:
                ClaudeProvider("claude-3-sonnet")
            assert "API key not found" in str(exc_info.value)
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    def test_api_key_from_environment(self, mock_anthropic):
        """Test that API key is loaded from environment variable."""
        mock_client = MockAnthropicClient()
        mock_anthropic.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-api-key'}):
            provider = ClaudeProvider("claude-3-sonnet")
            mock_anthropic.assert_called_once_with(api_key='test-api-key')
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.BOTO3_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    @patch('agentkit.core.model_interface.boto3.Session')
    def test_api_key_from_aws_secrets(self, mock_boto3_session, mock_anthropic):
        """Test that API key is loaded from AWS Secrets Manager."""
        # Mock boto3 session and secrets manager
        mock_session = Mock()
        mock_secrets_client = Mock()
        mock_session.client.return_value = mock_secrets_client
        mock_boto3_session.return_value = mock_session
        
        # Mock secret retrieval
        mock_secrets_client.get_secret_value.return_value = {
            'SecretString': json.dumps({'anthropic_api_key': 'aws-secret-key'})
        }
        
        mock_client = MockAnthropicClient()
        mock_anthropic.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_SECRET_ARN': 'arn:aws:secretsmanager:us-east-1:123456789012:secret:test'}):
            provider = ClaudeProvider("claude-3-sonnet")
            mock_anthropic.assert_called_once_with(api_key='aws-secret-key')
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    def test_generate_with_valid_inputs(self, mock_anthropic):
        """Test successful generation with valid inputs."""
        mock_client = MockAnthropicClient("Test response from Claude")
        mock_anthropic.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = ClaudeProvider("claude-3-sonnet")
            response = provider.generate(
                system_prompt="You are helpful",
                task_prompt="Say hello",
                max_tokens=100
            )
            
            assert response == "Test response from Claude"
            mock_client.messages.create.assert_called_once()
            
            # Verify call parameters
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs['model'] == "claude-3-sonnet-20240229"
            assert call_kwargs['max_tokens'] == 100
            assert call_kwargs['system'] == "You are helpful"
            assert len(call_kwargs['messages']) == 1
            assert call_kwargs['messages'][0]['role'] == 'user'
            assert call_kwargs['messages'][0]['content'] == "Say hello"
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    def test_generate_with_invalid_inputs(self, mock_anthropic):
        """Test generation with invalid inputs raises appropriate errors."""
        mock_client = MockAnthropicClient()
        mock_anthropic.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = ClaudeProvider("claude-3-sonnet")
            
            # Test empty prompts
            with pytest.raises(ModelError) as exc_info:
                provider.generate("", "task", 100)
            assert "required" in str(exc_info.value)
            
            with pytest.raises(ModelError) as exc_info:
                provider.generate("system", "", 100)
            assert "required" in str(exc_info.value)
            
            # Test invalid max_tokens
            with pytest.raises(ModelError) as exc_info:
                provider.generate("system", "task", 0)
            assert "must be between" in str(exc_info.value)
            
            with pytest.raises(ModelError) as exc_info:
                provider.generate("system", "task", 5000)
            assert "must be between" in str(exc_info.value)
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    def test_generate_with_api_failure(self, mock_anthropic):
        """Test generation handles API failures gracefully."""
        mock_client = MockAnthropicClient(should_fail=True)
        mock_anthropic.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = ClaudeProvider("claude-3-sonnet")
            
            with pytest.raises(ModelError) as exc_info:
                provider.generate("system", "task", 100)
            assert "Unexpected error during generation" in str(exc_info.value)
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    def test_rate_limit_retry(self, mock_anthropic):
        """Test retry logic for rate limit errors."""
        import anthropic
        
        # Mock client that fails with rate limit then succeeds
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Success after retry"
        mock_response.content = [mock_content]
        
        # Create proper mock response for the exception
        mock_error_response = Mock()
        mock_error_response.status_code = 429
        
        mock_client.messages.create.side_effect = [
            anthropic.RateLimitError("Rate limited", response=mock_error_response, body={}),
            mock_response
        ]
        mock_anthropic.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = ClaudeProvider("claude-3-sonnet")
            
            with patch('time.sleep'):  # Speed up test by mocking sleep
                response = provider.generate("system", "task", 100)
                
            assert response == "Success after retry"
            assert mock_client.messages.create.call_count == 2


class TestModelFactory:
    """Test model factory functions."""
    
    def test_get_model_provider_with_claude_models(self):
        """Test factory creates Claude providers for supported models."""
        claude_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        
        for model in claude_models:
            with patch('agentkit.core.model_interface.ClaudeProvider') as mock_provider:
                get_model_provider(model)
                mock_provider.assert_called_once_with(model, None)
    
    def test_get_model_provider_with_config(self):
        """Test factory passes config to providers."""
        config = Config()
        
        with patch('agentkit.core.model_interface.ClaudeProvider') as mock_provider:
            get_model_provider("claude-3-sonnet", config)
            mock_provider.assert_called_once_with("claude-3-sonnet", config)
    
    def test_get_model_provider_with_unsupported_model(self):
        """Test factory raises error for unsupported models."""
        with pytest.raises(ModelError) as exc_info:
            get_model_provider("gpt-4")
        assert "Unsupported model" in str(exc_info.value)
        assert "claude-3" in str(exc_info.value)
    
    def test_get_supported_models(self):
        """Test get_supported_models returns correct structure."""
        models = get_supported_models()
        
        assert isinstance(models, dict)
        assert "claude" in models
        assert isinstance(models["claude"], list)
        assert "claude-3-opus" in models["claude"]
        assert "claude-3-sonnet" in models["claude"]
        assert "claude-3-haiku" in models["claude"]


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    def test_long_prompts_handled_correctly(self, mock_anthropic):
        """Test that long prompts are handled correctly."""
        mock_client = MockAnthropicClient("Response to long prompt")
        mock_anthropic.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = ClaudeProvider("claude-3-sonnet")
            
            # Create long prompts
            long_system = "You are a helpful assistant. " * 100
            long_task = "Please help me with this complex task. " * 50
            
            response = provider.generate(
                system_prompt=long_system,
                task_prompt=long_task,
                max_tokens=500
            )
            
            assert response == "Response to long prompt"
            mock_client.messages.create.assert_called_once()
    
    @patch('agentkit.core.model_interface.ANTHROPIC_AVAILABLE', True)
    @patch('agentkit.core.model_interface.anthropic.Anthropic')
    def test_model_mapping_correct(self, mock_anthropic):
        """Test that model names are correctly mapped to API model names."""
        mock_client = MockAnthropicClient()
        mock_anthropic.return_value = mock_client
        
        test_cases = [
            ("claude-3-opus", "claude-3-opus-20240229"),
            ("claude-3-sonnet", "claude-3-sonnet-20240229"),
            ("claude-3-haiku", "claude-3-haiku-20240307")
        ]
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            for our_model, api_model in test_cases:
                provider = ClaudeProvider(our_model)
                provider.generate("system", "task", 100)
                
                call_kwargs = mock_client.messages.create.call_args[1]
                assert call_kwargs['model'] == api_model


if __name__ == "__main__":
    pytest.main([__file__])