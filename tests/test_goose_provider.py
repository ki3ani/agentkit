"""Tests for GooseProvider."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import httpx

from agentkit.core.model_interface import ModelError
from agentkit.models.goose_provider import GooseProvider


class MockHTTPXResponse:
    """Mock httpx response for testing."""
    
    def __init__(self, status_code: int, json_data: dict = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
    
    def json(self):
        return self._json_data


class TestGooseProvider:
    """Test GooseProvider functionality."""
    
    def test_init_success(self):
        """Test successful GooseProvider initialization."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.get.return_value = MockHTTPXResponse(200)
            mock_client_class.return_value = mock_client
            
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-api-key'}):
                provider = GooseProvider("gpt-4")
                
                assert provider.model_name == "gpt-4"
                assert provider.goose_model_name == "gpt-4"
                assert provider.api_key == "test-api-key"
                assert provider.base_url == "https://api.goose.ai/v1"
    
    def test_init_with_custom_params(self):
        """Test GooseProvider with custom parameters."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.get.return_value = MockHTTPXResponse(200)
            mock_client_class.return_value = mock_client
            
            provider = GooseProvider(
                "claude-3-opus",
                api_key="custom-key",
                base_url="https://custom.api.com/v1"
            )
            
            assert provider.model_name == "claude-3-opus"
            assert provider.goose_model_name == "claude-3-opus-20240229"  # Mapped
            assert provider.api_key == "custom-key"
            assert provider.base_url == "https://custom.api.com/v1"
    
    def test_init_httpx_not_available(self):
        """Test initialization when httpx is not available."""
        with patch('agentkit.models.goose_provider.HTTPX_AVAILABLE', False):
            with pytest.raises(ModelError) as exc_info:
                GooseProvider("gpt-4")
            
            assert "httpx not available" in str(exc_info.value)
    
    def test_init_no_api_key(self):
        """Test initialization with no API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ModelError) as exc_info:
                GooseProvider("gpt-4")
            
            assert "Goose API key not found" in str(exc_info.value)
    
    def test_model_name_resolution_known_models(self):
        """Test model name resolution for known models."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("claude-3-sonnet")
                assert provider.goose_model_name == "claude-3-sonnet-20240229"
                
                provider = GooseProvider("gpt-4o")
                assert provider.goose_model_name == "gpt-4o"
    
    def test_model_name_resolution_unknown_models(self):
        """Test model name resolution for unknown models (pass-through)."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("custom-model-123")
                assert provider.goose_model_name == "custom-model-123"
    
    def test_connection_test_success(self):
        """Test successful connection test."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.get.return_value = MockHTTPXResponse(200)
            mock_client_class.return_value = mock_client
            
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                # If we get here without exception, connection test passed
                assert provider.client is not None
    
    def test_connection_test_invalid_api_key(self):
        """Test connection test with invalid API key."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.get.return_value = MockHTTPXResponse(401)
            mock_client_class.return_value = mock_client
            
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'invalid-key'}):
                with pytest.raises(ModelError) as exc_info:
                    GooseProvider("gpt-4")
                
                assert "Invalid Goose API key" in str(exc_info.value)
    
    def test_connection_test_timeout(self):
        """Test connection test with timeout."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client_class.return_value = mock_client
            
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                with pytest.raises(ModelError) as exc_info:
                    GooseProvider("gpt-4")
                
                assert "Timeout connecting to Goose API" in str(exc_info.value)
    
    def test_prepare_request_body(self):
        """Test request body preparation."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                body = provider._prepare_request_body(
                    system_prompt="You are helpful",
                    task_prompt="Hello world", 
                    max_tokens=100
                )
                
                assert body["model"] == "gpt-4"
                assert body["max_tokens"] == 100
                assert body["temperature"] == 0.7
                assert body["stream"] is False
                
                assert len(body["messages"]) == 2
                assert body["messages"][0]["role"] == "system"
                assert body["messages"][0]["content"] == "You are helpful"
                assert body["messages"][1]["role"] == "user"
                assert body["messages"][1]["content"] == "Hello world"
    
    def test_prepare_request_body_no_system_prompt(self):
        """Test request body preparation without system prompt."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                body = provider._prepare_request_body(
                    system_prompt="",
                    task_prompt="Hello world",
                    max_tokens=100
                )
                
                # Should only have user message, no system message
                assert len(body["messages"]) == 1
                assert body["messages"][0]["role"] == "user"
                assert body["messages"][0]["content"] == "Hello world"
    
    def test_parse_response_openai_format(self):
        """Test response parsing with OpenAI-compatible format."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                response_data = {
                    "choices": [
                        {
                            "message": {
                                "content": "Hello! How can I help you?"
                            }
                        }
                    ]
                }
                
                result = provider._parse_response(response_data)
                assert result == "Hello! How can I help you?"
    
    def test_parse_response_text_format(self):
        """Test response parsing with text format."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                response_data = {
                    "choices": [
                        {
                            "text": "This is a text response"
                        }
                    ]
                }
                
                result = provider._parse_response(response_data)
                assert result == "This is a text response"
    
    def test_parse_response_alternative_formats(self):
        """Test response parsing with alternative formats."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                # Test direct content format
                response_data = {"content": "Direct content response"}
                result = provider._parse_response(response_data)
                assert result == "Direct content response"
                
                # Test direct text format
                response_data = {"text": "Direct text response"}
                result = provider._parse_response(response_data)
                assert result == "Direct text response"
    
    def test_parse_response_no_content(self):
        """Test response parsing with no content."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                response_data = {}
                
                with pytest.raises(ModelError) as exc_info:
                    provider._parse_response(response_data)
                
                assert "No text content found in Goose response" in str(exc_info.value)
    
    def test_generate_success(self):
        """Test successful text generation."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            
            # Mock connection test
            mock_client.get.return_value = MockHTTPXResponse(200)
            
            # Mock successful generation
            mock_response = MockHTTPXResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Generated response text"
                            }
                        }
                    ]
                }
            )
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                result = provider.generate(
                    system_prompt="You are helpful",
                    task_prompt="Say hello",
                    max_tokens=100
                )
                
                assert result == "Generated response text"
                
                # Verify the API call
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                
                assert call_args[0][0] == "/chat/completions"
                assert "json" in call_args[1]
                
                request_body = call_args[1]["json"]
                assert request_body["model"] == "gpt-4"
                assert request_body["max_tokens"] == 100
    
    def test_generate_invalid_parameters(self):
        """Test generation with invalid parameters."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                # Test empty prompts
                with pytest.raises(ModelError) as exc_info:
                    provider.generate("", "task", 100)
                assert "Both system_prompt and task_prompt are required" in str(exc_info.value)
                
                # Test invalid max_tokens
                with pytest.raises(ModelError) as exc_info:
                    provider.generate("system", "task", 0)
                assert "max_tokens must be between 1 and 8192" in str(exc_info.value)
                
                with pytest.raises(ModelError) as exc_info:
                    provider.generate("system", "task", 10000)
                assert "max_tokens must be between 1 and 8192" in str(exc_info.value)
    
    def test_generate_rate_limiting_retry(self):
        """Test generation with rate limiting and retry logic."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            with patch('agentkit.models.goose_provider.time.sleep'):  # Speed up tests
                mock_client = Mock()
                mock_client.get.return_value = MockHTTPXResponse(200)
                
                # Mock rate limiting then success
                rate_limit_response = MockHTTPXResponse(429)
                success_response = MockHTTPXResponse(
                    200,
                    {
                        "choices": [
                            {"message": {"content": "Success after retry"}}
                        ]
                    }
                )
                
                mock_client.post.side_effect = [rate_limit_response, success_response]
                mock_client_class.return_value = mock_client
                
                with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                    provider = GooseProvider("gpt-4")
                    
                    result = provider.generate("system", "task", 100)
                    assert result == "Success after retry"
                    assert mock_client.post.call_count == 2
    
    def test_generate_bad_request(self):
        """Test generation with bad request error."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.get.return_value = MockHTTPXResponse(200)
            
            error_response = MockHTTPXResponse(
                400,
                {"error": {"message": "Invalid request parameters"}}
            )
            mock_client.post.return_value = error_response
            mock_client_class.return_value = mock_client
            
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("gpt-4")
                
                with pytest.raises(ModelError) as exc_info:
                    provider.generate("system", "task", 100)
                
                assert "Bad request" in str(exc_info.value)
                assert "Invalid request parameters" in str(exc_info.value)
    
    def test_generate_invalid_api_key(self):
        """Test generation with invalid API key."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.get.return_value = MockHTTPXResponse(200)
            
            auth_error_response = MockHTTPXResponse(401)
            mock_client.post.return_value = auth_error_response
            mock_client_class.return_value = mock_client
            
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'invalid-key'}):
                provider = GooseProvider("gpt-4")
                
                with pytest.raises(ModelError) as exc_info:
                    provider.generate("system", "task", 100)
                
                assert "Invalid Goose API key" in str(exc_info.value)
    
    def test_generate_model_not_found(self):
        """Test generation with model not found error."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.get.return_value = MockHTTPXResponse(200)
            
            not_found_response = MockHTTPXResponse(404)
            mock_client.post.return_value = not_found_response
            mock_client_class.return_value = mock_client
            
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                provider = GooseProvider("unknown-model")
                
                with pytest.raises(ModelError) as exc_info:
                    provider.generate("system", "task", 100)
                
                assert "Model 'unknown-model' not found" in str(exc_info.value)
    
    def test_generate_server_error_retry(self):
        """Test generation with server error and retry."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            with patch('agentkit.models.goose_provider.time.sleep'):
                mock_client = Mock()
                mock_client.get.return_value = MockHTTPXResponse(200)
                
                # Mock server error then success
                server_error_response = MockHTTPXResponse(500)
                success_response = MockHTTPXResponse(
                    200,
                    {"choices": [{"message": {"content": "Success after server error"}}]}
                )
                
                mock_client.post.side_effect = [server_error_response, success_response]
                mock_client_class.return_value = mock_client
                
                with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                    provider = GooseProvider("gpt-4")
                    
                    result = provider.generate("system", "task", 100)
                    assert result == "Success after server error"
                    assert mock_client.post.call_count == 2
    
    def test_generate_timeout_retry(self):
        """Test generation with timeout and retry."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            with patch('agentkit.models.goose_provider.time.sleep'):
                mock_client = Mock()
                mock_client.get.return_value = MockHTTPXResponse(200)
                
                # Mock timeout then success
                success_response = MockHTTPXResponse(
                    200,
                    {"choices": [{"message": {"content": "Success after timeout"}}]}
                )
                
                mock_client.post.side_effect = [
                    httpx.TimeoutException("Request timeout"),
                    success_response
                ]
                mock_client_class.return_value = mock_client
                
                with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                    provider = GooseProvider("gpt-4")
                    
                    result = provider.generate("system", "task", 100)
                    assert result == "Success after timeout"
                    assert mock_client.post.call_count == 2
    
    def test_generate_max_retries_exceeded(self):
        """Test generation when max retries are exceeded."""
        with patch('agentkit.models.goose_provider.httpx.Client') as mock_client_class:
            with patch('agentkit.models.goose_provider.time.sleep'):
                mock_client = Mock()
                mock_client.get.return_value = MockHTTPXResponse(200)
                
                # Mock persistent rate limiting
                rate_limit_response = MockHTTPXResponse(429)
                mock_client.post.return_value = rate_limit_response
                mock_client_class.return_value = mock_client
                
                with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                    provider = GooseProvider("gpt-4")
                    
                    with pytest.raises(ModelError) as exc_info:
                        provider.generate("system", "task", 100)
                    
                    assert "Rate limit exceeded after 3 attempts" in str(exc_info.value)
                    assert mock_client.post.call_count == 3


class TestGooseProviderModelMapping:
    """Test model name mapping functionality."""
    
    def test_common_models_mapping(self):
        """Test that common models have correct mappings."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                # Test OpenAI models
                provider = GooseProvider("gpt-4")
                assert provider.goose_model_name == "gpt-4"
                
                provider = GooseProvider("gpt-4o")
                assert provider.goose_model_name == "gpt-4o"
                
                # Test Claude models
                provider = GooseProvider("claude-3-opus")
                assert provider.goose_model_name == "claude-3-opus-20240229"
                
                provider = GooseProvider("claude-3-sonnet")
                assert provider.goose_model_name == "claude-3-sonnet-20240229"
    
    def test_model_agnostic_behavior(self):
        """Test that Goose provider is truly model agnostic."""
        with patch('agentkit.models.goose_provider.httpx.Client'):
            with patch.dict('os.environ', {'GOOSE_API_KEY': 'test-key'}):
                # Test custom/unknown models pass through
                provider = GooseProvider("my-custom-model-v2")
                assert provider.goose_model_name == "my-custom-model-v2"
                
                provider = GooseProvider("llama-3-instruct-custom")
                assert provider.goose_model_name == "llama-3-instruct-custom"
                
                provider = GooseProvider("experimental-model-123")
                assert provider.goose_model_name == "experimental-model-123"


if __name__ == "__main__":
    pytest.main([__file__])