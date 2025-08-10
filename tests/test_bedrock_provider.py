"""Tests for BedrockProvider."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError

from agentkit.core.model_interface import ModelError
from agentkit.models.bedrock_provider import BedrockProvider


class TestBedrockProvider:
    """Test BedrockProvider functionality."""
    
    def test_init_success(self):
        """Test successful BedrockProvider initialization."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            # Mock successful boto3 session creation
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            
            # Mock successful credentials test
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            assert provider.model_name == "claude-3-sonnet"
            assert provider.bedrock_model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert provider.region == "us-east-1"
    
    def test_init_custom_region(self):
        """Test BedrockProvider with custom region."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-haiku", region="us-west-2")
            
            assert provider.region == "us-west-2"
            mock_session.client.assert_called_with(
                service_name="bedrock-runtime",
                region_name="us-west-2"
            )
    
    def test_init_boto3_not_available(self):
        """Test initialization when boto3 is not available."""
        with patch('agentkit.models.bedrock_provider.BOTO3_AVAILABLE', False):
            with pytest.raises(ModelError) as exc_info:
                BedrockProvider("claude-3-sonnet")
            
            assert "boto3 not available" in str(exc_info.value)
    
    def test_init_unsupported_model(self):
        """Test initialization with unsupported model."""
        with patch('agentkit.models.bedrock_provider.boto3'):
            with pytest.raises(ModelError) as exc_info:
                BedrockProvider("gpt-4")
            
            assert "Unsupported Bedrock model" in str(exc_info.value)
    
    def test_init_no_credentials(self):
        """Test initialization with no AWS credentials."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_boto3.Session.side_effect = NoCredentialsError()
            
            with pytest.raises(ModelError) as exc_info:
                BedrockProvider("claude-3-sonnet")
            
            assert "AWS credentials not found" in str(exc_info.value)
    
    def test_init_insufficient_permissions(self):
        """Test initialization with insufficient permissions."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            
            # Mock permission error
            error_response = {
                'Error': {
                    'Code': 'UnauthorizedOperation',
                    'Message': 'Access denied'
                }
            }
            mock_client.list_foundation_models.side_effect = ClientError(
                error_response, 'ListFoundationModels'
            )
            
            with pytest.raises(ModelError) as exc_info:
                BedrockProvider("claude-3-sonnet")
            
            assert "Insufficient permissions" in str(exc_info.value)
    
    def test_prepare_request_body(self):
        """Test request body preparation."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            body = provider._prepare_request_body(
                system_prompt="You are helpful",
                task_prompt="Hello world",
                max_tokens=100
            )
            
            parsed_body = json.loads(body)
            
            assert parsed_body["anthropic_version"] == "bedrock-2023-05-31"
            assert parsed_body["max_tokens"] == 100
            assert parsed_body["system"] == "You are helpful"
            assert parsed_body["messages"][0]["role"] == "user"
            assert parsed_body["messages"][0]["content"] == "Hello world"
    
    def test_parse_response_success(self):
        """Test successful response parsing."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            response_data = {
                "content": [
                    {"text": "Hello! How can I help you?"}
                ]
            }
            response_bytes = json.dumps(response_data).encode('utf-8')
            
            result = provider._parse_response(response_bytes)
            assert result == "Hello! How can I help you?"
    
    def test_parse_response_legacy_format(self):
        """Test parsing response with legacy completion format."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            response_data = {
                "completion": "This is a legacy response format"
            }
            response_bytes = json.dumps(response_data).encode('utf-8')
            
            result = provider._parse_response(response_bytes)
            assert result == "This is a legacy response format"
    
    def test_parse_response_invalid_json(self):
        """Test response parsing with invalid JSON."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            with pytest.raises(ModelError) as exc_info:
                provider._parse_response(b"invalid json")
            
            assert "Failed to parse Bedrock response JSON" in str(exc_info.value)
    
    def test_parse_response_no_content(self):
        """Test response parsing with no content."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            response_data = {}
            response_bytes = json.dumps(response_data).encode('utf-8')
            
            with pytest.raises(ModelError) as exc_info:
                provider._parse_response(response_bytes)
            
            assert "No text content found in Bedrock response" in str(exc_info.value)
    
    def test_generate_success(self):
        """Test successful text generation."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            # Mock successful invoke_model response
            response_data = {
                "content": [
                    {"text": "Generated response text"}
                ]
            }
            mock_response = {
                'body': Mock()
            }
            mock_response['body'].read.return_value = json.dumps(response_data).encode('utf-8')
            mock_client.invoke_model.return_value = mock_response
            
            provider = BedrockProvider("claude-3-sonnet")
            
            result = provider.generate(
                system_prompt="You are helpful",
                task_prompt="Say hello",
                max_tokens=100
            )
            
            assert result == "Generated response text"
            
            # Verify the invoke_model call
            mock_client.invoke_model.assert_called_once()
            call_args = mock_client.invoke_model.call_args
            
            assert call_args[1]["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert call_args[1]["contentType"] == "application/json"
            assert call_args[1]["accept"] == "application/json"
            
            # Check request body
            body = json.loads(call_args[1]["body"])
            assert body["system"] == "You are helpful"
            assert body["messages"][0]["content"] == "Say hello"
            assert body["max_tokens"] == 100
    
    def test_generate_invalid_parameters(self):
        """Test generation with invalid parameters."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            # Test empty prompts
            with pytest.raises(ModelError) as exc_info:
                provider.generate("", "task", 100)
            assert "Both system_prompt and task_prompt are required" in str(exc_info.value)
            
            # Test invalid max_tokens
            with pytest.raises(ModelError) as exc_info:
                provider.generate("system", "task", 0)
            assert "max_tokens must be between 1 and 4096" in str(exc_info.value)
    
    def test_generate_throttling_retry(self):
        """Test generation with throttling and retry logic."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            with patch('agentkit.models.bedrock_provider.time.sleep'):  # Speed up tests
                mock_session = Mock()
                mock_client = Mock()
                mock_session.client.return_value = mock_client
                mock_boto3.Session.return_value = mock_session
                mock_client.list_foundation_models.return_value = {}
                
                provider = BedrockProvider("claude-3-sonnet")
                
                # Mock throttling error then success
                error_response = {
                    'Error': {
                        'Code': 'ThrottlingException',
                        'Message': 'Rate exceeded'
                    }
                }
                throttling_error = ClientError(error_response, 'InvokeModel')
                
                success_response = {
                    'body': Mock()
                }
                response_data = {
                    "content": [{"text": "Success after retry"}]
                }
                success_response['body'].read.return_value = json.dumps(response_data).encode('utf-8')
                
                mock_client.invoke_model.side_effect = [throttling_error, success_response]
                
                result = provider.generate("system", "task", 100)
                assert result == "Success after retry"
                assert mock_client.invoke_model.call_count == 2
    
    def test_generate_validation_error(self):
        """Test generation with validation error (no retry)."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            error_response = {
                'Error': {
                    'Code': 'ValidationException',
                    'Message': 'Invalid request parameters'
                }
            }
            validation_error = ClientError(error_response, 'InvokeModel')
            mock_client.invoke_model.side_effect = validation_error
            
            with pytest.raises(ModelError) as exc_info:
                provider.generate("system", "task", 100)
            
            assert "Invalid request" in str(exc_info.value)
            # Should not retry validation errors
            assert mock_client.invoke_model.call_count == 1
    
    def test_generate_access_denied_error(self):
        """Test generation with access denied error."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            error_response = {
                'Error': {
                    'Code': 'AccessDeniedException',
                    'Message': 'You do not have permission'
                }
            }
            access_error = ClientError(error_response, 'InvokeModel')
            mock_client.invoke_model.side_effect = access_error
            
            with pytest.raises(ModelError) as exc_info:
                provider.generate("system", "task", 100)
            
            assert "Access denied" in str(exc_info.value)
    
    def test_generate_model_not_found_error(self):
        """Test generation with model not found error."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            mock_session = Mock()
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            mock_boto3.Session.return_value = mock_session
            mock_client.list_foundation_models.return_value = {}
            
            provider = BedrockProvider("claude-3-sonnet")
            
            error_response = {
                'Error': {
                    'Code': 'ResourceNotFoundException',
                    'Message': 'Model not found'
                }
            }
            not_found_error = ClientError(error_response, 'InvokeModel')
            mock_client.invoke_model.side_effect = not_found_error
            
            with pytest.raises(ModelError) as exc_info:
                provider.generate("system", "task", 100)
            
            assert "Model not found" in str(exc_info.value)
    
    def test_generate_max_retries_exceeded(self):
        """Test generation when max retries are exceeded."""
        with patch('agentkit.models.bedrock_provider.boto3') as mock_boto3:
            with patch('agentkit.models.bedrock_provider.time.sleep'):  # Speed up tests
                mock_session = Mock()
                mock_client = Mock()
                mock_session.client.return_value = mock_client
                mock_boto3.Session.return_value = mock_session
                mock_client.list_foundation_models.return_value = {}
                
                provider = BedrockProvider("claude-3-sonnet")
                
                # Mock persistent throttling
                error_response = {
                    'Error': {
                        'Code': 'ThrottlingException',
                        'Message': 'Rate exceeded'
                    }
                }
                throttling_error = ClientError(error_response, 'InvokeModel')
                mock_client.invoke_model.side_effect = throttling_error
                
                with pytest.raises(ModelError) as exc_info:
                    provider.generate("system", "task", 100)
                
                assert "Rate limit exceeded after 3 attempts" in str(exc_info.value)
                assert mock_client.invoke_model.call_count == 3


class TestBedrockProviderModelMapping:
    """Test model name mapping functionality."""
    
    def test_all_models_mapped(self):
        """Test that all supported models have correct Bedrock mappings."""
        expected_mappings = {
            "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0", 
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0"
        }
        
        assert BedrockProvider.MODEL_MAPPING == expected_mappings
    
    def test_model_mapping_accuracy(self):
        """Test that model mappings match expected Bedrock model IDs."""
        # These should match AWS Bedrock's actual model IDs
        opus_id = BedrockProvider.MODEL_MAPPING["claude-3-opus"]
        sonnet_id = BedrockProvider.MODEL_MAPPING["claude-3-sonnet"]
        haiku_id = BedrockProvider.MODEL_MAPPING["claude-3-haiku"]
        
        assert opus_id.startswith("anthropic.claude-3-opus")
        assert sonnet_id.startswith("anthropic.claude-3-sonnet")
        assert haiku_id.startswith("anthropic.claude-3-haiku")
        
        assert all(model_id.endswith(":0") for model_id in BedrockProvider.MODEL_MAPPING.values())


if __name__ == "__main__":
    pytest.main([__file__])