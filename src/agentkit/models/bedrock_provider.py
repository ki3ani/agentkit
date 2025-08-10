"""AWS Bedrock model provider for AgentKit."""

import json
import time
import os
from typing import Optional, Dict, Any

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from ..core.model_interface import ModelProvider, ModelError
from ..core.logger import get_logger
from ..core.config import Config

logger = get_logger(__name__)


class BedrockProvider(ModelProvider):
    """AWS Bedrock model provider for Claude models."""
    
    # Mapping from our model names to Bedrock model IDs
    MODEL_MAPPING = {
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0"
    }
    
    def __init__(self, model_name: str, region: Optional[str] = None, config: Optional[Config] = None):
        """Initialize Bedrock provider.
        
        Args:
            model_name: Name of the Claude model to use
            region: AWS region (defaults to us-east-1)
            config: Optional configuration instance
            
        Raises:
            ModelError: If boto3 is not available or model is unsupported
        """
        super().__init__(model_name)
        
        if not BOTO3_AVAILABLE:
            raise ModelError(
                "boto3 not available. Install with: pip install boto3"
            )
        
        if model_name not in self.MODEL_MAPPING:
            available = ", ".join(self.MODEL_MAPPING.keys())
            raise ModelError(
                f"Unsupported Bedrock model: {model_name}. "
                f"Available models: {available}"
            )
        
        self.config = config or Config()
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.bedrock_model_id = self.MODEL_MAPPING[model_name]
        self.client = self._create_client()
    
    def _create_client(self) -> Any:
        """Create Bedrock Runtime client with AWS credentials.
        
        Returns:
            Configured boto3 Bedrock Runtime client
            
        Raises:
            ModelError: If client creation fails
        """
        try:
            # Let boto3 handle credential discovery automatically
            # It will check environment variables, AWS CLI config, IAM roles, etc.
            session = boto3.Session()
            client = session.client(
                service_name="bedrock-runtime",
                region_name=self.region
            )
            
            # Test credentials by listing foundation models (or attempting a basic call)
            # This will fail fast if credentials are invalid
            try:
                # Make a test call to validate credentials and region
                client.list_foundation_models()
                self.logger.info(f"Successfully connected to Bedrock in region {self.region}")
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == 'UnauthorizedOperation':
                    raise ModelError(
                        f"Insufficient permissions for Bedrock in region {self.region}. "
                        "Ensure your AWS credentials have bedrock:InvokeModel permissions."
                    )
                else:
                    raise ModelError(f"Failed to connect to Bedrock: {str(e)}")
            
            return client
            
        except NoCredentialsError:
            raise ModelError(
                "AWS credentials not found. Please configure credentials using:\n"
                "1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION\n"
                "2. AWS CLI: aws configure\n"
                "3. IAM role (for Lambda/EC2)\n"
                "4. AWS credentials file"
            )
        except Exception as e:
            raise ModelError(f"Failed to create Bedrock client: {str(e)}", e)
    
    def _prepare_request_body(self, system_prompt: str, task_prompt: str, max_tokens: int) -> str:
        """Prepare request body for Bedrock Claude model.
        
        Args:
            system_prompt: System instruction for Claude
            task_prompt: User task/query for Claude  
            max_tokens: Maximum tokens to generate
            
        Returns:
            JSON string for Bedrock request body
        """
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": task_prompt
                }
            ]
        }
        
        return json.dumps(request_body)
    
    def _parse_response(self, response_body: bytes) -> str:
        """Parse Bedrock response body.
        
        Args:
            response_body: Raw response bytes from Bedrock
            
        Returns:
            Generated text content
            
        Raises:
            ModelError: If response parsing fails
        """
        try:
            response_data = json.loads(response_body.decode('utf-8'))
            
            # Claude response format in Bedrock
            if 'content' in response_data:
                content_blocks = response_data['content']
                if content_blocks and len(content_blocks) > 0:
                    first_block = content_blocks[0]
                    if 'text' in first_block:
                        return first_block['text'].strip()
            
            # Fallback for different response formats
            if 'completion' in response_data:
                return response_data['completion'].strip()
            
            raise ModelError("No text content found in Bedrock response")
            
        except json.JSONDecodeError as e:
            raise ModelError(f"Failed to parse Bedrock response JSON: {str(e)}", e)
        except Exception as e:
            raise ModelError(f"Unexpected error parsing Bedrock response: {str(e)}", e)
    
    def generate(
        self, 
        system_prompt: str, 
        task_prompt: str, 
        max_tokens: int = 1024
    ) -> str:
        """Generate response using Bedrock Claude model.
        
        Args:
            system_prompt: System instruction for Claude
            task_prompt: User task/query for Claude
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            ModelError: If generation fails
        """
        if not system_prompt or not task_prompt:
            raise ModelError("Both system_prompt and task_prompt are required")
        
        if max_tokens <= 0 or max_tokens > 4096:
            raise ModelError("max_tokens must be between 1 and 4096")
        
        self.logger.info(f"Generating response with Bedrock {self.model_name} in {self.region}")
        self.logger.debug(f"System prompt length: {len(system_prompt)}")
        self.logger.debug(f"Task prompt length: {len(task_prompt)}")
        
        # Prepare request
        request_body = self._prepare_request_body(system_prompt, task_prompt, max_tokens)
        
        # Retry logic for transient failures
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.client.invoke_model(
                    modelId=self.bedrock_model_id,
                    body=request_body,
                    contentType="application/json",
                    accept="application/json"
                )
                
                duration = time.time() - start_time
                self.logger.info(f"Generated response in {duration:.2f}s")
                
                # Parse response
                response_body = response['body'].read()
                return self._parse_response(response_body)
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                if error_code == 'ThrottlingException':
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(f"Throttled, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise ModelError(f"Rate limit exceeded after {max_retries} attempts", e)
                
                elif error_code == 'ModelTimeoutException':
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"Model timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise ModelError(f"Model timeout after {max_retries} attempts", e)
                
                elif error_code == 'ValidationException':
                    # Don't retry validation errors
                    raise ModelError(f"Invalid request: {error_message}", e)
                
                elif error_code == 'AccessDeniedException':
                    # Don't retry permission errors
                    raise ModelError(f"Access denied: {error_message}", e)
                
                elif error_code == 'ResourceNotFoundException':
                    # Don't retry model not found errors
                    raise ModelError(f"Model not found: {error_message}", e)
                
                else:
                    # Don't retry for other client errors
                    raise ModelError(f"Bedrock API error: {error_message}", e)
            
            except BotoCoreError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Network error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise ModelError(f"Network error after {max_retries} attempts: {str(e)}", e)
            
            except Exception as e:
                # Don't retry for unexpected errors
                raise ModelError(f"Unexpected error during generation: {str(e)}", e)
        
        raise ModelError(f"Failed to generate response after {max_retries} attempts")