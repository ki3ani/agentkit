"""Model abstraction layer for AgentKit with cloud-ready Claude integration."""

import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .logger import get_logger
from .config import Config

logger = get_logger(__name__)


class ModelError(Exception):
    """Exception raised for model-related errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class ModelProvider(ABC):
    """Abstract base class for AI model providers."""
    
    def __init__(self, model_name: str):
        """Initialize the model provider.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def generate(
        self, 
        system_prompt: str, 
        task_prompt: str, 
        max_tokens: int = 1024
    ) -> str:
        """Generate a response using the model.
        
        Args:
            system_prompt: System instruction for the model
            task_prompt: User task/query for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            ModelError: If generation fails
        """
        pass


class ClaudeProvider(ModelProvider):
    """Claude model provider with AWS Lambda support."""
    
    # Mapping from our model names to Anthropic API model names
    MODEL_MAPPING = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229", 
        "claude-3-haiku": "claude-3-haiku-20240307"
    }
    
    def __init__(self, model_name: str, config: Optional[Config] = None):
        """Initialize Claude provider.
        
        Args:
            model_name: Name of the Claude model to use
            config: Optional configuration instance
        """
        super().__init__(model_name)
        
        if not ANTHROPIC_AVAILABLE:
            raise ModelError(
                "Anthropic SDK not available. Install with: pip install anthropic"
            )
        
        if model_name not in self.MODEL_MAPPING:
            available = ", ".join(self.MODEL_MAPPING.keys())
            raise ModelError(
                f"Unsupported Claude model: {model_name}. "
                f"Available models: {available}"
            )
        
        self.config = config or Config()
        self.api_model_name = self.MODEL_MAPPING[model_name]
        self.client = self._create_client()
    
    def _create_client(self) -> anthropic.Anthropic:
        """Create Anthropic client with API key from env or AWS Secrets Manager.
        
        Returns:
            Configured Anthropic client
            
        Raises:
            ModelError: If API key cannot be retrieved
        """
        # Try environment variable first
        api_key = self.config.get_api_key("anthropic")
        
        if not api_key:
            # Try AWS Secrets Manager if in Lambda environment
            api_key = self._get_secret_from_aws()
        
        if not api_key:
            raise ModelError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or configure ANTHROPIC_SECRET_ARN for AWS Secrets Manager."
            )
        
        try:
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            raise ModelError(f"Failed to create Anthropic client: {str(e)}", e)
    
    def _get_secret_from_aws(self) -> Optional[str]:
        """Retrieve API key from AWS Secrets Manager.
        
        Returns:
            API key if found, None otherwise
        """
        if not BOTO3_AVAILABLE:
            self.logger.warning("boto3 not available, cannot retrieve from AWS Secrets Manager")
            return None
        
        secret_arn = os.getenv("ANTHROPIC_SECRET_ARN")
        if not secret_arn:
            return None
        
        try:
            session = boto3.Session()
            secrets_client = session.client('secretsmanager')
            
            response = secrets_client.get_secret_value(SecretId=secret_arn)
            secret_data = json.loads(response['SecretString'])
            
            # Support both direct string and JSON object formats
            if isinstance(secret_data, str):
                return secret_data
            elif isinstance(secret_data, dict):
                return secret_data.get('anthropic_api_key') or secret_data.get('api_key')
            
        except ClientError as e:
            self.logger.error(f"Failed to retrieve secret from AWS: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in AWS secret: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving AWS secret: {str(e)}")
        
        return None
    
    def generate(
        self, 
        system_prompt: str, 
        task_prompt: str, 
        max_tokens: int = 1024
    ) -> str:
        """Generate response using Claude model.
        
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
        
        self.logger.info(f"Generating response with {self.model_name}")
        self.logger.debug(f"System prompt length: {len(system_prompt)}")
        self.logger.debug(f"Task prompt length: {len(task_prompt)}")
        
        # Retry logic for transient failures
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.client.messages.create(
                    model=self.api_model_name,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": task_prompt
                        }
                    ]
                )
                
                duration = time.time() - start_time
                self.logger.info(f"Generated response in {duration:.2f}s")
                
                # Extract text content from response
                if response.content and len(response.content) > 0:
                    content = response.content[0]
                    if hasattr(content, 'text'):
                        return content.text.strip()
                
                raise ModelError("No text content in response")
                
            except anthropic.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise ModelError(f"Rate limit exceeded after {max_retries} attempts", e)
            
            except anthropic.APITimeoutError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"API timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise ModelError(f"API timeout after {max_retries} attempts", e)
            
            except anthropic.APIError as e:
                # Don't retry for non-transient API errors
                raise ModelError(f"Anthropic API error: {str(e)}", e)
            
            except Exception as e:
                # Don't retry for unexpected errors
                raise ModelError(f"Unexpected error during generation: {str(e)}", e)
        
        raise ModelError(f"Failed to generate response after {max_retries} attempts")


def get_model_provider(
    model_name: str, 
    provider: Optional[str] = None,
    region: Optional[str] = None,
    config: Optional[Config] = None,
    **kwargs
) -> ModelProvider:
    """Factory function to create model providers.
    
    Args:
        model_name: Name of the model to use
        provider: Provider name ('anthropic', 'bedrock', or 'goose'). Auto-detected if not specified.
        region: AWS region for Bedrock provider
        config: Optional configuration instance
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Configured model provider
        
    Raises:
        ModelError: If provider is unsupported
    """
    # Auto-detect provider if not specified
    if provider is None:
        # Default to anthropic for backwards compatibility
        provider = "anthropic"
    
    provider = provider.lower()
    
    if provider == "anthropic":
        # Validate Claude models for Anthropic provider
        claude_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        if model_name not in claude_models:
            available_models = ", ".join(claude_models)
            raise ModelError(
                f"Unsupported Claude model: {model_name}. "
                f"Available models: {available_models}"
            )
        return ClaudeProvider(model_name, config)
    
    elif provider == "bedrock":
        # Validate Claude models for Bedrock provider
        claude_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        if model_name not in claude_models:
            available_models = ", ".join(claude_models)
            raise ModelError(
                f"Unsupported Bedrock model: {model_name}. "
                f"Available models: {available_models}"
            )
        from ..models.bedrock_provider import BedrockProvider
        return BedrockProvider(model_name, region, config)
    
    elif provider == "goose":
        # Goose is model agnostic - accept any model name
        from ..models.goose_provider import GooseProvider
        return GooseProvider(
            model_name=model_name, 
            config=config,
            **kwargs
        )
    
    else:
        raise ModelError(
            f"Unsupported provider: {provider}. "
            f"Available providers: anthropic, bedrock, goose"
        )


def get_supported_models() -> Dict[str, list[str]]:
    """Get dictionary of supported models by provider.
    
    Returns:
        Dictionary mapping provider names to lists of model names
    """
    claude_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
    
    # Import GooseProvider to get common models
    try:
        from ..models.goose_provider import GooseProvider
        goose_common_models = list(GooseProvider.COMMON_MODELS.keys())
    except ImportError:
        goose_common_models = ["gpt-4", "gpt-4-turbo", "gpt-4o", "claude-3-opus", "claude-3-sonnet"]
    
    return {
        "anthropic": claude_models,
        "bedrock": claude_models,
        "goose": goose_common_models
    }