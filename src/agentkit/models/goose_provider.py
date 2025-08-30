"""Goose multi-model orchestration provider for AgentKit."""

import json
import time
import os
from typing import Optional, Dict, Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from ..core.model_interface import ModelProvider, ModelError
from ..core.logger import get_logger
from ..core.config import Config

logger = get_logger(__name__)


class GooseProvider(ModelProvider):
    """Goose multi-model orchestration provider."""
    
    # Goose-supported models (model agnostic - users can specify any model)
    # This is not an exhaustive list, just common examples
    COMMON_MODELS = {
        # OpenAI models
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        
        # Anthropic models
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        
        # Other models (examples)
        "gemini-pro": "gemini-pro",
        "llama-2-70b": "llama-2-70b-chat",
        "mixtral-8x7b": "mixtral-8x7b-instruct",
    }
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, config: Optional[Config] = None):
        """Initialize Goose provider.
        
        Args:
            model_name: Name of the model to use (any model supported by Goose)
            api_key: Goose API key (optional, can be set via environment)
            base_url: Goose API base URL (optional, defaults to standard endpoint)
            config: Optional configuration instance
            
        Raises:
            ModelError: If httpx is not available or configuration is invalid
        """
        super().__init__(model_name)
        
        if not HTTPX_AVAILABLE:
            raise ModelError(
                "httpx not available. Install with: pip install httpx"
            )
        
        self.config = config or Config()
        
        # API configuration
        self.api_key = api_key or os.getenv("GOOSE_API_KEY")
        self.base_url = base_url or os.getenv("GOOSE_BASE_URL", "https://api.goose.ai/v1")
        
        if not self.api_key:
            raise ModelError(
                "Goose API key not found. Set GOOSE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Model name can be any string - Goose is model agnostic
        # We don't restrict to a specific list to allow flexibility
        self.goose_model_name = self._resolve_model_name(model_name)
        
        self.client = self._create_client()
    
    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model name for Goose API.
        
        Args:
            model_name: User-specified model name
            
        Returns:
            Resolved model name for Goose API
        """
        # Check if it's a known model with a specific mapping
        if model_name in self.COMMON_MODELS:
            resolved = self.COMMON_MODELS[model_name]
            self.logger.info(f"Resolved model '{model_name}' to '{resolved}'")
            return resolved
        
        # For unknown models, pass through as-is (model agnostic)
        self.logger.info(f"Using model name as-is: '{model_name}'")
        return model_name
    
    def _create_client(self) -> httpx.Client:
        """Create HTTP client for Goose API.
        
        Returns:
            Configured httpx client
            
        Raises:
            ModelError: If client creation fails
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AgentKit/1.0"
            }
            
            client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0
            )
            
            # Test connection with a simple request
            self._test_connection(client)
            
            return client
            
        except Exception as e:
            raise ModelError(f"Failed to create Goose client: {str(e)}", e)
    
    def _test_connection(self, client: httpx.Client) -> None:
        """Test connection to Goose API.
        
        Args:
            client: HTTP client to test
            
        Raises:
            ModelError: If connection test fails
        """
        try:
            # Try to get models list or make a simple API call
            response = client.get("/models", timeout=10.0)
            
            if response.status_code == 200:
                self.logger.info("Successfully connected to Goose API")
            elif response.status_code == 401:
                raise ModelError("Invalid Goose API key")
            elif response.status_code == 404:
                # Models endpoint might not exist, that's OK
                self.logger.info("Connected to Goose API (models endpoint not available)")
            else:
                self.logger.warning(f"Goose API responded with status {response.status_code}")
                # Don't raise error for other status codes - API might still work for completions
                
        except httpx.TimeoutException:
            raise ModelError("Timeout connecting to Goose API")
        except httpx.ConnectError:
            raise ModelError("Cannot connect to Goose API. Check your network connection and base URL.")
        except ModelError:
            # Re-raise ModelError exceptions
            raise
        except Exception as e:
            self.logger.warning(f"Connection test failed: {str(e)}")
            # Don't fail completely - the API might still work for completions
    
    def _prepare_request_body(self, system_prompt: str, task_prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Prepare request body for Goose API.
        
        Args:
            system_prompt: System instruction for the model
            task_prompt: User task/query for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Request body dictionary
        """
        # OpenAI-compatible format (most common)
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user", 
            "content": task_prompt
        })
        
        request_body = {
            "model": self.goose_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        return request_body
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Parse Goose API response.
        
        Args:
            response_data: Response data from Goose API
            
        Returns:
            Generated text content
            
        Raises:
            ModelError: If response parsing fails
        """
        try:
            # OpenAI-compatible response format
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                
                # Handle different response formats
                if "message" in choice:
                    return choice["message"]["content"].strip()
                elif "text" in choice:
                    return choice["text"].strip()
            
            # Alternative formats
            if "content" in response_data:
                return response_data["content"].strip()
            
            if "text" in response_data:
                return response_data["text"].strip()
            
            raise ModelError("No text content found in Goose response")
            
        except Exception as e:
            raise ModelError(f"Failed to parse Goose response: {str(e)}", e)
    
    def generate(
        self, 
        system_prompt: str, 
        task_prompt: str, 
        max_tokens: int = 1024
    ) -> str:
        """Generate response using Goose multi-model orchestration.
        
        Args:
            system_prompt: System instruction for the model
            task_prompt: User task/query for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            ModelError: If generation fails
        """
        if not system_prompt or not task_prompt:
            raise ModelError("Both system_prompt and task_prompt are required")
        
        if max_tokens <= 0 or max_tokens > 8192:
            raise ModelError("max_tokens must be between 1 and 8192")
        
        self.logger.info(f"Generating response with Goose model {self.goose_model_name}")
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
                
                response = self.client.post(
                    "/chat/completions",
                    json=request_body,
                    timeout=60.0
                )
                
                duration = time.time() - start_time
                self.logger.info(f"Generated response in {duration:.2f}s")
                
                # Handle different response status codes
                if response.status_code == 200:
                    response_data = response.json()
                    return self._parse_response(response_data)
                
                elif response.status_code == 429:
                    # Rate limiting
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise ModelError(f"Rate limit exceeded after {max_retries} attempts")
                
                elif response.status_code == 400:
                    error_msg = "Invalid request"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"].get("message", error_msg)
                    except:
                        pass
                    raise ModelError(f"Bad request: {error_msg}")
                
                elif response.status_code == 401:
                    raise ModelError("Invalid Goose API key")
                
                elif response.status_code == 404:
                    raise ModelError(f"Model '{self.goose_model_name}' not found or not supported")
                
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"Server error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise ModelError(f"Server error after {max_retries} attempts")
                
                else:
                    raise ModelError(f"Unexpected response status: {response.status_code}")
                
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Request timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise ModelError(f"Request timeout after {max_retries} attempts")
            
            except httpx.ConnectError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Connection error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise ModelError(f"Connection error after {max_retries} attempts: {str(e)}")
            
            except Exception as e:
                # Don't retry for unexpected errors
                raise ModelError(f"Unexpected error during generation: {str(e)}", e)
        
        raise ModelError(f"Failed to generate response after {max_retries} attempts")
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except:
                pass