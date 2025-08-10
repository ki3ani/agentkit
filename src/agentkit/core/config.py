"""Configuration management for AgentKit."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


class Config:
    """Configuration manager for AgentKit."""
    
    def __init__(self, env_file: Optional[Path] = None):
        """Initialize configuration.
        
        Args:
            env_file: Optional path to .env file
        """
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            # Try to load .env from current directory or parent directories
            load_dotenv()
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a model provider.
        
        Args:
            provider: Provider name (e.g., 'anthropic', 'openai', 'mistral')
            
        Returns:
            API key if found, None otherwise
        """
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY", 
            "mistral": "MISTRAL_API_KEY"
        }
        
        env_var = key_map.get(provider.lower())
        if env_var:
            return os.getenv(env_var)
        
        return None
    
    def get_aws_config(self) -> Dict[str, Optional[str]]:
        """Get AWS configuration from environment variables.
        
        Returns:
            Dictionary with AWS configuration
        """
        return {
            "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "session_token": os.getenv("AWS_SESSION_TOKEN")
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from environment variables.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return os.getenv(key, default)