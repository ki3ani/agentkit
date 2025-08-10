"""AgentKit - A framework for creating and running AI agents defined through YAML configuration."""

__version__ = "0.1.0"
__author__ = "AgentKit Team"
__description__ = "A framework for creating and running AI agents defined through YAML configuration"

from .core.config import Config
from .core.logger import get_logger

__all__ = ["Config", "get_logger", "__version__"]