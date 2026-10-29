"""
FORECAST model API layer.
"""

from forecast.agents.models.utils import (
    ModelClient,
    ModelResponse,
    save_trajectory,
    get_model_client,
)
from forecast.plugins import register_model_provider
from forecast.agents.models.anthropic import AnthropicClient
from forecast.agents.models.openai import OpenAIClient
from forecast.agents.models.helmholtz import HelmholtzClient
from forecast.agents.models.ollama import OllamaClient

__all__ = [
    "ModelClient",
    "ModelResponse",
    "save_trajectory",
    "get_model_client",
    "register_model_provider",
    "AnthropicClient",
    "OpenAIClient",
    "HelmholtzClient",
    "OllamaClient",
]
