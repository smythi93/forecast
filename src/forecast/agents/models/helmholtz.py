"""
Helmholtz client for the FORECAST model API layer.

The Helmholtz AI center hosts OpenAI-compatible model endpoints.
This client extends :class:`OpenAIClient` with the Helmholtz base URL
and API key environment variable.
"""

import os
from typing import Optional

from forecast.agents.models.openai import OpenAIClient

HELMHOLTZ_BASE_URL = os.environ.get(
    "HELMHOLTZ_BASE_URL", "https://api.helmholtz-blablador.fz-juelich.de/v1"
)


class HelmholtzClient(OpenAIClient):
    """
    LLM client for the Helmholtz AI OpenAI-compatible endpoint.

    Inherits :class:`OpenAIClient` and overrides the base URL and API key
    source.

    :ivar model: Helmholtz model identifier.
    :ivar temperature: Sampling temperature (0.0 = deterministic).
    :ivar max_tokens: Maximum tokens in the completion.
    :ivar api_key: Explicit API key; falls back to ``HELMHOLTZ_API_KEY``
        environment variable.
    :ivar base_url: Helmholtz endpoint URL.
    :ivar system_prompt: System-level instruction prepended to every call.
    """

    def __init__(
        self,
        model: str = "1 - MiniMax-M2.1 - our best model as of December 26, 2025",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialise the Helmholtz client.

        :param model: Helmholtz model ID.
        :param temperature: Sampling temperature.  Defaults to 0.0.
        :param max_tokens: Max completion tokens.  Defaults to 4096.
        :param api_key: Helmholtz API key.  ``None`` reads from
            ``HELMHOLTZ_API_KEY``.
        :param base_url: Override the endpoint URL.  ``None`` uses
            ``HELMHOLTZ_BASE_URL`` env var or the built-in default.
        :param system_prompt: Override the default system prompt.  ``None``
            uses the built-in default from :class:`OpenAIClient`.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or os.environ.get("HELMHOLTZ_API_KEY"),
            base_url=base_url or HELMHOLTZ_BASE_URL,
            system_prompt=system_prompt,
        )
