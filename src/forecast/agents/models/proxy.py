"""
Models served through a LiteLLM proxy (e.g. CISPA's).

A proxy exposes models under its own aliases. FORECAST addresses them as
``proxy/<alias>``; the proxy's URL and key come from ``LITELLM_PROXY_API_BASE``
and ``LITELLM_PROXY_API_KEY``. Behaviour that depends on *which* model it
is -- prices, context window, an agent's model-specific defaults -- follows the
canonical model id: ``LITELLM_PROXY_CANONICAL_MODEL``, defaulting to the alias.
"""

from __future__ import annotations

import os
from typing import Optional

from forecast.agents.models.utils import ModelClient, ModelResponse, model_traits


class LitellmProxyClient(ModelClient):
    """
    A model behind a LiteLLM proxy.

    :param model: ``proxy/<alias>`` or the bare alias.
    :param temperature: Default temperature for :meth:`complete`.
    :param max_tokens: Default completion cap.
    :param api_key: Proxy key; ``None`` reads ``LITELLM_PROXY_API_KEY``.
    :param base_url: Proxy URL; ``None`` reads ``LITELLM_PROXY_API_BASE``.
    :param canonical_model: The model the alias serves; ``None`` reads
        ``LITELLM_PROXY_CANONICAL_MODEL`` or uses the alias.
    """

    LITELLM_PROVIDER = "litellm_proxy"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        canonical_model: Optional[str] = None,
    ):
        alias = model.split("/", 1)[1] if model.startswith("proxy/") else model
        self.alias = alias
        self.base_url = base_url or os.environ.get("LITELLM_PROXY_API_BASE")
        if not self.base_url:
            raise ValueError("LITELLM_PROXY_API_BASE is not set.")
        canonical = canonical_model or os.environ.get("LITELLM_PROXY_CANONICAL_MODEL") or alias
        super().__init__(
            model=canonical,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or os.environ.get("LITELLM_PROXY_API_KEY"),
        )
        traits = model_traits(canonical)
        self.context_window = traits.context_window if traits else None

    def litellm_model_name(self) -> str:
        return f"{self.LITELLM_PROVIDER}/{self.alias}"

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> ModelResponse:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}
        ]
        reply = self.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)[0]
        return ModelResponse(
            content=reply.content,
            model=reply.model,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            total_tokens=reply.total_tokens,
            finish_reason=reply.finish_reason,
            elapsed_seconds=reply.elapsed_seconds,
        )
