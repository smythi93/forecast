"""
Anthropic Claude client for the FORECAST model API layer.

Wraps the ``anthropic`` Python SDK to implement :class:`ModelClient`.
"""

import os
import time
from typing import Optional

from forecast.agents.models.utils import ModelClient, ModelResponse, model_traits


class AnthropicClient(ModelClient):
    """
    LLM client backed by the Anthropic Messages API.

    The API key is read from the ``ANTHROPIC_API_KEY`` environment variable
    unless supplied explicitly.

    :ivar model: Claude model identifier (e.g. ``"claude-haiku-4-5-20251001"``).
    :ivar temperature: Sampling temperature (0.0 = deterministic).
    :ivar max_tokens: Maximum tokens in the completion.
    :ivar api_key: Explicit API key; falls back to env var.
    :ivar system_prompt: System-level instruction prepended to every call.
    """

    LITELLM_PROVIDER = "anthropic"
    #: Claude's standard context window, for models without :class:`ModelTraits`.
    context_window = 200_000

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert software engineer. "
        "You will be given a description of a bug and the relevant code. "
        "Your task is to produce a minimal, correct unified-diff patch that fixes the bug."
    )

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialise the Anthropic client.

        :param model: Claude model ID.  Defaults to
            ``"claude-haiku-4-5-20251001"``.
        :param temperature: Sampling temperature.  Defaults to 0.0.
        :param max_tokens: Max completion tokens.  Defaults to 4096.
        :param api_key: Anthropic API key.  ``None`` reads from
            ``ANTHROPIC_API_KEY``.
        :param system_prompt: Override the default system prompt.  ``None``
            uses the built-in default.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        traits = model_traits(model)
        if traits:
            self.context_window = traits.context_window

    def complete(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """
        Send *prompt* to the Claude API and return the response.

        :param prompt: The full user-turn prompt text.
        :param system_prompt: Override the client-level system prompt for this
            call.  ``None`` uses :attr:`system_prompt`.
        :returns: :class:`ModelResponse` with content, usage, and timing.
        :raises ImportError: If the ``anthropic`` package is not installed.
        :raises RuntimeError: If the API call fails.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required: pip install anthropic")

        effective_system = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        client = anthropic.Anthropic(api_key=self.api_key)
        t0 = time.time()
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=effective_system,
                messages=[{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

        elapsed = time.time() - t0
        content = message.content[0].text if message.content else ""
        usage = message.usage
        return ModelResponse(
            content=content,
            model=message.model,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            finish_reason=message.stop_reason or "stop",
            raw=message.model_dump() if hasattr(message, "model_dump") else None,
            elapsed_seconds=elapsed,
        )
