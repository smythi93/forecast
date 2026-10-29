"""
OpenAI client for the FORECAST model API layer.

Wraps the ``openai`` Python SDK to implement :class:`ModelClient`.
Also, compatible with any OpenAI-API-compatible endpoint (e.g. Azure, vLLM)
via the ``base_url`` parameter.
"""

import os
import time
from typing import Optional

from forecast.agents.models.utils import ModelClient, ModelResponse


class OpenAIClient(ModelClient):
    """
    LLM client backed by the OpenAI Chat Completions API.

    The API key is read from the ``OPENAI_API_KEY`` environment variable
    unless supplied explicitly.

    :ivar model: OpenAI model identifier (e.g. ``"gpt-4o"``).
    :ivar temperature: Sampling temperature (0.0 = deterministic).
    :ivar max_tokens: Maximum tokens in the completion.
    :ivar api_key: Explicit API key; falls back to env var.
    :ivar base_url: Override the API base URL (e.g. for Azure or
        OpenAI-compatible local servers).
    :ivar system_prompt: System-level instruction prepended to every call.
    """

    LITELLM_PROVIDER = "openai"

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert software engineer. "
        "You will be given a description of a bug and the relevant code. "
        "Your task is to produce a minimal, correct unified-diff patch that fixes the bug."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialise the OpenAI client.

        :param model: OpenAI model ID.  Defaults to ``"gpt-4o"``.
        :param temperature: Sampling temperature.  Defaults to 0.0.
        :param max_tokens: Max completion tokens.  Defaults to 4096.
        :param api_key: OpenAI API key.  ``None`` reads from
            ``OPENAI_API_KEY``.
        :param base_url: Override the API base URL.  ``None`` uses the
            OpenAI default.
        :param system_prompt: Override the default system prompt.  ``None``
            uses the built-in default.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self.base_url = base_url
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def complete(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """
        Send *prompt* to the OpenAI Chat Completions API and return the
        response.

        :param prompt: The full user-turn prompt text.
        :param system_prompt: Override the client-level system prompt for this
            call.  ``None`` uses :attr:`system_prompt`.
        :returns: :class:`ModelResponse` with content, usage, and timing.
        :raises ImportError: If the ``openai`` package is not installed.
        :raises RuntimeError: If the API call fails.
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required: pip install openai")

        kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = openai.OpenAI(**kwargs)

        effective_system = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        t0 = time.time()
        try:
            # noinspection PyTypeChecker
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": prompt},
                ],  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

        elapsed = time.time() - t0
        choice = resp.choices[0]
        usage = resp.usage
        return ModelResponse(
            content=choice.message.content or "",
            model=resp.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            finish_reason=choice.finish_reason or "stop",
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
            elapsed_seconds=elapsed,
        )
