"""
Ollama client for the FORECAST model API layer.

Uses Ollama's local HTTP API (``/api/chat``) to implement
:class:`~forecast.agents.models.utils.ModelClient`.
"""

import json
import os
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from forecast.agents.models.utils import ModelClient, ModelResponse

DEFAULT_OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)


class OllamaClient(ModelClient):
    """
    LLM client backed by the Ollama chat API.

    :ivar model: Ollama model identifier (e.g. ``"llama3.1:8b"``).
    :ivar temperature: Sampling temperature (0.0 = deterministic).
    :ivar max_tokens: Maximum generated tokens (mapped to Ollama ``num_predict``).
    :ivar base_url: Ollama server base URL.
    :ivar timeout: HTTP timeout in seconds.
    :ivar system_prompt: System-level instruction prepended to every call.
    """

    LITELLM_PROVIDER = "ollama_chat"

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert software engineer. "
        "You will be given a description of a bug and the relevant code. "
        "Your task is to produce a minimal, correct unified-diff patch that fixes the bug."
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        timeout: int = 120,
        system_prompt: Optional[str] = None,
        num_ctx: Optional[int] = None,
    ):
        """
        Initialise the Ollama client.

        :param model: Ollama model name.
        :param temperature: Sampling temperature.  Defaults to 0.0.
        :param max_tokens: Max generated tokens.  Defaults to 4096.
        :param base_url: Ollama base URL.  ``None`` uses ``OLLAMA_BASE_URL``.
        :param timeout: HTTP timeout in seconds.  Defaults to 120.
        :param system_prompt: Override the default system prompt.  ``None``
            uses the built-in default.
        :param num_ctx: Context window to load the model with for
            :meth:`chat`.  ``None`` uses ``OLLAMA_NUM_CTX`` or 131072.  Ollama
            otherwise uses its small server default and *silently* drops the
            start of longer conversations -- fatal for multi-turn agents.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=None,
        )
        self.base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.num_ctx = int(num_ctx or os.environ.get("OLLAMA_NUM_CTX", 131072))
        # Ollama counts prompt and completion against num_ctx.
        self.context_window = self.num_ctx

    def chat(self, messages, **kwargs):
        """
        :meth:`ModelClient.chat` with an explicit context window, and with
        thinking off unless ``OLLAMA_THINK=1``: thinking models otherwise
        spend the agents' output budgets (Agentless caps some stages at a few
        hundred tokens) on reasoning and return empty answers. Off matches the
        Claude models the evaluation uses, which do not think by default.
        """
        kwargs.setdefault("num_ctx", self.num_ctx)
        kwargs.setdefault("think", os.environ.get("OLLAMA_THINK", "0") == "1")
        kwargs.setdefault("timeout", max(self.timeout, 600))
        return super().chat(messages, **kwargs)

    def complete(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """
        Send *prompt* to Ollama and return the response.

        :param prompt: The full user-turn prompt text.
        :param system_prompt: Override the client-level system prompt for this
            call.  ``None`` uses :attr:`system_prompt`.
        :returns: :class:`ModelResponse` with content, usage, and timing.
        :raises RuntimeError: If the HTTP request fails or the response is invalid.
        """
        effective_system = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        request = Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama API call failed with status {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.base_url}: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama API call failed: {exc}") from exc

        elapsed = time.time() - t0
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned a non-JSON response: {body[:200]}"
            ) from exc

        message = raw.get("message", {}) if isinstance(raw, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        prompt_tokens = (
            int(raw.get("prompt_eval_count", 0)) if isinstance(raw, dict) else 0
        )
        completion_tokens = (
            int(raw.get("eval_count", 0)) if isinstance(raw, dict) else 0
        )
        finish_reason = (
            raw.get("done_reason", "stop") if isinstance(raw, dict) else "stop"
        )
        model = raw.get("model", self.model) if isinstance(raw, dict) else self.model

        return ModelResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=finish_reason,
            raw=raw if isinstance(raw, dict) else None,
            elapsed_seconds=elapsed,
        )
