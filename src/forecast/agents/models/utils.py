"""
Shared model API utilities: :class:`ModelResponse`, :class:`ModelClient`,
:func:`save_trajectory`, and the :func:`get_model_client` factory.
"""

import abc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from forecast.plugins.registry import register_model_provider, resolve_model_factory

if TYPE_CHECKING:
    from forecast.agents.models.openai import OpenAIClient
    from forecast.agents.models.anthropic import AnthropicClient
    from forecast.agents.models.helmholtz import HelmholtzClient
    from forecast.agents.models.ollama import OllamaClient


@dataclass
class ModelResponse:
    """
    A single LLM response together with its provenance metadata.

    :ivar content: The raw text content returned by the model.
    :ivar model: The model identifier that produced the response.
    :ivar prompt_tokens: Number of tokens in the prompt.
    :ivar completion_tokens: Number of tokens in the completion.
    :ivar total_tokens: Total tokens consumed (prompt + completion).
    :ivar finish_reason: Stop reason reported by the API (e.g. ``"stop"``).
    :ivar raw: Full raw API response object for debugging.
    :ivar elapsed_seconds: Wall-clock time of the API call in seconds.
    """

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    raw: Optional[dict] = field(default=None, repr=False)
    elapsed_seconds: float = 0.0


#: USD per token -- (input, output, cache write, cache read) -- of the models
#: the evaluation prices against. Kept here rather than taken from litellm,
#: whose table changes between versions and has dropped retired models
#: (``claude-sonnet-4-20250514`` among them): a budget must not silently turn
#: into "unlimited" when a dependency is upgraded.
PRICES: dict[str, tuple[float, float, float, float]] = {
    # Anthropic list prices.
    "claude-haiku-4-5-20251001": (1e-6, 5e-6, 1.25e-6, 0.1e-6),
    "claude-sonnet-5-5": (2e-6, 10e-6, 2.5e-6, 0.2e-6),
    "claude-sonnet-4-20250514": (3e-6, 15e-6, 3.75e-6, 0.3e-6),
}


@dataclass(frozen=True)
class ModelTraits:
    """
    What FORECAST must know about a model to send every agent's calls to it.

    :ivar context_window: Prompt plus completion tokens the model accepts.
    :ivar max_output: Largest completion the model generates in one call.
    :ivar sampling: Whether the API accepts ``temperature``/``top_p``/``top_k``;
        Claude 5 rejects them, and samples at its own fixed setting.
    :ivar always_thinks: Whether the model thinks before every answer, with
        no way to switch it off; the thinking counts against ``max_tokens``.
    :ivar one_sampling_param: Whether the API rejects ``temperature`` and
        ``top_p`` in the same request (Claude 4.5); ``top_p`` is then left out.
    """

    context_window: int
    max_output: int
    sampling: bool = True
    always_thinks: bool = False
    one_sampling_param: bool = False


#: Traits of the models the evaluation runs, kept here for the same reason
#: as :data:`PRICES`.
MODEL_TRAITS: dict[str, ModelTraits] = {
    "claude-haiku-4-5-20251001": ModelTraits(200_000, 64_000, one_sampling_param=True),
    "claude-sonnet-5-5": ModelTraits(1_000_000, 128_000, sampling=False, always_thinks=True),
    "claude-sonnet-4-20250514": ModelTraits(200_000, 64_000),
}

#: Tokens added to an agent's completion cap for a model that always thinks,
#: so the thinking does not use up the budget the agent set for its answer
#: (AutoCodeRover allows 1024). ``FORECAST_THINKING_TOKENS`` overrides it.
THINKING_TOKENS = 16_000


def model_traits(model: str) -> Optional[ModelTraits]:
    """:data:`MODEL_TRAITS` of *model*, provider prefix ignored; ``None`` if unknown."""
    return MODEL_TRAITS.get(model.split("/", 1)[-1])


def price(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """
    Cost in USD of one call, at the prices of *model* (see :data:`PRICES`).

    :param model: Key of :data:`PRICES`; an unknown model costs 0.
    :param prompt_tokens: All prompt tokens, cache writes and reads included.
    :param completion_tokens: Generated tokens.
    :param cache_write_tokens: Prompt tokens written to the provider's cache.
    :param cache_read_tokens: Prompt tokens served from the provider's cache.
    """
    rates = PRICES.get(model.split("/", 1)[-1])
    if rates is None:
        return 0.0
    fresh = max(0, prompt_tokens - cache_write_tokens - cache_read_tokens)
    return (
        fresh * rates[0]
        + completion_tokens * rates[1]
        + cache_write_tokens * rates[2]
        + cache_read_tokens * rates[3]
    )


@dataclass
class ChatResponse:
    """
    One choice of a multi-turn :meth:`ModelClient.chat` call.

    Messages and tool calls use the OpenAI chat format, the format every
    upstream repair agent we integrate (SWE-agent, mini-swe-agent, Agentless)
    already speaks, so their own parsers can consume a response unchanged.

    :ivar content: Text of the assistant message (``""`` if it only called tools).
    :ivar model: The model identifier that produced the response.
    :ivar tool_calls: Tool calls of the message, as OpenAI-format dicts.
    :ivar thinking_blocks: Provider thinking blocks, to be echoed back verbatim.
    :ivar prompt_tokens: Prompt tokens of the call (shared by all its choices).
    :ivar completion_tokens: Completion tokens of this choice.
    :ivar cost: Cost of the call in USD at the prices of :attr:`ModelClient.price_as`.
    :ivar finish_reason: Stop reason reported by the API.
    :ivar elapsed_seconds: Wall-clock time of the API call.
    :ivar raw: The provider response object, for adapters whose upstream
        parser needs it (mini-swe-agent parses the litellm response itself).
    """

    content: str
    model: str
    tool_calls: list = field(default_factory=list)
    thinking_blocks: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    finish_reason: str = "stop"
    elapsed_seconds: float = 0.0
    raw: Optional[object] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class JsonlRecorder:
    """
    Append every :meth:`ModelClient.chat` call to a JSON-lines file.

    Each upstream agent keeps its own trajectory format; this one is
    FORECAST's, identical for every agent, so tokens and cost can be compared
    across agents and modes without parsing four formats.

    :param path: File to append to; parent directories are created.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class ModelClient(abc.ABC):
    """
    Abstract interface for an LLM client.

    All concrete backends (OpenAI, Anthropic, Helmholtz) subclass this and
    implement :meth:`complete` for single-turn prompts.  Multi-turn agents use
    :meth:`chat`, which every backend inherits.

    :ivar model: The model identifier forwarded to the API.
    :ivar temperature: Sampling temperature (0.0 = deterministic).
    :ivar max_tokens: Maximum tokens to generate in the completion.
    :ivar api_key: Explicit API key; ``None`` means the implementation falls
        back to the relevant environment variable.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        """
        Initialise the client.

        :param model: Model identifier (e.g. ``"claude-haiku-4-5-20251001"``).
        :param temperature: Sampling temperature.  Defaults to 0.0.
        :param max_tokens: Maximum completion tokens.  Defaults to 4096.
        :param api_key: Explicit API key.  ``None`` means the backend reads
            its environment variable.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        #: Called with one dict per :meth:`chat` call; ``None`` records nothing.
        self.recorder: Optional[Callable[[dict], None]] = None
        #: Model whose prices :meth:`chat` charges tokens at. Defaults to this
        #: model; ``FORECAST_PRICE_AS`` overrides it, so that a model without a
        #: price (a local one) runs under the same budgets as the one it is
        #: compared with.
        self.price_as: str = os.environ.get("FORECAST_PRICE_AS") or model

    #: Prompt plus completion tokens the model accepts; ``None`` if unknown.
    #: Agents with their own context check (SWE-agent) are given this limit,
    #: so they stop cleanly instead of being truncated by the server.
    context_window: Optional[int] = None

    #: litellm provider prefix of this backend; ``None`` passes the name as is.
    LITELLM_PROVIDER: Optional[str] = None

    def litellm_model_name(self) -> str:
        """
        The identifier litellm routes this client's model by.

        :returns: ``<provider>/<model>``, or the bare model name for backends
            without a provider prefix.
        """
        if self.LITELLM_PROVIDER and not self.model.startswith(
            f"{self.LITELLM_PROVIDER}/"
        ):
            return f"{self.LITELLM_PROVIDER}/{self.model}"
        return self.model

    def chat(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        retries: int = 0,
        **kwargs: Any,
    ) -> list[ChatResponse]:
        """
        Send a multi-turn conversation and return *n* sampled replies.

        Upstream agents call this instead of their own LLM layer, so every
        request of every agent passes through FORECAST: one model choice, one
        record of tokens and cost, regardless of which agent made the call.

        Sampling is the caller's: *temperature* is sent only when given, so an
        agent that relies on the provider default keeps it.  Providers without
        native multi-sampling (Anthropic) are queried *n* times.

        :class:`ModelTraits` of the model change the request, the same way for
        every agent and both modes, and the record shows what was sent:
        sampling parameters are left out for a model that rejects them (all of
        them, or ``top_p`` beside a temperature), and a model that always
        thinks gets :data:`THINKING_TOKENS` on top of the agent's completion
        cap, up to the model's maximum.

        :param messages: OpenAI-format messages, system message included.
        :param tools: OpenAI-format tool definitions, or ``None``.
        :param temperature: Sampling temperature; ``None`` sends none.
        :param max_tokens: Completion cap; ``None`` uses :attr:`max_tokens`.
        :param n: Number of replies to sample.
        :param retries: Extra attempts after a failed call, with exponential
            backoff. 0 by default: upstream agents retry in their own model
            layer, and retrying here as well would multiply their policy.
        :param kwargs: Further litellm completion arguments.
        :returns: One :class:`ChatResponse` per sampled reply.
        :raises Exception: The provider's own exception (litellm's types) if
            the call still fails after *retries*.
        """
        import litellm

        params: dict[str, Any] = {
            "model": self.litellm_model_name(),
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
        if temperature is not None:
            params["temperature"] = temperature
        traits = model_traits(self.model)
        if traits and not traits.sampling:
            for name in ("temperature", "top_p", "top_k"):
                params.pop(name, None)
        if traits and traits.one_sampling_param and "temperature" in params:
            # SWE-agent sends top_p=1.0, i.e. no nucleus truncation, beside
            # its temperature.
            params.pop("top_p", None)
        if traits and traits.always_thinks:
            headroom = int(os.environ.get("FORECAST_THINKING_TOKENS") or THINKING_TOKENS)
            params["max_tokens"] = min(traits.max_output, params["max_tokens"] + headroom)
        if self.api_key:
            params["api_key"] = self.api_key
        base_url = getattr(self, "base_url", None)
        if base_url:
            params["api_base"] = base_url

        replies: list[ChatResponse] = []
        for _ in range(max(1, n)):
            t0 = time.time()
            for attempt in range(retries + 1):
                try:
                    raw = litellm.completion(**params)
                    break
                except Exception:
                    # Re-raised as is: upstream agents react to the provider's
                    # exception types (a context-window overflow makes
                    # SWE-agent submit what it has instead of failing).
                    if attempt == retries:
                        raise
                    time.sleep(min(60, 2 ** (attempt + 1)))
            elapsed = time.time() - t0
            usage = getattr(raw, "usage", None)
            details = getattr(usage, "prompt_tokens_details", None)
            cost = price(
                self.price_as,
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
                int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
                int(getattr(details, "cached_tokens", 0) or 0),
            )
            choice = raw.choices[0]
            message = choice.message
            reply = ChatResponse(
                content=message.content or "",
                model=getattr(raw, "model", self.model) or self.model,
                tool_calls=[call.to_dict() for call in (message.tool_calls or [])],
                thinking_blocks=list(getattr(message, "thinking_blocks", None) or []),
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                cost=cost,
                finish_reason=str(choice.finish_reason or "stop"),
                elapsed_seconds=elapsed,
                raw=raw,
            )
            replies.append(reply)
            if self.recorder is not None:
                self.recorder(
                    {
                        "model": reply.model,
                        "messages": messages,
                        "tools": tools or [],
                        "temperature": params.get("temperature"),
                        "requested_temperature": temperature,
                        "top_p": params.get("top_p"),
                        "max_tokens": params["max_tokens"],
                        "requested_max_tokens": max_tokens or self.max_tokens,
                        "n": n,
                        "price_as": self.price_as,
                        "content": reply.content,
                        "tool_calls": reply.tool_calls,
                        "thinking_blocks": reply.thinking_blocks,
                        "prompt_tokens": reply.prompt_tokens,
                        "completion_tokens": reply.completion_tokens,
                        "cost": reply.cost,
                        "finish_reason": reply.finish_reason,
                        "elapsed_seconds": elapsed,
                    }
                )
        return replies

    @abc.abstractmethod
    def complete(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """
        Send *prompt* to the model and return the response.

        :param prompt: The full user-turn prompt string to send.
        :param system_prompt: Override the client-level system prompt for this
            single call.  ``None`` uses the client's stored default.
        :returns: :class:`ModelResponse` with content and usage metadata.
        :raises RuntimeError: If the API call fails after all retries.
        """


def save_trajectory(
    prompt: str,
    response: ModelResponse,
    traj_path: Path,
    extra: Optional[dict] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """
    Persist a prompt/response trajectory to a JSON file.

    The file (and its parent directories) is created if it does not exist;
    existing files are overwritten.

    :param prompt: The full user-turn prompt text that was sent to the model.
    :param response: The model response object.
    :param traj_path: Destination file path (should end in ``.json``).
    :param extra: Optional additional fields merged into the top-level
        trajectory dict (e.g. ``instance_id``, ``agent``).  ``None`` adds
        nothing.
    :param system_prompt: The system prompt used for the call, stored under
        ``"system_prompt"``.  ``None`` means the field is omitted.
    """
    traj_path = Path(traj_path)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "system_prompt": system_prompt,
        "prompt": prompt,
        "response": response.content,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        },
        "finish_reason": response.finish_reason,
        "elapsed_seconds": response.elapsed_seconds,
    }
    if extra:
        record.update(extra)
    traj_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


def get_model_client(model: str, **kwargs) -> ModelClient:
    """
    Return the appropriate :class:`ModelClient` for the given *model* name.

    Resolution order:

    * Names starting with ``"gpt"``, ``"o1"``, ``"o3"``, or ``"o4"``
      → :class:`~forecast.agents.models.openai.OpenAIClient`
    * Names starting with ``"claude"``
      → :class:`~forecast.agents.models.anthropic.AnthropicClient`
    * Names containing ``"helmholtz"``
      → :class:`~forecast.agents.models.helmholtz.HelmholtzClient`
    * Names starting with ``"proxy/"``
      → :class:`~forecast.agents.models.proxy.LitellmProxyClient` (a LiteLLM
      proxy such as CISPA's)
    * Names starting with ``"ollama/"`` or ``"ollama:"``
      → :class:`~forecast.agents.models.ollama.OllamaClient`
    * Anything else → :class:`~forecast.agents.models.openai.OpenAIClient`
      (OpenAI-compatible API assumed)

    :param model: Model identifier (e.g. ``"claude-haiku-4-5-20251001"``).
    :param kwargs: Forwarded to the client constructor (e.g. ``api_key``,
        ``base_url``, ``temperature``, ``max_tokens``).
    :returns: A ready-to-use :class:`ModelClient` instance.
    :raises ImportError: If the required third-party SDK is not installed.
    """
    _register_builtin_model_providers()
    factory = resolve_model_factory(model)
    if factory is not None:
        return factory(model=model, **kwargs)

    # Backward-compatible fallback: treat unknown names as OpenAI-compatible.
    from forecast.agents.models.openai import OpenAIClient

    return OpenAIClient(model=model, **kwargs)


_BUILTINS_REGISTERED = False


def _register_builtin_model_providers() -> None:
    """
    Ensure built-in model providers are registered exactly once.
    """

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    def _openai_factory(**factory_kwargs):
        from forecast.agents.models.openai import OpenAIClient

        return OpenAIClient(**factory_kwargs)

    def _anthropic_factory(**factory_kwargs):
        from forecast.agents.models.anthropic import AnthropicClient

        return AnthropicClient(**factory_kwargs)

    def _helmholtz_factory(**factory_kwargs):
        from forecast.agents.models.helmholtz import HelmholtzClient

        return HelmholtzClient(**factory_kwargs)

    def _ollama_factory(**factory_kwargs):
        from forecast.agents.models.ollama import OllamaClient

        model_name = str(factory_kwargs.get("model", ""))
        if model_name.lower().startswith("ollama/"):
            factory_kwargs["model"] = model_name.split("/", 1)[1]
        elif model_name.lower().startswith("ollama:"):
            factory_kwargs["model"] = model_name.split(":", 1)[1]
        return OllamaClient(**factory_kwargs)

    def _proxy_factory(**factory_kwargs):
        from forecast.agents.models.proxy import LitellmProxyClient

        return LitellmProxyClient(**factory_kwargs)

    register_model_provider(
        name="litellm-proxy",
        factory=_proxy_factory,
        matcher=lambda name: name.startswith("proxy/"),
        priority=60,
    )
    register_model_provider(
        name="ollama",
        factory=_ollama_factory,
        matcher=lambda name: name.startswith(("ollama/", "ollama:")),
        priority=50,
    )
    register_model_provider(
        name="openai",
        factory=_openai_factory,
        matcher=lambda name: name.startswith(("gpt", "o1", "o3", "o4")),
        priority=20,
    )
    register_model_provider(
        name="anthropic",
        factory=_anthropic_factory,
        matcher=lambda name: name.startswith("claude"),
        priority=30,
    )
    register_model_provider(
        name="helmholtz",
        factory=_helmholtz_factory,
        matcher=lambda name: "helmholtz" in name,
        priority=40,
    )
    _BUILTINS_REGISTERED = True
