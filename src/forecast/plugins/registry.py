"""
Core registries for FORECAST plugins.

This module keeps extension points in one place so plugins can register
additional repair agents, model providers, localization formulas/techniques,
and subject classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, overload

AgentClass = type
ModelFactory = Callable[..., Any]
ModelMatcher = Callable[[str], bool]
Formula = Callable[[int, int, int, int], float]
LocalizationTechnique = Callable[..., Any]
SubjectClass = type


@dataclass
class ModelProvider:
    """
    One model-provider registration entry.

    :param name: Human-readable provider name.
    :param factory: Callable that builds a model client.
    :param matcher: Optional matcher for model-name-based dispatch.
    :param aliases: Exact model names served by this provider.
    :param priority: Higher values are matched first for matcher-based dispatch.
    :param order: Internal insertion order used as a stable tiebreaker.
    """

    name: str
    factory: ModelFactory
    matcher: ModelMatcher | None = None
    aliases: set[str] = field(default_factory=set)
    priority: int = 0
    order: int = 0


_AGENTS: dict[str, AgentClass] = {}
_SUBJECTS: dict[str, SubjectClass] = {}
_FORMULAS: dict[str, tuple[str, Formula]] = {}
_LOCALIZATION_TECHNIQUES: dict[str, LocalizationTechnique] = {}
_MODEL_PROVIDER_BY_ALIAS: dict[str, ModelProvider] = {}
_MODEL_PROVIDERS: list[ModelProvider] = []
_MODEL_PROVIDER_COUNTER = 0


def clear_registries() -> None:
    """
    Clear all plugin registries.

    Primarily useful for tests.
    """

    global _MODEL_PROVIDER_COUNTER
    _AGENTS.clear()
    _SUBJECTS.clear()
    _FORMULAS.clear()
    _LOCALIZATION_TECHNIQUES.clear()
    _MODEL_PROVIDER_BY_ALIAS.clear()
    _MODEL_PROVIDERS.clear()
    _MODEL_PROVIDER_COUNTER = 0


@overload
def register_agent_class(
    cls: AgentClass,
    *,
    names: list[str] | None = None,
    replace: bool = False,
) -> AgentClass: ...


@overload
def register_agent_class(
    cls: None = None,
    *,
    names: list[str] | None = None,
    replace: bool = False,
) -> Callable[[AgentClass], AgentClass]: ...


def register_agent_class(
    cls: AgentClass | None = None,
    *,
    names: list[str] | None = None,
    replace: bool = False,
) -> AgentClass | Callable[[AgentClass], AgentClass]:
    """
    Register an agent class under one or more names.

    Supports both direct and decorator style:

    .. code-block:: python

        register_agent_class(MyAgent)

        @register_agent_class(names=["my-agent"])
        class MyAgent(RepairAgent): ...

    :param cls: Agent class to register.
    :param names: Optional explicit names. When omitted, ``cls.names()`` is used.
    :param replace: Whether to overwrite existing registrations.
    :returns: The input class (decorator-friendly).
    :raises ValueError: If a name already exists and *replace* is ``False``.
    """

    if cls is None:
        def _decorator(decorated_cls: AgentClass) -> AgentClass:
            return register_agent_class(
                decorated_cls,
                names=names,
                replace=replace,
            )

        return _decorator

    from forecast.agents.base import RepairAgent

    if not issubclass(cls, RepairAgent):
        raise ValueError("Can only register subclasses of RepairAgent")
    if not hasattr(cls, "names"):
        raise ValueError("Registered agent class must have a 'names' function")

    labels = names if names is not None else list(cls.names())
    for label in labels:
        key = label.lower()
        if key in _AGENTS and not replace:
            raise ValueError(f"Agent name '{label}' is already registered")
        _AGENTS[key] = cls
    return cls


def get_agent_class(name: str) -> AgentClass | None:
    """
    Resolve an agent class by name.

    :param name: Agent name.
    :returns: Registered class or ``None``.
    """

    return _AGENTS.get(name.lower())


def list_agents() -> list[str]:
    """
    Return sorted registered agent names.
    """

    return sorted(_AGENTS.keys())


@overload
def register_model_provider(
    name: str,
    factory: ModelFactory,
    *,
    matcher: ModelMatcher | None = None,
    aliases: list[str] | tuple[str, ...] | set[str] = (),
    priority: int = 0,
    replace: bool = False,
) -> ModelFactory: ...


@overload
def register_model_provider(
    name: str,
    factory: None = None,
    *,
    matcher: ModelMatcher | None = None,
    aliases: list[str] | tuple[str, ...] | set[str] = (),
    priority: int = 0,
    replace: bool = False,
) -> Callable[[ModelFactory], ModelFactory]: ...


def register_model_provider(
    name: str,
    factory: ModelFactory | None = None,
    *,
    matcher: ModelMatcher | None = None,
    aliases: list[str] | tuple[str, ...] | set[str] = (),
    priority: int = 0,
    replace: bool = False,
) -> ModelFactory | Callable[[ModelFactory], ModelFactory]:
    """
    Register a model provider.

    Providers can be selected either by exact alias match or by matcher.

    Supports both direct and decorator style:

    .. code-block:: python

        register_model_provider(name="openai", factory=build_openai)

        @register_model_provider(name="openai", aliases=["gpt-4o"])
        def build_openai(**kwargs): ...

    :param name: Provider name.
    :param factory: Callable returning a model client.
    :param matcher: Optional matcher for model-name dispatch.
    :param aliases: Exact model names served by this provider.
    :param priority: Matcher priority (higher first).
    :param replace: Whether to replace existing alias registrations.
    :raises ValueError: If an alias already exists and *replace* is ``False``.
    """

    if factory is None:
        def _decorator(decorated_factory: ModelFactory) -> ModelFactory:
            register_model_provider(
                name=name,
                factory=decorated_factory,
                matcher=matcher,
                aliases=aliases,
                priority=priority,
                replace=replace,
            )
            return decorated_factory

        return _decorator

    global _MODEL_PROVIDER_COUNTER
    provider = ModelProvider(
        name=name,
        factory=factory,
        matcher=matcher,
        aliases={a.lower() for a in aliases},
        priority=priority,
        order=_MODEL_PROVIDER_COUNTER,
    )
    _MODEL_PROVIDER_COUNTER += 1

    for alias in provider.aliases:
        if alias in _MODEL_PROVIDER_BY_ALIAS and not replace:
            raise ValueError(f"Model alias '{alias}' is already registered")
        _MODEL_PROVIDER_BY_ALIAS[alias] = provider
    _MODEL_PROVIDERS.append(provider)
    return factory


def resolve_model_factory(model_name: str) -> ModelFactory | None:
    """
    Resolve a model factory for *model_name*.

    Resolution order:
    1) exact alias match
    2) matcher registrations by ``priority`` descending, then insertion order

    :param model_name: Model identifier string.
    :returns: Matching factory or ``None``.
    """

    key = model_name.lower()
    exact = _MODEL_PROVIDER_BY_ALIAS.get(key)
    if exact is not None:
        return exact.factory

    candidates = sorted(_MODEL_PROVIDERS, key=lambda p: (-p.priority, p.order))
    for provider in candidates:
        if provider.matcher is not None and provider.matcher(key):
            return provider.factory
    return None


def list_model_providers() -> list[str]:
    """
    Return provider names sorted by registration order.
    """

    seen: set[str] = set()
    names: list[str] = []
    for provider in sorted(_MODEL_PROVIDERS, key=lambda p: p.order):
        if provider.name not in seen:
            names.append(provider.name)
            seen.add(provider.name)
    return names


@overload
def register_formula(
    name: str,
    formula: Formula,
    *,
    replace: bool = False,
) -> Formula: ...


@overload
def register_formula(
    name: str,
    formula: None = None,
    *,
    replace: bool = False,
) -> Callable[[Formula], Formula]: ...


def register_formula(
    name: str,
    formula: Formula | None = None,
    *,
    replace: bool = False,
) -> Formula | Callable[[Formula], Formula]:
    """
    Register a localization formula by name.

    Supports both direct and decorator style:

    .. code-block:: python

        register_formula("MyFormula", my_formula)

        @register_formula("MyFormula")
        def my_formula(ep, np, ef, nf): ...

    :param name: Formula name.
    :param formula: Callable ``(ep, np, ef, nf) -> float``.
    :param replace: Whether to overwrite existing registrations.
    :raises ValueError: If the name already exists and *replace* is ``False``.
    """

    if formula is None:
        def _decorator(decorated_formula: Formula) -> Formula:
            register_formula(name, decorated_formula, replace=replace)
            return decorated_formula

        return _decorator

    key = name.lower()
    if key in _FORMULAS and not replace:
        raise ValueError(f"Formula '{name}' is already registered")
    _FORMULAS[key] = (name, formula)
    return formula


def get_formula(name: str) -> Formula | None:
    """
    Resolve a registered formula by name (case-insensitive).

    :param name: Formula name.
    :returns: Formula callable or ``None``.
    """

    record = _FORMULAS.get(name.lower())
    return record[1] if record is not None else None


def list_registered_formulas() -> list[str]:
    """
    Return canonical names of plugin-registered formulas.
    """

    return sorted(record[0] for record in _FORMULAS.values())


@overload
def register_localization_technique(
    name: str,
    handler: LocalizationTechnique,
    *,
    replace: bool = False,
) -> LocalizationTechnique: ...


@overload
def register_localization_technique(
    name: str,
    handler: None = None,
    *,
    replace: bool = False,
) -> Callable[[LocalizationTechnique], LocalizationTechnique]: ...


def register_localization_technique(
    name: str,
    handler: LocalizationTechnique | None = None,
    *,
    replace: bool = False,
) -> LocalizationTechnique | Callable[[LocalizationTechnique], LocalizationTechnique]:
    """
    Register a localization technique handler.

    Technique handlers are called by :meth:`forecast.subject.Subject.sfl`,
    including the default ``"sfl"`` technique.

    Supports both direct and decorator style:

    .. code-block:: python

        register_localization_technique("llm", handler)

        @register_localization_technique("llm")
        def handler(*, subject, location_type, formula="Tarantula", **kwargs): ...

    :param name: Technique name.
    :param handler: Callable that performs localization.
    :param replace: Whether to overwrite existing registrations.
    :raises ValueError: If a technique already exists and *replace* is ``False``.
    """

    if handler is None:
        def _decorator(
            decorated_handler: LocalizationTechnique,
        ) -> LocalizationTechnique:
            register_localization_technique(
                name=name,
                handler=decorated_handler,
                replace=replace,
            )
            return decorated_handler

        return _decorator

    key = name.lower()
    if key in _LOCALIZATION_TECHNIQUES and not replace:
        raise ValueError(f"Localization technique '{name}' is already registered")
    _LOCALIZATION_TECHNIQUES[key] = handler
    return handler


def get_localization_technique(name: str) -> LocalizationTechnique | None:
    """
    Resolve a registered localization technique by name.

    :param name: Technique name.
    :returns: Technique handler or ``None``.
    """

    return _LOCALIZATION_TECHNIQUES.get(name.lower())


def list_localization_techniques() -> list[str]:
    """
    Return sorted localization technique names.
    """

    return sorted(_LOCALIZATION_TECHNIQUES.keys())


@overload
def register_subject_class(
    name: str,
    cls: SubjectClass,
    *,
    replace: bool = False,
) -> SubjectClass: ...


@overload
def register_subject_class(
    name: str,
    cls: None = None,
    *,
    replace: bool = False,
) -> Callable[[SubjectClass], SubjectClass]: ...


def register_subject_class(
    name: str,
    cls: SubjectClass | None = None,
    *,
    replace: bool = False,
) -> SubjectClass | Callable[[SubjectClass], SubjectClass]:
    """
    Register a subject class under a string key.

    Supports both direct and decorator style:

    .. code-block:: python

        register_subject_class("repo", RepoSubject)

        @register_subject_class("repo")
        class RepoSubject(Subject): ...

    :param name: Subject kind key.
    :param cls: Subject class.
    :param replace: Whether to overwrite existing registrations.
    :returns: The input class (decorator-friendly).
    :raises ValueError: If the name already exists and *replace* is ``False``.
    """

    if cls is None:
        def _decorator(decorated_cls: SubjectClass) -> SubjectClass:
            return register_subject_class(
                name=name,
                cls=decorated_cls,
                replace=replace,
            )

        return _decorator

    key = name.lower()
    if key in _SUBJECTS and not replace:
        raise ValueError(f"Subject '{name}' is already registered")
    _SUBJECTS[key] = cls
    return cls


def get_subject_class(name: str) -> SubjectClass | None:
    """
    Resolve a registered subject class by name.

    :param name: Subject kind key.
    :returns: Subject class or ``None``.
    """

    return _SUBJECTS.get(name.lower())


def list_subjects() -> list[str]:
    """
    Return sorted registered subject kinds.
    """

    return sorted(_SUBJECTS.keys())


def create_subject(name: str, **kwargs) -> Any:
    """
    Instantiate a registered subject class by kind.

    :param name: Subject kind key.
    :param kwargs: Constructor arguments forwarded to the subject class.
    :returns: Constructed subject instance.
    :raises ValueError: If *name* is unknown.
    """

    cls = get_subject_class(name)
    if cls is None:
        raise ValueError(f"Unknown subject kind '{name}'.")
    return cls(**kwargs)
