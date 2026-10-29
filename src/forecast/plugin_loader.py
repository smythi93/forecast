"""
Plugin loading for FORECAST.

Discovers and loads all third-party FORECAST plugins registered under the
``forecast.plugins`` entry-point group.  Call :func:`load_plugins` early in
the process lifecycle (e.g., at CLI start-up) so that plugins can register
custom agents, formulas, or subjects before any pipeline step runs.
"""

import inspect
import importlib
from dataclasses import dataclass
from importlib.metadata import entry_points
from types import ModuleType
from typing import Callable, Any

from forecast.plugins.registry import (
    list_agents,
    register_agent_class,
    register_formula,
    register_localization_technique,
    register_model_provider,
    register_subject_class,
)


@dataclass(frozen=True)
class PluginContext:
    """
    Registration API exposed to plugin entry points.

    All register methods support direct-call and decorator style.
    """

    register_agent = staticmethod(register_agent_class)
    register_model_provider = staticmethod(register_model_provider)
    register_formula = staticmethod(register_formula)
    register_localization_technique = staticmethod(register_localization_technique)
    register_subject = staticmethod(register_subject_class)


_LOADED = False
_LOADED_PLUGIN_NAMES: list[str] = []
_FALLBACK_PLUGIN_MODULES = [
    # Safe built-in fallback when package entry points are unavailable.
    ("MockAgent", "forecast.agents.mock_agent.mock_agent"),
]


def _call_plugin_callable(func: Callable[..., Any], context: PluginContext) -> None:
    """
    Call *func* either as ``func(context)`` or ``func()`` based on signature.
    """

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        func()
        return

    params = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = [p for p in params if p.default is p.empty]
    if len(required) >= 1:
        func(context)
    else:
        func()


def _initialize_plugin_object(obj: Any, context: PluginContext) -> None:
    """
    Initialize one loaded plugin object.

    Supported styles:
    - module with ``register(context)`` / ``register_plugin(context)`` /
      ``setup(context)`` / ``setup_plugin(context)``
    - callable entry point (called with context if it requires one arg)
    - plain import side effects (no further call)
    """

    if isinstance(obj, ModuleType):
        for hook_name in ("register_plugin", "register", "setup_plugin", "setup"):
            hook = getattr(obj, hook_name, None)
            if callable(hook):
                _call_plugin_callable(hook, context)
                return
        return
    if callable(obj):
        _call_plugin_callable(obj, context)


def load_plugins(force: bool = False) -> list[str]:
    """
    Discover and load all installed FORECAST plugins.

    Iterates over all entry points registered under the ``forecast.plugins``
    group and loads each plugin object.  Supports both the legacy
    side-effect-based style and explicit registration hooks.

    :raises Exception: Any exception raised by a plugin's ``load()`` call
        propagates to the caller unchanged; it is the plugin's responsibility
        to handle its own initialisation errors.
    """
    global _LOADED, _LOADED_PLUGIN_NAMES
    if _LOADED and not force:
        return list(_LOADED_PLUGIN_NAMES)

    eps = list(entry_points(group="forecast.plugins"))
    context = PluginContext()
    loaded: list[str] = []
    for ep in eps:
        obj = ep.load()
        _initialize_plugin_object(obj, context)
        loaded.append(ep.name)
    if not eps and not list_agents():
        for name, module_path in _FALLBACK_PLUGIN_MODULES:
            obj = importlib.import_module(module_path)
            _initialize_plugin_object(obj, context)
            if name not in loaded:
                loaded.append(name)
    _LOADED = True
    _LOADED_PLUGIN_NAMES = loaded
    return list(loaded)
