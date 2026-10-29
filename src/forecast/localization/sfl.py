"""
SFL formula registry for FORECAST.

Provides :func:`get_formula` as the sole public entry-point for obtaining
``sflkit`` similarity-formula callables by name, and :func:`list_formulas`
to enumerate all available formulas at runtime.

The formulas are looked up lazily from :mod:`sflkit.analysis.similarity`
and cached after the first access to avoid repeated attribute lookups.
"""

import importlib
from typing import Callable, overload

from forecast.plugins.registry import (
    get_formula as get_registered_formula,
    list_registered_formulas,
    register_formula as register_plugin_formula,
)

# Lazily populated cache of {name -> callable}
_FORMULA_CACHE: dict[str, Callable[[int, int, int, int], float]] = {}


def get_formula(name: str) -> Callable[[int, int, int, int], float]:
    """
    Look up an SFL similarity formula from :mod:`sflkit.analysis.similarity`
    by name (case-sensitive as the attribute appears in the module).

    The result is cached after the first look-up so repeated calls are cheap.

    :param name: Formula name, e.g. ``"Tarantula"``, ``"Ochiai"``, ``"DStar"``.
    :returns: A callable ``f(pass_observed, pass_not_observed,
        fail_observed, fail_not_observed) -> float`` that scores one
        program element.
    :raises ImportError: If ``sflkit`` is not installed.
    :raises ValueError: If *name* is not an attribute of
        :mod:`sflkit.analysis.similarity`.

    Example::

        formula = get_formula("Ochiai")
        score = formula(ep=5, np=2, ef=8, nf=1)
    """
    if name in _FORMULA_CACHE:
        return _FORMULA_CACHE[name]

    plugin_formula = get_registered_formula(name)
    if plugin_formula is not None:
        _FORMULA_CACHE[name] = plugin_formula
        return plugin_formula

    try:
        mod = importlib.import_module("sflkit.analysis.similarity")
    except ImportError as e:
        raise ImportError("sflkit is not installed. Run: pip install sflkit") from e

    formula = getattr(mod, name, None)
    if formula is None:
        available = [n for n in dir(mod) if not n.startswith("_")]
        raise ValueError(
            f"Formula '{name}' not found in sflkit.analysis.similarity. "
            f"Available: {available}"
        )

    _FORMULA_CACHE[name] = formula
    return formula


def list_formulas() -> list[str]:
    """
    Return all callable formula names available in :mod:`sflkit.analysis.similarity`.

    :returns: Sorted list of formula names that can be passed to
        :func:`get_formula`.
    :raises ImportError: If ``sflkit`` is not installed.
    """
    plugin_names = set(list_registered_formulas())
    try:
        mod = importlib.import_module("sflkit.analysis.similarity")
    except ImportError as e:
        # When sflkit is unavailable we still expose plugin formulas.
        if plugin_names:
            return sorted(plugin_names)
        raise ImportError("sflkit is not installed. Run: pip install sflkit") from e
    builtin_names = {
        n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))
    }
    return sorted(builtin_names | plugin_names)


@overload
def register_formula(
    name: str,
    formula: Callable[[int, int, int, int], float],
    *,
    replace: bool = False,
) -> Callable[[int, int, int, int], float]: ...


@overload
def register_formula(
    name: str,
    formula: None = None,
    *,
    replace: bool = False,
) -> Callable[
    [Callable[[int, int, int, int], float]],
    Callable[[int, int, int, int], float],
]: ...


def register_formula(
    name: str,
    formula: Callable[[int, int, int, int], float] | None = None,
    *,
    replace: bool = False,
) -> (
    Callable[[int, int, int, int], float]
    | Callable[
        [Callable[[int, int, int, int], float]],
        Callable[[int, int, int, int], float],
    ]
):
    """
    Register a plugin-provided localization formula.

    :param name: Formula name.
    Supports both direct and decorator style:

    .. code-block:: python

        register_formula("MyFormula", my_formula, replace=True)

        @register_formula("MyFormula", replace=True)
        def my_formula(ep, np, ef, nf): ...

    :param formula: Callable implementing ``(ep, np, ef, nf) -> float``.
        When omitted, returns a decorator.
    :param replace: Whether to overwrite an existing registration.
    :returns: The registered formula callable, or a decorator when *formula*
        is omitted.
    """
    if formula is None:
        def _decorator(
            decorated_formula: Callable[[int, int, int, int], float],
        ) -> Callable[[int, int, int, int], float]:
            register_formula(name, decorated_formula, replace=replace)
            return decorated_formula

        return _decorator

    register_plugin_formula(name, formula, replace=replace)
    # Keep local cache coherent, including case variations.
    for cached_name in list(_FORMULA_CACHE.keys()):
        if cached_name.lower() == name.lower():
            _FORMULA_CACHE[cached_name] = formula
    _FORMULA_CACHE[name] = formula
    return formula
