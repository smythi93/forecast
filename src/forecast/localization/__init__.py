"""
Fault localization utilities for FORECAST.

This sub-package contains the data structures and algorithms used to rank
suspicious source-code locations after test-coverage data has been collected:

* :mod:`~forecast.localization.spectra` — :class:`~forecast.localization.spectra.TestResult`
  and :class:`~forecast.localization.spectra.Spectra` for aggregating per-test
  line/function coverage hit counts.
* :mod:`~forecast.localization.locations` — :class:`~forecast.localization.locations.Line`
  and :class:`~forecast.localization.locations.Function` result types.
* :mod:`~forecast.localization.sfl` — thin wrapper around ``sflkit`` that resolves
  similarity-formula names at runtime via :func:`~forecast.localization.sfl.get_formula`.
"""

from forecast.localization.localization import (
    Localization,
    SpectrumLocalization,
    default_localize,
)
from forecast.localization.sfl import get_formula, list_formulas, register_formula

__all__ = [
    "Localization",
    "SpectrumLocalization",
    "default_localize",
    "get_formula",
    "list_formulas",
    "register_formula",
]
