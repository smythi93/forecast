"""
Test runners for FORECAST.

This sub-package provides two families of test runners:

* :mod:`~forecast.runner.coverage_runner` — runs tests under ``coverage.py`` and
  returns :class:`~forecast.localization.spectra.TestResult` objects carrying both
  line-level and function-level coverage data.
* :mod:`~forecast.runner.event_runner` — runs instrumented tests and collects
  ``sflkit`` execution-trace event files.

Both families have local (subprocess) and Docker (SDK) variants.
Utility helpers for Docker container I/O live in :mod:`~forecast.runner.utils`.
"""
