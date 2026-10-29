"""
Default fault-localization implementations.

This module provides a concrete, reusable default localizer based on spectra
and SFL formulas, plus an abstract base to support additional techniques.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING

from forecast.localization.locations import Function, Line, Location, LocationType
from forecast.localization.sfl import get_formula
from forecast.localization.spectra import Spectra
from forecast.plugins.registry import register_localization_technique

if TYPE_CHECKING:
    from forecast.subject import Subject


class Localization(abc.ABC):
    """
    Abstract interface for fault-localization techniques.

    Implementations operate on a subject and produce ranked suspicious
    locations. This supports both spectra-based and non-spectra techniques
    (e.g. LLM-driven localization).
    """

    def __init__(self, subject: "Subject"):
        """
        Initialize a localizer for one subject.

        :param subject: Subject to localize.
        """
        self.subject = subject

    @abc.abstractmethod
    def localize_line(self, formula: str = "Tarantula") -> list[Line]:
        """
        Rank suspicious source lines.
        """

    @abc.abstractmethod
    def localize_function(self, formula: str = "Tarantula") -> list[Function]:
        """
        Rank suspicious functions.
        """

    def localize(
        self,
        location_type: LocationType = LocationType.FUNCTION,
        formula: str = "Tarantula",
    ) -> list[Location]:
        """
        Dispatch localization by granularity.

        :param location_type: ``LINE`` or ``FUNCTION``.
        :param formula: Formula name used by the technique.
        :returns: Ranked suspicious locations.
        :raises ValueError: If *location_type* is unsupported.
        """

        match location_type:
            case LocationType.LINE:
                return self.localize_line(formula=formula)
            case LocationType.FUNCTION:
                return self.localize_function(formula=formula)
            case _:
                raise ValueError(
                    f"Unsupported location type: {location_type}. "
                    f"Expected one of {list(LocationType)}."
                )


class SpectrumLocalization(Localization):
    """
    Default SFL-based localizer.

    Uses :class:`Spectra` plus formula lookup from
    :mod:`forecast.localization.sfl`.
    """

    def __init__(self, subject: "Subject"):
        super().__init__(subject)
        self._spectra: Spectra | None = None

    @property
    def test_base(self) -> Path | None:
        base = getattr(self.subject, "test_base", None)
        return Path(base) if base is not None else None

    def _get_spectra(self) -> Spectra:
        if self._spectra is None:
            self._spectra = self.subject.collect_spectra()
        return self._spectra

    def localize_line(self, formula: str = "Tarantula") -> list[Line]:
        scores = self._get_spectra().score_line(get_formula(formula))
        if self.test_base is not None:
            scores = {
                (file_path, line_no): score
                for (file_path, line_no), score in scores.items()
                if not Path(file_path).is_relative_to(self.test_base)
            }
        return sorted(
            [
                Line(file=file_path, line=line_no, suspiciousness=score)
                for (file_path, line_no), score in scores.items()
            ],
            reverse=True,
        )

    def localize_function(self, formula: str = "Tarantula") -> list[Function]:
        scores = self._get_spectra().score_function(get_formula(formula))
        if self.test_base is not None:
            scores = {
                fkey: score
                for fkey, score in scores.items()
                if not Path(fkey.file).is_relative_to(self.test_base)
            }
        return sorted(
            [
                Function(
                    file=fkey.file,
                    line=fkey.start_line,
                    suspiciousness=score,
                    name=fkey.name,
                    class_name=fkey.class_name,
                    end_line=fkey.end_line,
                )
                for fkey, score in scores.items()
            ],
            reverse=True,
        )


def default_localize(
    subject: "Subject",
    location_type: LocationType = LocationType.FUNCTION,
    formula: str = "Tarantula",
) -> list[Location]:
    """
    Convenience wrapper around :class:`SpectrumLocalization`.
    """

    return SpectrumLocalization(subject=subject).localize(
        location_type=location_type,
        formula=formula,
    )


def _builtin_sfl_technique(
    *,
    subject: "Subject",
    location_type: LocationType = LocationType.FUNCTION,
    formula: str = "Tarantula",
    **_kwargs,
) -> list[Location]:
    """
    Built-in localization-technique adapter backed by spectra-based SFL.
    """

    return SpectrumLocalization(subject=subject).localize(
        location_type=location_type,
        formula=formula,
    )


register_localization_technique("sfl", _builtin_sfl_technique, replace=True)
