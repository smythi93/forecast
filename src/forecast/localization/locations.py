"""
Fault-localization location types for FORECAST.

Provides the result types returned by :meth:`~forecast.subject.Subject.sfl`,
:meth:`~forecast.subject.Subject.sfl_line`, and
:meth:`~forecast.subject.Subject.sfl_function`.

Classes
-------
* :class:`LocationType` — enum selecting line vs. function granularity.
* :class:`Location`     — abstract base carrying file, line, and suspiciousness.
* :class:`Line`         — a single source line identified as suspicious.
* :class:`Function`     — a function/method identified as suspicious.
"""

import abc
import enum
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class LocationType(enum.Enum):
    """
    Granularity level for spectrum-based fault localization (SFL).

    :cvar LINE: Rank individual source lines.
    :cvar FUNCTION: Rank functions / methods.
    """

    LINE = "line"
    FUNCTION = "function"


@dataclass
class Location(abc.ABC):
    """
    Abstract base class for a source-code location ranked by SFL.

    Ordering operators compare by :attr:`suspiciousness` so that a
    plain ``sorted(..., reverse=True)`` call produces the most-suspicious
    locations first.

    :param file: Path to the source file containing this location.
    :param line: 1-based line number within *file*.
    :param suspiciousness: Suspiciousness score assigned by the SFL formula
        (higher = more suspicious).
    """

    file: str | Path | os.PathLike[str]
    line: int
    suspiciousness: float = 0.0

    def __gt__(self, other: "Location") -> bool:
        """Return ``True`` if this location is *more* suspicious than *other*."""
        if not isinstance(other, Location):
            raise TypeError(f"Cannot compare Location with {type(other)}")
        return self.suspiciousness > other.suspiciousness

    def __lt__(self, other: "Location") -> bool:
        """Return ``True`` if this location is *less* suspicious than *other*."""
        if not isinstance(other, Location):
            raise TypeError(f"Cannot compare Location with {type(other)}")
        return self.suspiciousness < other.suspiciousness

    def __ge__(self, other: "Location") -> bool:
        """Return ``True`` if this location is at least as suspicious as *other*."""
        if not isinstance(other, Location):
            raise TypeError(f"Cannot compare Location with {type(other)}")
        return self.suspiciousness >= other.suspiciousness

    def __le__(self, other: "Location") -> bool:
        """Return ``True`` if this location is at most as suspicious as *other*."""
        if not isinstance(other, Location):
            raise TypeError(f"Cannot compare Location with {type(other)}")
        return self.suspiciousness <= other.suspiciousness

    def __hash__(self) -> int:
        """Hash based on ``(file, line)`` so locations can be used in sets/dicts."""
        return hash((self.file, self.line))


@dataclass
class Line(Location):
    """
    A single source line identified as suspicious by SFL.

    :param file: Path to the source file.
    :param line: 1-based line number.
    :param suspiciousness: Suspiciousness score (higher = more suspicious).
    """

    def __eq__(self, other: "Line") -> bool:
        """Two :class:`Line` objects are equal when they refer to the same file and line."""
        if not isinstance(other, Line):
            return NotImplemented
        return self.file == other.file and self.line == other.line


@dataclass
class Function(Location):
    """
    A function or method identified as suspicious by SFL.

    :param file: Path to the source file containing the function.
    :param line: 1-based start line of the function definition.
    :param suspiciousness: Suspiciousness score (higher = more suspicious).
    :param name: Simple function name (without class prefix).
    :param class_name: Class that owns this method, or ``None`` for
        module-level functions.
    :param end_line: Last line of the function body, or ``None`` if unknown.
    """

    name: str = ""
    class_name: Optional[str] = None
    end_line: Optional[int] = None

    def __hash__(self) -> int:
        """Hash based on ``(file, line, name, class_name)``."""
        return hash((self.file, self.line, self.name, self.class_name))

    def __eq__(self, other: "Function") -> bool:
        """Two :class:`Function` objects are equal when all four key fields match."""
        if not isinstance(other, Function):
            return NotImplemented
        return (
            self.file == other.file
            and self.line == other.line
            and self.name == other.name
            and self.class_name == other.class_name
        )
