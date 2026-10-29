"""
Coverage spectra for FORECAST.

A :class:`Spectra` aggregates per-test :class:`TestResult` objects into
per-element hit/miss counts and exposes helpers to score each element with
any SFL similarity formula.

The four counts passed to every formula are::

    formula(pass_observed, pass_not_observed, fail_observed, fail_not_observed)

where *observed* means the element (line or function) was executed by that test.

:class:`FunctionKey` uniquely identifies a function/method within the project
and is produced directly by the coverage runner — no post-hoc AST mapping is
needed.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Function identifier (collected at coverage time via AST + executed lines)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionKey:
    """
    Immutable identifier for a function or method within a project.

    Instances are collected directly from the ``coverage.py`` JSON report by
    the coverage runner, so no additional AST pass is required.

    :param file: Absolute or project-relative path to the source file.
    :param name: Simple function name (without class prefix).
    :param class_name: Class that owns this method, or ``None`` for
        module-level functions.
    :param start_line: 1-based line number of the ``def`` statement.
    :param end_line: Last line of the function body as inferred from
        ``executed_lines + missing_lines``, or ``None`` if unavailable.
    """

    file: str
    name: str
    class_name: Optional[str]
    start_line: int
    end_line: Optional[int]

    def serialize(self) -> dict[str, str | int | dict[str, str | int | None]]:
        """
        Convert to a JSON-serializable dictionary.

        :returns: Dictionary with keys ``file``, ``name``, ``class_name``,
            ``start_line``, ``end_line``.
        """
        return {
            "file": self.file,
            "name": self.name,
            "class_name": self.class_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @staticmethod
    def load(data: dict[str, str | int | None]) -> "FunctionKey":
        """
        Reconstruct a :class:`FunctionKey` from a serialized dictionary.

        :param data: Dictionary as produced by :meth:`serialize`.
        :returns: A new :class:`FunctionKey` instance.
        """
        return FunctionKey(
            file=str(data["file"]),
            name=str(data["name"]),
            class_name=data.get("class_name"),
            start_line=int(data["start_line"]),  # type: ignore[argument]
            end_line=data.get("end_line"),
        )


# ---------------------------------------------------------------------------
# Raw test result (produced by any subject's coverage runner)
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """
    Coverage data for a single test execution.

    Produced by the coverage runner and consumed by :class:`Spectra` to
    build hit-count tables for SFL scoring.

    :param test_id: Pytest node ID (e.g. ``tests/test_foo.py::test_bar``)
        or a system-test identifier.
    :param passed: ``True`` if the test exited with return code 0.
    :param line_coverage: Mapping from absolute file path to the set of
        line numbers executed during this test.
    :param function_coverage: Set of :class:`FunctionKey` objects for every
        function/method entered during this test.
    """

    test_id: str
    passed: bool
    # {absolute_file_path: set_of_executed_line_numbers}
    line_coverage: dict[str, set[int]] = field(default_factory=dict)
    # set of functions that were entered during this test
    function_coverage: set[FunctionKey] = field(default_factory=set)

    def __hash__(self) -> int:
        """Hash based on :attr:`test_id` so results can be used in sets/dicts."""
        return hash(self.test_id)

    def __eq__(self, other: object) -> bool:
        """Two results are equal when their :attr:`test_id` values match."""
        if not isinstance(other, TestResult):
            return NotImplemented
        return self.test_id == other.test_id

    def serialize(
        self,
    ) -> dict[str, str | int | dict[str, set[int]] | list[dict[str, str | int | None]]]:
        """
        Convert to a JSON-serializable dictionary.

        :returns: Dictionary with keys ``test_id``, ``passed`` (as int),
            ``line_coverage`` (lines sorted), and ``function_coverage``.
        """
        return {
            "test_id": self.test_id,
            "passed": int(self.passed),
            "line_coverage": {
                fpath: sorted(lines) for fpath, lines in self.line_coverage.items()
            },
            "function_coverage": [fkey.serialize() for fkey in self.function_coverage],
        }

    @staticmethod
    def load(
        data: dict[
            str, str | int | dict[str, list[int]] | list[dict[str, str | int | None]]
        ],
    ) -> "TestResult":
        """
        Reconstruct a :class:`TestResult` from a serialized dictionary.

        :param data: Dictionary as produced by :meth:`serialize`.
        :returns: A new :class:`TestResult` instance.
        """
        return TestResult(
            test_id=data["test_id"],
            passed=bool(data["passed"]),
            line_coverage={
                fpath: set(map(int, lines))
                for fpath, lines in data["line_coverage"].items()
            },
            function_coverage=set(
                FunctionKey.load(fkey_data) for fkey_data in data["function_coverage"]
            ),
        )


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------


class Spectra:
    """
    Aggregates :class:`TestResult` objects into per-element hit/miss counts
    and exposes helpers to score each element with any SFL similarity formula.

    After construction the spectra can be queried via :meth:`score_line` and
    :meth:`score_function` to obtain suspiciousness scores for every covered
    line or function respectively.

    :param results: List of :class:`TestResult` objects from a test run.
    """

    def __init__(self, results: list[TestResult]):
        self.results = results
        # Count passing and failing tests for normalization inside SFL formulas
        self.total_pass = sum(1 for r in results if r.passed)
        self.total_fail = sum(1 for r in results if not r.passed)

        # (file, line) -> {pass_hit, fail_hit}
        self._line_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
            lambda: {"pass_hit": 0, "fail_hit": 0}
        )
        # FunctionKey -> {pass_hit, fail_hit}
        self._func_counts: dict[FunctionKey, dict[str, int]] = defaultdict(
            lambda: {"pass_hit": 0, "fail_hit": 0}
        )

        # Accumulate hit counts from every test result
        for r in results:
            bucket = "pass_hit" if r.passed else "fail_hit"
            for fpath, lines in r.line_coverage.items():
                for line in lines:
                    self._line_counts[(fpath, line)][bucket] += 1
            for fkey in r.function_coverage:
                self._func_counts[fkey][bucket] += 1

    def serialize(self) -> dict[str, list[dict[str, str | int | dict[str, int]]]]:
        """
        Convert to a JSON-serializable dictionary.

        :returns: Dictionary with a ``results`` key containing the serialized
            list of :class:`TestResult` objects.
        """
        return {
            "results": [r.serialize() for r in self.results],
        }

    @staticmethod
    def load(data: dict[str, list[dict[str, str | int | dict[str, int]]]]) -> "Spectra":
        """
        Reconstruct a :class:`Spectra` from a serialized dictionary.

        :param data: Dictionary as produced by :meth:`serialize`.
        :returns: A new :class:`Spectra` instance.
        """
        return Spectra(results=[TestResult.load(r_data) for r_data in data["results"]])

    # ------------------------------------------------------------------
    # Core scoring helpers
    # ------------------------------------------------------------------

    def _score(
        self,
        counts: dict,
        formula: Callable[[int, int, int, int], float],
    ) -> dict:
        """
        Compute suspiciousness scores for all elements tracked in *counts*.

        :param counts: Mapping from element key to ``{pass_hit, fail_hit}``
            count dictionary (either ``_line_counts`` or ``_func_counts``).
        :param formula: SFL similarity formula callable with signature
            ``f(ep, np, ef, nf) -> float`` where:

            - *ep* — passing tests that covered the element
            - *np* — passing tests that did **not** cover the element
            - *ef* — failing tests that covered the element
            - *nf* — failing tests that did **not** cover the element

        :returns: Mapping from element key to suspiciousness score.
        """
        scores = {}
        for key, c in counts.items():
            ep = c["pass_hit"]  # passing tests that covered element
            np_ = self.total_pass - ep  # passing tests that did NOT cover it
            ef = c["fail_hit"]  # failing tests that covered element
            nf = self.total_fail - ef  # failing tests that did NOT cover it
            scores[key] = formula(ep, np_, ef, nf)
        return scores

    def score_line(
        self,
        formula: Callable[[int, int, int, int], float],
    ) -> dict[tuple[str, int], float]:
        """
        Score every covered line using *formula*.

        :param formula: SFL similarity formula — see :meth:`_score`.
        :returns: Mapping from ``(file_path, line_number)`` tuple to
            suspiciousness score.
        """
        return self._score(self._line_counts, formula)

    def score_function(
        self,
        formula: Callable[[int, int, int, int], float],
    ) -> dict[FunctionKey, float]:
        """
        Score every covered function using *formula*.

        :param formula: SFL similarity formula — see :meth:`_score`.
        :returns: Mapping from :class:`FunctionKey` to suspiciousness score.
        """
        return self._score(self._func_counts, formula)
