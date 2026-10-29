"""
Test selection for FORECAST diagnosis.

A diagnosis contrasts what failing runs do against what passing runs do.  Not
every passing test contributes to that contrast: FORECAST only compares
executions at *discriminative* nodes, which are functions reached by both a
passing and a failing run.  A passing test that never executes any code the
failing tests execute produces no feature vector at any such node, so it
cannot change the diagnosis — it only costs a full instrumented run.

Real benchmarks make this expensive.  A SWE-bench instance can list hundreds of
``PASS_TO_PASS`` tests, and tracing each one is far costlier than running it,
so tracing them all is often the difference between a feasible and an
infeasible instance.

This module therefore selects the passing tests worth tracing from cheap
coverage data:

* :func:`select_passing_tests` keeps the tests that overlap the code executed
  by the failing tests, ranked so that the sharpest contrasts come first.
  It uses no information beyond the tests themselves.
* :func:`select_tests_by_patch` additionally restricts to tests that reach the
  code a reference patch changes.  This consults the fix, so it is an
  **oracle-based** fallback for instances that remain infeasible otherwise; the
  result is flagged accordingly so that reporting can keep such instances
  separate.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from forecast.localization.spectra import TestResult
from forecast.logger import LOGGER

#: ``diff --git`` / ``+++`` style header of a changed file.
_DIFF_FILE_RE = re.compile(r"^\+\+\+ [ab]/(?P<path>.+?)\s*$")
#: Hunk header carrying the line range in the post-image.
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


@dataclass(frozen=True)
class SelectedTests:
    """
    The outcome of a test-selection decision.

    :param failing: Failing tests to trace.  Never reduced — every failing test
        is evidence about the fault.
    :param passing: Passing tests to trace, most informative first.
    :param dropped: Passing tests that were excluded.
    :param strategy: Name of the strategy that produced this selection.
    :param uses_oracle: ``True`` when the selection consulted information that
        would not be available to a repair agent, such as the reference patch.
    :param statistics: Diagnostic counters describing the decision, suitable
        for inclusion in a run report.
    """

    failing: list[str]
    passing: list[str]
    dropped: list[str] = field(default_factory=list)
    strategy: str = "all"
    uses_oracle: bool = False
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def test_ids(self) -> list[str]:
        """All tests to trace, failing first."""
        return list(self.failing) + list(self.passing)


def _covered_lines(result: TestResult) -> set[tuple[str, int]]:
    """Return the ``(file, line)`` pairs a test executed."""
    return {
        (file, line)
        for file, lines in (result.line_coverage or {}).items()
        for line in lines
    }


def failing_footprint(results: Iterable[TestResult]) -> set[tuple[str, int]]:
    """
    Return the union of the code executed by the failing tests.

    :param results: Coverage results of the failing tests.
    :returns: The ``(file, line)`` pairs reached by at least one failing test.
    """
    footprint: set[tuple[str, int]] = set()
    for result in results:
        footprint |= _covered_lines(result)
    return footprint


def select_passing_tests(
    *,
    failing_results: Sequence[TestResult],
    passing_results: Sequence[TestResult],
    max_passing_tests: int = 0,
    min_overlap: int = 1,
) -> SelectedTests:
    """
    Select the passing tests worth tracing, by coverage overlap.

    A passing test qualifies when it executes at least *min_overlap* lines that
    a failing test also executes; only such tests can share a function — and
    therefore a feature vector — with a failing run.  Qualifying tests are
    ranked by how much of the failing code they reach, and ties are broken in
    favour of the test that executes the least code besides it: that test gives
    the sharpest contrast and is also the cheapest to trace.

    :param failing_results: Coverage results of the failing tests.
    :param passing_results: Coverage results of the candidate passing tests.
    :param max_passing_tests: Keep at most this many passing tests.  ``0``
        keeps every qualifying test.
    :param min_overlap: Minimum number of shared lines required to qualify.
    :returns: The selection, with the failing tests kept unchanged.
    """
    failing_ids = [result.test_id for result in failing_results]
    footprint = failing_footprint(failing_results)

    if not footprint:
        # Without failing coverage there is nothing to compare against, so we
        # cannot justify dropping anything.
        LOGGER.warning(
            "No coverage recorded for the failing tests; keeping all passing tests."
        )
        return SelectedTests(
            failing=failing_ids,
            passing=[result.test_id for result in passing_results],
            strategy="all",
            statistics={"reason": "no failing coverage"},
        )

    scored: list[tuple[int, int, str]] = []
    unrelated: list[str] = []
    for result in passing_results:
        covered = _covered_lines(result)
        overlap = len(covered & footprint)
        if overlap < max(1, min_overlap):
            unrelated.append(result.test_id)
            continue
        # Rank by shared code first, then by how little unrelated code the test
        # drags along.
        scored.append((-overlap, len(covered), result.test_id))

    scored.sort()
    ranked = [test_id for _, _, test_id in scored]

    dropped = list(unrelated)
    if max_passing_tests and len(ranked) > max_passing_tests:
        dropped.extend(ranked[max_passing_tests:])
        ranked = ranked[:max_passing_tests]

    statistics = {
        "failing_lines": len(footprint),
        "candidates": len(passing_results),
        "selected": len(ranked),
        "dropped_without_overlap": len(unrelated),
        "dropped_over_limit": len(dropped) - len(unrelated),
        "max_passing_tests": max_passing_tests,
        "min_overlap": min_overlap,
    }
    LOGGER.info(
        "Coverage-overlap selection kept %d of %d passing tests "
        "(%d share no code with the failure).",
        len(ranked),
        len(passing_results),
        len(unrelated),
    )
    return SelectedTests(
        failing=failing_ids,
        passing=ranked,
        dropped=dropped,
        strategy="coverage-overlap",
        statistics=statistics,
    )


def changed_regions(patch: str) -> dict[str, set[int]]:
    """
    Return the lines a unified diff changes, per file.

    :param patch: Unified diff text.
    :returns: Mapping from repository-relative file path to the set of
        post-image line numbers the patch adds or replaces.  Files whose hunks
        only delete lines map to the line the deletion occurs at.
    """
    regions: dict[str, set[int]] = {}
    current: str | None = None
    line_number = 0
    for raw in (patch or "").splitlines():
        header = _DIFF_FILE_RE.match(raw)
        if header:
            path = header.group("path")
            current = None if path == "/dev/null" else path
            if current is not None:
                regions.setdefault(current, set())
            continue
        hunk = _DIFF_HUNK_RE.match(raw)
        if hunk:
            line_number = int(hunk.group("start"))
            continue
        if current is None:
            continue
        if raw.startswith("+"):
            regions[current].add(line_number)
            line_number += 1
        elif raw.startswith("-"):
            # The deleted line is gone from the post-image, but the surrounding
            # code is what a test would execute.
            regions[current].add(line_number)
        elif raw.startswith(" "):
            line_number += 1
    return {path: lines for path, lines in regions.items() if lines}


def _matches_changed_region(
    result: TestResult,
    regions: Mapping[str, set[int]],
    context: int,
) -> bool:
    """Return whether *result* executes code near a changed region."""
    for covered_file, covered_lines in (result.line_coverage or {}).items():
        normalized = covered_file.replace("\\", "/")
        for changed_file, changed_lines in regions.items():
            # Coverage paths are absolute; patch paths are repository-relative.
            if not normalized.endswith(changed_file):
                continue
            if context <= 0:
                if covered_lines & changed_lines:
                    return True
                continue
            for line in covered_lines:
                if any(abs(line - changed) <= context for changed in changed_lines):
                    return True
    return False


def select_tests_by_patch(
    *,
    patch: str,
    failing_results: Sequence[TestResult],
    passing_results: Sequence[TestResult],
    max_passing_tests: int = 0,
    context: int = 0,
) -> SelectedTests:
    """
    Restrict the passing tests to those reaching the code a patch changes.

    This is the fallback for instances that stay infeasible under
    :func:`select_passing_tests`.  It consults the reference fix, which a
    repair agent would not have, so the returned selection is marked with
    :attr:`SelectedTests.uses_oracle` and results obtained with it must be
    reported separately from the main evaluation.

    :param patch: Reference (gold) patch as a unified diff.
    :param failing_results: Coverage results of the failing tests.
    :param passing_results: Coverage results of the candidate passing tests.
    :param max_passing_tests: Keep at most this many passing tests.  ``0``
        keeps every qualifying test.
    :param context: Number of lines around a changed line that still count as
        reaching it.  ``0`` requires the executed line to be a changed line.
    :returns: The oracle-based selection.
    """
    failing_ids = [result.test_id for result in failing_results]
    regions = changed_regions(patch)
    if not regions:
        LOGGER.warning(
            "Reference patch changed no identifiable lines; "
            "falling back to coverage-overlap selection."
        )
        return select_passing_tests(
            failing_results=failing_results,
            passing_results=passing_results,
            max_passing_tests=max_passing_tests,
        )

    footprint = failing_footprint(failing_results)
    scored: list[tuple[int, int, str]] = []
    dropped: list[str] = []
    for result in passing_results:
        if not _matches_changed_region(result, regions, context):
            dropped.append(result.test_id)
            continue
        covered = _covered_lines(result)
        scored.append((-len(covered & footprint), len(covered), result.test_id))

    scored.sort()
    ranked = [test_id for _, _, test_id in scored]
    if max_passing_tests and len(ranked) > max_passing_tests:
        dropped.extend(ranked[max_passing_tests:])
        ranked = ranked[:max_passing_tests]

    statistics = {
        "changed_files": len(regions),
        "changed_lines": sum(len(lines) for lines in regions.values()),
        "candidates": len(passing_results),
        "selected": len(ranked),
        "dropped": len(dropped),
        "context": context,
    }
    LOGGER.warning(
        "Oracle-based patch selection kept %d of %d passing tests. "
        "This uses the reference fix and must be reported separately.",
        len(ranked),
        len(passing_results),
    )
    return SelectedTests(
        failing=failing_ids,
        passing=ranked,
        dropped=dropped,
        strategy="reference-patch",
        uses_oracle=True,
        statistics=statistics,
    )
