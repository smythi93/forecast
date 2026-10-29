"""
Subject hierarchy for FORECAST.

A *subject* is the program-under-test from which coverage spectra are
collected for Spectrum-Based Fault Localization (SFL), diagnoses are constructed,
and repair agents are applied to. The hierarchy is:

* :class:`Subject` — abstract base class; defines the SFL/event-collection
  API and shared state.
* :class:`LocalSubject` — subject accessible directly on the local file system.
* :class:`RepoSubject` — extends :class:`LocalSubject` with ``git clone`` /
  ``git checkout`` setup.
* :class:`DockerSubject` — subject running inside a Docker container.
* :class:`SWESubject` — extends :class:`DockerSubject` for SWE-bench Lite
  instances, including HuggingFace dataset loading helpers.
* :class:`SystemTestSubject` — abstract base for system-test (non-pytest)
  subjects driven by raw command-line inputs.
* :class:`LocalSystemTestSubject` / :class:`RepoSystemTestSubject` /
  :class:`DockerSystemTestSubject` — concrete system-test variants.
"""

import abc
import importlib
import io
import json
import os
import platform
import shlex
import string
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar, cast

import docker

from forecast.logger import LOGGER
from forecast.localization.locations import Location, LocationType, Line, Function
from forecast.localization.spectra import Spectra, TestResult
from forecast.plugins.registry import (
    get_localization_technique,
    register_subject_class,
)
from forecast.test_ids import normalize_pytest_test_id, normalize_pytest_test_ids
from forecast.runner.coverage_runner import (
    collect_test_ids,
    run_test_with_coverage,
    docker_collect_test_ids,
    docker_run_test_with_coverage,
    docker_run_tests_with_coverage_batch,
    run_test_without_coverage,
    docker_run_test_without_coverage,
    run_input_with_coverage,
    docker_run_input_with_coverage,
)
from forecast.runner.event_runner import (
    run_test_and_collect_events,
    docker_run_test_and_collect_events,
    run_input_and_collect_events,
    docker_run_input_and_collect_events,
)
from forecast.runner.trace_budget import TraceBudget
from forecast.runner.utils import (
    read_file_from_client,
    write_file_to_container,
    docker_install_all,
)

if TYPE_CHECKING:
    from swebench.harness.constants import SWEbenchInstance

LOG = LOGGER.getChild("subject")


def safe(s: str | os.PathLike[str]) -> str:
    """Make a string safe for use as an identifier by replacing problematic characters."""
    s = str(s)
    for c in string.whitespace + string.punctuation:
        s = s.replace(c, "_")
    return s


def split_image_reference(image: str) -> tuple[str, str | None]:
    """Split a Docker image reference into ``(repository, tag)``."""
    ref = (image or "").strip()
    if not ref:
        raise ValueError("Docker image reference is empty.")

    last_colon = ref.rfind(":")
    last_slash = ref.rfind("/")
    if last_colon > last_slash:
        return ref[:last_colon], ref[last_colon + 1 :]
    return ref, None


def build_modification_reference(image: str) -> str:
    """Build a modification image reference for in-place snapshots."""
    repository, tag = split_image_reference(image)
    if tag:
        return f"{repository}-mod:{tag}"
    return f"{repository}-mod:latest"


def _resilient_repo_install_script(repo_script_list: list[str]) -> list[str]:
    """
    Normalize flaky SWE-bench ``pip install`` steps.

    1) ``--no-use-pep517`` is no longer accepted by modern pip versions and can
       make setup scripts fail with exit code ``2``.
    2) Editable installs can fail for some old projects; retrying with a regular
       install keeps image setup moving.
    """
    updated: list[str] = []

    def _as_pip_install_tokens(command: str) -> list[str] | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if len(tokens) < 4:
            return None
        if tokens[:4] != ["python", "-m", "pip", "install"]:
            return None
        return tokens

    def _drop_no_use_pep517(tokens: list[str]) -> list[str]:
        return [token for token in tokens if token != "--no-use-pep517"]

    def _is_editable_install(tokens: list[str]) -> bool:
        for token in tokens:
            if token in {"-e", "--editable"} or token.startswith("--editable="):
                return True
        return False

    def _drop_editable_flag(tokens: list[str]) -> list[str]:
        normalized: list[str] = []
        for token in tokens:
            if token in {"-e", "--editable"}:
                continue
            if token.startswith("--editable="):
                requirement = token.partition("=")[2]
                if requirement:
                    normalized.append(requirement)
                continue
            normalized.append(token)
        return normalized

    for line in repo_script_list:
        tokens = _as_pip_install_tokens(line)
        if tokens is None:
            updated.append(line)
            continue

        normalized_tokens = _drop_no_use_pep517(tokens)
        if _is_editable_install(normalized_tokens):
            editable_cmd = shlex.join(normalized_tokens)
            non_editable_cmd = shlex.join(
                _drop_editable_flag(normalized_tokens)
            )
            updated.append(f"if ! {editable_cmd}; then {non_editable_cmd}; fi")
            continue

        if normalized_tokens != tokens:
            updated.append(shlex.join(normalized_tokens))
            continue

        updated.append(line)
    return updated


SUBJECT_DIR = Path("subjects")

T = TypeVar("T")


@dataclass
class Subject(abc.ABC):
    """
    Abstract base class for a subject under test, which can be localized with SFL.

    :param path: The file system path to the subject's source base, containing the tests and code to be analyzed.
                 For a LocalSubject, this is the direct path to the source.
                 For a RepoSubject, this is the path to the local clone of the repository.
                 For a DockerSubject, this is the path inside the container where the source is located (e.g., "/app").
    :param test_base: Optional specific test directory or file to search (e.g., "tests/" or "test/").
                      If not provided, pytest will use its default discovery rules.
    :param workers: The number of parallel workers to use when running tests and collecting coverage.
                    This can speed up the process for larger subjects with many tests, but may increase resource usage.
                    Defaults to 4.
    :param test_ids: Optional list of test identifiers to run. If empty, all tests will be collected and run.
    :param fail_ids: Optional set of test identifiers that are expected to fail. If empty, all tests will be treated
                     according to their actual outcomes.
    :param timeout: The maximum time (in seconds) to allow for running tests and collecting coverage. This can be useful
                    to prevent hanging indefinitely on subjects that have issues or long setup times. Defaults to 120
                    seconds.
    """

    path: str | Path = ""
    issue: str | None = None
    patch: str | None = None
    test_base: str | Path | None = None
    workers: int = 4
    test_ids: list[str] = field(default_factory=list)
    fail_ids: set[str] = field(default_factory=set)
    passing_test_ids: set[str] = field(default_factory=set)
    failing_test_ids: set[str] = field(default_factory=set)
    timeout: int = 120
    trace_budget: TraceBudget | None = None
    _coverage: dict[str, TestResult] = field(default_factory=dict)

    def __post_init__(self):
        self.path = Path(self.path)
        if self.test_base is not None:
            self.test_base = Path(self.test_base)

    @property
    def identifier(self) -> str:
        """A unique identifier for this subject, which can be used for logging, reporting, etc."""
        return safe(str(self.path))

    @property
    def cache_path(self) -> Path:
        """The path where cached data for this subject should be stored."""
        return SUBJECT_DIR / f"{self.identifier}.json"

    def serialize(self) -> dict:
        """
        Return a dictionary representation of the subject that can be easily serialized to JSON or other formats.
        This can be useful for logging, reporting, or caching purposes.
        """
        return {
            "path": str(self.path),
            "test_base": str(self.test_base) if self.test_base else None,
            "issue": self.issue,
            "patch": self.patch,
            "test_ids": self.test_ids,
            "fail_ids": list(self.fail_ids),
            "passing_test_ids": list(self.passing_test_ids),
            "failing_test_ids": list(self.failing_test_ids),
            "coverage": {c: self._coverage[c].serialize() for c in self._coverage},
        }

    def save(self):
        """
        Save the current state of the subject, e.g., collected coverage data, spectra, etc.
        This can be used for caching or checkpointing intermediate results.
        """
        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.serialize(), f, indent=1)

    def load(self, ignore_errors: bool = True):
        """
        Load a subject from a saved state.
        This should be implemented by subclasses to match their specific serialization format.

        :param ignore_errors: If ``True``, catch and log any exceptions that occur during
            loading (e.g., file not found, JSON decode error) and continue with an empty state.
            If ``False``, propagate exceptions to the caller.
        """
        try:
            if not self.cache_path.exists():
                raise FileNotFoundError(
                    f"No cache found for subject {self.identifier} at {self.cache_path}"
                )
            with open(self.cache_path, "r") as f:
                data = json.load(f)
            self.load_from_data(data)
        except Exception as e:
            if ignore_errors:
                print(
                    f"Warning: Failed to load cache for subject {self.identifier}: {e}"
                )
            else:
                raise

    def load_from_data(self, data: dict):
        """
        Load a subject from a provided data dictionary. This can be used to load from a cache or other serialized
        format. The implementation should match the structure produced by the serialize method.

        :param data: Dictionary containing the subject state, typically produced by the serialize method.
        """
        if "issue" in data:
            self.issue = data["issue"]
        if "patch" in data:
            self.patch = data["patch"]
        if "test_ids" in data:
            self.test_ids = data["test_ids"]
        if "fail_ids" in data:
            self.fail_ids = set(data["fail_ids"])
        if "passing_test_ids" in data:
            self.passing_test_ids = set(data["passing_test_ids"])
        if "failing_test_ids" in data:
            self.failing_test_ids = set(data["failing_test_ids"])
        if "coverage" in data:
            self._coverage = {
                c: TestResult.load(data["coverage"][c])
                for c in data.get("coverage", {})
            }
        if self.failing_test_ids and not self.fail_ids:
            self.fail_ids = set(self.failing_test_ids)
        if self.fail_ids and not self.failing_test_ids:
            self.failing_test_ids = set(self.fail_ids)
        if self.test_ids and not self.passing_test_ids:
            self.passing_test_ids = set(self.test_ids) - self.failing_test_ids

    def setup(self):
        """
        Optional setup step (clone repo, pull image, etc.).
        """

    def instrument(
        self,
        *,
        src: os.PathLike[str] | str,
        dst: os.PathLike[str] | str,
        mapping_path: os.PathLike[str] | str,
        excludes: list[str] | None = None,
    ) -> Path:
        """
        Instrument source code for event collection and return the instrumented root.

        Base behavior instruments *src* into *dst* using
        :func:`forecast.instrument.instrument`. Subclasses can override this
        to additionally apply instrumentation to their own execution substrate
        (e.g., Docker images).

        :param src: Source repository root.
        :param dst: Output directory for instrumented sources.
        :param mapping_path: Output path for ``mapping.json``.
        :param excludes: Optional instrumentation exclude patterns.
        :returns: Resolved path to the instrumented source root.
        """
        from forecast.instrument import instrument as instrument_fn

        src_path = Path(src).resolve()
        dst_path = Path(dst).resolve()
        mapping = Path(mapping_path).resolve()
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        mapping.parent.mkdir(parents=True, exist_ok=True)
        instrument_fn(
            src=src_path,
            dst=dst_path,
            mapping_path=mapping,
            excludes=excludes,
        )
        return dst_path

    @abc.abstractmethod
    def run_single_test_uncached(self, test_id: str) -> TestResult:
        """
        Run a single test without consulting the cache.

        Must be implemented by subclasses to execute the test in the
        appropriate environment (local, Docker, etc.) and return its result.

        :param test_id: Pytest node ID or system-test identifier.
        :returns: :class:`~forecast.localization.spectra.TestResult` for *test_id*.
        """

    def run_single_test(self, test_id: str) -> TestResult:
        """
        Return a cached :class:`~forecast.localization.spectra.TestResult` if
        available, otherwise delegate to :meth:`run_single_test_uncached`.

        :param test_id: Pytest node ID or system-test identifier.
        :returns: :class:`~forecast.localization.spectra.TestResult` for *test_id*.
        """
        if test_id in self._coverage:
            return self._coverage[test_id]
        return self.run_single_test_uncached(test_id)

    def run_tests_modular(
        self,
        test_ids: list[str] | set[str],
        runner: Callable[..., T],
        *args,
        **kwargs,
    ) -> list[T]:
        """
        Run the specified test IDs in parallel using a process pool and return
        the ordered results.

        Uses a :class:`~concurrent.futures.ProcessPoolExecutor` with up to
        :attr:`workers` worker processes.  Results are reordered to match
        the input *test_ids* sequence even though futures complete out of order.

        :param test_ids: Test identifiers to run.
        :param runner: Callable that accepts a single test ID as its first
            positional argument (followed by *args* and *kwargs*) and returns
            a result of type *T*.
        :param args: Extra positional arguments forwarded to every *runner* call.
        :param kwargs: Extra keyword arguments forwarded to every *runner* call.
        :returns: List of results in the same order as *test_ids*.
        """
        LOG.debug(
            "Running modular test execution",
            extra={"count": len(test_ids), "workers": self.workers},
        )
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(runner, test_id, *args, **kwargs): i
                for i, test_id in enumerate(test_ids)
            }
            ordered: list[T] = [None] * len(test_ids)  # type: ignore[list-item]
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        return ordered

    def run_tests(self, test_ids: list[str] | set[str]) -> list[TestResult]:
        """
        Run the given test IDs and return ordered results.

        Delegates to :meth:`run_tests_modular` using :meth:`run_single_test`
        as the runner, so results benefit from the in-memory cache.

        :param test_ids: Collection of test identifiers to run.
        :returns: Ordered list of :class:`~forecast.localization.spectra.TestResult`.
        """
        return self.run_tests_modular(test_ids, self.run_single_test)

    @abc.abstractmethod
    def collect_events_single_uncached(
        self, test_id: str, events_file: os.PathLike[str] | str
    ) -> bool:
        """
        Collect ``sflkit`` execution-trace events for *test_id* and write them
        to *events_file*, bypassing any caching.

        Must be implemented by subclasses.

        :param test_id: Pytest node ID or system-test identifier.
        :param events_file: Host path where the binary event file should be
            written.
        :returns: ``True`` if events were collected successfully, ``False``
            otherwise.
        """

    def collect_events_single(
        self,
        test_id: str,
        file_mapping: dict[str, os.PathLike[str] | str],
        on_complete: Callable[[str, bool, Path], None] | None = None,
    ) -> bool:
        """
        Collect events for a single test identifier without using any caching mechanism. This method must be
        implemented by subclasses to define how to run a single test and collect its events, and it will be called
        by the default implementation of collect_events_single, which can include caching or other optimizations.

        :param test_id: The identifier of the test for which to collect events
            (e.g., pytest node ID or system-test identifier).
        :param file_mapping: A mapping from test identifiers to file paths where
            the collected events should be stored.
        :returns: A boolean indicating whether the event collection was successful.
        """
        events_file = file_mapping.get(test_id, "")
        if events_file:
            event_path = Path(events_file)
            if event_path.exists() and event_path.is_file():
                LOGGER.debug(
                    f"Events file already exists for test {test_id}, skipping event collection."
                )
                ok = True
            else:
                ok = self.collect_events_single_uncached(test_id, events_file)
            if on_complete is not None:
                try:
                    on_complete(test_id, ok, event_path)
                except Exception as exc:
                    LOGGER.warning(
                        "Event completion callback failed for test %s: %s",
                        test_id,
                        exc,
                    )
            return ok
        else:
            LOGGER.warning(
                f"No events file found for test {test_id}, skipping event collection."
            )
            return False

    def collect_events(
        self,
        test_ids: list[str] | set[str],
        file_mapping: dict[str, os.PathLike[str] | str],
        on_complete: Callable[[str, bool, Path], None] | None = None,
    ) -> list[bool]:
        """
        Collect events for the specified test identifiers and return a list of event objects. This is an abstract
        method that must be implemented by subclasses to define how to collect events for multiple tests, and it will
        depend on the specific context of the subject (e.g., local file system, Docker container, etc.). Subclasses
        should implement this method to run the tests according to their specific environment and return the appropriate
        event objects.

        :param test_ids: A list or set of test identifiers for which to collect events.
        :param file_mapping: A mapping from test identifiers to file paths where
            the collected events should be stored.
        :returns: A list of boolean values indicating the success of event collection for each test identifier.
        """
        return self.run_tests_modular(
            test_ids,
            self.collect_events_single,
            file_mapping,
            on_complete=on_complete,
        )

    def collect_all_events(
        self,
        file_mapping: dict[str, os.PathLike[str] | str],
        on_complete: Callable[[str, bool, Path], None] | None = None,
    ) -> list[bool]:
        """
        Collect events for all tests and return a list of event objects. This is a convenience method that calls
        collect_events with the full list of test identifiers. Subclasses can override this if they have a more
        efficient way to collect events for all tests at once, but the default implementation simply collects all test
        identifiers and collects events for them using collect_events.

        :param file_mapping: A mapping from test identifiers to file paths where the collected events should be stored.
        :returns: A list of boolean values indicating the success of event collection for each test identifier.
        """
        if not self.test_ids and (self.passing_test_ids or self.failing_test_ids):
            self.test_ids = sorted(self.passing_test_ids | self.failing_test_ids)
        if not self.test_ids:
            self.test_ids = self.collect_test_ids()
        if not self.passing_test_ids:
            self.passing_test_ids = set(self.test_ids) - self.failing_test_ids
        return self.collect_events(
            self.test_ids,
            file_mapping,
            on_complete=on_complete,
        )

    def run_all_tests(self) -> list[TestResult]:
        """
        Run all tests and return a list of TestResult objects. This is a convenience method that calls run_tests with
        the full list of test identifiers. Subclasses can override this if they have a more efficient way to run all
        tests at once, but the default implementation simply collects all test identifiers and runs them using
        run_tests.

        :returns: A list of TestResult objects for all tests.
        """
        if not self.test_ids and (self.passing_test_ids or self.failing_test_ids):
            self.test_ids = sorted(self.passing_test_ids | self.failing_test_ids)
        if not self.test_ids:
            self.test_ids = self.collect_test_ids()
        return self.run_tests(self.test_ids)

    def find_failing_tests(self) -> set[str]:
        """
        Run all tests and return a set of test identifiers that are expected to fail. This can be used to guide the
        localization process by identifying which tests are relevant for distinguishing between passing and failing
        behavior. The default implementation runs all tests and collects their results, but subclasses can override
        this method if they have a more efficient way to identify failing tests (e.g., based on known bugs, provided
        metadata, etc.).

        :returns: A set of test identifiers that are expected to fail.
        """
        results = self.run_all_tests()
        return {r.test_id for r in results if not r.passed}

    def get_failing_tests(self) -> set[str]:
        """
        Return a set of test identifiers that are expected to fail. This can be used to guide the localization process
        by identifying which tests are relevant for distinguishing between passing and failing behavior. Subclasses
        should implement this method to return the appropriate set of failing test identifiers based on their specific
        context (e.g., based on known bugs, provided metadata, etc.).

        :returns: A set of test identifiers that are expected to fail.
        """
        if self.failing_test_ids:
            self.fail_ids = set(self.failing_test_ids)
            return self.failing_test_ids
        if self.fail_ids:
            self.failing_test_ids = set(self.fail_ids)
            return self.fail_ids
        if not self.test_ids:
            self.test_ids = self.collect_test_ids()
        self.fail_ids = self.find_failing_tests()
        self.failing_test_ids = set(self.fail_ids)
        self.passing_test_ids = set(self.test_ids) - self.failing_test_ids
        return self.fail_ids

    @abc.abstractmethod
    def collect_coverage_single_uncached(self, test_id: str) -> TestResult:
        """
        Collect coverage data for *test_id* without consulting the cache.

        Must be implemented by subclasses to run the test under
        ``coverage.py`` (or equivalent) and return the result.

        :param test_id: Pytest node ID or system-test identifier.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            line-level and function-level coverage populated.
        """

    def collect_coverage_single(self, test_id: str) -> TestResult:
        """
        Return a cached coverage result if available, otherwise collect it.

        Caches the result in :attr:`_coverage` after the first run so
        subsequent calls for the same *test_id* are free.

        :param test_id: Pytest node ID or system-test identifier.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            coverage data.
        """
        if test_id in self._coverage:
            return self._coverage[test_id]
        result = self.collect_coverage_single_uncached(test_id)
        self._coverage[test_id] = result
        return result

    def collect_coverage(
        self, test_ids: list[str] | set[str]
    ) -> tuple[list[TestResult], list[str]]:
        """
        Collect coverage data for *test_ids* in parallel using a process pool.

        Results are cached in :attr:`_coverage` after collection.
        Subclasses (e.g. :class:`DockerSubject`) override this to use a
        thread pool instead, because each Docker test run is already isolated.

        :param test_ids: Test identifiers to collect coverage for.
        :returns: Tuple of ``(ordered_results, failing_ids)`` where
            *ordered_results* preserves the input order and *failing_ids* is
            the list of tests whose :attr:`~forecast.localization.spectra.TestResult.passed`
            field is ``False``.
        """
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self.collect_coverage_single, tid): i
                for i, tid in enumerate(test_ids)
            }
            ordered: list[TestResult] = [None] * len(test_ids)  # type: ignore[list-item]
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        for result in ordered:
            if result.test_id not in self._coverage:
                self._coverage[result.test_id] = result
        failing = [r.test_id for r in ordered if not r.passed]
        self.fail_ids = set(failing)
        self.failing_test_ids = set(failing)
        self.passing_test_ids = set(test_ids) - self.failing_test_ids
        return ordered, failing

    @abc.abstractmethod
    def collect_test_ids(self) -> list[str]:
        """
        Discover and return all test identifiers for this subject.

        Must be implemented by subclasses.  For pytest-based subjects this
        typically runs ``pytest --collect-only``; for system-test subjects it
        returns the pre-configured :attr:`test_ids`.

        :returns: List of test identifiers.
        """

    def collect_spectra(self) -> Spectra:
        """
        Collect coverage for every test and return the resulting
        :class:`~forecast.localization.spectra.Spectra`.

        Auto-discovers test IDs if :attr:`test_ids` is empty.  Also
        populates :attr:`fail_ids` from the results when not already set.

        :returns: Populated :class:`~forecast.localization.spectra.Spectra`
            ready for SFL scoring.
        """
        if not self.test_ids:
            self.test_ids = self.collect_test_ids()
        results, fail_ids = self.collect_coverage(
            test_ids=self.test_ids,
        )
        if not self.fail_ids:
            self.fail_ids = set(fail_ids)
        self.failing_test_ids = set(self.fail_ids)
        self.passing_test_ids = set(self.test_ids) - self.failing_test_ids
        return Spectra(results)

    def sfl_line(self, formula: str = "Tarantula") -> list[Line]:
        """
        Perform SFL at the line level and return a ranked list.

        :param formula: Name of the ``sflkit`` similarity formula to use
            (e.g. ``"Tarantula"``, ``"Ochiai"``).
        :returns: List of :class:`~forecast.localization.locations.Line` objects
            sorted by suspiciousness in descending order.  Lines inside
            :attr:`test_base` are excluded when *test_base* is set.
        """
        return cast(
            list[Line],
            self.sfl(
                location_type=LocationType.LINE,
                formula=formula,
                technique="sfl",
            ),
        )

    def sfl_function(self, formula: str = "Tarantula") -> list[Function]:
        """
        Perform SFL at the function level and return a ranked list.

        :param formula: Name of the ``sflkit`` similarity formula to use
            (e.g. ``"Tarantula"``, ``"Ochiai"``).
        :returns: List of :class:`~forecast.localization.locations.Function` objects
            sorted by suspiciousness in descending order.  Functions inside
            :attr:`test_base` are excluded when *test_base* is set.
        """
        return cast(
            list[Function],
            self.sfl(
                location_type=LocationType.FUNCTION,
                formula=formula,
                technique="sfl",
            ),
        )

    def sfl(
        self,
        location_type: LocationType = LocationType.FUNCTION,
        formula: str = "Tarantula",
        technique: str = "sfl",
        **kwargs,
    ) -> list[Location]:
        """
        Perform SFL and return a ranked list of suspicious locations.

        Dispatches to a registered localization technique handler.
        The built-in ``"sfl"`` technique is provided by
        :class:`~forecast.localization.localization.SpectrumLocalization`.
        Alternative techniques can be provided by plugins.

        :param location_type: Granularity — ``LINE`` or ``FUNCTION``.
        :param formula: Name of the ``sflkit`` similarity formula.
        :param technique: Localization technique name. ``"sfl"`` uses the
            built-in spectra-based implementation.
        :returns: Sorted list of :class:`~forecast.localization.locations.Location`
            objects (most suspicious first).
        :raises ValueError: If *location_type* is not a recognized
            :class:`~forecast.localization.locations.LocationType`.
        """
        if not isinstance(location_type, LocationType):
            raise ValueError(
                f"Unsupported location type: {location_type}, "
                f"consider using one of {list(LocationType)}."
            )
        if not isinstance(technique, str) or not technique.strip():
            raise ValueError("Localization technique must be a non-empty string.")

        handler = get_localization_technique(technique)
        if handler is None and technique.lower() == "sfl":
            # Lazily import built-in localization to ensure default technique
            # registration even when no module imported it yet.
            importlib.import_module("forecast.localization.localization")
            handler = get_localization_technique(technique)
        if handler is None:
            raise ValueError(
                f"Unsupported localization technique: {technique}. "
                "Register it via forecast.plugin_loader plugins."
            )
        return handler(
            subject=self,
            location_type=location_type,
            formula=formula,
            **kwargs,
        )

    def read_file(self, file_path: os.PathLike[str]) -> str:
        """
        Read a source file relative to the subject's :attr:`path`.

        :param file_path: Path relative to the subject's base directory.
        :returns: File contents as a string.
        :raises FileNotFoundError: If the file does not exist.
        """
        full_path = self.path / file_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"File {full_path} does not exist for subject {self.identifier}"
            )
        with open(full_path, "r") as f:
            return f.read()

    @abc.abstractmethod
    def modify(self, files_to_modify: dict[str, str]):
        """
        Overwrite files in the subject with new content.

        Must be implemented by subclasses.  For :class:`LocalSubject` this
        writes files on disk; for :class:`DockerSubject` it writes files into
        a container and commits a new image layer.

        :param files_to_modify: Mapping from file path (relative to
            :attr:`path`) to new file content.
        """
        pass

    def restore(self, files_to_restore: dict[str, str]):
        """
        Restore files to their original content by calling :meth:`modify`.

        The default implementation simply calls :meth:`modify` with the
        original contents.  :class:`DockerSubject` overrides this as a no-op
        because a fresh container always starts from the unmodified image.

        :param files_to_restore: Mapping from file path to original content.
        """
        self.modify(files_to_restore)


@dataclass
class LocalSubject(Subject):
    """
    A subject that is available locally on the file system. The path points to the source base that contains the tests
    and code to be analyzed.

    :param python: The Python executable to use for running tests and collecting coverage. Defaults to the current
                   Python interpreter. This can be useful if the subject requires a specific Python version or
                   environment.
    :param env: Optional dictionary of environment variables to set when running tests. Defaults to the current
                environment. This can be useful for subjects that require specific environment variables to run
                correctly.
    """

    python: str = sys.executable
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))

    def instrument(
        self,
        *,
        src: os.PathLike[str] | str,
        dst: os.PathLike[str] | str,
        mapping_path: os.PathLike[str] | str,
        excludes: list[str] | None = None,
    ) -> Path:
        """
        Instrument a local repository copy and switch this subject to it.

        :returns: Path to the instrumented source root.
        """
        instrumented = super().instrument(
            src=src,
            dst=dst,
            mapping_path=mapping_path,
            excludes=excludes,
        )
        self.path = instrumented
        return instrumented

    def collect_test_ids(self) -> list[str]:
        """
        Discover test IDs by running ``pytest --collect-only`` locally.

        :returns: List of pytest node IDs found under :attr:`path`
            (or :attr:`test_base` when set).
        """
        return collect_test_ids(
            src_root=str(self.path),
            python=self.python,
            test_base=None if self.test_base is None else str(self.test_base),
            env=self.env,
        )

    def run_single_test_uncached(self, test_id: str) -> TestResult:
        """
        Run *test_id* locally without coverage.

        :param test_id: Pytest node ID.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            empty coverage fields.
        """
        return run_test_without_coverage(
            test_id=test_id,
            src_root=str(self.path),
            python=self.python,
            env=self.env,
            timeout=self.timeout,
        )

    def collect_events_single_uncached(
        self, test_id: str, events_file: os.PathLike[str] | str
    ) -> bool:
        """
        Run *test_id* locally with the ``sflkit`` tracer and write events to
        *events_file*.

        :param test_id: Pytest node ID.
        :param events_file: Host path for the output event file.
        :returns: ``True`` on success, ``False`` if an exception occurs.
        """
        # noinspection PyBroadException
        try:
            run_test_and_collect_events(
                test_id=test_id,
                src_root=str(self.path),
                event_file_path=events_file,
                python=self.python,
                env=self.env,
                timeout=self.timeout,
                budget=self.trace_budget,
            )
            return True
        except:
            LOGGER.warning(
                f"Failed to collect events for test {test_id} in subject {self.identifier}"
            )
            return False

    def collect_coverage_single_uncached(self, test_id: str) -> TestResult:
        """
        Run *test_id* locally under ``coverage.py`` and return the result.

        :param test_id: Pytest node ID.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            line-level and function-level coverage populated.
        """
        return run_test_with_coverage(
            test_id=test_id,
            src_root=str(self.path),
            python=self.python,
            env=self.env,
            timeout=self.timeout,
        )

    def modify(self, files_to_modify: dict[str, str]):
        """
        Overwrite local source files with new content.

        :param files_to_modify: Mapping from file path (relative to
            :attr:`path`) to new content string.
        Missing parent directories are created automatically.
        """
        LOG.debug("LocalSubject modifying files", extra={"files": len(files_to_modify)})
        for file_path, content in files_to_modify.items():
            file = self.path / file_path
            file.parent.mkdir(parents=True, exist_ok=True)
            with open(file, "w") as fp:
                fp.write(content)


#: How long to wait on a Docker API call before giving up.
#:
#: docker-py defaults to 60 seconds and, as of 7.1.0, ignores
#: ``DOCKER_CLIENT_TIMEOUT`` entirely, so it has to be passed explicitly. Sixty
#: seconds is not enough on a busy machine: the daemon's own bookkeeping is
#: disk-bound, and when the host is saturated with I/O a call that would
#: succeed instead raises ReadTimeout, collection is abandoned, and the
#: instance is recorded as a failure that has nothing to do with the subject.
#: Waiting longer costs nothing when the daemon is healthy.
DOCKER_API_TIMEOUT = int(os.environ.get("FORECAST_DOCKER_TIMEOUT", "600"))


def docker_client() -> "docker.DockerClient":
    """
    Build a Docker client that tolerates a busy daemon.

    :returns: A client with :data:`DOCKER_API_TIMEOUT` as its API timeout.
    """
    return docker.from_env(timeout=DOCKER_API_TIMEOUT)


@dataclass
class RepoSubject(LocalSubject):
    """
    A subject that is available as a git repository. The path points to the local clone of the repository. The repo
    will be checked out to the specified commit before collecting spectra.

    :param repo: The URL of the git repository (e.g., "https://www.github.com/smythi93/forecast.git").
                 This is used to clone the repository if it doesn't already exist at the specified path.
    :param commit: The git commit hash to check out before collecting spectra. This allows you to analyze a specific
                   version of the codebase.
    :param should_install: Whether to attempt to install the package in editable mode if a setup file is present.
                           This can be useful for subjects that require installation before tests can be run, but may
                           be set to False for subjects that have complex or non-standard installation processes.
    """

    repo: str = ""
    commit: str = ""
    should_install: bool = True

    @property
    def identifier(self) -> str:
        """Unique identifier composed of the repo URL slug and commit hash."""
        return f"{safe(self.repo)}_{self.commit}"

    def serialize(self) -> dict:
        """
        Serialize to a JSON-compatible dictionary, extending the base
        :meth:`~Subject.serialize` with repo-specific fields.

        :returns: Dictionary containing all :class:`Subject` fields plus
            ``repo``, ``commit``, ``issue``, and ``should_install``.
        """
        base = super().serialize()
        base.update(
            {
                "repo": self.repo,
                "commit": self.commit,
                "issue": self.issue,
                "should_install": self.should_install,
            }
        )
        return base

    def load_from_data(self, data: dict):
        super().load_from_data(data)
        if "repo" in data:
            self.repo = data["repo"]
        if "commit" in data:
            self.commit = data["commit"]
        if "issue" in data:
            self.issue = data["issue"]

    def setup(self):
        """
        Clone the repository, check out :attr:`commit`, and install the
        package in editable mode when a setup file is present.

        Cloning is skipped when a ``.git`` directory already exists at
        :attr:`path`.  Installation is skipped when :attr:`should_install` is
        ``False`` or no ``pyproject.toml`` / ``setup.py`` / ``setup.cfg``
        is found.
        """
        if not (self.path / ".git").exists():
            subprocess.run(
                ["git", "clone", self.repo, str(self.path)],
                check=True,
            )
        subprocess.run(["git", "checkout", self.commit], cwd=self.path, check=True)
        if any(
            (self.path / setup_file).exists()
            for setup_file in ("pyproject.toml", "setup.py", "setup.cfg")
        ):
            subprocess.run(
                [self.python, "-m", "pip", "install", "-e", ".", "-q"],
                cwd=self.path,
                check=False,
                env=self.env,
            )


@dataclass

class DockerSubject(Subject):
    """
    A subject whose tests run inside a Docker container.

    :attr:`path` refers to the absolute path *inside the container* where the
    project source lives (default ``"/app"``).  Each test gets its own
    short-lived container that is removed after the run; parallelism is
    achieved with a :class:`~concurrent.futures.ThreadPoolExecutor` (one
    container per thread).

    Before collecting, :meth:`setup` installs ``pytest`` and ``coverage``
    into the image and commits the result as a new layer so subsequent runs
    do not repeat the installation step.

    :param image: Docker image name/tag, e.g. ``"python:3.11-slim"``.
    :param conda_env: Optional conda environment name to activate via
        ``conda run -n <env>`` for all exec calls.
    :param modification: Repository name used when committing the modified
        image layer.  Defaults to ``"<image>_mod"``.
    :param warmup: Whether :meth:`setup` runs one test before committing so
        that first-run caches are baked into the image.  See
        :meth:`_warm_container`.
    """

    image: str = ""
    client: docker.DockerClient = field(default_factory=docker_client, init=False)
    conda_env: str | None = None
    modification: str | None = None
    warmup: bool = True
    _base_image: str = field(default="", init=False, repr=False)

    @property
    def identifier(self) -> str:
        """Unique identifier derived from the Docker image name."""
        return safe(self.image)

    def serialize(self) -> dict:
        """
        Serialize to a JSON-compatible dictionary, extending the base
        :meth:`~Subject.serialize` with Docker-specific fields.

        :returns: Dictionary containing all :class:`Subject` fields plus
            ``image``, ``timeout``, and ``conda_env``.
        """
        base = super().serialize()
        base.update(
            {
                "image": self.image,
                "timeout": self.timeout,
                "conda_env": self.conda_env,
                "warmup": self.warmup,
            }
        )
        return base

    def load_from_data(self, data: dict):
        super().load_from_data(data)
        if "image" in data:
            self.image = data["image"]
        if "timeout" in data:
            self.timeout = data["timeout"]
        if "conda_env" in data:
            self.conda_env = data["conda_env"]
        if "warmup" in data:
            self.warmup = data["warmup"]

    def __post_init__(self):
        self.path = self.path or "/app"
        super().__post_init__()
        self._base_image = self.image
        if self.modification is None and self.image:
            self.modification = build_modification_reference(self.image)

    @staticmethod
    def _decode_exec_output(output: bytes | bytearray | str | None) -> str:
        if isinstance(output, (bytes, bytearray)):
            return output.decode("utf-8", errors="replace")
        return str(output or "")

    def _container_has_project_metadata(self, container) -> bool:
        """
        Return True when the container workdir looks pip-installable.
        """
        if not hasattr(container, "exec_run"):
            return False
        for marker in ("pyproject.toml", "setup.py", "setup.cfg"):
            exit_code, _ = container.exec_run(
                ["test", "-f", marker],
                workdir=str(self.path),
            )
            if exit_code == 0:
                return True
        return False

    #: Modules a subject imports at module scope but never declares as a
    #: dependency, mapped to the pip package that provides them. sphinx 3.x
    #: does ``from roman import toRoman`` in ``sphinx/writers/latex.py`` while
    #: naming ``roman`` nowhere in ``setup.py``, so the SWE-bench image builds
    #: exactly as the project specifies and still cannot import
    #: ``sphinx.testing.fixtures``. Every test that touches sphinx's fixtures
    #: then dies during collection -- and the run still exits in a way the
    #: harness tolerated, so all 17 runs of sphinx-7686 wrote the same
    #: import-only trace and the diagnosis came out empty.
    UNDECLARED_DEPENDENCIES = {"roman": "roman"}

    def _install_undeclared_dependencies(self, container) -> None:
        """
        Install imports the subject needs but does not declare.

        Detected rather than hard-coded per instance: a module is installed
        only when the subject's own sources import it *and* the environment
        cannot import it. That way the repair applies to whichever instances
        actually carry the defect, and does nothing everywhere else.

        :param container: Container to install into.
        """
        if not hasattr(container, "exec_run"):
            return
        for module, package in self.UNDECLARED_DEPENDENCIES.items():
            probe: list[str] = []
            if self.conda_env:
                probe += ["conda", "run", "-n", self.conda_env]
            probe += ["python", "-c", f"import {module}"]
            exit_code, _ = container.exec_run(probe, workdir=str(self.path))
            if int(exit_code) == 0:
                continue
            grep_code, _ = container.exec_run(
                ["grep", "-rqE", f"^\\s*(from|import) {module}\\b", "."],
                workdir=str(self.path),
            )
            if int(grep_code) != 0:
                continue
            cmd: list[str] = []
            if self.conda_env:
                cmd += ["conda", "run", "-n", self.conda_env]
            cmd += ["python", "-m", "pip", "install", package, "-q"]
            install_code, output = container.exec_run(cmd, workdir=str(self.path))
            if int(install_code) == 0:
                LOG.warning(
                    "Installed %s: the subject imports it but declares it nowhere, "
                    "so its test suite could not start.",
                    package,
                    extra={"image": self.image},
                )
            else:
                LOG.error(
                    "Could not install undeclared dependency %s; the subject's "
                    "tests will not run and its traces will be import-only.",
                    package,
                    extra={
                        "image": self.image,
                        "detail": self._decode_exec_output(output),
                    },
                )

    def _reinstall_project_in_container(self, container) -> None:
        """
        Reinstall the project after instrumentation so imports resolve to
        the instrumented code path (important for non-editable SWE installs).
        """
        if not hasattr(container, "exec_run"):
            return
        if not self._container_has_project_metadata(container):
            return

        def _pip_install(args: list[str]) -> tuple[int, str]:
            cmd: list[str] = []
            if self.conda_env:
                cmd += ["conda", "run", "-n", self.conda_env]
            cmd += ["python", "-m", "pip", "install", *args, "-q"]
            exit_code, output = container.exec_run(
                cmd,
                workdir=str(self.path),
            )
            return int(exit_code), self._decode_exec_output(output)

        exit_code, detail = _pip_install(["-e", "."])
        if exit_code == 0:
            return

        LOG.warning(
            "Editable reinstall failed after instrumentation; trying non-editable install.",
            extra={"image": self.image, "detail": detail},
        )
        fallback_code, fallback_detail = _pip_install(["."])
        if fallback_code != 0:
            LOG.warning(
                "Failed to reinstall project after instrumentation; continuing with existing environment.",
                extra={"image": self.image, "detail": fallback_detail},
            )

    def _warm_container(self, container) -> None:
        """
        Run one test so that first-run caches end up in the committed image.

        A fresh container starts with empty caches, and the first test to run
        in it pays to build them — matplotlib, for instance, scans and parses
        every system font to build its font cache.  That cost is small on its
        own, but each test gets its own container, so it is paid once per test,
        and under instrumentation it is *traced*: on a matplotlib subject the
        same test took 65s and produced 1.1 GiB cold, against 4s and 54 MiB
        once the caches existed.

        Running a test here, before the image is committed and before it is
        instrumented, moves that work into the image, where it happens once and
        is never traced.  Failures are ignored: the point is the side effect on
        the file system, not the test result.

        :param container: Running container to warm, committed by the caller.
        """
        if not self.warmup or not hasattr(container, "exec_run"):
            return
        test_id = next(iter(self.test_ids), None)
        if not test_id:
            return

        cmd: list[str] = ["timeout", f"{self.timeout}s"]
        if self.conda_env:
            cmd += ["conda", "run", "-n", self.conda_env]
        cmd += [
            "python",
            "-m",
            "pytest",
            normalize_pytest_test_id(test_id),
            "-rN",
            "--tb=no",
            "-q",
        ]
        try:
            container.exec_run(cmd, workdir=str(self.path))
        except Exception as e:
            LOG.debug(
                "Container warm-up did not complete",
                extra={"image": self.image, "error": str(e)},
            )

    def _next_iteration_image(self) -> str:
        """Build a unique image reference for one generated iteration."""
        if self.modification is None:
            self.modification = build_modification_reference(self.image)
        repository, _ = split_image_reference(self.modification)
        unique_tag = f"iter-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        return f"{repository}:{unique_tag}"

    def _remove_image_if_generated(self, image_ref: str) -> None:
        """Remove one generated iteration image if it belongs to this subject."""
        if not image_ref or image_ref == self._base_image or self.modification is None:
            return
        generated_repository, _ = split_image_reference(self.modification)
        repository, _ = split_image_reference(image_ref)
        if repository != generated_repository:
            return
        # noinspection PyBroadException
        try:
            self.client.images.remove(image_ref, force=True)
            LOG.debug("Removed previous generated image", extra={"image": image_ref})
        except Exception:
            LOG.warning(
                "Failed to remove previous generated image",
                extra={"image": image_ref},
            )

    def instrument(
        self,
        *,
        src: os.PathLike[str] | str,
        dst: os.PathLike[str] | str,
        mapping_path: os.PathLike[str] | str,
        excludes: list[str] | None = None,
    ) -> Path:
        """
        Instrument source via a temporary host mirror and commit a new image.

        The source tree under :attr:`path` is copied out of a temporary
        container, instrumented on the host via :class:`Subject`'s default
        instrumentation path, then uploaded back into the container at the same
        location. The container is committed to a new iteration image used for
        subsequent test runs.

        :param src: Unused for Docker subjects (kept for polymorphic API).
        :param dst: Host-side instrumentation output directory (created for
            compatibility with callers).
        :param mapping_path: Host-side output path for ``mapping.json``.
        :param excludes: Optional instrumentation exclude patterns.
        :returns: Resolved *dst* path.
        """
        dst_path = Path(dst).resolve()
        mapping = Path(mapping_path).resolve()
        dst_path.mkdir(parents=True, exist_ok=True)
        mapping.parent.mkdir(parents=True, exist_ok=True)

        previous_image = self.image
        next_image = self._next_iteration_image()
        container = self.client.containers.run(
            self.image,
            ["sleep", "infinity"],
            detach=True,
        )
        try:
            container.start()
            # Ensure runtime dependencies used by later event collection are present.
            docker_install_all(container, conda_env=self.conda_env)

            with tempfile.TemporaryDirectory(
                prefix="forecast-docker-instrument-"
            ) as tmp:
                tmp_root = Path(tmp)
                extracted_root = tmp_root / "source"
                extracted_root.mkdir(parents=True, exist_ok=True)

                archive_stream, _ = container.get_archive(str(self.path))
                archive_buffer = io.BytesIO()
                for chunk in archive_stream:
                    archive_buffer.write(chunk)
                archive_buffer.seek(0)
                with tarfile.open(fileobj=archive_buffer, mode="r") as tar:
                    tar.extractall(path=extracted_root)

                candidate_root = extracted_root / self.path.name
                if candidate_root.exists() and candidate_root.is_dir():
                    source_root = candidate_root
                else:
                    entries = [p for p in extracted_root.iterdir()]
                    if len(entries) == 1 and entries[0].is_dir():
                        source_root = entries[0]
                    else:
                        source_root = extracted_root

                instrumented_root = super().instrument(
                    src=source_root,
                    dst=tmp_root / "instrumented",
                    mapping_path=mapping,
                    excludes=excludes,
                )

                tar_buffer = io.BytesIO()
                with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                    for child in instrumented_root.iterdir():
                        tar.add(child, arcname=child.name)
                tar_buffer.seek(0)
                if not container.put_archive(str(self.path), tar_buffer):
                    raise RuntimeError(
                        "Failed to copy instrumented source back into container "
                        f"for image '{self.image}'."
                    )
                self._install_undeclared_dependencies(container)
                self._reinstall_project_in_container(container)

            repository, tag = split_image_reference(next_image)
            container.commit(
                repository=repository,
                tag=tag,
            )
            self.image = next_image
            LOG.info(
                "Committed instrumented image",
                extra={"image": self.image, "path": str(self.path)},
            )
        finally:
            container.stop()
            container.remove()

        self._remove_image_if_generated(previous_image)
        return dst_path

    def setup(self):
        """
        Pull the image (if not already present locally), install test
        dependencies, and commit a new image layer.

        Pulls :attr:`image` from the registry when it cannot be found via
        ``docker images``. Then starts a temporary container, installs
        runtime/test dependencies (``pytest``, ``coverage``, ``sflkitlib``)
        into the configured conda environment, and commits the
        result back to the same image tag so subsequent runs skip setup.
        """
        LOG.info("DockerSubject setup started", extra={"image": self.image})
        # Check if image is available locally, and pull it if not
        if not self.client.images.list(name=self.image):
            self.client.images.pull(self.image)
        if self.modification is None:
            self.modification = build_modification_reference(self.image)
        container = self.client.containers.run(
            self.image,
            ["sleep", "infinity"],
            detach=True,
        )
        try:
            docker_install_all(container, conda_env=self.conda_env)
            # Warm before committing so every later container starts with the
            # caches already built, and never traces building them.
            self._warm_container(container)
            repository, tag = split_image_reference(self.image)
            container.commit(
                repository=repository,
                tag=tag,
            )
        finally:
            container.stop()
            container.remove()
        LOG.info("DockerSubject setup completed", extra={"image": self.image})

    def run(self, cmd: list[str]) -> str:
        """
        Run *cmd* in a one-shot container and return its stdout as a string.

        The container is created from :attr:`image`, runs to completion, and
        is removed automatically by the Docker daemon.

        :param cmd: Command and arguments to execute, e.g.
            ``["python", "--version"]``.
        :returns: Decoded stdout of the container.
        """
        LOG.debug("DockerSubject running command", extra={"cmd": " ".join(cmd)})
        return self.client.containers.run(
            self.image,
            cmd,
        ).decode("utf-8")

    def run_single_test_uncached(self, test_id: str) -> TestResult:
        """
        Run *test_id* inside its own container without coverage.

        :param test_id: Pytest node ID.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            empty coverage fields.
        """
        return docker_run_test_without_coverage(
            client=self.client,
            image=self.image,
            test_id=test_id,
            src_root=str(self.path),
            timeout=self.timeout,
            conda_env=self.conda_env,
        )

    def collect_events_single_uncached(
        self, test_id: str, events_file: os.PathLike[str] | str
    ) -> bool:
        """
        Run *test_id* inside its own container with the ``sflkit`` tracer
        and copy the resulting event file to *events_file* on the host.

        :param test_id: Pytest node ID.
        :param events_file: Host path for the output event file.
        :returns: ``True`` on success, ``False`` if an exception occurs.
        """
        try:
            docker_run_test_and_collect_events(
                client=self.client,
                image=self.image,
                test_id=test_id,
                src_root=str(self.path),
                event_file_path=events_file,
                timeout=self.timeout,
                conda_env=self.conda_env,
                budget=self.trace_budget,
            )
            return True
        except Exception as exc:
            LOGGER.warning(
                "Failed to collect events for test %s in subject %s: %s",
                test_id,
                self.identifier,
                exc,
            )
            return False

    def run_tests_modular(
        self,
        test_ids: list[str] | set[str],
        runner: Callable[..., T],
        *args,
        **kwargs,
    ) -> list[T]:
        """
        Run test IDs in parallel using a thread pool.

        Overrides the base-class process pool with a
        :class:`~concurrent.futures.ThreadPoolExecutor` because Docker
        containers are isolated by design, so parallel container launches
        are safe from a thread and I/O perspective without needing separate
        processes.

        :param test_ids: Test identifiers to run.
        :param runner: Callable with signature
            ``runner(test_id, *args, **kwargs) -> T``.
        :param args: Extra positional arguments forwarded to *runner*.
        :param kwargs: Extra keyword arguments forwarded to *runner*.
        :returns: Results in the same order as *test_ids*.
        """
        LOG.debug(
            "DockerSubject running modular tests",
            extra={"count": len(test_ids), "workers": self.workers},
        )
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(runner, test_id, *args, **kwargs): i
                for i, test_id in enumerate(test_ids)
            }
            ordered: list[TestResult] = [None] * len(test_ids)  # type: ignore[list-item]
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        return ordered

    def collect_test_ids(self) -> list[str]:
        """
        Discover test IDs by running ``pytest --collect-only`` inside a
        short-lived container.

        The test patch is applied first, for the same reason the coverage
        sweep applies it: a SWE-bench instance's tests may not exist in the
        base commit at all. django-12113's ``test_patch`` *creates*
        ``tests/backends/sqlite/test_creation.py``, the only file holding its
        ``FAIL_TO_PASS`` test, so discovery against the unpatched checkout
        found neither that test nor any sibling to contrast it against -- and
        the instance produced an empty cause chain.

        :returns: List of pytest node IDs found at :attr:`path` (or
            :attr:`test_base` when set).
        """
        return docker_collect_test_ids(
            client=self.client,
            image=self.image,
            src_root=str(self.path),
            test_base=None if self.test_base is None else str(self.test_base),
            patch=self._coverage_test_patch(),
            conda_env=self.conda_env,
        )

    def _coverage_test_patch(self) -> str | None:
        """
        Patch to apply before measuring coverage.  ``None`` for a plain Docker
        subject; SWE-bench subjects override this to apply their ``test_patch``
        so that patch-added tests exist during the coverage sweep.
        """
        return None

    def collect_coverage(
        self, test_ids: list[str]
    ) -> tuple[list[TestResult], list[str]]:
        """
        Collect coverage for *test_ids* in parallel using a thread pool.

        Collects all tests in a single container and a single pytest run using
        per-test coverage contexts, which avoids paying the container and
        interpreter start-up cost once per test — the dominant cost for
        instances with hundreds of passing tests.  If that batch path is not
        usable (e.g. coverage contexts are unavailable), it falls back to the
        per-test path, one isolated container each.

        :param test_ids: Test identifiers to collect coverage for.
        :returns: Tuple of ``(ordered_results, failing_ids)``.
        """
        ids = list(test_ids)
        ordered: list[TestResult] | None = None
        if ids:
            try:
                ordered = docker_run_tests_with_coverage_batch(
                    client=self.client,
                    image=self.image,
                    test_ids=ids,
                    src_root=str(self.path),
                    timeout=self.timeout,
                    patch=self._coverage_test_patch(),
                    conda_env=self.conda_env,
                )
            except Exception as exc:
                LOG.warning(
                    "Batched coverage failed; falling back to per-test coverage.",
                    extra={"image": self.image, "error": str(exc)},
                )
                ordered = None

        if ordered is None:
            ordered = [None] * len(ids)  # type: ignore[list-item]
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(self.collect_coverage_single, tid): i
                    for i, tid in enumerate(ids)
                }
                for future in as_completed(futures):
                    ordered[futures[future]] = future.result()

        failing = [r.test_id for r in ordered if not r.passed]
        self.fail_ids = set(failing)
        self.failing_test_ids = set(failing)
        self.passing_test_ids = set(ids) - self.failing_test_ids
        return ordered, failing

    def collect_coverage_single_uncached(self, test_id: str) -> TestResult:
        """
        Run *test_id* inside its own container under ``coverage.py``.

        :param test_id: Pytest node ID.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            line-level and function-level coverage populated.
        """
        return docker_run_test_with_coverage(
            client=self.client,
            image=self.image,
            test_id=test_id,
            src_root=str(self.path),
            timeout=self.timeout,
            conda_env=self.conda_env,
        )

    def read_file(self, file_path: str) -> str:
        """
        Read a file from inside the Docker image.

        Creates a temporary container, reads the file, and removes the
        container.  The path is resolved relative to :attr:`path`.

        :param file_path: Path relative to the subject's base directory inside
            the container.
        :returns: File contents as a string.
        """
        return read_file_from_client(
            client=self.client,
            image=self.image,
            path=str(self.path / file_path),
        )

    def modify(self, files_to_modify: dict[str, str]):
        """
        Write modified files into a container and commit a new image layer.

        Starts a fresh container from :attr:`image`, overwrites each file in
        *files_to_modify*, re-installs all required packages via
        :func:`~forecast.runner.utils.docker_install_all`, and commits the
        result to :attr:`modification`.

        :param files_to_modify: Mapping from container-relative file path to
            new content.
        """
        LOG.debug(
            "DockerSubject modifying files", extra={"files": len(files_to_modify)}
        )
        previous_image = self.image
        next_image = self._next_iteration_image()
        container = self.client.containers.run(
            self.image,
            ["sleep", "infinity"],
            detach=True,
        )
        try:
            container.start()
            for file_path, content in files_to_modify.items():
                write_file_to_container(
                    container=container,
                    path=str(self.path / file_path),
                    content=content,
                )
            docker_install_all(
                container=container,
                conda_env=self.conda_env,
            )
            repository, tag = split_image_reference(next_image)
            container.commit(
                repository=repository,
                tag=tag,
            )
            self.image = next_image
            LOG.info(
                "Committed generated-suite iteration image",
                extra={"image": self.image, "files": len(files_to_modify)},
            )
        finally:
            container.stop()
            container.remove()
        self._remove_image_if_generated(previous_image)

    def restore(self, files_to_restore: dict[str, str]):
        # No need to restore files in a DockerSubject since we can just run a new
        # container from the original image for each test or use the modified image.
        pass


@dataclass
class SWESubject(DockerSubject):
    """
    A subject that is available as a git repository and is meant to be run in the SWE environment. This class extends
    DockerSubject to include additional metadata fields that are relevant for SWE problems, such as the repository URL,
    the specific commit to analyze, the problem statement, and any hints or patches that may be provided with the
    problem. The setup and spectra collection are the same as DockerSubject, but the additional metadata can be used
    for logging, reporting, or guiding the localization and repair process in the context of SWE problems.

    :param repo: The SWE-bench repository slug (e.g., "sqlfluff/sqlfluff").
    :param instance_id: The unique identifier for this SWE problem instance (e.g., "sqlfluff__sqlfluff-1625").
    :param base_commit: The git commit hash that represents the base version of the code for this problem instance.
    :param patch: The code patch that, when applied to the base commit, produces the buggy version of the code that
                  fails the tests. This is typically provided in the problem statement and can be used for analysis or
                  as part of the repair process.
    :param test_patch: The code patch that, when applied to the base commit, produces the version of the code with the
                       passing tests. This is typically provided in the problem statement and can be used for analysis
                       or as part of the repair process.
    :param problem_statement: The text of the problem statement that describes the issue to be solved. This may include
                              a description of the bug, the expected behavior, and any constraints or requirements for
                              the solution.
    :param hints_text: Any hints or additional information provided with the problem statement that may assist in
                       solving the problem. This could include insights about the nature of the bug, suggestions for
                       where to look in the code, or other relevant details that are not part of the main problem
                       statement but could be helpful for understanding or solving the problem.
    :param created_at: The timestamp when this problem instance was created. This can be useful for tracking the age of
                       the problem, analyzing trends over time, or for logging and reporting purposes.
    :param version: The version of the SWE problem instance format or schema. This can be useful for ensuring
                    compatibility with different versions of the problem specification, especially if the format
                    evolves over time.
    :param fail_to_pass: A list of test identifiers that are expected to fail on the buggy version of the code (after
                         applying the patch) but pass on the fixed version of the code (after applying the test_patch).
                         This can be used to guide the localization and repair process by identifying which tests are
                         relevant for distinguishing between the buggy and fixed versions.
    :param pass_to_pass: A list of test identifiers that are expected to pass on both the buggy and fixed versions of
                         the code. This can be used to identify tests that are not affected by the bug and can serve as
                         a baseline for understanding the behavior of the code and for validating that any proposed
                         fixes do not break existing functionality.
    :param environment_setup_commit: The git commit hash that represents the version of the code that includes any
                                     necessary environment setup for running the tests. This can be useful if there are
                                     specific dependencies, configurations, or setup steps that are required to run the
                                     tests successfully, and it may differ from the base_commit if the environment
                                     setup was done in a separate commit.
    """

    repo: str = ""
    instance_id: str = ""
    base_commit: str = ""
    patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    hints_text: str = ""
    created_at: str = ""
    version: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    environment_setup_commit: str = ""
    golden_code_patch: str = ""
    golden_test_patch: str = ""

    @property
    def identifier(self) -> str:
        """Unique identifier — the SWE-bench instance ID."""
        return self.instance_id

    @property
    def issue_patch(self) -> str:
        """Backward-compatible alias used by the updated dripht-style schema."""
        return self.patch

    @issue_patch.setter
    def issue_patch(self, value: str) -> None:
        self.patch = value

    def serialize(self) -> dict:
        """
        Serialize to a JSON-compatible dictionary, extending
        :meth:`DockerSubject.serialize` with all SWE-bench metadata fields.

        :returns: Dictionary containing Docker fields plus ``repo``,
            ``instance_id``, ``base_commit``, ``patch``, ``test_patch``,
            ``problem_statement``, ``hints_text``, ``created_at``,
            ``version``, ``fail_to_pass``, ``pass_to_pass``, and
            ``environment_setup_commit``.
        """
        base = super().serialize()
        base.update(
            {
                "repo": self.repo,
                "instance_id": self.instance_id,
                "base_commit": self.base_commit,
                "patch": self.patch,
                "issue_patch": self.patch,
                "test_patch": self.test_patch,
                "problem_statement": self.problem_statement,
                "hints_text": self.hints_text,
                "created_at": self.created_at,
                "version": self.version,
                "fail_to_pass": json.dumps(self.fail_to_pass),
                "pass_to_pass": json.dumps(self.pass_to_pass),
                "environment_setup_commit": self.environment_setup_commit,
                "golden_code_patch": self.golden_code_patch,
                "golden_test_patch": self.golden_test_patch,
            }
        )
        return base

    def load_from_data(self, data: dict):
        super().load_from_data(data)
        if "repo" in data:
            self.repo = data["repo"]
        if "instance_id" in data:
            self.instance_id = data["instance_id"]
        if "base_commit" in data:
            self.base_commit = data["base_commit"]
        if "patch" in data:
            self.patch = data["patch"]
        if "issue_patch" in data:
            self.patch = data["issue_patch"]
        if "test_patch" in data:
            self.test_patch = data["test_patch"]
        if "problem_statement" in data:
            self.problem_statement = data["problem_statement"]
        if "hints_text" in data:
            self.hints_text = data["hints_text"]
        if "created_at" in data:
            self.created_at = data["created_at"]
        if "version" in data:
            self.version = data["version"]
        if "fail_to_pass" in data:
            self.fail_to_pass = normalize_pytest_test_ids(
                json.loads(data["fail_to_pass"])
            )
        if "pass_to_pass" in data:
            self.pass_to_pass = normalize_pytest_test_ids(
                json.loads(data["pass_to_pass"])
            )
        if "environment_setup_commit" in data:
            self.environment_setup_commit = data["environment_setup_commit"]
        if "golden_code_patch" in data:
            self.golden_code_patch = data["golden_code_patch"]
        if "golden_test_patch" in data:
            self.golden_test_patch = data["golden_test_patch"]
        self.test_ids = self.fail_to_pass + self.pass_to_pass
        self.fail_ids = set(self.fail_to_pass)
        self.failing_test_ids = set(self.fail_to_pass)
        self.passing_test_ids = set(self.pass_to_pass)

    def __post_init__(self):
        """
        Initialize :attr:`path` to ``"/testbed"`` (the SWE-bench default),
        pre-populate :attr:`test_ids` from ``fail_to_pass + pass_to_pass``,
        seed :attr:`fail_ids`, and set the conda environment to ``"testbed"``.
        """
        self.path = self.path or "/testbed"
        super().__post_init__()
        self.fail_to_pass = normalize_pytest_test_ids(self.fail_to_pass)
        self.pass_to_pass = normalize_pytest_test_ids(self.pass_to_pass)
        self.test_ids = self.fail_to_pass + self.pass_to_pass
        self.fail_ids = set(self.fail_to_pass)
        self.failing_test_ids = set(self.fail_to_pass)
        self.passing_test_ids = set(self.pass_to_pass)
        # All SWE-bench Docker images use the "testbed" conda environment
        self.conda_env = "testbed"
        self.issue = self.problem_statement

    def setup(self):
        """
        Build the SWE-bench instance Docker image via ``swebench`` and install
        test dependencies into it.

        Uses :func:`swebench.build_instance_images` to create the image for
        this instance, then installs ``pytest`` and ``coverage`` via
        :func:`~forecast.runner.utils.docker_install_all` (including
        ``sflkitlib``) before committing so runtime event collection works in
        the configured conda environment.

        :raises RuntimeError: If the SWE-bench image build fails or does not
            yield exactly one usable image.
        """
        from swebench import build_instance_images
        from swebench.harness.test_spec.test_spec import make_test_spec

        spec = make_test_spec(
            self._as_swebench_instance(),
            arch=platform.machine(),
        )
        spec.repo_script_list = _resilient_repo_install_script(spec.repo_script_list)
        s, failed = build_instance_images(
            self.client,
            [spec],
        )
        if len(s) != 1:
            details = self._format_swe_build_failures(failed)
            raise RuntimeError(
                "Failed to build SWE-bench image for instance "
                f"{self.instance_id}. build_instance_images returned "
                f"{len(s)} successful image(s) and {len(failed or [])} failure(s). "
                f"{details}"
            )
        first = s[0]
        if isinstance(first, tuple) and first:
            first = first[0]
        image_key = getattr(first, "instance_image_key", "")
        if not image_key:
            image_key = getattr(spec, "instance_image_key", "")
        if not image_key:
            raise RuntimeError(
                f"SWE-bench returned an unexpected image build result for {self.instance_id}: {first!r}"
            )
        self.image = image_key
        if self.modification is None or self.modification == "_mod":
            self.modification = build_modification_reference(self.image)
        container = self.client.containers.run(
            self.image,
            ["sleep", "infinity"],
            detach=True,
        )
        try:
            container.start()
            docker_install_all(container, conda_env=self.conda_env)
            # Warm before committing so every later container starts with the
            # caches already built, and never traces building them.
            self._warm_container(container)
            repository, tag = split_image_reference(self.image)
            container.commit(
                repository=repository,
                tag=tag,
            )
        finally:
            container.stop()
            container.remove()

    @staticmethod
    def _format_swe_build_failures(failures) -> str:
        """
        Convert SWE-bench build failure payloads into one concise message.
        """
        if not failures:
            return "No detailed build failure payload was returned by swebench."

        snippets: list[str] = []
        for item in failures:
            text = str(item).strip()
            if not text:
                continue
            snippets.append(" ".join(text.split()))
            if len(snippets) >= 3:
                break

        if not snippets:
            return "SWE-bench reported build failures, but no textual details were available."
        return "Build failure details: " + " | ".join(snippets)

    def _as_swebench_instance(self) -> "SWEbenchInstance":
        """Return a dict matching SWE-bench's SWEbenchInstance TypedDict."""
        return {
            "repo": self.repo,
            "instance_id": self.instance_id,
            "base_commit": self.base_commit,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "created_at": self.created_at,
            "version": self.version,
            "FAIL_TO_PASS": json.dumps(self.fail_to_pass),
            "PASS_TO_PASS": json.dumps(self.pass_to_pass),
            "environment_setup_commit": self.environment_setup_commit,
        }

    @classmethod
    def from_dataset_row(cls, row: dict, **kwargs) -> "SWESubject":
        """
        Build a :class:`SWESubject` from a HuggingFace SWE-bench dataset row.

        :param row: A single dataset row as returned by ``datasets.load_dataset``.
        :param kwargs: Extra keyword arguments forwarded to the constructor
            (e.g. ``workers=8``).  ``path`` defaults to ``"/testbed"``.
        :returns: A fully-initialized :class:`SWESubject`.
        """

        def _parse(v):
            # Dataset rows may store list fields as JSON strings; handle both
            return json.loads(v) if isinstance(v, str) else (v or [])

        return cls(
            path=kwargs.pop("path", "/testbed"),
            repo=row.get("repo", ""),
            instance_id=row["instance_id"],
            base_commit=row.get("base_commit", ""),
            patch=row.get("issue_patch", row.get("patch", "")),
            test_patch=row.get("test_patch", ""),
            problem_statement=row.get("problem_statement", ""),
            hints_text=row.get("hints_text", ""),
            created_at=row.get("created_at", ""),
            version=row.get("version", ""),
            fail_to_pass=_parse(row.get("FAIL_TO_PASS", [])),
            pass_to_pass=_parse(row.get("PASS_TO_PASS", [])),
            environment_setup_commit=row.get("environment_setup_commit", ""),
            golden_code_patch=row.get("golden_code_patch", ""),
            golden_test_patch=row.get("golden_test_patch", ""),
            **kwargs,
        )

    @classmethod
    def from_instance_id(
        cls,
        instance_id: str,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        **kwargs,
    ) -> "SWESubject":
        """
        Load a single :class:`SWESubject` from a HuggingFace dataset by ID.

        :param instance_id: SWE-bench instance identifier, e.g.
            ``"astropy__astropy-13236"``.
        :param dataset: HuggingFace dataset name.  Common values:
            ``"princeton-nlp/SWE-bench_Lite"`` (default),
            ``"princeton-nlp/SWE-bench_Verified"``,
            ``"princeton-nlp/SWE-bench"``.
        :param split: Dataset split — ``"test"`` (default), ``"dev"``, or
            ``"train"``.
        :param kwargs: Extra keyword arguments forwarded to the constructor.
        :returns: A fully-initialized :class:`SWESubject`.
        :raises ValueError: If *instance_id* is not found in the dataset.
        """
        from datasets import load_dataset

        ds = load_dataset(dataset, split=split)
        rows = [r for r in ds if r["instance_id"] == instance_id]
        if not rows:
            raise ValueError(
                f"Instance '{instance_id}' not found in {dataset} (split='{split}')."
            )
        return cls.from_dataset_row(rows[0], **kwargs)

    @classmethod
    def from_dataset(
        cls,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        **kwargs,
    ) -> "list[SWESubject]":
        """
        Load all instances from a HuggingFace SWE-bench dataset split.

        :param dataset: HuggingFace dataset name.  Common values:
            ``"princeton-nlp/SWE-bench_Lite"`` (300 instances, default),
            ``"princeton-nlp/SWE-bench_Verified"`` (500 instances),
            ``"princeton-nlp/SWE-bench"`` (full benchmark).
        :param split: Dataset split — ``"test"`` (default), ``"dev"``, or
            ``"train"``.
        :param kwargs: Extra keyword arguments forwarded to the constructor
            for every instance.
        :returns: List of :class:`SWESubject` instances.

        Example::

            subjects = SWESubject.from_dataset(
                "princeton-nlp/SWE-bench_Verified", split="test",
            )
            for subj in subjects:
                subj.setup()
                print(subj.sfl(LocationType.FUNCTION))
        """
        from datasets import load_dataset

        ds = load_dataset(dataset, split=split)
        return [cls.from_dataset_row(row, **kwargs) for row in ds]

    def run_single_test_uncached(self, test_id: str) -> TestResult:
        """
        Run *test_id* with :attr:`test_patch` applied, without coverage.

        The patch is applied so that the tests in ``PASS_TO_PASS`` and
        ``FAIL_TO_PASS`` can be run against the patched codebase as intended
        by the SWE-bench evaluation harness.

        :param test_id: Pytest node ID.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            empty coverage.
        """
        return docker_run_test_without_coverage(
            client=self.client,
            image=self.image,
            test_id=test_id,
            src_root=str(self.path),
            timeout=self.timeout,
            patch=self.test_patch,
            conda_env=self.conda_env,
        )

    def collect_events_single_uncached(
        self, test_id: str, events_file: os.PathLike[str] | str
    ) -> bool:
        """
        Run *test_id* with :attr:`test_patch` applied and the ``sflkit``
        tracer active, writing events to *events_file*.

        :param test_id: Pytest node ID.
        :param events_file: Host path for the output event file.
        :returns: ``True`` on success, ``False`` if an exception occurs.
        """
        try:
            docker_run_test_and_collect_events(
                client=self.client,
                image=self.image,
                test_id=test_id,
                src_root=str(self.path),
                event_file_path=events_file,
                timeout=self.timeout,
                patch=self.test_patch,
                conda_env=self.conda_env,
                budget=self.trace_budget,
            )
            return True
        except Exception as exc:
            LOGGER.warning(
                "Failed to collect events for test %s in subject %s: %s",
                test_id,
                self.identifier,
                exc,
            )
            return False

    def _coverage_test_patch(self) -> str | None:
        """Apply the instance's ``test_patch`` during the coverage sweep."""
        return self.test_patch or None

    def collect_coverage_single_uncached(self, test_id: str) -> TestResult:
        """
        Run *test_id* with :attr:`test_patch` applied under ``coverage.py``.

        :param test_id: Pytest node ID.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            line-level and function-level coverage populated.
        """
        return docker_run_test_with_coverage(
            client=self.client,
            image=self.image,
            test_id=test_id,
            src_root=str(self.path),
            timeout=self.timeout,
            patch=self.test_patch,
            conda_env=self.conda_env,
        )


@dataclass
class SystemTestSubject(Subject, abc.ABC):
    """
    Abstract base class for subjects driven by raw command-line inputs rather
    than pytest test IDs.

    System tests are specified as lists of arguments (``passing_tests`` and
    ``failing_tests``); each list is mapped to a stable string identifier by
    :meth:`get_test_id`.  Subclasses implement :meth:`collect_events_single_uncached`
    and :meth:`collect_coverage_single_uncached` to run the program with a
    given input.

    :param access: Path to the entry-point script or module name used to run
        the program under test.
    :param module: When ``True`` the program is invoked as
        ``python -m <access>``; otherwise as ``python <access>``.
    :param passing_tests: List of input argument lists expected to produce a
        passing (zero exit-code) result.
    :param failing_tests: List of input argument lists expected to produce a
        failing (non-zero exit-code) result.
    :param test_id_mapping: Populated automatically — maps test ID string to
        its argument list.
    :param failing_id_mapping: Populated automatically — maps failing test ID
        to its argument list.
    """

    access: str | None = None
    module: bool = False
    passing_tests: list[list[str]] = field(default_factory=list)
    failing_tests: list[list[str]] = field(default_factory=list)
    test_id_mapping: dict[str, list[str]] = field(default_factory=dict)
    failing_id_mapping: dict[str, list[str]] = field(default_factory=dict)

    def serialize(self) -> dict:
        base = super().serialize()
        base.update(
            {
                "access": self.access,
                "module": self.module,
                "passing_tests": self.passing_tests,
                "failing_tests": self.failing_tests,
            }
        )
        return base

    def load_from_data(self, data: dict):
        super().load_from_data(data)
        if "access" in data:
            self.access = data["access"]
        if "module" in data:
            self.module = data["module"]
        if "passing_tests" in data:
            self.passing_tests = data["passing_tests"]
        if "failing_tests" in data:
            self.failing_tests = data["failing_tests"]
        if self.access is None:
            raise ValueError(
                "Access script or file must be specified for system test subjects, as they may require special "
                "setup or execution steps compared to unit-level tests."
            )
        if not self.passing_tests and not self.failing_tests:
            raise ValueError(
                "At least one passing or failing test must be specified for system test subjects, as they are "
                "needed to guide the localization process and to distinguish between passing and failing behavior "
                "in the context of system tests."
            )
        self.test_ids = []
        self.fail_ids = set()
        for passing_test in self.passing_tests:
            test_id = self.get_test_id(passing_test)
            self.test_ids.append(test_id)
            self.test_id_mapping[test_id] = passing_test
        for failing_test in self.failing_tests:
            test_id = self.get_test_id(failing_test)
            self.test_ids.append(test_id)
            self.test_id_mapping[test_id] = failing_test
            self.failing_id_mapping[test_id] = failing_test
            self.fail_ids.add(test_id)
        self.failing_test_ids = set(self.fail_ids)
        self.passing_test_ids = set(self.test_ids) - self.failing_test_ids

    @staticmethod
    def get_test_id(test: list[str]) -> str:
        """Generate a unique test identifier from a list of strings that represent the test
        (e.g., test name, parameters, etc.)."""
        return " ".join(test)

    def __post_init__(self):
        super().__post_init__()
        if self.access is None:
            raise ValueError(
                "Access script or file must be specified for system test subjects, as they may require special "
                "setup or execution steps compared to unit-level tests."
            )
        if not self.passing_tests and not self.failing_tests:
            raise ValueError(
                "At least one passing or failing test must be specified for system test subjects, as they are "
                "needed to guide the localization process and to distinguish between passing and failing behavior "
                "in the context of system tests."
            )
        for passing_test in self.passing_tests:
            test_id = self.get_test_id(passing_test)
            self.test_ids.append(test_id)
            self.test_id_mapping[test_id] = passing_test
        for failing_test in self.failing_tests:
            test_id = self.get_test_id(failing_test)
            self.test_ids.append(test_id)
            self.test_id_mapping[test_id] = failing_test
            self.failing_id_mapping[test_id] = failing_test
            self.fail_ids.add(test_id)
        self.failing_test_ids = set(self.fail_ids)
        self.passing_test_ids = set(self.test_ids) - self.failing_test_ids


@dataclass
class LocalSystemTestSubject(SystemTestSubject, LocalSubject):
    """
    A system-test subject available on the local file system.

    :attr:`access` points to a script or module that accepts the test inputs
    as command-line arguments.  :attr:`test_id_mapping` is populated
    automatically from :attr:`passing_tests` and :attr:`failing_tests`.
    """

    def collect_test_ids(self) -> list[str]:
        """
        Return all pre-configured test IDs (no discovery needed).

        :returns: Keys of :attr:`test_id_mapping`.
        """
        return list(self.test_id_mapping.keys())

    def run_single_test_uncached(self, test_id: str) -> TestResult:
        """
        Return a :class:`~forecast.localization.spectra.TestResult` whose
        ``passed`` field is derived from the pre-configured outcome.

        No subprocess is launched — the pass/fail outcome is known statically
        from :attr:`failing_id_mapping`.

        :param test_id: System-test identifier.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            empty coverage.
        """
        return TestResult(
            test_id=test_id, passed=test_id not in self.failing_id_mapping
        )

    def collect_events_single_uncached(
        self, test_id: str, events_file: os.PathLike[str] | str
    ) -> bool:
        """
        Run the program locally with the input for *test_id* and write events.

        :param test_id: System-test identifier.
        :param events_file: Host path for the output event file.
        :returns: ``True`` on success, ``False`` on any exception.
        """
        # noinspection PyBroadException
        try:
            run_input_and_collect_events(
                access=self.access,
                test=self.test_id_mapping[test_id],
                test_id=test_id,
                src_root=str(self.path),
                event_file_path=events_file,
                module=self.module,
                python=self.python,
                env=self.env,
                timeout=self.timeout,
                budget=self.trace_budget,
            )
            return True
        except:
            LOGGER.warning(
                f"Failed to collect events for test {test_id} in subject {self.identifier}"
            )
            return False

    def collect_coverage_single_uncached(self, test_id: str) -> TestResult:
        """
        Run the program locally under ``coverage.py`` with the input for
        *test_id*.

        :param test_id: System-test identifier.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            line-level and function-level coverage populated.
        """
        return run_input_with_coverage(
            access=self.access,
            test=self.test_id_mapping[test_id],
            test_id=test_id,
            src_root=str(self.path),
            passing=test_id not in self.failing_id_mapping,
            module=self.module,
            python=self.python,
            env=self.env,
            timeout=self.timeout,
        )


@dataclass
class RepoSystemTestSubject(LocalSystemTestSubject, RepoSubject):
    """
    A system-test subject backed by a git repository.

    Combines :class:`LocalSystemTestSubject` (input-driven testing) with
    :class:`RepoSubject` (``git clone`` / ``git checkout`` setup).  The
    repository is cloned and checked out to :attr:`~RepoSubject.commit`
    before spectra collection begins.
    """

    @property
    def identifier(self) -> str:
        """Unique identifier composed of the repo URL slug and commit hash."""
        return f"{safe(self.repo)}_{self.commit}"

    def setup(self):
        RepoSubject.setup(self)

    def serialize(self) -> dict:
        return RepoSubject.serialize(self)

    def load_from_data(self, data: dict):
        RepoSubject.load_from_data(self, data)
        LocalSystemTestSubject.load_from_data(self, data)


@dataclass
class DockerSystemTestSubject(SystemTestSubject, DockerSubject):
    """
    A system-test subject that runs inside a Docker container.

    Combines :class:`SystemTestSubject` (input-driven testing) with
    :class:`DockerSubject` (each test gets its own isolated container).
    ``pytest`` and ``coverage`` are installed by :meth:`DockerSubject.setup`
    and committed to the image before collection begins.
    """

    @property
    def identifier(self) -> str:
        """Unique identifier derived from the Docker image name."""
        return safe(self.image)

    def __post_init__(self):
        """Initialize Docker state first, then system-test state."""
        DockerSubject.__post_init__(self)
        super().__post_init__()

    def setup(self):
        """Delegate to :meth:`DockerSubject.setup`."""
        DockerSubject.setup(self)

    def run_single_uncached(self, test_id: str) -> TestResult:
        """
        Return a static pass/fail result without launching a container.

        The pass/fail outcome is derived from :attr:`failing_id_mapping`.

        :param test_id: System-test identifier.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            empty coverage.
        """
        return TestResult(
            test_id=test_id, passed=test_id not in self.failing_id_mapping
        )

    def collect_events_single_uncached(
        self, test_id: str, events_file: os.PathLike[str] | str
    ) -> bool:
        """
        Run the program inside its own container with the ``sflkit`` tracer
        and copy the resulting event file to the host.

        :param test_id: System-test identifier.
        :param events_file: Host path for the output event file.
        :returns: ``True`` on success, ``False`` on any exception.
        """
        # noinspection PyBroadException
        try:
            docker_run_input_and_collect_events(
                client=self.client,
                image=self.image,
                access=self.access,
                test=self.test_id_mapping[test_id],
                src_root=str(self.path),
                event_file_path=events_file,
                module=self.module,
                timeout=self.timeout,
                conda_env=self.conda_env,
                budget=self.trace_budget,
            )
            return True
        except:
            LOGGER.warning(
                f"Failed to collect events for test {test_id} in subject {self.identifier}"
            )
            return False

    def run_tests_modular(
        self,
        test_ids: list[str] | set[str],
        runner: Callable[..., T],
        *args,
        **kwargs,
    ) -> list[T]:
        """Delegate to :meth:`DockerSubject.run_tests_modular` (thread pool)."""
        return DockerSubject.run_tests_modular(self, test_ids, runner, *args, **kwargs)

    def collect_coverage(
        self, test_ids: list[str]
    ) -> tuple[list[TestResult], list[str]]:
        """Delegate to :meth:`DockerSubject.collect_coverage` (thread pool)."""
        return DockerSubject.collect_coverage(self, test_ids)

    def collect_coverage_single_uncached(self, test_id: str) -> TestResult:
        """
        Run the program inside its own container under ``coverage.py``.

        :param test_id: System-test identifier.
        :returns: :class:`~forecast.localization.spectra.TestResult` with
            line-level and function-level coverage populated.
        """
        return docker_run_input_with_coverage(
            client=self.client,
            image=self.image,
            access=self.access,
            test=self.test_id_mapping[test_id],
            test_id=test_id,
            src_root=str(self.path),
            passing=test_id not in self.failing_id_mapping,
            module=self.module,
            timeout=self.timeout,
            conda_env=self.conda_env,
        )

    def read_file(self, file_path: str) -> str:
        """Delegate to :meth:`DockerSubject.read_file`."""
        return DockerSubject.read_file(self, file_path)

    def modify(self, files_to_modify: dict[str, str]):
        """Delegate to :meth:`DockerSubject.modify`."""
        DockerSubject.modify(self, files_to_modify)

    def restore(self, files_to_restore: dict[str, str]):
        """Delegate to :meth:`DockerSubject.restore` (no-op for Docker subjects)."""
        DockerSubject.restore(self, files_to_restore)


# Register built-in subject kinds for plugin-aware discovery and extension.
register_subject_class("local", LocalSubject, replace=True)
register_subject_class("repo", RepoSubject, replace=True)
register_subject_class("docker", DockerSubject, replace=True)
register_subject_class("swe", SWESubject, replace=True)
register_subject_class("swt", SWESubject, replace=True)
register_subject_class("local-system", LocalSystemTestSubject, replace=True)
register_subject_class("repo-system", RepoSystemTestSubject, replace=True)
register_subject_class("docker-system", DockerSystemTestSubject, replace=True)
