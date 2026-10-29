"""
Source-code instrumentation for FORECAST.

Wraps the ``sflkit`` instrumentation API to insert execution-trace probes into
a Python project.  The instrumented copy is written to a separate output
directory so the original source is never modified.
"""

import ast
import os
import re
import tempfile
import tokenize
from pathlib import Path
from typing import List

from sflkit.language.language import Language
from sflkit.language.meta import CombinationVisitor, IDGenerator, TmpGenerator
from sflkit.language.python.visitor import PythonInstrumentation
from sflkit.logger import LOGGER
from sflkitlib.events import EventType


class Instrumenter(PythonInstrumentation):
    """
    Source-code instrumenter for FORECAST.

    Thin wrapper around :class:`~sflkit.language.python.visitor.PythonInstrumentation`
    that initializes a combined visitor for all requested ``sflkit`` event types.
    When no *events* are specified every available :class:`~sflkitlib.events.EventType`
    is instrumented.

    :param events: Optional list of event types to instrument.  Defaults to
        all available event types when ``None``.
    """

    def __init__(self, events: list[EventType] | None = None):
        super().__init__(
            CombinationVisitor(
                Language.PYTHON,
                IDGenerator(),
                IDGenerator(),
                TmpGenerator(),
                [Language.PYTHON.meta_visitors[e] for e in (events or list(EventType))],
            )
        )

    def instrument_source(
        self, source: str, dst: os.PathLike[str], file: os.PathLike[str]
    ):
        """
        Parse *source*, instrument it, and write the result to *dst*.

        The *file* path is recorded on each probe event so that the event
        mapping can later resolve which source file an event belongs to.

        :param source: Original source code as a string.
        :param dst: Destination path where the instrumented source will be
            written.
        :param file: Logical path of the source file (used as the file
            identifier on probe events).
        """
        self.file = file
        tree = self.parse(source)
        prev_events = len(self.events)
        # Notify the meta visitor that we are entering / exiting a new file
        self.meta_visitor.enter_file(self.file)
        instrumented_tree = self.start_visit(tree)
        self.meta_visitor.exit_file(self.file)
        with open(dst, "w") as fp:
            fp.write(self.unparse(instrumented_tree))
        LOGGER.info(f"I found {len(self.events) - prev_events} events in {file}.")


class DummyInstrumenter(Instrumenter):
    """
    No-op instrumenter used when instrumentation has already been applied.

    Initialised with an empty event list so the parent ``CombinationVisitor``
    has no visitors, and :meth:`instrument_source` is a deliberate no-op — the
    file is simply not modified.
    """

    def __init__(self):
        super().__init__([])

    def instrument_source(
        self, source: str, dst: os.PathLike[str], file: os.PathLike[str]
    ):
        """
        Skip instrumentation entirely.

        This method intentionally does nothing: when a pre-existing event
        mapping is found the source files are already instrumented and must
        not be overwritten.

        :param source: Original source code (ignored).
        :param dst: Destination path (not written).
        :param file: Logical source file path (ignored).
        """
        pass


DEFAULT_EXCLUDES = [
    r"^test",
    r"^tests",
    r"^conftest",
    r".*conftest\.py$",
    r".*/tests/.*",
    r".*/test/.*",
    "setup.py",
    "env",
    "build",
    "bin",
    "docs",
    "examples",
    "hacking",
    ".git",
    ".github",
    "extras",
    "profiling",
    "plugin",
    "gallery",
    "blib2to3",
    "docker",
    "contrib",
    "changelogs",
    "licenses",
    "packaging",
    # Build-time machinery. It is not part of the program under test, and
    # instrumenting it actively breaks projects that generate code from their
    # own source: astropy's setup.py builds astropy/version.py by inlining
    # helper functions read back with inspect.getsource(). Rewriting a module
    # through ast.unparse renumbers its lines, getsource then returns nothing,
    # and the generated version.py calls helpers it never defines -- so
    # `import astropy` raises NameError, the test runs against uninstrumented
    # code, and the trace holds nothing but the failed import.
    r".*_helpers/.*",
    r".*setup_package\.py$",
    r".*versioneer\.py$",
]


def _is_excluded_path(rel_path: str, excludes: list[str]) -> bool:
    """
    Return True when *rel_path* matches one of the ``sflkit`` exclude regexes.
    """
    return any(re.match(pattern, rel_path) for pattern in excludes)


def _discover_syntax_error_excludes(
    src: os.PathLike[str] | str,
    excludes: list[str],
) -> list[str]:
    """
    Find ``.py`` files that cannot be parsed and should be excluded.

    ``sflkit`` fails hard on syntax errors while instrumenting Python files.
    This helper proactively detects problematic files and returns relative paths
    that can be appended to ``exclude`` so instrumentation can continue.
    """
    src_path = Path(src).resolve()
    invalid: list[str] = []
    for file in src_path.rglob("*.py"):
        rel = file.relative_to(src_path).as_posix()
        if _is_excluded_path(rel, excludes):
            continue
        try:
            with tokenize.open(file) as fp:
                source = fp.read()
            ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            invalid.append(rel)
    return sorted(set(invalid))


def _resolve_event_types(events: list[str] | None) -> list[EventType] | None:
    """
    Resolve event names to :class:`~sflkitlib.events.EventType` values.
    """
    if not events:
        return None
    resolved: list[EventType] = []
    for name in events:
        try:
            resolved.append(EventType[str(name)])
        except Exception:
            continue
    return resolved or None


def _discover_instrumentation_error_excludes(
    src: os.PathLike[str] | str,
    excludes: list[str],
    events: list[str] | None = None,
) -> list[str]:
    """
    Find files that parse but still crash ``sflkit`` instrumentation.
    """
    src_path = Path(src).resolve()
    failing: list[str] = []
    event_types = _resolve_event_types(events)

    with tempfile.TemporaryDirectory(prefix="forecast-instrument-probe-") as tmp:
        tmp_dst = Path(tmp) / "probe.py"
        for file in src_path.rglob("*.py"):
            rel = file.relative_to(src_path).as_posix()
            if _is_excluded_path(rel, excludes):
                continue
            try:
                with tokenize.open(file) as fp:
                    source = fp.read()
            except (UnicodeDecodeError, ValueError, OSError):
                continue

            detector = Instrumenter(event_types)
            try:
                detector.instrument_source(source, tmp_dst, rel)
            except Exception:
                failing.append(rel)

    return sorted(set(failing))


def instrument(
    src: os.PathLike,
    dst: os.PathLike,
    mapping_path: os.PathLike,
    excludes: List[str] | None = None,
    events: List[str] | None = None,
):
    """
    Backward-compatible instrumentation entry point used by the legacy CLI.

    :param src: Source repository root.
    :param dst: Destination directory for instrumented sources.
    :param mapping_path: Output path for ``mapping.json``.
    :param excludes: Optional path filters excluded from instrumentation.
    :param events: Optional event names; defaults to all available events.
    """
    from sflkit import Config, instrument_config

    if excludes is None:
        excludes = DEFAULT_EXCLUDES
    else:
        excludes = list(excludes)

    syntax_error_excludes = _discover_syntax_error_excludes(src, excludes)
    if syntax_error_excludes:
        LOGGER.warning(
            "Skipping non-parseable Python files during instrumentation: "
            f"{len(syntax_error_excludes)}"
        )
        excludes.extend(syntax_error_excludes)
        excludes = list(dict.fromkeys(excludes))

    if events is None:
        # A campaign can narrow the event set without touching the code, which
        # is what makes it practical to find out which kind of probe is
        # responsible when instrumentation changes a subject's behaviour:
        # instrument once per candidate set and see which one restores the
        # subject's own result.
        override = os.environ.get("FORECAST_EVENTS", "").strip()
        if override:
            events = [name.strip() for name in override.split(",") if name.strip()]
            LOGGER.warning(
                "Instrumenting with a restricted event set from FORECAST_EVENTS: %s",
                ",".join(events),
            )
        else:
            event_iter = getattr(EventType, "events", None)
            if callable(event_iter):
                events = [e.name for e in event_iter()]
            else:
                events = [e.name for e in EventType]

    def _run_instrumentation(active_excludes: list[str]) -> None:
        instrument_config(
            Config.create(
                path=str(src),
                language="Python",
                events=",".join(events),
                working=str(dst),
                exclude=",".join(active_excludes),
                mapping_path=str(mapping_path),
                thread_support=True,
            ),
        )

    try:
        _run_instrumentation(excludes)
    except Exception:
        failing_excludes = _discover_instrumentation_error_excludes(
            src=src,
            excludes=excludes,
            events=events,
        )
        new_excludes = [f for f in failing_excludes if f not in excludes]
        if not new_excludes:
            raise
        sample = ", ".join(new_excludes[:5])
        LOGGER.warning(
            "Skipping files that crash sflkit instrumentation: "
            f"{len(new_excludes)} ({sample})"
        )
        excludes.extend(new_excludes)
        excludes = list(dict.fromkeys(excludes))
        _run_instrumentation(excludes)
