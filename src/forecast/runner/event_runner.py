"""
Event runners for FORECAST.

Runs instrumented test suites or system-test inputs and writes per-test
``sflkit`` execution-trace event files to the host file system. These files
are later consumed by :class:`~forecast.handler.ForecastHandler` to build the
:class:`~forecast.tree.TracedRelevanceTree`.

Two families of runners are provided:

* **Local** (``run_test_and_collect_events``, ``run_input_and_collect_events``) —
  subprocess-based, suitable for locally available subjects.
* **Docker** (``docker_run_test_and_collect_events``,
  ``docker_run_input_and_collect_events``) — Docker SDK-based, used by
  :class:`~forecast.subject.DockerSubject` and its subclasses.

The event file path is communicated to the instrumented process via the
``EVENTS_PATH`` environment variable, which ``sflkitlib`` monitors at runtime
to write the binary event stream.
"""

import errno
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from forecast.logger import LOGGER
from forecast.test_ids import normalize_pytest_test_id
from forecast.runner.trace_budget import TraceBudget, resolve_budget
from forecast.runner.utils import (
    write_file_to_container,
)


def _is_disk_full_error(exc: Exception) -> bool:
    """
    Check if an exception is due to disk space being exhausted.
    """
    if isinstance(exc, OSError):
        return exc.errno == errno.ENOSPC
    return False


def _try_free_disk_space(target_path: Path) -> bool:
    """
    Attempt to free disk space near the target path by removing temporary files.

    :param target_path: The path where we're trying to write.
    :return: True if any space was freed, False otherwise.
    """
    freed_any = False

    # Only scratch directories are reclaimed here.  Already-collected event
    # files must never be deleted: they belong to other tests of the same
    # instance, and removing them would silently corrupt the diagnosis or force
    # an expensive re-collection.  If scratch space is not enough, the write
    # fails and the caller reports it.
    parent = target_path.parent
    try:
        if parent.exists():
            for tmp_dir in parent.glob("tmp_*"):
                if tmp_dir.is_dir():
                    try:
                        LOGGER.info(
                            "Cleaning up temporary directory to free disk space",
                            extra={"path": str(tmp_dir)},
                        )
                        shutil.rmtree(tmp_dir)
                        freed_any = True
                    except Exception as e:
                        LOGGER.debug(
                            "Failed to remove temporary directory",
                            extra={"path": str(tmp_dir), "error": str(e)},
                        )
    except Exception as e:
        LOGGER.debug(
            "Error during disk space cleanup",
            extra={"error": str(e)},
        )

    return freed_any


#: Outcome of trying to fetch a trace out of a container.
TRACE_MISSING = "missing"
TRACE_EMPTY = "empty"
TRACE_OK = "ok"
TRACE_ERROR = "error"


def _remove_container_file(container, file_path: str) -> None:
    """
    Delete a file inside the container, ignoring failures.

    Called after every collection attempt so that a retry never leaves the
    previous (possibly very large) trace behind in the container's filesystem.
    """
    try:
        container.exec_run(["rm", "-f", file_path])
    except Exception as e:
        LOGGER.debug(
            "Failed to remove event file inside container",
            extra={"path": file_path, "error": str(e)},
        )


def _stream_event_file_from_container(
    container,
    file_path: str,
    destination: str | Path,
) -> tuple[str, int]:
    """
    Copy one trace out of a container, streaming it straight to disk.

    The Docker API returns files as a tar stream.  Buffering that stream in
    memory is not viable for traces of realistic size, so it is unpacked
    incrementally and written directly to *destination*.

    :param container: Docker container object.
    :param file_path: Path of the trace inside the container.
    :param destination: Host path to write the trace to.
    :returns: A ``(status, size)`` pair where *status* is one of
        :data:`TRACE_OK`, :data:`TRACE_EMPTY`, :data:`TRACE_MISSING` or
        :data:`TRACE_ERROR`, and *size* is the number of bytes written.
    """
    destination = Path(destination)
    try:
        stream, _ = container.get_archive(file_path)
    except Exception as e:
        LOGGER.debug(
            "No trace archive available inside container",
            extra={"path": file_path, "error": str(e)},
        )
        return TRACE_MISSING, 0

    class _ChunkReader:
        """Adapts the chunked archive generator to a readable file object."""

        def __init__(self, chunks):
            self._chunks = chunks
            self._buffer = b""
            self._closed = False

        def read(self, size: int = -1) -> bytes:
            if size is None or size < 0:
                parts = [self._buffer]
                self._buffer = b""
                parts.extend(self._chunks)
                return b"".join(parts)
            while len(self._buffer) < size and not self._closed:
                try:
                    self._buffer += next(self._chunks)
                except StopIteration:
                    self._closed = True
            head, self._buffer = self._buffer[:size], self._buffer[size:]
            return head

    written = 0
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Stream mode ("r|*") never seeks, so the archive is consumed as it
        # arrives instead of being materialised first.
        with tarfile.open(fileobj=_ChunkReader(stream), mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                with open(destination, "wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        written += len(chunk)
                break
    except OSError as e:
        if _is_disk_full_error(e):
            raise
        LOGGER.warning(
            "Failed to stream trace out of container",
            extra={"path": file_path, "error": str(e)},
        )
        return TRACE_ERROR, written
    except Exception as e:
        LOGGER.warning(
            "Failed to stream trace out of container",
            extra={"path": file_path, "error": str(e)},
        )
        return TRACE_ERROR, written

    if written == 0:
        destination.unlink(missing_ok=True)
        return TRACE_EMPTY, 0
    return TRACE_OK, written


#: Command used when a campaign does not override it.
DEFAULT_TEST_CMD = ["python", "-m", "pytest", "{nodeid}", "-rN", "--tb=no", "-q"]


def pytest_id_to_dotted_label(node_id: str) -> str:
    """
    Convert a pytest node ID to the dotted label unittest-style runners expect.

    ``tests/migrations/test_writer.py::WriterTests::test_serialize_enum_flags``
    becomes ``migrations.test_writer.WriterTests.test_serialize_enum_flags``.

    The leading test-root directory is dropped because runners such as Django's
    ``tests/runtests.py`` are invoked from inside that directory and label tests
    relative to it.

    :param node_id: Pytest node ID.
    :returns: Dotted test label.
    """
    path, _, rest = node_id.partition("::")
    if path.endswith(".py"):
        path = path[: -len(".py")]
    parts = [p for p in path.split("/") if p]
    if len(parts) > 1 and parts[0] in ("tests", "test"):
        parts = parts[1:]
    label = ".".join(parts)
    if rest:
        label = f"{label}.{rest.replace('::', '.')}"
    return label


def build_test_command(test_id: str) -> list[str]:
    """
    Build the command that runs a single test inside the subject.

    Defaults to ``pytest <node-id>``, which suits most projects.  Some test
    suites cannot run that way at all: Django's builds ``INSTALLED_APPS``
    dynamically per test module, so a bare ``pytest`` invocation fails during
    collection (``AppRegistryNotReady`` / missing ``app_label``) and the traced
    run dies before reaching the test body.  Such subjects need their own
    runner, supplied by the campaign rather than hard-coded per project::

        FORECAST_TEST_CMD="python tests/runtests.py --settings=test_sqlite --parallel 1 -v0 {label}"

    Placeholders: ``{nodeid}`` is the pytest node ID, ``{label}`` the dotted
    form (see :func:`pytest_id_to_dotted_label`).  A template that expands to
    nothing falls back to the default rather than running an empty command.

    :param test_id: Test identifier, in any form
        :func:`~forecast.test_ids.normalize_pytest_test_id` accepts.
    :returns: Argument vector to execute.
    """
    node_id = normalize_pytest_test_id(test_id)
    label = pytest_id_to_dotted_label(node_id)
    template = os.getenv("FORECAST_TEST_CMD", "").strip()
    if template:
        try:
            parts = shlex.split(template)
        except ValueError:
            LOGGER.warning(
                "FORECAST_TEST_CMD is not valid shell syntax; using the default pytest command",
                extra={"template": template},
            )
            parts = []
        cmd = [p.format(nodeid=node_id, label=label) for p in parts]
        if cmd:
            return cmd
    return [p.format(nodeid=node_id, label=label) for p in DEFAULT_TEST_CMD]


#: How much of a failed run's output to carry into the log. Enough to hold a
#: traceback, short enough not to bury the log in test chatter.
_EXEC_OUTPUT_TAIL = 2000


def _run_container_with_env(
    *,
    container,
    timeout: int,
    conda_env: str | None,
    workdir: str,
    inner_cmd: list[str],
    runtime_env: dict[str, str] | None = None,
) -> tuple[int, bytes | bytearray | str | tuple[bytes, bytes] | None]:
    """
    Execute one command in a container while preserving env vars for conda runs.
    """
    cmd = ["timeout", f"{timeout}s"]
    docker_env = dict(runtime_env or {})
    if conda_env:
        cmd += ["conda", "run", "-n", conda_env]
        if docker_env:
            cmd += ["env"]
            cmd += [f"{key}={value}" for key, value in docker_env.items()]
        docker_env = {}
    cmd += inner_cmd
    return container.exec_run(
        cmd,
        workdir=workdir,
        environment=docker_env or None,
    )


def _decode_exec_output(
    output: bytes | bytearray | str | tuple[bytes, bytes] | None,
) -> str:
    if isinstance(output, tuple):
        left = output[0] or b""
        right = output[1] or b""
        return (
            left.decode("utf-8", errors="replace")
            + "\n"
            + right.decode("utf-8", errors="replace")
        ).strip()
    if isinstance(output, (bytes, bytearray)):
        return output.decode("utf-8", errors="replace")
    return str(output or "")


def _best_effort_reinstall_project(
    *,
    container,
    src_root: str,
    conda_env: str | None,
) -> None:
    """
    Try to reinstall the project in-place to avoid stale site-packages imports.
    """
    has_setup = False
    for marker in ("pyproject.toml", "setup.py", "setup.cfg"):
        code, _ = container.exec_run(["test", "-f", marker], workdir=src_root)
        if int(code) == 0:
            has_setup = True
            break
    if not has_setup:
        return

    code, output = _run_container_with_env(
        container=container,
        timeout=300,
        conda_env=conda_env,
        workdir=src_root,
        inner_cmd=["python", "-m", "pip", "install", "-e", ".", "-q"],
    )
    if int(code) == 0:
        return
    LOGGER.warning(
        "Editable reinstall failed while collecting events; retrying non-editable.",
        extra={"detail": _decode_exec_output(output)},
    )
    code2, output2 = _run_container_with_env(
        container=container,
        timeout=300,
        conda_env=conda_env,
        workdir=src_root,
        inner_cmd=["python", "-m", "pip", "install", ".", "-q"],
    )
    if int(code2) != 0:
        LOGGER.warning(
            "Fallback reinstall failed while collecting events.",
            extra={"detail": _decode_exec_output(output2)},
        )


def run_test_and_collect_events(
    test_id: str,
    src_root: str | Path,
    event_file_path: str | Path,
    python: str = sys.executable,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    budget: TraceBudget | None = None,
):
    """
    Run an instrumented pytest test locally and write its event file.

    Configures the ``sflkitlib`` runtime tracer through the environment before
    launching pytest: where to write the event stream, and how large that
    stream may grow (see :class:`~forecast.runner.trace_budget.TraceBudget`).
    Timeouts are silently swallowed — events collected up to the timeout are
    still written.

    :param test_id: Pytest node ID, e.g. ``"tests/test_foo.py::test_bar"``.
    :param src_root: Root directory of the subject project; used as the
        ``cwd`` for pytest.
    :param event_file_path: Host path where the binary event file should be
        written.
    :param python: Python executable to use.  Defaults to the current
        interpreter.
    :param env: Optional base environment variables.  Defaults to
        ``os.environ`` when ``None``.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :param budget: Limits applied to the trace.  ``None`` resolves the budget
        from the environment.
    """
    src_root = Path(src_root).resolve()
    pytest_test_id = normalize_pytest_test_id(test_id)
    # Inherit the caller's environment and configure the tracer for this run
    environ = dict(env) if env else dict(os.environ)
    environ.update(
        resolve_budget(budget).environment(str(Path(event_file_path).resolve()))
    )
    # Same command resolution as the container path, so a campaign that needs a
    # project's own test runner gets it locally too.
    local_cmd = build_test_command(pytest_test_id)
    if local_cmd and local_cmd[0] == "python":
        local_cmd = [python, *local_cmd[1:]]
    try:
        subprocess.run(
            local_cmd,
            capture_output=True,
            text=True,
            cwd=src_root,
            timeout=timeout,
            env=environ,
        )
    except subprocess.TimeoutExpired:
        LOGGER.info(
            f"Test {test_id} timed out after {timeout} seconds. Continuing with collected events."
        )


def run_input_and_collect_events(
    access: str,
    test_id: str,
    test: list[str],
    src_root: str | Path,
    event_file_path: str | Path,
    module: bool = False,
    python: str = sys.executable,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    budget: TraceBudget | None = None,
):
    """
    Run an instrumented program locally with the given input and write its event file.

    Used for system-test subjects (:class:`~forecast.subject.LocalSystemTestSubject`)
    that execute a script or module rather than a pytest test.

    :param access: Path to the entry-point script, or the module name when
        *module* is ``True``.
    :param test_id: Logical test identifier (used only for logging on timeout).
    :param test: Command-line arguments passed to the program after the
        access/module argument.
    :param src_root: Root directory of the subject project; used as ``cwd``.
    :param event_file_path: Host path where the binary event file should be
        written.
    :param module: When ``True`` the program is invoked as
        ``python -m <access>``; otherwise as ``python <access>``.
    :param python: Python executable to use.  Defaults to the current
        interpreter.
    :param env: Optional base environment variables.  Defaults to
        ``os.environ`` when ``None``.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :param budget: Limits applied to the trace.  ``None`` resolves the budget
        from the environment.
    """
    src_root = Path(src_root).resolve()
    # Inherit the caller's environment and configure the tracer for this run
    environ = dict(env) if env else dict(os.environ)
    environ.update(
        resolve_budget(budget).environment(str(Path(event_file_path).resolve()))
    )
    cmd = [python]
    if module:
        cmd += ["-m"]
    cmd += [access]
    cmd += test
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=src_root,
            timeout=timeout,
            env=environ,
        )
    except subprocess.TimeoutExpired:
        LOGGER.info(
            f"Input {test_id} timed out after {timeout} seconds. Continuing with collected events."
        )


#: Discovers the project's top-level packages and reports where importing each
#: one actually lands. Run inside the container, where the answer is decided by
#: that interpreter's ``sys.path`` and not by anything we can inspect from the
#: host.
_IMPORT_RESOLUTION_PROBE = r"""
import importlib, json, os, sys
root = os.path.realpath(sys.argv[1])
skip = {"test", "tests", "doc", "docs", "example", "examples", "bench", "benchmarks", "build"}
names = []
for base in (root, os.path.join(root, "src")):
    if not os.path.isdir(base):
        continue
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        continue
    for name in entries:
        if name.lower() in skip or name.startswith("."):
            continue
        if os.path.isfile(os.path.join(base, name, "__init__.py")):
            names.append(name)
resolved = {}
for name in names[:8]:
    try:
        module = importlib.import_module(name)
        path = getattr(module, "__file__", "") or ""
        resolved[name] = os.path.realpath(path) if path else ""
    except BaseException as error:
        import traceback
        frames = traceback.extract_tb(sys.exc_info()[2])
        where = ""
        for frame in reversed(frames):
            if frame.filename.startswith(root):
                where = "%s:%s in %s | %s" % (
                    os.path.relpath(frame.filename, root),
                    frame.lineno,
                    frame.name,
                    (frame.line or "").strip(),
                )
                break
        head = ""
        for frame in reversed(frames):
            if frame.filename.startswith(root):
                try:
                    with open(frame.filename) as handle:
                        head = "".join(handle.readlines()[:24])
                except OSError:
                    head = ""
                break
        resolved[name] = "ERROR:%s: %s [%s]\n----- head -----\n%s----------" % (
            type(error).__name__, error, where or "outside the source tree", head
        )
print("FORECAST_IMPORT_PROBE " + json.dumps({"root": root, "resolved": resolved}))
"""


def _instrumented_sources_are_importable(
    *,
    container,
    src_root: str,
    conda_env: str | None,
    timeout: int,
    description: str,
) -> bool:
    """
    Check that importing the project reaches the instrumented tree.

    Instrumentation rewrites the sources in place, but the test process imports
    whatever its ``sys.path`` resolves first -- and an installed copy in
    ``site-packages`` shadows the tree we just rewrote. When that happens the
    test still runs and still exits zero, yet none of the code under test is
    instrumented, so the trace holds only whatever the interpreter happened to
    import from the source tree on the way past. Such a trace looks like a
    successful collection and is worthless: every run records the same import
    sequence, so nothing separates a failing run from a passing one.

    Checking before the run is what makes the existing retry ladder usable --
    a ``PYTHONPATH`` override or a reinstall is exactly the remedy, but the
    ladder only ever fired when a run produced no trace whatsoever.

    :param container: A started Docker container object.
    :param src_root: Project root inside the container.
    :param conda_env: Optional conda environment to activate.
    :param timeout: Timeout in seconds for the probe.
    :param description: Human-readable label used in log messages.
    :returns: ``True`` when at least one of the project's own packages imports
        from inside *src_root*, and also when the probe could not reach a
        verdict -- an unreadable answer is not evidence of a broken
        environment, and refusing to collect on it would be worse than
        collecting a trace we can still inspect afterwards.
    """
    try:
        _, output = _run_container_with_env(
            container=container,
            timeout=timeout,
            conda_env=conda_env,
            workdir=src_root,
            inner_cmd=["python", "-c", _IMPORT_RESOLUTION_PROBE, src_root],
        )
    except Exception:
        LOGGER.debug(
            "Import-resolution probe could not be run; assuming the sources are "
            "importable.",
            extra={"target": description},
        )
        return True

    text = _decode_exec_output(output)
    marker = "FORECAST_IMPORT_PROBE "
    index = text.rfind(marker)
    if index < 0:
        LOGGER.debug(
            "Import-resolution probe returned no verdict; assuming the sources "
            "are importable. Output tail: %s",
            text[-_EXEC_OUTPUT_TAIL:],
            extra={"target": description},
        )
        return True
    try:
        report = json.loads(text[index + len(marker) :].splitlines()[0])
    except (ValueError, IndexError):
        return True

    root = report.get("root") or src_root
    resolved = report.get("resolved") or {}
    if not resolved:
        # Nothing that looks like a package of its own -- a single-module
        # project, say. There is nothing to shadow, so nothing to check.
        return True

    prefix = root.rstrip("/") + "/"
    inside = {
        name: path
        for name, path in resolved.items()
        if isinstance(path, str) and path.startswith(prefix)
    }
    if inside:
        LOGGER.debug(
            "Instrumented sources are what the project imports (%s).",
            ", ".join(sorted(inside)),
            extra={"target": description},
        )
        return True

    LOGGER.warning(
        "The project does not import from its instrumented sources; a trace "
        "collected now would not cover the test. Resolved to: %s",
        ", ".join(f"{name} -> {path or '<none>'}" for name, path in sorted(resolved.items())),
        extra={"target": description, "src_root": src_root},
    )
    return False


def _collect_events_in_container(
    *,
    container,
    inner_cmd: list[str],
    src_root: str,
    event_file_path: str | Path,
    timeout: int,
    conda_env: str | None,
    budget: TraceBudget,
    description: str,
) -> None:
    """
    Run *inner_cmd* in a container and stream the resulting trace to the host.

    The command is retried only when the run produced **no trace at all**,
    which indicates that the instrumented sources were never imported.  A trace
    that exists but could not be transferred is *not* retried: re-running the
    test would produce the same oversized trace again while leaving the first
    one behind in the container.  Every attempt deletes its trace inside the
    container before the next one starts, so a container never accumulates more
    than one.

    :param container: A started Docker container object.
    :param inner_cmd: Command to execute inside the container.
    :param src_root: Project root inside the container, used as ``workdir``.
    :param event_file_path: Host path to write the trace to.
    :param timeout: Per-attempt timeout in seconds.
    :param conda_env: Optional conda environment to activate.
    :param budget: Limits applied to the trace.
    :param description: Human-readable label used in log messages.
    :raises FileNotFoundError: If no attempt produced a trace.
    :raises RuntimeError: If a trace was produced but could not be transferred.
    """
    # A fixed path keeps the bookkeeping simple: there is exactly one trace in
    # the container at any time, and it is removed after every attempt.
    container_path = "/tmp/forecast_events"

    attempts: list[tuple[str, dict[str, str], bool]] = [
        ("default environment", {}, False),
        ("PYTHONPATH override", {"PYTHONPATH": src_root}, False),
        ("project reinstall", {"PYTHONPATH": src_root}, True),
    ]

    last_status = TRACE_MISSING
    for index, (label, extra_env, reinstall) in enumerate(attempts):
        if index > 0:
            LOGGER.info(
                "No trace produced; retrying with %s.",
                label,
                extra={"target": description, "src_root": src_root},
            )
        if reinstall:
            _best_effort_reinstall_project(
                container=container,
                src_root=src_root,
                conda_env=conda_env,
            )

        runtime_env = budget.environment(container_path)
        runtime_env.update(extra_env)
        # A run whose imports miss the instrumented tree cannot produce a
        # useful trace, so spend the next rung of the ladder rather than a full
        # test run. The last rung still runs: by then there is nothing left to
        # escalate to, and a trace we can inspect beats no trace at all.
        if index < len(attempts) - 1 and not _instrumented_sources_are_importable(
            container=container,
            src_root=src_root,
            conda_env=conda_env,
            timeout=timeout,
            description=description,
        ):
            last_status = TRACE_MISSING
            continue
        try:
            exit_code, exec_output = _run_container_with_env(
                container=container,
                timeout=timeout,
                conda_env=conda_env,
                workdir=src_root,
                inner_cmd=inner_cmd,
                runtime_env=runtime_env,
            )
            # A test command reports 0 when everything passed and 1 when a test
            # failed; the run we are tracing is expected to do either. Anything
            # else means the command did not get as far as running tests --
            # a collection error, an unimportable module, a killed process --
            # and the trace it leaves behind is whatever the interpreter
            # managed before giving up. That trace is not a recording of the
            # test, so say so: it used to be accepted in silence, which left
            # an instance with a plausible-looking but import-only trace and no
            # way to find out why.
            if exit_code not in (0, 1):
                LOGGER.warning(
                    "Instrumented run exited with %s (expected 0 or 1); "
                    "its trace may not cover the test. Output tail: %s",
                    exit_code,
                    _decode_exec_output(exec_output)[-_EXEC_OUTPUT_TAIL:],
                    extra={"target": description, "src_root": src_root},
                )
            try:
                status, size = _stream_event_file_from_container(
                    container,
                    container_path,
                    event_file_path,
                )
            except OSError as exc:
                if not _is_disk_full_error(exc):
                    raise
                # Reclaim scratch space and copy again. The run itself is not
                # repeated: the trace is already sitting in the container.
                LOGGER.warning(
                    "Ran out of disk while copying a trace; "
                    "reclaiming scratch space and retrying the copy.",
                    extra={"target": description},
                )
                _try_free_disk_space(Path(event_file_path))
                status, size = _stream_event_file_from_container(
                    container,
                    container_path,
                    event_file_path,
                )
        finally:
            _remove_container_file(container, container_path)

        last_status = status
        if status == TRACE_OK:
            LOGGER.debug(
                "Collected trace",
                extra={
                    "target": description,
                    "bytes": size,
                    "path": str(event_file_path),
                },
            )
            return
        if status in (TRACE_EMPTY, TRACE_ERROR):
            # The instrumented code did run, so retrying changes nothing.
            break

    if last_status == TRACE_ERROR:
        raise RuntimeError(
            f"Trace for '{description}' was produced but could not be copied out "
            "of the container. It is most likely too large for the available "
            "disk; lower FORECAST_TRACE_MAX_BYTES or trace fewer tests."
        )
    raise FileNotFoundError(
        f"Event file was not generated for '{description}'. "
        "Tried default run, PYTHONPATH fallback, and reinstall fallback."
    )


def docker_run_test_and_collect_events(
    client,
    image: str,
    test_id: str,
    src_root: str,
    event_file_path: str | Path,
    timeout: int = 120,
    patch: str | None = None,
    conda_env: str | None = None,
    budget: TraceBudget | None = None,
):
    """
    Run an instrumented pytest test inside a fresh Docker container and copy
    the resulting event file to the host.

    The trace is written inside the container and streamed out to
    *event_file_path*, so its size is bounded by disk rather than by memory.
    How large it may grow is controlled by *budget*.

    :param client: Docker SDK client (e.g. ``docker.from_env()``).
    :param image: Docker image name/tag to run.
    :param test_id: Pytest node ID.
    :param src_root: Absolute path inside the container to the project root
        (used as ``workdir`` for exec calls).
    :param event_file_path: Host path where the binary event file should be
        written after the run.
    :param timeout: Per-container timeout in seconds (passed to ``timeout``
        command inside the container).  Defaults to 120.
    :param patch: Optional unified diff to apply with ``git apply`` before
        running the test.
    :param conda_env: Optional conda environment to activate via
        ``conda run -n <env>``.
    :param budget: Limits applied to the trace.  ``None`` resolves the budget
        from the environment.
    :raises FileNotFoundError: If no attempt produced a trace.
    :raises RuntimeError: If a trace was produced but could not be transferred.
    """
    container = client.containers.create(image, command="sleep infinity", tty=True)
    pytest_test_id = normalize_pytest_test_id(test_id)
    try:
        container.start()
        if patch:
            # Write the diff to a temp file and apply it with git
            patch_file = "/tmp/patch.diff"
            write_file_to_container(container, patch_file, patch)
            container.exec_run(["git", "apply", patch_file], workdir=src_root)

        _collect_events_in_container(
            container=container,
            inner_cmd=build_test_command(pytest_test_id),
            src_root=src_root,
            event_file_path=event_file_path,
            timeout=timeout,
            conda_env=conda_env,
            budget=resolve_budget(budget),
            description=test_id,
        )
    finally:
        container.stop()
        container.remove()


def docker_run_input_and_collect_events(
    client,
    image: str,
    access: str,
    test: list[str],
    src_root: str,
    event_file_path: str | Path,
    module: bool = False,
    timeout: int = 120,
    patch: str | None = None,
    conda_env: str | None = None,
    budget: TraceBudget | None = None,
):
    """
    Run an instrumented program inside a fresh Docker container with the given
    input and copy the resulting event file to the host.

    Used for system-test subjects (:class:`~forecast.subject.DockerSystemTestSubject`)
    that execute a script or module rather than a pytest test.

    :param client: Docker SDK client.
    :param image: Docker image name/tag to run.
    :param access: Entry-point script path or module name inside the container.
    :param test: Command-line arguments passed to the program.
    :param src_root: Absolute path inside the container to the project root.
    :param event_file_path: Host path where the binary event file should be
        written.
    :param module: When ``True`` the program is invoked as
        ``python -m <access>``; otherwise as ``python <access>``.
    :param timeout: Per-container timeout in seconds.  Defaults to 120.
    :param patch: Optional unified diff to apply before running.
    :param conda_env: Optional conda environment name.
    :param budget: Limits applied to the trace.  ``None`` resolves the budget
        from the environment.
    :raises FileNotFoundError: If no attempt produced a trace.
    :raises RuntimeError: If a trace was produced but could not be transferred.
    """
    container = client.containers.create(image, command="sleep infinity", tty=True)
    try:
        container.start()
        if patch:
            # Write the diff to a temp file and apply it with git
            patch_file = "/tmp/patch.diff"
            write_file_to_container(container, patch_file, patch)
            container.exec_run(["git", "apply", patch_file], workdir=src_root)

        cmd: list[str] = ["python"]
        if module:
            cmd += ["-m"]
        cmd += [access]
        cmd += test

        _collect_events_in_container(
            container=container,
            inner_cmd=cmd,
            src_root=src_root,
            event_file_path=event_file_path,
            timeout=timeout,
            conda_env=conda_env,
            budget=resolve_budget(budget),
            description=access,
        )
    finally:
        container.stop()
        container.remove()
