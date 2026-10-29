"""
coverage_runner.py
------------------
Runs individual pytest tests (or system-test inputs) with ``coverage.py``
and returns :class:`~forecast.localization.spectra.TestResult` objects
carrying both line-level and function-level coverage.

Function coverage is read directly from the ``coverage.py`` JSON report
(``--show-functions`` format), which already contains per-function
``executed_lines``, ``missing_lines``, and ``start_line``.  No post-hoc
AST pass is needed.

Two families of runners are provided:

* **Local** — subprocess-based wrappers using the system Python interpreter.
* **Docker** — Docker SDK-based wrappers that spin up a fresh container per
  test, run the test, read back the JSON report, and remove the container.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from forecast.localization.spectra import FunctionKey, TestResult
from forecast.runner.utils import (
    docker_install_tests,
    write_file_to_container,
    read_file_from_container,
)
from forecast.test_ids import normalize_pytest_test_id

# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------


def collect_test_ids(
    src_root: str | Path,
    python: str = sys.executable,
    test_base: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """
    Use ``pytest --collect-only`` to discover all test node IDs under *src_root*.

    :param src_root: The root directory from which pytest is run.  Also used
        as the ``--rootdir`` to prevent pytest from walking up to a parent
        conftest.
    :param python: Python interpreter to use.  Defaults to the current
        interpreter.
    :param test_base: Optional test directory or file to search (e.g.
        ``"tests/"``).  When omitted pytest searches *src_root* itself.
    :param env: Optional environment variables.  Defaults to the current
        process environment when ``None``.
    :returns: List of pytest node IDs, e.g.
        ``["tests/test_foo.py::TestBar::test_baz", ...]``.
    """
    src_root = Path(src_root).resolve()
    cmd = [
        python,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "--tb=no",
        f"--rootdir={src_root}",
    ]
    if test_base:
        cmd.append(str(test_base))
    else:
        cmd.append(str(src_root))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=src_root,
        env=dict(env) if env else None,
    )
    ids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(("no tests ran", "=", "ERROR")):
            continue
        normalized = normalize_pytest_test_id(line)
        if "::" in normalized:
            ids.append(normalized)
    return ids


# ---------------------------------------------------------------------------
# Public entry point (local execution)
# ---------------------------------------------------------------------------


def run_test_without_coverage(
    test_id: str,
    src_root: str | Path,
    python: str = sys.executable,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> TestResult:
    """
    Run a single pytest test *without* coverage and return an empty
    :class:`~forecast.localization.spectra.TestResult`.

    Useful as a fast pre-pass to determine which tests fail before running
    the more expensive coverage instrumentation.

    :param test_id: Pytest node ID, e.g. ``"tests/test_foo.py::test_bar"``.
    :param src_root: Root directory of the subject project.
    :param python: Python interpreter to use.  Defaults to the current
        interpreter.
    :param env: Optional environment variables.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :returns: :class:`~forecast.localization.spectra.TestResult` with
        ``passed`` reflecting pytest's exit code and empty coverage dicts.
    """
    src_root = Path(src_root).resolve()
    pytest_test_id = normalize_pytest_test_id(test_id)
    try:
        run = subprocess.run(
            [
                python,
                "-m",
                "pytest",
                pytest_test_id,
                "-rN",
                "--tb=no",
                "-q",
            ],
            capture_output=True,
            text=True,
            cwd=src_root,
            timeout=timeout,
            env=dict(env) if env else None,
        )
        passed = run.returncode == 0

        return TestResult(
            test_id=test_id,
            passed=passed,
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            test_id=test_id,
            passed=False,
        )


def run_input_without_coverage(
    access: str,
    test: list[str],
    src_root: str | Path,
    passing: bool,
    module: bool = False,
    python: str = sys.executable,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> TestResult:
    """
    Run a single system-test input *without* coverage and return a
    :class:`~forecast.localization.spectra.TestResult`.

    Useful as a fast pre-pass before the more expensive coverage runs.

    :param access: Path to the entry-point script, or the module name when
        *module* is ``True``.
    :param test: Command-line arguments to pass to the program.
    :param src_root: Root directory of the subject project.
    :param passing: Expected outcome — when ``True`` the test is considered
        passing if the process exits with code 0.
    :param module: When ``True`` the program is invoked as
        ``python -m <access>``; otherwise as ``python <access>``.
    :param python: Python interpreter to use.  Defaults to the current
        interpreter.
    :param env: Optional environment variables.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :returns: :class:`~forecast.localization.spectra.TestResult` with empty
        coverage dicts and ``passed`` set from the exit code vs *passing*.
    """
    src_root = Path(src_root).resolve()
    cmd = [
        python,
    ]
    if module:
        cmd += [
            "-m",
            access,
        ]
    else:
        cmd += [
            access,
        ]
    cmd += test
    try:
        run = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=src_root,
            timeout=timeout,
            env=dict(env) if env else None,
        )
        passed = run.returncode == 0

        return TestResult(
            test_id=access,
            passed=passed == passing,
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            test_id=access,
            passed=False,  # Consider it a failure if it times out
        )


def run_test_with_coverage(
    test_id: str,
    src_root: str | Path,
    python: str = sys.executable,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> TestResult:
    """
    Run a single pytest test with coverage and return a :class:`TestResult`.

    Invokes ``coverage run --branch --source=<src_root> -m pytest <test_id>``
    in a temporary directory, then exports the ``.coverage`` data file to
    JSON via ``coverage json``.  The JSON report is parsed to extract both
    line-level and function-level coverage.

    :param test_id: Pytest node ID, e.g. ``"tests/test_foo.py::test_bar"``.
    :param src_root: Root of the source tree to measure (``--source`` flag).
    :param python: Python interpreter to use.  Defaults to the current
        interpreter.
    :param env: Optional environment variables.  Defaults to the current
        process environment when ``None``.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :returns: A :class:`~forecast.localization.spectra.TestResult` with
        ``passed`` set according to pytest's exit code and ``line_coverage``
        / ``function_coverage`` populated from the JSON report.
    """
    src_root = Path(src_root).resolve()
    pytest_test_id = normalize_pytest_test_id(test_id)
    with tempfile.TemporaryDirectory() as tmp:
        cov_file = f"{tmp}/.coverage"
        json_file = f"{tmp}/cov.json"

        try:
            run = subprocess.run(
                [
                    python,
                    "-m",
                    "coverage",
                    "run",
                    f"--data-file={cov_file}",
                    "--branch",
                    f"--source={src_root}",
                    "-m",
                    "pytest",
                    pytest_test_id,
                    "-rN",
                    "--tb=no",
                    "-q",
                ],
                capture_output=True,
                text=True,
                cwd=src_root,
                timeout=timeout,
                env=dict(env) if env else None,
            )
            passed = run.returncode == 0
        except subprocess.TimeoutExpired:
            passed = False

        subprocess.run(
            [
                python,
                "-m",
                "coverage",
                "json",
                f"--data-file={cov_file}",
                "-o",
                json_file,
                "-q",
            ],
            capture_output=True,
            cwd=src_root,
        )

        line_coverage: dict[str, set[int]] = {}
        function_coverage: set[FunctionKey] = set()
        try:
            with open(json_file) as f:
                data = json.load(f)
            for fpath, fdata in data.get("files", {}).items():
                lines = set(fdata.get("executed_lines", []))
                if lines:
                    line_coverage[fpath] = lines
                function_coverage.update(_extract_function_coverage(fpath, fdata))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    return TestResult(
        test_id=test_id,
        passed=passed,
        line_coverage=line_coverage,
        function_coverage=function_coverage,
    )


def run_input_with_coverage(
    access: str,
    test_id: str,
    test: list[str],
    src_root: str | Path,
    passing: bool,
    module: bool = False,
    python: str = sys.executable,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> TestResult:
    """
    Run a single system-test input with coverage and return a
    :class:`~forecast.localization.spectra.TestResult`.

    :param access: Path to the entry-point script or module name.
    :param test_id: Logical test identifier stored on the returned result.
    :param test: Command-line arguments to pass to the program.
    :param src_root: Root of the source tree to measure.
    :param passing: Expected outcome — used as the ``passed`` field on the
        returned result (the process exit code is *not* used here because
        system tests may exit non-zero even on success).
    :param module: When ``True`` the program is invoked as
        ``python -m <access>``; otherwise as ``python <access>``.
    :param python: Python interpreter to use.  Defaults to the current
        interpreter.
    :param env: Optional environment variables.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :returns: :class:`~forecast.localization.spectra.TestResult` populated
        with line-level and function-level coverage from the JSON report.
    """
    src_root = Path(src_root).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        cov_file = f"{tmp}/.coverage"
        json_file = f"{tmp}/cov.json"
        cmd = [
            python,
            "-m",
            "coverage",
            "run",
            f"--data-file={cov_file}",
            "--branch",
            f"--source={src_root}",
        ]
        if module:
            cmd += [
                "-m",
                access,
            ]
        else:
            cmd += [
                access,
            ]
        cmd += test

        try:
            _ = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=src_root,
                timeout=timeout,
                env=dict(env) if env else None,
            )
        except subprocess.TimeoutExpired:
            pass

        subprocess.run(
            [
                python,
                "-m",
                "coverage",
                "json",
                f"--data-file={cov_file}",
                "-o",
                json_file,
                "-q",
            ],
            capture_output=True,
            cwd=src_root,
        )

        line_coverage: dict[str, set[int]] = {}
        function_coverage: set[FunctionKey] = set()
        try:
            with open(json_file) as f:
                data = json.load(f)
            for fpath, fdata in data.get("files", {}).items():
                lines = set(fdata.get("executed_lines", []))
                if lines:
                    line_coverage[fpath] = lines
                function_coverage.update(_extract_function_coverage(fpath, fdata))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    return TestResult(
        test_id=test_id,
        passed=passing,
        line_coverage=line_coverage,
        function_coverage=function_coverage,
    )


# ---------------------------------------------------------------------------
# Single-test runner
# ---------------------------------------------------------------------------


def _extract_function_coverage(
    fpath: str,
    fdata: dict,
) -> list[FunctionKey]:
    """
    Build :class:`~forecast.localization.spectra.FunctionKey` objects from a
    single file's coverage JSON entry.

    The ``coverage.py`` JSON ``functions`` dict has entries keyed by:

    * ``""``                  — module-level code (skipped)
    * ``"func_name"``         — a top-level function
    * ``"ClassName.method"``  — a method inside a class

    Each entry provides ``start_line``, ``executed_lines``, and
    ``missing_lines``.  The end line is inferred as
    ``max(executed_lines + missing_lines)``.  Only functions that have at
    least one executed line (i.e. were actually called) are returned.

    :param fpath: Project-relative or absolute path of the source file, used
        as-is on the returned :class:`~forecast.localization.spectra.FunctionKey`.
    :param fdata: The per-file dict from the coverage JSON report.
    :returns: List of :class:`~forecast.localization.spectra.FunctionKey`
        objects for every function that was called during the test.
    """
    keys: list[FunctionKey] = []
    for func_key, func_data in fdata.get("functions", {}).items():
        if not func_key:  # skip the module-level "" pseudo-function
            continue

        executed = func_data.get("executed_lines", [])
        if not executed:  # function was not hit at all
            continue

        start_line: int = func_data["start_line"]
        all_lines = executed + func_data.get("missing_lines", [])
        end_line: int = max(all_lines) if all_lines else start_line

        # Parse "ClassName.method" vs plain "method"
        if "." in func_key:
            class_name, name = func_key.rsplit(".", 1)
        else:
            class_name, name = None, func_key

        keys.append(
            FunctionKey(
                file=fpath,
                name=name,
                class_name=class_name,
                start_line=start_line,
                end_line=end_line,
            )
        )
    return keys


# ---------------------------------------------------------------------------
# Docker helpers  (operate via the Docker SDK)
# ---------------------------------------------------------------------------


def docker_collect_test_ids(
    client,
    image: str,
    src_root: str,
    test_base: str | None = None,
    patch: str | None = None,
    conda_env: str | None = None,
) -> list[str]:
    """
    Run ``pytest --collect-only`` in a short-lived container and return test
    node IDs.

    Creates a fresh container from *image*, installs ``pytest`` and
    ``coverage``, optionally applies a patch, collects test IDs, then stops
    and removes the container.

    :param client: Docker SDK client (e.g. ``docker.from_env()``).
    :param image: Docker image name/tag.
    :param src_root: Absolute path inside the container to the project root;
        used as ``--rootdir`` and ``workdir``.
    :param test_base: Optional test directory / file to pass to pytest.
        When omitted pytest searches *src_root*.
    :param patch: Optional unified diff to apply with ``git apply`` before
        collecting.
    :param conda_env: Optional conda environment to activate via
        ``conda run -n <env>``.
    :returns: List of pytest node IDs discovered inside the container.
    """
    cmd = []
    if conda_env:
        cmd += [
            "conda",
            "run",
            "-n",
            conda_env,
        ]
    cmd += [
        "python",
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "--tb=no",
        f"--rootdir={src_root}",
    ]
    if test_base:
        cmd.append(test_base)
    else:
        cmd.append(src_root)

    container = client.containers.create(image, command="sleep infinity", tty=True)
    try:
        container.start()
        docker_install_tests(container, conda_env)
        if patch:
            patch_file = "/tmp/patch.diff"
            write_file_to_container(container, patch_file, patch)
            container.exec_run(["git", "apply", patch_file], workdir=src_root)
        _, output = container.exec_run(cmd, workdir=src_root)
    finally:
        container.stop()
        container.remove()

    ids = []
    for line in output.decode().splitlines():
        line = line.strip()
        if not line or line.startswith(("no tests ran", "=", "ERROR")):
            continue
        normalized = normalize_pytest_test_id(line)
        if "::" in normalized:
            ids.append(normalized)
    return ids


from forecast.runner.event_runner import build_test_command


def docker_run_test_without_coverage(
    client,
    image: str,
    test_id: str,
    src_root: str,
    timeout: int = 120,
    patch: str | None = None,
    conda_env: str | None = None,
) -> TestResult:
    """
    Run a single pytest test *without* coverage inside its own container and
    return a :class:`~forecast.localization.spectra.TestResult`.

    Each call creates a fresh container, runs the test, then stops and removes
    the container.

    :param client: Docker SDK client.
    :param image: Docker image name/tag.
    :param test_id: Pytest node ID.
    :param src_root: Absolute path inside the container to the project root.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :param patch: Optional unified diff to apply before running.
    :param conda_env: Optional conda environment name.
    :returns: :class:`~forecast.localization.spectra.TestResult` with empty
        coverage and ``passed`` reflecting the container's exit code.
    """
    pytest_test_id = normalize_pytest_test_id(test_id)
    container = client.containers.create(image, command="sleep infinity", tty=True)
    try:
        container.start()
        docker_install_tests(container, conda_env)
        if patch:
            patch_file = "/tmp/patch.diff"
            write_file_to_container(container, patch_file, patch)
            container.exec_run(["git", "apply", patch_file], workdir=src_root)

        cmd = ["timeout", f"{timeout}s"]
        if conda_env:
            cmd += [
                "conda",
                "run",
                "-n",
                conda_env,
            ]
        # Use the same command the traced run uses. Hard-coding pytest here
        # meant a subject with its own runner was probed with a runner it does
        # not have: every django candidate "failed", so django-12113 found 200
        # sibling tests and reported none of them passing, leaving the
        # diagnosis with nothing to contrast against.
        cmd += build_test_command(pytest_test_id)

        exit_code, _ = container.exec_run(
            cmd,
            workdir=src_root,
        )
        passed = exit_code == 0

    finally:
        container.stop()
        container.remove()

    return TestResult(
        test_id=test_id,
        passed=passed,
    )


def docker_run_input_without_coverage(
    client,
    image: str,
    access: str,
    test_id: str,
    test: list[str],
    src_root: str,
    passing: bool,
    module: bool = False,
    timeout: int = 120,
    patch: str | None = None,
    conda_env: str | None = None,
) -> TestResult:
    """
    Run a single system-test input *without* coverage inside its own container
    and return a :class:`~forecast.localization.spectra.TestResult`.

    Each call creates a fresh container, runs the program, then stops and
    removes the container.

    :param client: Docker SDK client.
    :param image: Docker image name/tag.
    :param access: Entry-point script path or module name inside the container.
    :param test_id: Logical test identifier stored on the returned result.
    :param test: Command-line arguments to pass to the program.
    :param src_root: Absolute path inside the container to the project root.
    :param passing: Expected outcome — ``passed`` is set to
        ``(exit_code == 0) == passing``.
    :param module: When ``True`` the program is invoked as
        ``python -m <access>``; otherwise as ``python <access>``.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :param patch: Optional unified diff to apply before running.
    :param conda_env: Optional conda environment name.
    :returns: :class:`~forecast.localization.spectra.TestResult` with empty
        coverage.
    """
    container = client.containers.create(image, command="sleep infinity", tty=True)
    try:
        container.start()
        docker_install_tests(container, conda_env)
        if patch:
            patch_file = "/tmp/patch.diff"
            write_file_to_container(container, patch_file, patch)
            container.exec_run(["git", "apply", patch_file], workdir=src_root)

        cmd = ["timeout", f"{timeout}s"]
        if conda_env:
            cmd += [
                "conda",
                "run",
                "-n",
                conda_env,
            ]
        cmd += [
            "python",
        ]
        if module:
            cmd += [
                "-m",
                access,
            ]
        else:
            cmd += [
                access,
            ]
        cmd += test

        exit_code, _ = container.exec_run(
            cmd,
            workdir=src_root,
        )
        passed = exit_code == 0

    finally:
        container.stop()
        container.remove()

    return TestResult(
        test_id,
        passed=passed == passing,
    )


def docker_run_test_with_coverage(
    client,
    image: str,
    test_id: str,
    src_root: str,
    timeout: int = 120,
    patch: str | None = None,
    conda_env: str | None = None,
) -> TestResult:
    """
    Run a single pytest test with coverage inside its own container and return
    a :class:`~forecast.localization.spectra.TestResult`.

    Each call creates a fresh container, runs the test under
    ``coverage run --branch``, exports the ``.coverage`` data file to JSON
    via ``coverage json``, reads the JSON back from the container, then stops
    and removes the container.

    :param client: Docker SDK client.
    :param image: Docker image name/tag.
    :param test_id: Pytest node ID.
    :param src_root: Absolute path inside the container to the project root;
        used as ``--source`` and ``workdir``.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :param patch: Optional unified diff to apply before running.
    :param conda_env: Optional conda environment name.
    :returns: :class:`~forecast.localization.spectra.TestResult` populated
        with line-level and function-level coverage from the JSON report.
    """
    cov_file = "/tmp/.coverage"
    json_file = "/tmp/cov.json"
    pytest_test_id = normalize_pytest_test_id(test_id)

    line_coverage: dict[str, set[int]] = {}
    function_coverage: set[FunctionKey] = set()

    container = client.containers.create(image, command="sleep infinity", tty=True)
    try:
        container.start()
        docker_install_tests(container)
        if patch:
            patch_file = "/tmp/patch.diff"
            write_file_to_container(container, patch_file, patch)
            container.exec_run(["git", "apply", patch_file], workdir=src_root)

        # 1. Run the test under coverage
        # wrap in timeout and consider failed if it exceeds, to avoid hanging containers

        cmd = ["timeout", f"{timeout}s"]
        if conda_env:
            cmd += [
                "conda",
                "run",
                "-n",
                conda_env,
            ]
        cmd += [
            "python",
            "-m",
            "coverage",
            "run",
            f"--data-file={cov_file}",
            "--branch",
            f"--source={src_root}",
            "-m",
            "pytest",
            pytest_test_id,
            "-rN",
            "--tb=no",
            "-q",
        ]

        exit_code, _ = container.exec_run(
            cmd,
            workdir=src_root,
        )
        passed = exit_code == 0

        # 2. Export coverage to JSON
        container.exec_run(
            [
                "python",
                "-m",
                "coverage",
                "json",
                f"--data-file={cov_file}",
                "-o",
                json_file,
                "-q",
            ],
            workdir=src_root,
        )

        # 3. Read the JSON back from the container
        try:
            raw = read_file_from_container(container, json_file, decode=True)
            data = json.loads(raw)
            for fpath, fdata in data.get("files", {}).items():
                lines = set(fdata.get("executed_lines", []))
                if lines:
                    line_coverage[fpath] = lines
                function_coverage.update(_extract_function_coverage(fpath, fdata))
        except (json.JSONDecodeError, Exception):
            pass
    finally:
        container.stop()
        container.remove()

    return TestResult(
        test_id=test_id,
        passed=passed,
        line_coverage=line_coverage,
        function_coverage=function_coverage,
    )


def docker_run_tests_with_coverage_batch(
    client,
    image: str,
    test_ids: list[str],
    src_root: str,
    timeout: int = 120,
    patch: str | None = None,
    conda_env: str | None = None,
) -> list[TestResult]:
    """
    Collect per-test coverage for many tests in a *single* container and a
    single pytest run.

    Creating one container per test dominates coverage time for instances with
    hundreds of ``PASS_TO_PASS`` tests: the container lifecycle and interpreter
    start-up cost is paid for every test even though ``coverage.py`` itself is
    fast.  Here all tests run in one pytest process with per-test coverage
    contexts (``--cov-context=test``), so ``coverage.py`` records which test hit
    each line, and the fixed overhead is paid once instead of ``N`` times.

    Only line coverage is recovered (which is all test selection needs);
    function coverage is left empty.  Pass/fail is taken from pytest's
    ``FAILED``/``ERROR`` summary lines.

    :param client: Docker SDK client.
    :param image: Docker image name/tag (the clean, uninstrumented image).
    :param test_ids: Pytest node IDs to measure.
    :param src_root: Project root inside the container; used as ``--cov`` target
        and ``workdir``.
    :param timeout: Per-test timeout budget; the whole run is bounded by a
        multiple of it.
    :param patch: Optional unified diff to apply before running.
    :param conda_env: Optional conda environment name.
    :returns: One :class:`~forecast.localization.spectra.TestResult` per
        requested test id, in the same order.
    :raises RuntimeError: If the batch run produced no usable per-test
        coverage, so the caller can fall back to the per-test path.
    """
    ordered = [normalize_pytest_test_id(t) for t in test_ids]
    if not ordered:
        return []

    reader_path = "/tmp/forecast_cov_reader.py"
    # A ``coverage json --show-contexts`` report repeats every context on every
    # line and grows to gigabytes for hundreds of tests.  Instead read the
    # coverage database directly and emit compact ``{node: {file: [lines]}}``.
    reader_src = (
        "import json, coverage\n"
        "d = coverage.CoverageData()\n"
        "d.read()\n"
        "out = {}\n"
        "files = list(d.measured_files())\n"
        "for ctx in d.measured_contexts():\n"
        "    node = ctx.split('|', 1)[0]\n"
        "    if not node:\n"
        "        continue\n"
        # set_query_context matches the exact context; set_query_contexts would
        # treat it as a regex, and node ids contain |, [, ] and :: metacharacters.
        "    d.set_query_context(ctx)\n"
        "    bucket = out.setdefault(node, {})\n"
        "    for f in files:\n"
        "        lines = d.lines(f)\n"
        "        if lines:\n"
        "            bucket.setdefault(f, []).extend(lines)\n"
        "print(json.dumps(out))\n"
    )
    # coverage.py runs the suite once; a per-test container is not paying the
    # bill any more, so allow the whole run a generous, test-count-scaled budget.
    total_timeout = min(3600, max(600, len(ordered) * 8))

    def _conda(cmd: list[str]) -> list[str]:
        return (["conda", "run", "-n", conda_env] if conda_env else []) + cmd

    container = client.containers.create(image, command="sleep infinity", tty=True)
    try:
        container.start()
        docker_install_tests(container, conda_env)
        # pytest-cov keys coverage contexts by the exact pytest node id, which
        # is what the selection logic matches against.
        code, out = container.exec_run(
            _conda(["python", "-m", "pip", "install", "-q", "pytest-cov"]),
            workdir=src_root,
        )
        if int(code) != 0:
            raise RuntimeError(
                "pytest-cov could not be installed in the container: "
                + (out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out))
            )
        if patch:
            patch_file = "/tmp/patch.diff"
            write_file_to_container(container, patch_file, patch)
            container.exec_run(["git", "apply", patch_file], workdir=src_root)

        run_cmd = ["timeout", f"{total_timeout}s"] + _conda(
            [
                "python",
                "-m",
                "pytest",
                *ordered,
                f"--cov={src_root}",
                "--cov-branch",
                "--cov-context=test",
                "--cov-report=",
                "-rfE",
                "--tb=no",
                "-q",
                "-p",
                "no:cacheprovider",
            ]
        )
        run_code, run_out = container.exec_run(run_cmd, workdir=src_root)
        stdout = run_out.decode("utf-8", "replace") if isinstance(run_out, bytes) else str(run_out)

        # Read per-test coverage straight from the coverage database (default
        # ``.coverage`` written by pytest-cov in ``src_root``).
        write_file_to_container(container, reader_path, reader_src)
        read_code, read_out = container.exec_run(
            _conda(["python", reader_path]),
            workdir=src_root,
        )
        payload = read_out.decode("utf-8", "replace") if isinstance(read_out, bytes) else str(read_out)
        # The reader prints a single JSON line; ignore any preceding warnings.
        json_line = next(
            (ln for ln in reversed(payload.splitlines()) if ln.startswith("{")),
            "",
        )
        context_map = json.loads(json_line) if json_line else {}
    finally:
        container.stop()
        container.remove()

    # Tests that pytest reported as FAILED/ERROR (summary lines from -rfE).
    failed: set[str] = set()
    for line in stdout.splitlines():
        line = line.strip()
        for marker in ("FAILED ", "ERROR "):
            if line.startswith(marker):
                node = line[len(marker):].split(" ", 1)[0].strip()
                if node:
                    failed.add(normalize_pytest_test_id(node))

    per_test: dict[str, dict[str, set[int]]] = {t: {} for t in ordered}
    saw_context = False
    for node, file_lines in context_map.items():
        bucket = per_test.get(node)
        if bucket is None:
            bucket = per_test.setdefault(node, {})
        for fpath, lines in file_lines.items():
            if lines:
                bucket.setdefault(fpath, set()).update(int(x) for x in lines)
                saw_context = True

    if not saw_context:
        raise RuntimeError(
            "batch coverage produced no per-test contexts; "
            "coverage contexts may be unsupported here"
        )

    results: list[TestResult] = []
    for original, node in zip(test_ids, ordered):
        results.append(
            TestResult(
                test_id=original,
                passed=node not in failed,
                line_coverage=per_test.get(node, {}),
                function_coverage=set(),
            )
        )
    return results


def docker_run_input_with_coverage(
    client,
    image: str,
    access: str,
    test_id: str,
    test: list[str],
    src_root: str,
    passing: bool,
    module: bool = False,
    timeout: int = 120,
    patch: str | None = None,
    conda_env: str | None = None,
) -> TestResult:
    """
    Run a single system-test input with coverage inside its own container and
    return a :class:`~forecast.localization.spectra.TestResult`.

    Each call creates a fresh container, runs the program under
    ``coverage run``, reads back the JSON report, then stops and removes
    the container.

    :param client: Docker SDK client.
    :param image: Docker image name/tag.
    :param access: Entry-point script path or module name inside the container.
    :param test_id: Logical test identifier stored on the returned result.
    :param test: Command-line arguments to pass to the program.
    :param src_root: Absolute path inside the container to the project root.
    :param passing: Expected outcome — stored as-is in the returned result's
        ``passed`` field.
    :param module: When ``True`` the program is invoked as
        ``python -m <access>``; otherwise as ``python <access>``.
    :param timeout: Per-test timeout in seconds.  Defaults to 120.
    :param patch: Optional unified diff to apply before running.
    :param conda_env: Optional conda environment name.
    :returns: :class:`~forecast.localization.spectra.TestResult` populated
        with line-level and function-level coverage from the JSON report.
    """
    cov_file = "/tmp/.coverage"
    json_file = "/tmp/cov.json"

    line_coverage: dict[str, set[int]] = {}
    function_coverage: set[FunctionKey] = set()

    container = client.containers.create(image, command="sleep infinity", tty=True)
    try:
        container.start()
        if patch:
            patch_file = "/tmp/patch.diff"
            write_file_to_container(container, patch_file, patch)
            container.exec_run(["git", "apply", patch_file], workdir=src_root)

        cmd = ["timeout", f"{timeout}s"]
        if conda_env:
            cmd += [
                "conda",
                "run",
                "-n",
                conda_env,
            ]
        cmd += ["python"]
        if module:
            cmd += ["-m"]
        cmd += [access]
        cmd += test
        exit_code, _ = container.exec_run(
            cmd,
            workdir=src_root,
        )
        # 2. Export coverage to JSON
        container.exec_run(
            [
                "python",
                "-m",
                "coverage",
                "json",
                f"--data-file={cov_file}",
                "-o",
                json_file,
                "-q",
            ],
            workdir=src_root,
        )

        # 3. Read the JSON back from the container
        try:
            raw = read_file_from_container(container, json_file, decode=True)
            data = json.loads(raw)
            for fpath, fdata in data.get("files", {}).items():
                lines = set(fdata.get("executed_lines", []))
                if lines:
                    line_coverage[fpath] = lines
                function_coverage.update(_extract_function_coverage(fpath, fdata))
        except (json.JSONDecodeError, Exception):
            pass
    finally:
        container.stop()
        container.remove()

    return TestResult(
        test_id=test_id,
        passed=passing,
        line_coverage=line_coverage,
        function_coverage=function_coverage,
    )
