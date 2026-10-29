"""
The SWE-bench environment repair agents run and are judged in.

Every agent, in both modes, works inside the instance's SWE-bench image -- the
same image FORECAST traced the instance in -- and every patch is judged by the
official SWE-bench harness. This module is the one place that knows how:

* :func:`instance_image` names the image of an instance;
* :class:`InstanceContainer` runs commands, edits files and runs tests in a
  fresh container of it, for agents FORECAST drives itself (ACR, RepairAgent);
* :func:`harness_evaluate` returns the harness verdict on a patch: applied to
  the base commit, ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` run with the
  developer's test patch, ``resolved`` exactly as SWE-bench reports it.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

#: Where SWE-bench images keep the repository.
WORKDIR = "/testbed"


def image_namespace() -> Optional[str]:
    """
    Where instance images come from: ``None`` (default) for the images built
    locally during trace collection, or a registry namespace -- ``swebench``
    for the official prebuilt images on Docker Hub -- set through
    ``FORECAST_IMAGE_NAMESPACE``.
    """
    return os.environ.get("FORECAST_IMAGE_NAMESPACE") or None


def test_spec(task_info: dict):
    """
    The SWE-bench test spec of an instance, for the images of
    :func:`image_namespace`.

    :param task_info: The raw SWE-bench dataset row of the instance.
    :returns: A :class:`swebench.harness.test_spec.test_spec.TestSpec`.
    """
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec(task_info, namespace=image_namespace())


def instance_image(task_info: dict) -> str:
    """
    The image an instance's agents run in: the locally built SWE-bench
    instance image FORECAST collected the instance's traces in.

    :param task_info: The raw SWE-bench dataset row of the instance.
    :returns: Image reference, e.g. ``sweb.eval.x86_64.psf__requests-863:latest``.
    """
    return test_spec(task_info).instance_image_key


def image_digest(image: str) -> Optional[str]:
    """
    The content digest of a local image (registry digest if it was pulled,
    image id otherwise), so a run records exactly what it ran in.
    """
    try:
        import docker

        found = docker.from_env(timeout=60).images.get(image)
    except Exception:
        return None
    digests = found.attrs.get("RepoDigests") or []
    return digests[0] if digests else found.id


@dataclass
class ExecResult:
    """Outcome of one command run in an :class:`InstanceContainer`."""

    output: str
    returncode: int
    timed_out: bool = False


class InstanceContainer:
    """
    A throwaway container of an instance image, at the base commit.

    Used as a context manager; the container is removed on exit. Commands run
    in ``/testbed`` with the image's ``testbed`` conda environment active, as
    the SWE-bench harness runs them.

    :param task_info: The raw SWE-bench dataset row of the instance.
    :param client: Docker client; ``None`` creates one from the environment.
    """

    def __init__(self, task_info: dict, client=None):
        self.task_info = task_info
        self.image = instance_image(task_info)
        self._client = client
        self.container = None

    def __enter__(self) -> "InstanceContainer":
        import docker

        client = self._client or docker.from_env(timeout=600)
        self.container = client.containers.run(
            self.image,
            ["sleep", "infinity"],
            name=f"forecast-agent-{uuid.uuid4().hex[:12]}",
            detach=True,
            working_dir=WORKDIR,
        )
        return self

    def __exit__(self, *exc) -> None:
        if self.container is not None:
            try:
                self.container.remove(force=True)
            finally:
                self.container = None

    def exec(self, command: str, timeout: int = 300) -> ExecResult:
        """
        Run a shell command in ``/testbed`` under the ``testbed`` environment.

        :param command: Shell command line.
        :param timeout: Seconds before the command is killed.
        :returns: Combined stdout/stderr and exit code; exit code 124 and
            ``timed_out`` when the timeout hit.
        """
        script = (
            "source /opt/miniconda3/bin/activate testbed >/dev/null 2>&1; "
            f"cd {WORKDIR}; {command}"
        )
        result = self.container.exec_run(
            ["timeout", "--kill-after=10", str(timeout), "bash", "-c", script],
            workdir=WORKDIR,
        )
        output = (result.output or b"").decode("utf-8", errors="replace")
        return ExecResult(output, result.exit_code, result.exit_code == 124)

    def read_file(self, path: str) -> Optional[str]:
        """Contents of *path* (relative to ``/testbed``), or ``None``."""
        result = self.exec(f"cat -- {shlex.quote(path)}", timeout=60)
        return result.output if result.returncode == 0 else None

    def write_file(self, path: str, content: str) -> None:
        """Write *content* to *path* (relative to ``/testbed``)."""
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        quoted = shlex.quote(path)
        result = self.exec(
            f"mkdir -p \"$(dirname {quoted})\" && "
            f"echo {encoded} | base64 -d > {quoted}",
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"writing {path} failed: {result.output}")

    def apply_patch(self, patch: str) -> ExecResult:
        """``git apply`` *patch* to the working tree (exit 0 on success)."""
        if not patch.endswith("\n"):
            patch += "\n"
        self.write_file("/tmp/forecast.patch", patch)
        return self.exec("git apply --whitespace=nowarn /tmp/forecast.patch", 120)

    def diff(self) -> str:
        """The working tree's changes to tracked and new files, as a patch."""
        result = self.exec("git add -A && git diff --cached --no-color HEAD", 120)
        self.exec("git reset -q", 60)
        return result.output if result.returncode == 0 else ""

    def reset(self) -> None:
        """Discard every change, back to the base commit."""
        self.exec("git checkout -q -- . && git clean -fdq", 120)

    def apply_test_patch(self) -> ExecResult:
        """Add the developer's tests, for agents meant to see them."""
        return self.apply_patch(self.task_info["test_patch"])

    def run_tests(self, timeout: int = 1800) -> ExecResult:
        """
        Run the instance's test files with the project's own test command --
        the command and file selection the SWE-bench harness uses.

        :param timeout: Seconds before the run is killed.
        """
        return self.exec(test_command(self.task_info), timeout=timeout)

    def export(self, destination: Path) -> Path:
        """
        Copy the repository out of the container, ``.git`` included.

        :param destination: Directory to extract into.
        :returns: The repository root on the host.
        """
        import io
        import tarfile

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        stream, _ = self.container.get_archive(WORKDIR)
        buffer = io.BytesIO(b"".join(stream))
        with tarfile.open(fileobj=buffer) as archive:
            archive.extractall(destination, filter="tar")
        return destination / Path(WORKDIR).name


def test_verdict(task_info: dict, output: str) -> dict[str, Any]:
    """
    Judge a test run the way the SWE-bench harness does: parse *output* with
    the project's log parser and check ``FAIL_TO_PASS`` and ``PASS_TO_PASS``.

    :param task_info: The raw SWE-bench dataset row of the instance.
    :param output: Output of :meth:`InstanceContainer.run_tests`.
    :returns: ``passed`` (all required tests pass), and the tests of each list
        that do not pass.
    """
    from swebench.harness.grading import test_passed
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    status = MAP_REPO_TO_PARSER[task_info["repo"]](output, test_spec(task_info))
    lists = {}
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        tests = task_info[key]
        tests = json.loads(tests) if isinstance(tests, str) else list(tests)
        lists[key] = [t for t in tests if not test_passed(t, status)]
    return {
        "passed": not lists["FAIL_TO_PASS"] and not lists["PASS_TO_PASS"],
        "failing_fail_to_pass": lists["FAIL_TO_PASS"],
        "failing_pass_to_pass": lists["PASS_TO_PASS"],
    }


def test_command(task_info: dict) -> str:
    """
    The command SWE-bench runs an instance's tests with: the project's test
    command (``runtests.py`` for Django, ``bin/test`` for SymPy, pytest
    otherwise) over the test files the developer's test patch touches.

    :param task_info: The raw SWE-bench dataset row of the instance.
    """
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.python import get_test_directives

    spec = MAP_REPO_VERSION_TO_SPECS[task_info["repo"]][task_info["version"]]
    return " ".join([spec["test_cmd"], *get_test_directives(task_info)])


def harness_evaluate(
    task_info: dict,
    patch: str,
    *,
    run_id: str,
    model_name: str,
    log_root: Path,
    timeout: int = 1800,
) -> dict[str, Any]:
    """
    The official SWE-bench verdict on *patch*.

    Runs :func:`swebench.harness.run_evaluation.run_instance` against the
    instance's image: the patch is applied to the base commit, the developer's
    test patch is added, and ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` are run.

    :param task_info: The raw SWE-bench dataset row (unmodified test lists).
    :param patch: The model patch; ``""`` evaluates the unchanged base.
    :param run_id: Harness run identifier; part of the log path.
    :param model_name: Recorded as ``model_name_or_path``.
    :param log_root: Directory receiving the harness logs and report.
    :param timeout: Seconds the test run may take.
    :returns: ``completed``, ``resolved``, ``report`` (the harness report for
        the instance, or ``None``) and ``log_dir``.
    """
    import docker
    import swebench.harness.run_evaluation as harness

    spec = test_spec(task_info)
    instance_id = task_info["instance_id"]
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch or "",
    }
    client = docker.from_env(timeout=600)
    # run_instance writes under a module-level log root; point it at ours for
    # this call only -- agents (Agentless) run the harness with its default.
    previous_root = harness.RUN_EVALUATION_LOG_DIR
    harness.RUN_EVALUATION_LOG_DIR = Path(log_root)
    try:
        outcome = harness.run_instance(
            spec,
            prediction,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id=run_id,
            timeout=timeout,
        )
    finally:
        harness.RUN_EVALUATION_LOG_DIR = previous_root
    log_dir = Path(log_root) / run_id / model_name.replace("/", "__") / instance_id
    report_path = log_dir / "report.json"
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text()).get(instance_id)
    return {
        "completed": bool(outcome.get("completed")),
        "resolved": bool(outcome.get("resolved")),
        "report": report,
        "log_dir": str(log_dir),
    }
