"""
Abstract base class for repair agents.

The repair pipeline is split into five clearly-separated steps that can each
be executed and inspected independently:

1. :meth:`RepairAgent.get_prompt`     — build the raw agent-specific prompt
2. :meth:`RepairAgent.send_prompt`    — call the LLM via our model API, save traj
3. :meth:`RepairAgent.get_patch`      — post-process the response into a unified diff
4. :meth:`RepairAgent.validate_patch` — verify the patch passes tests
5. :meth:`RepairAgent.repair`         — convenience wrapper for steps 1–4

Models are **not** owned by individual agents.  Each agent receives a
:class:`~forecast.agents.models.utils.ModelClient` obtained from
:func:`~forecast.agents.models.utils.get_model_client`.  This makes it
trivial to swap backends without touching agent code.
"""

import abc
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class AgentMode(Enum):
    """
    Operating mode for a repair agent.

    :cvar BASELINE: Standard agent with no additional diagnostic information.
    :cvar FORECAST: Agent augmented with a FORECAST diagnosis.
    """

    BASELINE = "baseline"
    FORECAST = "forecast"


@dataclass
class RepairContext:
    """
    All information required to attempt a repair for one issue.

    :param instance_id: Unique identifier for this repair instance
        (e.g. ``"astropy__astropy-12907"``).
    :param repo: Absolute path to the checked-out repository root (local
        project) or the ``owner/name`` slug (SWE-bench context).
    :param version: Version string of the subject project.
    :param problem_statement: Free-text description of the bug to fix.
    :param failing_tests: Pytest node IDs of tests that currently fail.
    :param passing_tests: Pytest node IDs of currently-passing tests (used
        for regression detection).
    :param diagnosis: FORECAST cause-chain text to inject in FORECAST mode.
    :param diagnosis_discriminability: Discriminability score of the diagnosis
        (0.0–1.0).
    :param base_commit: Git commit SHA the repository is pinned to.
    :param relevant_files: Source files relevant to the issue.
    :param swe_task_info: Full raw SWE-bench instance dict; ``None`` for local
        projects.
    :param swe_setup_info: ACR ``setup_info`` dict; ``None`` for local projects.
    :param fault_localization: Method-level spectrum-based fault localization
        over the executions FORECAST traced (``file``, ``class``, ``method``,
        ``line``, ``suspiciousness``; most suspicious first). Available in both
        modes: it is a test-based baseline signal, not part of the diagnosis.
        Only agents that use the tests (AutoCodeRover) consume it.
    """

    instance_id: str
    repo: str
    version: str
    problem_statement: str
    failing_tests: List[str]
    passing_tests: Optional[List[str]] = None
    diagnosis: Optional[str] = None
    diagnosis_discriminability: Optional[float] = None
    base_commit: Optional[str] = None
    relevant_files: Optional[List[Path]] = None
    swe_task_info: Optional[dict] = None
    swe_setup_info: Optional[dict] = None
    fault_localization: Optional[List[dict]] = None


@dataclass
class RepairResult:
    """
    The outcome of one repair attempt.

    :param instance_id: Identifier of the repaired instance.
    :param agent_name: Name of the agent that produced this result.
    :param mode: Operating mode (``BASELINE`` or ``FORECAST``).
    :param success: Whether the patch resolves the failing tests.
    :param patch: Unified diff string; ``None`` if no patch was produced.
    :param lines_changed: Lines added and removed by the patch.
    :param files_modified: Number of files touched by the patch.
    :param test_pass_rate: Fraction of failing tests now passing (0.0–1.0).
    :param has_regression: ``True`` if the patch breaks previously-passing tests.
    :param attempts: Number of repair attempts made.
    :param time_seconds: Wall-clock seconds spent on the repair.
    :param tokens_used: LLM tokens consumed.
    :param diagnosis_used: Diagnosis text used (FORECAST mode only).
    :param diagnosis_discriminability: Discriminability of the diagnosis used.
    :param output_dir: Directory where trajectories and artifacts are stored.
    :param error: Error message if the repair attempt failed with an exception.
    """

    instance_id: str
    agent_name: str
    mode: AgentMode
    success: bool
    patch: Optional[str] = None
    lines_changed: int = 0
    files_modified: int = 0
    test_pass_rate: float = 0.0
    has_regression: bool = False
    attempts: int = 1
    time_seconds: float = 0.0
    tokens_used: int = 0
    diagnosis_used: Optional[str] = None
    diagnosis_discriminability: Optional[float] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None


def normalise_patch(patch: str, repo_path: str) -> str:
    """
    Normalize a unified diff patch.

    1. Strip trailing whitespace from every line.
    2. Validate with ``git apply --check`` (raises :exc:`ValueError` if the
       patch cannot be applied cleanly).

    :param patch: Unified diff string to normalize.
    :param repo_path: Absolute path of the repository root.
    :returns: The normalized patch string.
    :raises ValueError: If the patch does not apply cleanly to the repository.
    """
    normalised = "\n".join(line.rstrip() for line in patch.splitlines())
    if not normalised.strip():
        return normalised
    if not normalised.endswith("\n"):
        normalised += "\n"

    with tempfile.NamedTemporaryFile(suffix=".diff", mode="w", delete=False) as f:
        f.write(normalised)
        tmp = f.name

    result = subprocess.run(
        ["git", "apply", "--check", tmp],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )
    Path(tmp).unlink(missing_ok=True)
    if result.returncode != 0:
        raise ValueError(f"Patch does not apply cleanly: {result.stderr.strip()}")
    return normalised


def _count_lines_changed(patch: str) -> int:
    """
    Count lines added and removed in a unified diff.

    :param patch: Unified diff string.
    :returns: Total number of added and removed lines (``+`` and ``-`` lines,
        excluding the ``+++``/``---`` header lines).
    """
    return sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    )


def _count_files_modified(patch: str) -> int:
    """
    Count the number of distinct files touched by a unified diff.

    :param patch: Unified diff string.
    :returns: Number of files with at least one change, counted by
        ``--- a/`` header lines.
    """
    return sum(1 for line in patch.splitlines() if line.startswith("--- a/"))


class RepairAgent(abc.ABC):
    """
    Abstract base class for all repair agents.

    Subclasses must implement:

    * :meth:`get_prompt` — format the problem for their agent.
    * :meth:`get_patch`  — extract a diff from the LLM response.

    The LLM call (:meth:`send_prompt`) is implemented once here using the
    shared :class:`~forecast.agents.models.utils.ModelClient`, so swapping
    models never requires touching agent subclasses.

    :param name: Human-readable name of the agent.
    :param mode: Operating mode (``BASELINE`` or ``FORECAST``).
    """

    #: Whether the agent works on a host checkout of the repository
    #: (``RepairContext.repo``). Agents that run in the instance's image do not,
    #: and the pipeline then skips checking the repository out.
    needs_checkout: bool = True

    def __init__(self, name: str, mode: AgentMode = AgentMode.BASELINE):
        """
        Initialize the agent.

        :param name: Human-readable agent name.
        :param mode: Operating mode.  Defaults to
            :attr:`AgentMode.BASELINE`.
        """
        self.name = name
        self.mode = mode

    @classmethod
    @abc.abstractmethod
    def names(cls) -> List[str]:
        """
        Return all agent names registered by this class.

        Used for agent lookup when running experiments.

        :returns: List of agent name strings.
        """

    # ── Step 1 ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_prompt(self, context: RepairContext) -> str:
        """
        Build the raw user-turn prompt to send to the LLM.

        Build it from :meth:`issue_text`, never from
        ``context.problem_statement`` directly: that is the single place the
        FORECAST diagnosis enters, so baseline and FORECAST prompts differ in
        exactly the diagnosis and in nothing else.

        :param context: All information about the repair task.
        :returns: The agent-formatted user prompt.
        """

    def get_system_prompt(self, context: RepairContext) -> Optional[str]:
        """
        Return the agent-specific system prompt, or ``None``.

        The default implementation returns ``None``, which tells
        :meth:`send_prompt` to use the model client's built-in default.

        Override in subclasses to supply an agent-specific system prompt.
        When FORECAST mode is active the diagnosis can optionally be
        injected here instead of (or in addition to) the user prompt.

        :param context: All information about the repair task.
        :returns: Agent-specific system prompt string, or ``None`` to use
            the model client's default.
        """
        return None

    # ── Step 2 ────────────────────────────────────────────────────────

    def send_prompt(
        self,
        prompt: str,
        context: RepairContext,
        model_client,
        traj_path: Optional[Path] = None,
        log: Optional[logging.Logger] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Send *prompt* to the LLM via *model_client* and optionally save the
        trajectory.

        :param prompt: The fully-assembled user-turn prompt (with diagnosis
            injected when in FORECAST mode).
        :param context: Repair context — used for trajectory metadata.
        :param model_client: Ready-to-use client from
            :func:`~forecast.agents.models.utils.get_model_client`.
        :param traj_path: Where to write ``traj.json``.  ``None`` means no
            trajectory file is written.
        :param log: Logger for progress messages.  ``None`` is silent.
        :param system_prompt: Agent-specific system prompt to forward to the
            model client, overriding its built-in default.  ``None`` leaves
            the client default in place.
        :returns: The raw LLM ``ModelResponse``.
        :raises RuntimeError: If the underlying API call fails.
        """
        from forecast.agents.models.utils import save_trajectory

        if log:
            log.info(
                "Sending prompt to %s  (tokens≈%d)",
                model_client.model,
                len(prompt) // 4,
            )

        response = model_client.complete(prompt, system_prompt=system_prompt)

        if log:
            log.info(
                "Response received  finish_reason=%s  tokens=%d  elapsed=%.1fs",
                response.finish_reason,
                response.total_tokens,
                response.elapsed_seconds,
            )

        if traj_path is not None:
            save_trajectory(
                prompt=prompt,
                response=response,
                traj_path=traj_path,
                system_prompt=system_prompt,
                extra={
                    "instance_id": context.instance_id,
                    "agent": self.name,
                    "mode": self.mode.value,
                },
            )
            if log:
                log.debug("Trajectory saved to %s", traj_path)

        return response

    # ── Step 3 ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_patch(
        self,
        response,
        context: RepairContext,
        log: Optional[logging.Logger] = None,
    ) -> Optional[str]:
        """
        Extract a unified-diff patch from the LLM *response*.

        :param response: The raw ``ModelResponse`` from :meth:`send_prompt`.
        :param context: Repair context — may be needed for repo-relative path
            normalization.
        :param log: Logger for warnings.  ``None`` is silent.
        :returns: A unified diff string ready for ``git apply``, or ``None``
            if no valid patch could be extracted.
        """

    # ── Step 4 ────────────────────────────────────────────────────────

    def validate_patch(self, context: RepairContext, patch: str) -> bool:
        """
        Validate a patch by applying it and running the failing tests.

        Default implementation: ``git apply`` → ``pytest`` on
        :attr:`~RepairContext.failing_tests` → ``git apply -R``.
        Override in subclasses for SWE-bench Docker-based validation.

        :param context: Repair context supplying the repo path and test IDs.
        :param patch: Unified diff string to validate.
        :returns: ``True`` if the patch makes all failing tests pass.
        """
        tmp_patch: Optional[str] = None
        # noinspection PyBroadException
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".diff", mode="w", delete=False
            ) as f:
                f.write(patch)
                tmp_patch = f.name

            if (
                subprocess.run(
                    ["git", "apply", tmp_patch],
                    capture_output=True,
                    text=True,
                    cwd=context.repo,
                ).returncode
                != 0
            ):
                return False

            passed = (
                subprocess.run(
                    [sys.executable, "-m", "pytest", "--tb=no", "-q"]
                    + context.failing_tests,
                    capture_output=True,
                    text=True,
                    cwd=context.repo,
                ).returncode
                == 0
            )

            subprocess.run(
                ["git", "apply", "-R", tmp_patch],
                capture_output=True,
                text=True,
                cwd=context.repo,
            )
            return passed
        except Exception:
            return False
        finally:
            if tmp_patch:
                Path(tmp_patch).unlink(missing_ok=True)

    # ── Step 5 ────────────────────────────────────────────────────────

    def repair(
        self,
        context: RepairContext,
        model_client=None,
        traj_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> RepairResult:
        """
        End-to-end convenience wrapper that runs steps 1–4 in sequence.

        :param context: All information about the repair task.
        :param model_client: Model client to use.  If ``None``, one is created
            via :func:`~forecast.agents.models.utils.get_model_client` using
            the agent's ``model`` attribute, falling back to
            ``"claude-haiku-4-5-20251001"``.
        :param traj_dir: Directory where ``traj.json`` is written.  ``None``
            means no trajectory file is saved.
        :param log_dir: Directory for the ``run`` step log file.  ``None``
            means console-only logging.
        :returns: :class:`RepairResult` describing the outcome.
        """
        import time
        from forecast.agents.models.utils import get_model_client
        from forecast.logger import get_logger

        log = get_logger("run", log_dir=log_dir, instance_id=context.instance_id)
        log.info("Starting repair  agent=%s  mode=%s", self.name, self.mode.value)

        if model_client is None:
            model_name = (
                getattr(self, "model", "claude-haiku-4-5-20251001")
                or "claude-haiku-4-5-20251001"
            )
            model_client = get_model_client(model_name)

        t0 = time.time()

        # Step 1: build user prompt (carrying the diagnosis via issue_text) and
        # system prompt
        prompt = self.get_prompt(context)
        system_prompt = self.get_system_prompt(context)
        log.debug("Prompt length: %d chars", len(prompt))
        if system_prompt:
            log.debug("System prompt length: %d chars", len(system_prompt))

        # Step 2: LLM call
        traj_path: Optional[Path] = None
        if traj_dir is not None:
            traj_path = Path(traj_dir) / context.instance_id / "traj.json"
        try:
            response = self.send_prompt(
                prompt,
                context,
                model_client,
                traj_path,
                log,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            return RepairResult(
                instance_id=context.instance_id,
                agent_name=self.name,
                mode=self.mode,
                success=False,
                error=str(exc),
                time_seconds=time.time() - t0,
            )

        # Step 3: extract patch
        patch = self.get_patch(response, context, log)
        if patch:
            try:
                patch = normalise_patch(patch, context.repo)
            except ValueError as exc:
                log.warning("Patch normalisation failed: %s", exc)

        # Step 4: validate
        success = False
        if patch:
            try:
                success = self.validate_patch(context, patch)
            except Exception as exc:
                log.warning("Patch validation failed: %s", exc)

        log.info("Repair done  success=%s  elapsed=%.1fs", success, time.time() - t0)

        return RepairResult(
            instance_id=context.instance_id,
            agent_name=self.name,
            mode=self.mode,
            success=success,
            patch=patch,
            lines_changed=_count_lines_changed(patch) if patch else 0,
            files_modified=_count_files_modified(patch) if patch else 0,
            test_pass_rate=1.0 if success else 0.0,
            time_seconds=time.time() - t0,
            tokens_used=response.total_tokens,
            diagnosis_used=(
                context.diagnosis if self.mode == AgentMode.FORECAST else None
            ),
            diagnosis_discriminability=(
                context.diagnosis_discriminability
                if self.mode == AgentMode.FORECAST
                else None
            ),
        )

    #: A chain whose causes all score at or below this separate the runs so
    #: weakly that presenting them as "the cause" would overstate them.
    WEAK_DISCRIMINABILITY = 0.1

    @staticmethod
    def _discriminabilities(diagnosis: str) -> list[float]:
        """
        Every ``(discriminability: X)`` score in a rendered chain.

        :param diagnosis: Rendered cause chain.
        :returns: The scores, in the order they appear.
        """
        return [float(m) for m in re.findall(r"discriminability:\s*([0-9.]+)", diagnosis)]

    def issue_text(self, context: RepairContext) -> str:
        """
        The issue as this agent should see it: the problem statement, followed
        in ``FORECAST`` mode by the diagnosis.

        Every agent hands exactly this text to its repair tool wherever the
        tool expects the issue, so the diagnosis reaches every prompt that
        carries the issue -- once -- and baseline and FORECAST runs differ in
        nothing else.

        :param context: All information about the repair task.
        :returns: Problem statement plus :meth:`diagnosis_block`.
        """
        return context.problem_statement.rstrip() + self.diagnosis_block(context)

    def diagnosis_block(self, context: RepairContext) -> str:
        """
        Format the FORECAST diagnosis of *context* for injection into a prompt.

        An *empty* chain is not a failure to diagnose, and it is not nothing:
        it is the finding that no traced execution feature separates the
        failing run from the passing ones. That constrains the fault -- the
        failing run does not take a different path -- so it is reported rather
        than dropped. Saying nothing would leave the agent unable to tell
        "we looked and the runs agree" from "we never looked".

        A chain whose scores are all near zero is reported with that stated,
        for the same reason in the opposite direction: the causes are real
        observations but separate the runs barely at all, and presenting them
        as the root cause would invite the agent to chase noise.

        :param context: Repair context; its ``diagnosis`` is ``None`` when no
            diagnosis was produced and ``""`` when it ran and found nothing.
        :returns: A formatted ``## Failure Diagnosis`` Markdown block, or an
            empty string in ``BASELINE`` mode or without a diagnosis.
        """
        diagnosis = context.diagnosis
        if self.mode != AgentMode.FORECAST:
            return ""
        if diagnosis is None:
            # No diagnosis was produced at all -- collection failed, so we know
            # nothing about how the runs compare. Claiming they are
            # indistinguishable would assert something we did not observe.
            return ""
        if not diagnosis.strip():
            return (
                "\n\n## Failure Diagnosis\n\n"
                "FORECAST traced the failing run alongside the passing runs and "
                "found **no execution feature that separates them**: over the "
                "executed lines, branches, variable values, and the relations "
                "between them, the runs are indistinguishable.\n\n"
                "Treat that as evidence, not as a missing diagnosis. The failing "
                "run does not reach different code or take a different branch, "
                "so the defect is unlikely to be a wrong condition or a "
                "misordered statement on the path that ran. Look instead for:\n"
                "- values entering from outside the traced code -- arguments, "
                "configuration, environment, file or network content;\n"
                "- behaviour that is *absent* rather than wrong: a case never "
                "handled, a branch never taken, a call never made;\n"
                "- a wrong result computed by code that ran identically in both, "
                "where only the output value differs.\n"
            )
        scores = self._discriminabilities(diagnosis)
        weak = scores and max(scores) <= self.WEAK_DISCRIMINABILITY
        caveat = (
            "\n\n**These causes separate the failing and passing runs only "
            "weakly** (highest discriminability "
            f"{max(scores):.2f}). They are real observations, but they are "
            "close to what any run of this code would show, so treat them as "
            "a starting point rather than as the identified root cause."
            if weak
            else ""
        )
        return (
            "\n\n## Failure Diagnosis\n\n"
            "The following diagnosis shows the cause-effect chain leading to "
            f"the test failure:\n\n{diagnosis}{caveat}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, mode={self.mode.value!r})"
        )
