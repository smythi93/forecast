"""
Base for repair agents that run their own multi-turn loop.

Some agents FORECAST integrates are published tools run as they are --
SWE-agent, mini-swe-agent, Agentless; others are FORECAST re-implementations
of published approaches -- AutoCodeRover, RepairAgent. Either way, FORECAST
owns everything around the loop:

* the **issue text** the loop receives (:meth:`RepairAgent.issue_text`), which in
  ``FORECAST`` mode carries the diagnosis -- the only difference between modes;
* the **model**: every LLM call is routed through FORECAST's
  :meth:`~forecast.agents.models.utils.ModelClient.chat`, and recorded there;
* the **result**: the loop's patch comes back as a :class:`RepairResult`, and
  is judged later by the SWE-bench harness, not by the agent itself.

A subclass implements :meth:`run_loop`; this class does the rest.
"""

from __future__ import annotations

import abc
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from forecast.agents.base import (
    AgentMode,
    RepairAgent,
    RepairContext,
    RepairResult,
    _count_files_modified,
    _count_lines_changed,
)


class _UsageTally:
    """Wraps a recorder and sums the tokens and cost of every recorded call."""

    def __init__(self, inner=None):
        self.inner = inner
        self.calls = 0
        self.tokens = 0
        self.cost = 0.0

    def __call__(self, record: dict) -> None:
        self.calls += 1
        self.tokens += int(record.get("prompt_tokens", 0)) + int(
            record.get("completion_tokens", 0)
        )
        self.cost += float(record.get("cost", 0.0) or 0.0)
        if self.inner is not None:
            self.inner(record)


class LoopRepairAgent(RepairAgent):
    """
    A repair agent that runs its own multi-turn repair loop.

    :param name: Agent name.
    :param mode: ``BASELINE`` or ``FORECAST``.
    :param model: Model name; only used when :meth:`repair` gets no client.
    :param output_dir: Where the agent may keep its own artifacts.
    """

    #: Loop agents work in the instance's image, not on a host checkout.
    needs_checkout = False

    def __init__(
        self,
        name: str,
        mode: AgentMode = AgentMode.BASELINE,
        model: str = "claude-haiku-4-5-20251001",
        output_dir: str = ".",
    ):
        super().__init__(name=name, mode=mode)
        self.model = model
        self.output_dir = Path(output_dir)

    def get_prompt(self, context: RepairContext) -> str:
        """
        The issue text handed to the loop -- the part of its prompts FORECAST
        controls. The loop wraps it in its own templates.
        """
        return self.issue_text(context)

    def get_patch(self, response, context, log=None) -> Optional[str]:
        """The loop produces its patch itself; see :meth:`run_loop`."""
        return None

    @abc.abstractmethod
    def run_loop(
        self,
        context: RepairContext,
        model_client,
        work_dir: Path,
        log: logging.Logger,
    ) -> tuple[str, dict[str, Any]]:
        """
        Run the agent's loop on one instance.

        :param context: The repair task; the loop receives
            :meth:`issue_text` as its issue.
        :param model_client: FORECAST client every LLM call must go through.
        :param work_dir: Directory for the agent's own trajectory and files.
        :param log: Logger for progress.
        :returns: The final patch (``""`` if none) and a dict of
            agent-specific outcome information (exit status, steps, ...).
        """

    def repair(
        self,
        context: RepairContext,
        model_client=None,
        traj_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> RepairResult:
        from forecast.agents.models.utils import JsonlRecorder, get_model_client
        from forecast.logger import get_logger

        log = get_logger("run", log_dir=log_dir, instance_id=context.instance_id)
        log.info("Starting %s  mode=%s", self.name, self.mode.value)
        if model_client is None:
            model_client = get_model_client(self.model)
        work_dir = Path(traj_dir or self.output_dir) / context.instance_id
        work_dir.mkdir(parents=True, exist_ok=True)

        self._write_run_info(context, model_client, work_dir)
        previous = getattr(model_client, "recorder", None)
        tally = _UsageTally(JsonlRecorder(work_dir / "calls.jsonl"))
        model_client.recorder = tally
        t0 = time.time()
        patch, info, error = "", {}, None
        try:
            patch, info = self.run_loop(context, model_client, work_dir, log)
        except Exception as exc:  # the agent crashing is a result, not a pipeline failure
            log.exception("%s failed on %s", self.name, context.instance_id)
            error = f"{type(exc).__name__}: {exc}"
        finally:
            model_client.recorder = previous
        (work_dir / "outcome.json").write_text(
            json.dumps(
                {"patch": patch, "info": info, "error": error, "calls": tally.calls,
                 "tokens": tally.tokens, "cost": tally.cost, "seconds": time.time() - t0},
                indent=2, default=str,
            )
        )
        patch = patch or ""
        if patch and not patch.endswith("\n"):
            patch += "\n"
        log.info(
            "%s done  patch=%s  calls=%d  tokens=%d  cost=$%.3f  elapsed=%.0fs",
            self.name,
            bool(patch.strip()),
            tally.calls,
            tally.tokens,
            tally.cost,
            time.time() - t0,
        )
        return RepairResult(
            instance_id=context.instance_id,
            agent_name=self.name,
            mode=self.mode,
            # Whether the agent produced a patch; whether it *resolves* the
            # issue is the SWE-bench harness's verdict in the evaluate stage.
            success=bool(patch.strip()),
            patch=patch or None,
            lines_changed=_count_lines_changed(patch) if patch else 0,
            files_modified=_count_files_modified(patch) if patch else 0,
            time_seconds=time.time() - t0,
            tokens_used=tally.tokens,
            diagnosis_used=(
                context.diagnosis if self.mode == AgentMode.FORECAST else None
            ),
            diagnosis_discriminability=(
                context.diagnosis_discriminability
                if self.mode == AgentMode.FORECAST
                else None
            ),
            output_dir=str(work_dir),
            error=error or info.get("error"),
        )

    def _write_run_info(self, context: RepairContext, model_client, work_dir: Path) -> None:
        """
        Record what this run was: agent, mode, model, pricing, image, and the
        versions of FORECAST and the upstream tools -- enough to rerun it.
        """
        import platform
        import subprocess

        from forecast.agents.environment import image_digest, instance_image

        def commit(path: Path) -> Optional[str]:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
            return result.stdout.strip() or None

        repo_root = Path(__file__).resolve().parents[3]
        versions = {}
        for module in ("sweagent", "minisweagent", "swebench", "litellm", "sflkit"):
            try:
                versions[module] = getattr(__import__(module), "__version__", "?")
            except Exception:
                pass
        info = {
            "instance_id": context.instance_id,
            "agent": self.name,
            "mode": self.mode.value,
            "model": getattr(model_client, "model", None),
            "litellm_model": getattr(model_client, "litellm_model_name", lambda: None)(),
            "price_as": getattr(model_client, "price_as", None),
            "context_window": getattr(model_client, "context_window", None),
            "diagnosis": context.diagnosis if self.mode == AgentMode.FORECAST else None,
            "fault_localization": context.fault_localization,
            "forecast_commit": commit(repo_root),
            "upstream_commits": {
                name: commit(repo_root / "agents" / "repos" / name)
                for name in ("swe-agent", "agentless")
                if (repo_root / "agents" / "repos" / name / ".git").exists()
            },
            "versions": versions,
            "python": platform.python_version(),
            "host": platform.node(),
        }
        if isinstance(context.swe_task_info, dict):
            try:
                image = instance_image(context.swe_task_info)
            except Exception:  # not a SWE-bench instance (e.g. a test fixture)
                image = None
            info["image"] = image
            info["image_digest"] = image_digest(image) if image else None
        (work_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))

    @staticmethod
    def task_info(context: RepairContext) -> dict:
        """The raw SWE-bench row of *context*; the agents need it."""
        if not isinstance(context.swe_task_info, dict):
            raise ValueError(
                f"{context.instance_id}: loop agents need the SWE-bench row "
                "(RepairContext.swe_task_info)."
            )
        return context.swe_task_info
