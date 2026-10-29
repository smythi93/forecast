"""
mini-swe-agent, run as published, inside FORECAST.

The agent, its SWE-bench configuration (``benchmarks/swebench.yaml``: prompts,
step and cost limits, bash tool, observation templates) and its Docker
environment are upstream's, unmodified. FORECAST supplies three things:

* the task: :meth:`~forecast.agents.base.RepairAgent.issue_text`, i.e. the
  issue, plus the diagnosis in ``FORECAST`` mode;
* the model: :class:`ForecastMiniModel` sends every query through FORECAST's
  :meth:`~forecast.agents.models.utils.ModelClient.chat`;
* the image: the instance's SWE-bench image FORECAST traced it in.

Upstream: https://github.com/SWE-agent/mini-swe-agent (``mini-swe-agent==2.4.6``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from forecast.agents.base import AgentMode, RepairContext
from forecast.agents.loop import LoopRepairAgent

#: Upstream's SWE-bench configuration, used unmodified apart from the model.
CONFIG_SPEC = "swebench.yaml"


def _quiet_upstream() -> None:
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    # Cost is FORECAST's to record; a model litellm cannot price must not abort.
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")


def forecast_mini_model(model_client, **config: Any):
    """
    An upstream ``LitellmModel`` whose API call goes through FORECAST.

    Only the transport changes: message preparation, tool-call parsing,
    observation formatting and retries are upstream's. The class is built on
    demand so importing FORECAST does not require mini-swe-agent.

    :param model_client: FORECAST client to send queries through.
    :param config: Upstream model configuration (templates, ``model_kwargs``).
    """
    _quiet_upstream()
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    class ForecastMiniModel(LitellmModel):
        _last_cost = 0.0

        def _calculate_cost(self, response) -> dict[str, float]:
            # FORECAST priced the call (ModelClient.price_as); upstream's own
            # pricing would see no price for local models and count nothing,
            # which would switch its cost limit off.
            return {"cost": self._last_cost}

        def _query(self, messages, **kwargs):
            params = dict(self.config.model_kwargs) | kwargs
            temperature = params.pop("temperature", None)
            max_tokens = params.pop("max_tokens", None)
            reply = model_client.chat(
                messages,
                tools=[BASH_TOOL],
                temperature=temperature,
                max_tokens=max_tokens,
                **params,
            )[0]
            self._last_cost = reply.cost
            # Upstream parses the provider response itself.
            return reply.raw

    config = dict(config)
    config["model_name"] = model_client.litellm_model_name()
    config.setdefault("cost_tracking", "ignore_errors")
    # Upstream's get_model() turns on prompt caching for Anthropic models; the
    # model is built directly here, so the same rule is applied here.
    if any(s in config["model_name"].lower() for s in ("anthropic", "sonnet", "opus", "claude")):
        config.setdefault("set_cache_control", "default_end")
    return ForecastMiniModel(**config)


class MiniSWEAgentRepairAgent(LoopRepairAgent):
    """mini-swe-agent with FORECAST's issue text, model, and image."""

    @classmethod
    def names(cls) -> list[str]:
        return ["Mini-SWE-Agent", "MiniSWEAgent", "mini-swe-agent", "minisweagent"]

    def __init__(
        self,
        name: str = "Mini-SWE-Agent",
        mode: AgentMode = AgentMode.BASELINE,
        model: str = "claude-haiku-4-5-20251001",
        output_dir: str = ".",
    ):
        super().__init__(name=name, mode=mode, model=model, output_dir=output_dir)

    @staticmethod
    def upstream_config() -> dict:
        """Upstream's SWE-bench configuration."""
        _quiet_upstream()
        from minisweagent.config import get_config_from_spec

        return get_config_from_spec(CONFIG_SPEC)

    def make_environment(self, context: RepairContext, env_config: dict):
        """
        Upstream's Docker environment on the instance image.

        Overridable so tests can substitute a local environment.
        """
        from minisweagent.environments.docker import DockerEnvironment

        from forecast.agents.environment import instance_image

        env_config = {k: v for k, v in env_config.items() if k != "environment_class"}
        return DockerEnvironment(image=instance_image(self.task_info(context)), **env_config)

    def run_loop(
        self,
        context: RepairContext,
        model_client,
        work_dir: Path,
        log: logging.Logger,
    ) -> tuple[str, dict[str, Any]]:
        from minisweagent.agents.default import DefaultAgent

        config = self.upstream_config()
        env = self.make_environment(context, config.get("environment", {}))
        try:
            model = forecast_mini_model(model_client, **config.get("model", {}))
            agent = DefaultAgent(
                model,
                env,
                **(config.get("agent", {}) | {"output_path": work_dir / "mini.traj.json"}),
            )
            info = agent.run(self.get_prompt(context))
        finally:
            cleanup = getattr(env, "cleanup", None)
            if cleanup is not None:
                cleanup()
        log.info(
            "mini-swe-agent exit=%s  steps=%d", info.get("exit_status"), agent.n_calls
        )
        return info.get("submission") or "", {
            "exit_status": info.get("exit_status"),
            "steps": agent.n_calls,
        }


def register(context):
    """
    Plugin entry point for FORECAST's ``forecast.plugins`` group.
    """

    @context.register_agent(names=MiniSWEAgentRepairAgent.names(), replace=True)
    class _RegisteredMiniSWEAgentRepairAgent(MiniSWEAgentRepairAgent):
        pass

    return _RegisteredMiniSWEAgentRepairAgent


__all__ = ["MiniSWEAgentRepairAgent", "forecast_mini_model", "register"]
