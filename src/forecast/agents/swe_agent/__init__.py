"""
SWE-agent, run as published, inside FORECAST.

The agent is upstream's ``DefaultAgent`` with the configuration SWE-agent's
authors use for SWE-bench with Claude
(``config/benchmarks/250522_anthropic_filemap_simple_review.yaml``): their
templates, tool bundles (the agent-computer interface: file viewer/editor,
review-on-submit), history processing, cost and call limits, and decoding.
Only the model name is FORECAST's.
The environment is upstream's ``SWEEnv`` on a SWE-ReX Docker deployment of the
instance image. FORECAST supplies three things:

* the problem statement: :meth:`~forecast.agents.base.RepairAgent.issue_text`,
  i.e. the issue plus, in ``FORECAST`` mode, the diagnosis;
* the model: :func:`forecast_swe_agent_model` sends every query through
  FORECAST's :meth:`~forecast.agents.models.utils.ModelClient.chat`;
* the image: the instance's SWE-bench image FORECAST traced it in.

Upstream: https://github.com/SWE-agent/SWE-agent at the commit pinned in
``agents/swe-agent.json`` (SWE-agent 1.1.0), installed from that checkout.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from forecast.agents.base import AgentMode, RepairContext
from forecast.agents.loop import LoopRepairAgent

#: Upstream's SWE-bench configuration for Claude, relative to its
#: config directory.
CONFIG_FILE = "benchmarks/250522_anthropic_filemap_simple_review.yaml"


def upstream_config() -> dict:
    """Upstream's SWE-bench configuration, as a dict."""
    from sweagent import CONFIG_DIR

    return yaml.safe_load((Path(CONFIG_DIR) / CONFIG_FILE).read_text())


def forecast_swe_agent_model(model_client, model_config, tools):
    """
    An upstream ``LiteLLMModel`` whose API call goes through FORECAST.

    Only the transport changes. History-to-message conversion, retries,
    context-window checks, cost and call limits, and tool-call handling stay
    upstream's; the reply is returned in exactly the shape upstream's own call
    produces.

    :param model_client: FORECAST client to send queries through.
    :param model_config: Upstream ``GenericAPIModelConfig``.
    :param tools: Upstream ``ToolConfig`` of the agent.
    """
    from sweagent.agent.models import LiteLLMModel

    class ForecastSWEAgentModel(LiteLLMModel):
        def _single_query(self, messages, n=None, temperature=None):
            self._sleep()
            replies = model_client.chat(
                messages,
                tools=self.tools.tools if self.tools.use_function_calling else None,
                temperature=self.config.temperature if temperature is None else temperature,
                max_tokens=self.model_max_output_tokens,
                n=n or 1,
                top_p=self.config.top_p,
                **self.config.completion_kwargs,
            )
            outputs = []
            for reply in replies:
                output = {"message": reply.content}
                if self.tools.use_function_calling:
                    output["tool_calls"] = reply.tool_calls
                if reply.thinking_blocks:
                    output["thinking_blocks"] = reply.thinking_blocks
                outputs.append(output)
            self._update_stats(
                input_tokens=replies[0].prompt_tokens,
                output_tokens=sum(r.completion_tokens for r in replies),
                cost=sum(r.cost for r in replies),
            )
            return outputs

    return ForecastSWEAgentModel(model_config, tools)


#: The steps of SWE-ReX's Dockerfile (``DockerDeployment.glibc_dockerfile``)
#: that leave an image of their own, as ``docker history`` shows them.
SWEREX_STEPS = (
    "swerex-remote --version",
    "ln -s /root/python3.11/bin/swerex-remote ",
    "/root/python3.11/bin/pip3 install --no-cache-dir swe-rex",
    "/root/python3.11/bin/python3 --version",
    "ENV LD_LIBRARY_PATH=/root/python3.11/lib:",
    " in /root/python3.11",  # COPY of the standalone Python
    "apt-get install -y     libc6     && rm -rf /var/lib/apt/lists/*",
)


def remember_built_image(deployment) -> None:
    """
    Record on a SWE-ReX Docker deployment the image it builds for the run.

    SWE-ReX builds an image per run (a standalone Python and swe-rex on top of
    the instance image, about 400 MB of its own) and never removes it;
    :func:`remove_built_image` does.
    """
    build = deployment._build_image

    def recording_build() -> str:
        deployment.forecast_built_image = build()
        return deployment.forecast_built_image

    deployment._build_image = recording_build


def _built_chain(runtime: str, image: str) -> list[str]:
    """The image and the parents SWE-ReX's steps left, top first."""
    try:
        result = subprocess.run(
            [runtime, "history", "--no-trunc", "--format", "{{.ID}}|{{.CreatedBy}}", image],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return [image]
    chain = []
    for line in result.stdout.splitlines():
        layer, _, step = line.partition("|")
        if layer in ("", "<missing>") or not any(s in step for s in SWEREX_STEPS):
            break
        chain.append(layer)
    return chain or [image]


def _remove_image(runtime: str, image: str, attempts: int, delay: float) -> bool:
    """``docker rmi`` without force; True once the image is gone."""
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                [runtime, "rmi", image], capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0 or "No such image" in result.stderr:
            return True
        if attempt + 1 < attempts:
            time.sleep(delay)
    return False


def remove_built_image(deployment, attempts: int = 6, delay: float = 5.0) -> None:
    """
    Remove the image :func:`remember_built_image` recorded, if any, and the
    parent images SWE-ReX's build steps left under it.

    Docker refuses while a container still uses the image, and the run's
    container is removed asynchronously after it is killed, so this retries.
    A parent another build still uses stays. Failing only leaves images behind.
    """
    image = getattr(deployment, "forecast_built_image", None)
    if not image:
        return
    runtime = getattr(getattr(deployment, "_config", None), "container_runtime", "docker")
    chain = _built_chain(runtime, image)
    if not _remove_image(runtime, chain[0], attempts, delay):
        logging.getLogger(__name__).warning("Could not remove image %s", image)
        return
    for parent in chain[1:]:
        if not _remove_image(runtime, parent, 1, 0):
            break


class SWEAgentRepairAgent(LoopRepairAgent):
    """SWE-agent with FORECAST's issue text, model, and image."""

    #: Extra ``docker run`` arguments, as upstream's SWE-bench config sets them.
    DOCKER_ARGS = ["--memory=10g"]

    @classmethod
    def names(cls) -> list[str]:
        return ["SWE-Agent", "SWEAgent", "swe-agent", "sweagent"]

    def __init__(
        self,
        name: str = "SWE-Agent",
        mode: AgentMode = AgentMode.BASELINE,
        model: str = "claude-haiku-4-5-20251001",
        output_dir: str = ".",
    ):
        super().__init__(name=name, mode=mode, model=model, output_dir=output_dir)

    def make_agent(self, model_client):
        """Upstream's agent from its SWE-bench config, talking through FORECAST."""
        from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig

        agent_config = dict(upstream_config()["agent"])
        # The model is FORECAST's; upstream's key rotation variable is not ours.
        agent_config["model"] = dict(
            agent_config.get("model", {}), name=model_client.model, api_key=None
        )
        config = DefaultAgentConfig.model_validate(agent_config)
        agent = DefaultAgent.from_config(config)
        agent.model = model = forecast_swe_agent_model(model_client, config.model, config.tools)
        # Give upstream's context check the model's real window, so a long
        # trajectory ends the way upstream ends it (exit on context, submit
        # what is there) rather than being truncated by the server or failing.
        window = getattr(model_client, "context_window", None)
        if window and config.model.max_input_tokens is None:
            reserve = model.model_max_output_tokens or model_client.max_tokens
            model.model_max_input_tokens = window - reserve
        return agent

    def make_environment(self, context: RepairContext):
        """
        Upstream's environment on the instance image, set up the way upstream's
        SWE-bench runner sets up each instance. Overridable for tests.
        """
        from sweagent.environment.repo import PreExistingRepoConfig
        from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv
        from swerex.deployment.config import DockerDeploymentConfig

        from forecast.agents.environment import instance_image

        task_info = self.task_info(context)
        config = EnvironmentConfig(
            deployment=DockerDeploymentConfig(
                image=instance_image(task_info),
                docker_args=list(self.DOCKER_ARGS),
                python_standalone_dir="/root",
            ),
            repo=PreExistingRepoConfig(
                repo_name="testbed", base_commit=task_info["base_commit"]
            ),
        )
        env = SWEEnv.from_config(config)
        remember_built_image(env.deployment)
        return env

    def run_loop(
        self,
        context: RepairContext,
        model_client,
        work_dir: Path,
        log: logging.Logger,
    ) -> tuple[str, dict[str, Any]]:
        from sweagent.agent.problem_statement import TextProblemStatement

        agent = self.make_agent(model_client)
        problem = TextProblemStatement(
            text=self.get_prompt(context), id=context.instance_id
        )
        env = self.make_environment(context)
        try:
            env.start()
            try:
                result = agent.run(problem_statement=problem, env=env, output_dir=work_dir)
            finally:
                env.close()
        finally:
            remove_built_image(getattr(env, "deployment", None))
        info = dict(result.info or {})
        log.info(
            "SWE-agent exit=%s  steps=%d", info.get("exit_status"), len(result.trajectory)
        )
        return info.get("submission") or "", {
            "exit_status": info.get("exit_status"),
            "steps": len(result.trajectory),
        }


def register(context):
    """
    Plugin entry point for FORECAST's ``forecast.plugins`` group.
    """

    @context.register_agent(names=SWEAgentRepairAgent.names(), replace=True)
    class _RegisteredSWEAgentRepairAgent(SWEAgentRepairAgent):
        pass

    return _RegisteredSWEAgentRepairAgent


__all__ = [
    "SWEAgentRepairAgent",
    "forecast_swe_agent_model",
    "register",
    "remember_built_image",
    "remove_built_image",
]
