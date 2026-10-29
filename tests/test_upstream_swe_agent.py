"""
SWE-agent runs as upstream ships it, with FORECAST's issue text and model.

Without Docker these tests check the parts FORECAST adds -- the agent comes
from upstream's SWE-bench config, and its model talks through FORECAST. The
full loop in an instance container is exercised by
``tests/test_upstream_docker.py``.
"""

import unittest

from agent_fakes import ScriptedChatClient, tool_call

try:
    import sweagent  # noqa: F401
    from forecast.agents.swe_agent import SWEAgentRepairAgent, upstream_config
except ImportError:  # pragma: no cover - optional extra
    SWEAgentRepairAgent = None


@unittest.skipIf(SWEAgentRepairAgent is None, "SWE-agent not installed")
class SWEAgentAdapterTests(unittest.TestCase):
    def setUp(self):
        self.client = ScriptedChatClient([tool_call("ls -la")])
        self.agent = SWEAgentRepairAgent().make_agent(self.client)

    def test_agent_is_upstreams_swe_bench_configuration(self):
        upstream = upstream_config()["agent"]
        model = self.agent.model.config
        self.assertEqual(upstream["model"]["per_instance_cost_limit"], model.per_instance_cost_limit)
        self.assertEqual(upstream["model"]["per_instance_call_limit"], model.per_instance_call_limit)
        self.assertEqual(upstream["model"]["temperature"], model.temperature)
        self.assertEqual(
            upstream["templates"]["instance_template"],
            self.agent.templates.instance_template,
        )
        self.assertEqual(
            ["bash", "str_replace_editor", "submit"],
            [t["function"]["name"] for t in self.agent.model.tools.tools],
        )

    def test_context_window_becomes_upstreams_input_limit(self):
        from forecast.agents.models.ollama import OllamaClient

        client = OllamaClient(model="qwen3.6:35b", num_ctx=131072)
        agent = SWEAgentRepairAgent().make_agent(client)
        self.assertEqual(131072 - client.max_tokens, agent.model.model_max_input_tokens)
        # Claude Haiku 4.5: 200k window minus the 64k SWE-agent reserves for output.
        self.assertEqual(200_000 - 64_000, self.agent.model.model_max_input_tokens)

    def test_queries_go_through_forecast_with_upstream_tools(self):
        history = [
            {"role": "system", "content": "sys", "message_type": "system"},
            {"role": "user", "content": "ISSUE-TEXT", "message_type": "user"},
        ]
        output = self.agent.model.query(history)
        self.assertEqual(1, len(self.client.requests))
        request = self.client.requests[0]
        self.assertEqual("ISSUE-TEXT", request["messages"][-1]["content"])
        self.assertEqual(0.0, request["temperature"])
        self.assertEqual(
            {"bash", "str_replace_editor", "submit"},
            {t["function"]["name"] for t in request["tools"]},
        )
        # Upstream's own reply shape, so its parser takes over unchanged.
        self.assertEqual("bash", output["tool_calls"][0]["function"]["name"])
        self.assertEqual(1, self.agent.model.stats.api_calls)
        self.assertEqual(15, self.agent.model.stats.tokens_sent + self.agent.model.stats.tokens_received)


if __name__ == "__main__":
    unittest.main()


class BuiltImageCleanupTests(unittest.TestCase):
    """The image SWE-ReX builds per run is removed afterwards (no Docker needed)."""

    HISTORY = (
        "sha256:built|/bin/sh -c swerex-remote --version\n"
        "sha256:link|/bin/sh -c ln -s /root/python3.11/bin/swerex-remote /usr/local/bin/swerex-remote\n"
        "sha256:pip|/bin/sh -c /root/python3.11/bin/pip3 install --no-cache-dir swe-rex\n"
        "sha256:python|/bin/sh -c #(nop) COPY dir:3295 in /root/python3.11 \n"
        "sha256:apt|/bin/sh -c apt-get update && apt-get install -y     libc6     && rm -rf /var/lib/apt/lists/*\n"
        "sha256:instance|WORKDIR /testbed/\n"
        "<missing>|RUN |1 DEBIAN_FRONTEND=noninteractive /bin/sh -c /dev/pipes/EOF # buildkit\n"
    )

    class Deployment:
        def _build_image(self):
            return "sha256:built"

    def setUp(self):
        from forecast.agents.swe_agent import remember_built_image, remove_built_image

        self.remember, self.remove = remember_built_image, remove_built_image
        self.deployment = self.Deployment()
        self.remember(self.deployment)

    def docker(self, *rmi_results, history=HISTORY):
        """``subprocess.run`` answering ``docker history`` then each ``docker rmi``."""
        from unittest import mock

        replies = [mock.Mock(returncode=0, stdout=history, stderr="")]
        replies += [mock.Mock(returncode=code, stderr=err) for code, err in rmi_results]
        return mock.patch("subprocess.run", side_effect=replies)

    def removed(self, run):
        return [c.args[0][2] for c in run.call_args_list if c.args[0][1] == "rmi"]

    def test_build_is_recorded_and_returned(self):
        self.assertEqual("sha256:built", self.deployment._build_image())
        self.assertEqual("sha256:built", self.deployment.forecast_built_image)

    def test_nothing_built_nothing_removed(self):
        with self.docker() as run:
            self.remove(self.deployment)
            self.remove(None)
        run.assert_not_called()

    def test_image_and_swe_rex_parents_are_removed(self):
        self.deployment._build_image()
        with self.docker(*[(0, "")] * 5) as run:
            self.remove(self.deployment, delay=0)
        self.assertEqual(
            ["sha256:built", "sha256:link", "sha256:pip", "sha256:python", "sha256:apt"],
            self.removed(run),
        )

    def test_removal_retries_while_the_container_is_going(self):
        self.deployment._build_image()
        with self.docker((1, "image is being used by stopped container"), *[(0, "")] * 5) as run:
            self.remove(self.deployment, delay=0)
        self.assertEqual(["sha256:built"] * 2, self.removed(run)[:2])
        self.assertNotIn("sha256:instance", self.removed(run))

    def test_a_parent_in_use_stops_the_walk(self):
        self.deployment._build_image()
        with self.docker((0, ""), (1, "image has dependent child images")) as run:
            self.remove(self.deployment, delay=0)
        self.assertEqual(["sha256:built", "sha256:link"], self.removed(run))

    def test_failure_only_leaves_the_image(self):
        self.deployment._build_image()
        with self.docker(*[(1, "conflict")] * 3) as run:
            with self.assertLogs("forecast.agents.swe_agent", "WARNING"):
                self.remove(self.deployment, attempts=3, delay=0)
        self.assertEqual(["sha256:built"] * 3, self.removed(run))
