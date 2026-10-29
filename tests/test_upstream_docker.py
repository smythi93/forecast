"""
Upstream agents in a real SWE-bench instance image, judged by the harness.

A scripted model drives each agent to apply the developer's fix the way a model
would -- through the agent's own tools, inside the instance container -- and
submit it. The submitted patch must then be *resolved* by the official
SWE-bench harness, and the unpatched base must not be. That exercises
everything between the model and the verdict without an LLM: image selection,
the agent's environment and submission protocol, patch extraction, and
evaluation.

Needs Docker and the locally built instance images, so it only runs when
``FORECAST_DOCKER_TESTS=1`` (on se01/se02).
"""

import os
import tempfile
import unittest
from pathlib import Path

from agent_fakes import ScriptedChatClient, context_for, tool_call

from forecast.agents.base import AgentMode

INSTANCE = os.getenv("FORECAST_DOCKER_TEST_INSTANCE", "psf__requests-863")
ENABLED = os.getenv("FORECAST_DOCKER_TESTS") == "1"


def _row(instance_id: str) -> dict:
    from datasets import load_dataset

    for row in load_dataset("princeton-nlp/SWE-bench_Lite", split="test"):
        if row["instance_id"] == instance_id:
            return dict(row)
    raise KeyError(instance_id)


def _apply_gold(row: dict) -> str:
    return f"cd /testbed && git apply <<'FORECAST_EOF'\n{row['patch']}\nFORECAST_EOF"


@unittest.skipUnless(ENABLED, "set FORECAST_DOCKER_TESTS=1 on a host with SWE-bench images")
class UpstreamAgentsInDockerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = _row(INSTANCE)
        cls.out = Path(tempfile.mkdtemp(prefix="forecast_docker_"))

    def _verdict(self, patch: str, run_id: str) -> dict:
        from forecast.agents.environment import harness_evaluate

        return harness_evaluate(
            self.row, patch, run_id=run_id, model_name="scripted", log_root=self.out / "harness"
        )

    def _repair(self, agent, script):
        client = ScriptedChatClient(script)
        context = context_for(Path("/nonexistent"), self.row, "1. CHAINMARK (discriminability: 0.90)")
        result = agent.repair(context, model_client=client, traj_dir=self.out / agent.name)
        return result, client

    def test_harness_resolves_gold_and_rejects_base(self):
        self.assertTrue(self._verdict(self.row["patch"], "gold")["resolved"])
        self.assertFalse(self._verdict("", "empty")["resolved"])

    def test_mini_swe_agent_submission_is_resolved(self):
        from forecast.agents.mini_swe_agent import MiniSWEAgentRepairAgent

        result, client = self._repair(
            MiniSWEAgentRepairAgent(mode=AgentMode.FORECAST),
            [
                tool_call(_apply_gold(self.row)),
                tool_call("git diff > patch.txt && cat patch.txt"),
                tool_call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
            ],
        )
        self.assertIsNone(result.error, result.error)
        self.assertEqual(1, client.requests[0]["messages"][1]["content"].count("CHAINMARK"))
        self.assertTrue(self._verdict(result.patch, "mini")["resolved"], result.patch)

    def test_swe_agent_submission_is_resolved(self):
        from forecast.agents.swe_agent import SWEAgentRepairAgent

        result, client = self._repair(
            SWEAgentRepairAgent(mode=AgentMode.FORECAST),
            [
                tool_call(_apply_gold(self.row)),
                # review_on_submit asks for confirmation; submit twice as a model would.
                tool_call(None, name="submit"),
                tool_call(None, name="submit"),
            ],
        )
        self.assertIsNone(result.error, result.error)
        self.assertEqual(1, client.sent_text().count("CHAINMARK") // len(client.requests))
        self.assertTrue(self._verdict(result.patch, "swe-agent")["resolved"], result.patch)


if __name__ == "__main__":
    unittest.main()
