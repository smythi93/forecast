"""
mini-swe-agent runs as upstream ships it, with FORECAST's issue text and model.

The environment is upstream's ``LocalEnvironment`` on a toy repository instead
of the instance's Docker image; everything else -- agent, SWE-bench config,
tool-call parsing, submission protocol -- is upstream's.
"""

import tempfile
import unittest
from pathlib import Path

from agent_fakes import FIX_DIFF, ISSUE, TEST_PATCH_MARK, ScriptedChatClient, context_for, tool_call, toy_repo

from forecast.agents.base import AgentMode

try:
    from forecast.agents.mini_swe_agent import MiniSWEAgentRepairAgent
    import minisweagent  # noqa: F401
except ImportError:  # pragma: no cover - optional extra
    MiniSWEAgentRepairAgent = None

CHAIN = "1. CHAINMARK `a - b` in add (discriminability: 0.90)"


def _local_agent(mode, repo):
    class _Local(MiniSWEAgentRepairAgent):
        def make_environment(self, context, env_config):
            from minisweagent.environments.local import LocalEnvironment

            return LocalEnvironment(cwd=str(repo), timeout=60)

    return _Local(mode=mode)


SCRIPT = [
    tool_call("cat calc.py"),
    tool_call("sed -i.bak 's/a - b/a + b/' calc.py && rm calc.py.bak"),
    tool_call("git diff -- calc.py > patch.txt && cat patch.txt"),
    tool_call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
]


@unittest.skipIf(MiniSWEAgentRepairAgent is None, "mini-swe-agent not installed")
class MiniSWEAgentTests(unittest.TestCase):
    def _run(self, mode, diagnosis):
        repo, row = toy_repo()
        client = ScriptedChatClient(SCRIPT)
        with tempfile.TemporaryDirectory() as out:
            result = _local_agent(mode, repo).repair(
                context_for(repo, row, diagnosis), model_client=client, traj_dir=Path(out)
            )
            calls = (Path(out) / row["instance_id"] / "calls.jsonl").read_text().splitlines()
        return result, client, calls

    def test_upstream_loop_produces_the_submitted_patch(self):
        result, client, calls = self._run(AgentMode.BASELINE, None)
        self.assertEqual(4, len(client.requests))
        self.assertEqual(4, len(calls), "every call is recorded by FORECAST")
        self.assertIn("+    return a + b", result.patch)
        self.assertTrue(result.success)
        self.assertEqual(60, result.tokens_used)

    def test_issue_reaches_upstream_template_with_chain_once(self):
        _, client, _ = self._run(AgentMode.FORECAST, CHAIN)
        first = client.requests[0]["messages"]
        user = first[1]["content"]
        if isinstance(user, list):  # cache-marked content blocks
            user = "".join(block["text"] for block in user)
        self.assertIn("<pr_description>", user, "upstream's instance template wraps it")
        self.assertIn(ISSUE, user)
        self.assertEqual(1, user.count("CHAINMARK"))
        self.assertEqual(1, client.sent_text().count("CHAINMARK") // len(client.requests))

    def test_baseline_never_sees_the_chain(self):
        _, client, _ = self._run(AgentMode.BASELINE, CHAIN)
        self.assertNotIn("CHAINMARK", client.sent_text())
        self.assertNotIn("Failure Diagnosis", client.sent_text())

    def test_empty_chain_is_reported_as_a_finding(self):
        _, client, _ = self._run(AgentMode.FORECAST, "")
        self.assertIn("no execution feature that separates them", client.sent_text())

    def test_hidden_tests_stay_hidden(self):
        _, client, _ = self._run(AgentMode.FORECAST, CHAIN)
        sent = client.sent_text()
        self.assertNotIn(TEST_PATCH_MARK, sent)
        self.assertNotIn("test_add", sent)

    def test_forecasts_cost_drives_upstreams_cost_limit(self):
        # Each call costs $1.25 at FORECAST's prices; upstream's $3 limit must
        # stop the agent after the third call, as it would with real pricing.
        repo, row = toy_repo()
        client = ScriptedChatClient([tool_call("ls")], cost=1.25)
        with tempfile.TemporaryDirectory() as out:
            result = _local_agent(AgentMode.BASELINE, repo).repair(
                context_for(repo, row, None), model_client=client, traj_dir=Path(out)
            )
        self.assertEqual(3, len(client.requests))
        self.assertFalse(result.patch)

    def test_upstream_tool_definitions_are_sent(self):
        _, client, _ = self._run(AgentMode.BASELINE, None)
        self.assertEqual("bash", client.requests[0]["tools"][0]["function"]["name"])

    def test_claude_conversations_are_cached_as_upstream_does(self):
        import json

        _, client, _ = self._run(AgentMode.BASELINE, None)
        self.assertIn("cache_control", json.dumps(client.requests[-1]["messages"]))


if __name__ == "__main__":
    unittest.main()
