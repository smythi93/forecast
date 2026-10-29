"""
AutoCodeRover re-implementation: the paper's loop, driven by a scripted model.

The container is a local stand-in (the toy repository; tests "pass" when the
patch applies), so the test exercises retrieval, patch building, the retry
loop, and what reaches the model -- not the instance image.
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_fakes import ISSUE, ScriptedChatClient, context_for, toy_repo

from forecast.agents.acr import ACRRepairAgent, _last_json
from forecast.agents.acr.api import ContextRetrievalAPI
from forecast.agents.base import AgentMode

CHAIN = "1. CHAINMARK `a - b` in add (discriminability: 0.90)"


class _LocalContainer:
    def __init__(self, repo: Path):
        self.repo = repo
        self.validated: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def export(self, destination: Path) -> Path:
        target = Path(destination) / "testbed"
        shutil.copytree(self.repo, target)
        return target


def _plan(**plan) -> str:
    return "Analysis.\n```json\n" + json.dumps(plan) + "\n```"


PATCH_REPLY = (
    "# modification 1\n<file>calc.py</file>\n<original>\n    return a - b\n</original>\n"
    "<patched>\n    return a + b\n</patched>\n"
)


class ACRLoopTests(unittest.TestCase):
    def _agent(self, mode, repo, passes=(True,)):
        verdicts = list(passes)

        class _Agent(ACRRepairAgent):
            def make_container(self, task_info):
                self.container = _LocalContainer(repo)
                return self.container

            def _validate(self, container, task_info, diff):
                container.validated.append(diff)
                check = subprocess.run(
                    ["git", "apply", "--check", "-"], input=diff, text=True,
                    cwd=repo, capture_output=True,
                )
                ok = check.returncode == 0 and verdicts.pop(0)
                return ok, "" if ok else "The patch fails these tests:\n- tests/test_calc.py::test_add"

        return _Agent(mode=mode)

    def _run(self, script, mode=AgentMode.FORECAST, diagnosis=CHAIN, sbfl=None, passes=(True,)):
        repo, row = toy_repo()
        context = context_for(repo, row, diagnosis)
        context.fault_localization = sbfl
        client = ScriptedChatClient(script)
        agent = self._agent(mode, repo, passes)
        with tempfile.TemporaryDirectory() as out:
            result = agent.repair(context, model_client=client, traj_dir=Path(out))
        return result, client, agent

    def test_stratified_search_then_patch(self):
        result, client, agent = self._run([
            _plan(api_calls=[{"name": "search_method", "args": {"method_name": "add"}}],
                  sufficient=False, bug_locations=[]),
            _plan(api_calls=[], sufficient=True,
                  bug_locations=[{"file": "calc.py", "class": "", "method": "add"}]),
            PATCH_REPLY,
        ])
        self.assertIsNone(result.error, result.error)
        self.assertIn("+    return a + b", result.patch)
        self.assertEqual(3, len(client.requests))
        # The search result reached the model in the second stratum.
        second = client.requests[1]["messages"][-1]["content"]
        self.assertIn("search_method('add')", second)
        self.assertIn("return a - b", second)
        # Patch generation sees the location's exact code and the history.
        patch_prompt = client.requests[2]["messages"][1]["content"]
        self.assertIn("Location: calc.py, method add", patch_prompt)
        self.assertIn("Investigation so far", patch_prompt)
        self.assertEqual(0.2, client.requests[0]["temperature"])

    def test_issue_and_chain_reach_both_agents_once(self):
        _, client, _ = self._run([
            _plan(api_calls=[], sufficient=True, bug_locations=[{"file": "calc.py", "method": "add"}]),
            PATCH_REPLY,
        ])
        for request in client.requests:
            first_user = request["messages"][1]["content"]
            self.assertIn(ISSUE, first_user)
            self.assertEqual(1, first_user.count("CHAINMARK"))

    def test_baseline_has_no_chain(self):
        _, client, _ = self._run([
            _plan(api_calls=[], sufficient=True, bug_locations=[{"file": "calc.py", "method": "add"}]),
            PATCH_REPLY,
        ], mode=AgentMode.BASELINE)
        self.assertNotIn("CHAINMARK", client.sent_text())

    def test_sbfl_methods_are_shown_before_retrieval_in_both_modes(self):
        sbfl = [{"file": "calc.py", "class": "", "method": "add", "suspiciousness": 1.0}]
        for mode in (AgentMode.BASELINE, AgentMode.FORECAST):
            _, client, _ = self._run([
                _plan(api_calls=[], sufficient=True, bug_locations=[{"file": "calc.py", "method": "add"}]),
                PATCH_REPLY,
            ], mode=mode, sbfl=sbfl)
            first = client.requests[0]["messages"][1]["content"]
            self.assertIn("external analysis tool", first)
            self.assertIn("1. calc.py, method add", first)

    def test_unmatched_original_is_fed_back_and_retried(self):
        bad = PATCH_REPLY.replace("return a - b", "return a * b", 1)
        result, client, _ = self._run([
            _plan(api_calls=[], sufficient=True, bug_locations=[{"file": "calc.py", "method": "add"}]),
            bad,
            PATCH_REPLY,
        ])
        self.assertIn("does not match the file", client.requests[2]["messages"][-1]["content"])
        self.assertIn("+    return a + b", result.patch)

    def test_failing_tests_are_fed_back_and_retried(self):
        result, client, agent = self._run([
            _plan(api_calls=[], sufficient=True, bug_locations=[{"file": "calc.py", "method": "add"}]),
            PATCH_REPLY,
            PATCH_REPLY,
        ], passes=(False, True))
        self.assertIn("fails these tests", client.requests[2]["messages"][-1]["content"])
        self.assertEqual(2, len(agent.container.validated))
        self.assertTrue(result.patch)

    def test_syntax_errors_are_rejected(self):
        broken = PATCH_REPLY.replace("return a + b", "return a +")
        _, client, _ = self._run([
            _plan(api_calls=[], sufficient=True, bug_locations=[{"file": "calc.py", "method": "add"}]),
            broken,
            PATCH_REPLY,
        ])
        self.assertIn("does not parse", client.requests[2]["messages"][-1]["content"])

    def test_plan_parsing_takes_the_last_json_block(self):
        text = "x ```json\n{\"a\": 1}\n``` y ```json\n{\"b\": 2}\n```"
        self.assertEqual({"b": 2}, _last_json(text))


class SearchAPITests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "m.py").write_text(
            "@dec\nclass A(B):\n    x = 1\n    def f(self):\n        return 1\n"
            "    async def h(self): pass\n\n\ndef g():\n    return A().f()\n"
        )
        self.api = ContextRetrievalAPI(self.root)

    def test_class_signature_includes_header_attributes_and_methods(self):
        sig = self.api.search_class("A")[0].signatures
        for expected in ("@dec", "class A(B):", "x = 1", "def f(self):", "async def h(self)"):
            self.assertIn(expected, sig)
        self.assertNotIn("return 1", sig)

    def test_code_search_is_literal(self):
        self.assertEqual(1, len(self.api.search_code("A().f()")))

    def test_method_search_finds_async_and_nested(self):
        self.assertEqual(["h"], [r.function_name for r in self.api.search_method_in_class("h", "A")])
        self.assertIn("return 1", self.api.search_method("f")[0].code)

    def test_a_file_in_another_encoding_does_not_break_codebase_searches(self):
        (self.root / "latin1.py").write_bytes("# caf\xe9\ndef g2(): pass\n".encode("latin-1"))
        self.assertIn("return A().f()", self.api.search_method("g")[0].code)
        self.assertEqual(1, len(self.api.search_code("def g2")))


if __name__ == "__main__":
    unittest.main()
