"""
RepairAgent port: the state machine, middleware and tools, driven by a
scripted model on a toy repository.

The container is a local stand-in that applies patches with git and runs the
toy project's pytest, so ``write_fix`` really applies, tests, and reverts.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_fakes import ISSUE, ScriptedChatClient, context_for, toy_repo

from forecast.agents.base import AgentMode
from forecast.agents.environment import ExecResult
from forecast.agents.repair_agent import (
    COLLECT,
    TRY,
    UNDERSTAND,
    RepairAgentRepairAgent,
    _closest,
    _subtokens,
)

CHAIN = "1. CHAINMARK `a - b` in add (discriminability: 0.90)"


class _LocalContainer:
    def __init__(self, repo: Path, row: dict):
        self.repo = repo
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def _git(self, *args, stdin=None):
        return subprocess.run(["git", *args], cwd=self.repo, input=stdin, text=True, capture_output=True)

    def apply_patch(self, patch):
        r = self._git("apply", "-", stdin=patch if patch.endswith("\n") else patch + "\n")
        return ExecResult(r.stdout + r.stderr, r.returncode)

    def apply_test_patch(self):
        return self.apply_patch(self.row["test_patch"])

    def reset(self):
        self._git("checkout", "-q", "--", ".")
        self._git("clean", "-fdq")

    def export(self, destination):
        target = Path(destination) / "testbed"
        shutil.copytree(self.repo, target)
        return target

    def run_tests(self, timeout=0):
        r = subprocess.run([sys.executable, "-m", "pytest", "-rA", "-q", "tests"],
                           cwd=self.repo, capture_output=True, text=True)
        return ExecResult(r.stdout + r.stderr, r.returncode)

    def exec(self, command, timeout=0):
        r = subprocess.run(command, shell=True, cwd=self.repo, capture_output=True, text=True)
        return ExecResult(r.stdout + r.stderr, r.returncode)


def _pytest_verdict(task_info, output):
    from swebench.harness.log_parsers.python import parse_log_pytest
    from swebench.harness.grading import test_passed

    status = parse_log_pytest(output, None)
    result = {}
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        tests = json.loads(task_info[key])
        result[key] = [t for t in tests if not test_passed(t, status)]
    return {"passed": not result["FAIL_TO_PASS"] and not result["PASS_TO_PASS"],
            "failing_fail_to_pass": result["FAIL_TO_PASS"],
            "failing_pass_to_pass": result["PASS_TO_PASS"]}


def _cmd(name, **args):
    return json.dumps({"thoughts": "...", "command": {"name": name, "args": args}})


FIX = [{"file_path": "calc.py", "modifications": [{"line_number": 2, "modified_line": "    return a + b"}]}]
WRONG_FIX = [{"file_path": "calc.py", "modifications": [{"line_number": 2, "modified_line": "    return a * b"}]}]


class RepairAgentPortTests(unittest.TestCase):
    def _run(self, script, mode=AgentMode.FORECAST, diagnosis=CHAIN, sbfl=None):
        repo, row = toy_repo()
        context = context_for(repo, row, diagnosis)
        context.fault_localization = sbfl

        class _Agent(RepairAgentRepairAgent):
            def make_container(self, task_info):
                self.container = _LocalContainer(repo, row)
                return self.container

        agent = _Agent(mode=mode)
        client = ScriptedChatClient(script)
        with mock.patch("forecast.agents.environment.test_verdict", side_effect=_pytest_verdict), \
                tempfile.TemporaryDirectory() as out:
            result = agent.repair(context, model_client=client, traj_dir=Path(out))
        return result, client, repo

    def test_full_cycle_through_the_state_machine(self):
        result, client, repo = self._run([
            _cmd("read_range", filepath="calc.py", startline=1, endline=2),
            _cmd("express_hypothesis", hypothesis="add subtracts"),
            _cmd("write_fix", changes_dicts=WRONG_FIX),
            _cmd("write_fix", changes_dicts=FIX),
            _cmd("goals_accomplished", reason="tests pass"),
        ])
        self.assertIsNone(result.error, result.error)
        self.assertIn("+    return a + b", result.patch)
        self.assertNotIn("test_calc", result.patch, "the fix's diff excludes the developer tests")
        self.assertEqual(5, len(client.requests))
        prompts = [r["messages"][1]["content"] for r in client.requests]
        self.assertIn(f"## Current state\n{UNDERSTAND}", prompts[0])
        self.assertIn(f"## Current state\n{COLLECT}", prompts[2])
        self.assertIn(f"## Current state\n{TRY}", prompts[3])
        self.assertIn("1 failing test(s)", prompts[3])  # the wrong fix was tested and reverted
        self.assertIn("passed all test cases", prompts[4])
        # the working tree is back to base + tests after each fix
        self.assertIn("return a - b", (repo / "calc.py").read_text())

    def test_initial_prompt_has_issue_chain_failing_tests_and_test_run(self):
        sbfl = [{"file": "calc.py", "method": "add", "line": 1, "suspiciousness": 1.0}]
        _, client, _ = self._run([_cmd("goals_accomplished", reason="x")] * 1, sbfl=sbfl)
        first = client.requests[0]["messages"][1]["content"]
        self.assertIn(ISSUE, first)
        self.assertEqual(1, first.count("CHAINMARK"))
        self.assertIn("tests/test_calc.py::test_add", first)
        self.assertIn("1 failing test(s)", first)
        self.assertIn("1. calc.py, function add", first)
        self.assertIn("You have 40 commands left", first)

    def test_baseline_has_no_chain(self):
        _, client, _ = self._run([_cmd("express_hypothesis", hypothesis="h")] * 2, mode=AgentMode.BASELINE)
        self.assertNotIn("CHAINMARK", client.sent_text())

    def test_unavailable_command_is_refused(self):
        _, client, _ = self._run([
            _cmd("write_fix", changes_dicts=FIX),  # not available while understanding the bug
            _cmd("goals_accomplished", reason="x"),
        ])
        self.assertIn("is not available in the state", client.requests[1]["messages"][1]["content"])

    def test_repeated_command_is_refused(self):
        _, client, _ = self._run([
            _cmd("read_range", filepath="calc.py", startline=1, endline=2),
            _cmd("read_range", filepath="calc.py", startline=1, endline=2),
            _cmd("express_hypothesis", hypothesis="h"),
        ])
        self.assertIn("Repeating a command is not allowed", client.requests[2]["messages"][1]["content"])

    def test_budget_ends_the_run(self):
        result, client, _ = self._run([_cmd("nonsense")])
        self.assertEqual(RepairAgentRepairAgent.COMMANDS_LIMIT, len(client.requests))
        self.assertFalse(result.patch)

    def test_middleware_repairs_names(self):
        self.assertEqual("read_range", _closest("read_rang", ["read_range", "write_fix"]))
        self.assertEqual("filepath", _closest("filepath", ["filepath", "startline"]))
        self.assertIsNone(_closest("zzz", ["read_range"]))

    def test_search_splits_keywords_into_subtokens(self):
        self.assertEqual(["quick", "sort", "array"], _subtokens("quickSortArray"))
        self.assertEqual(["max", "value"], _subtokens("max_value"))


if __name__ == "__main__":
    unittest.main()
