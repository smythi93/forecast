"""
Agentless runs from its pinned checkout with FORECAST's issue text and model.

Without Docker these tests check what FORECAST adds around upstream's stages:
the session's dataset and model substitutions, the decoder's request shape
and retries, and the harness-API shim. The full recipe runs in an instance
image on se01/se02.
"""

import logging
import tempfile
import unittest
from pathlib import Path

from agent_fakes import ScriptedChatClient, toy_repo

try:
    from forecast.agents.agentless import agentless_session, checkout

    checkout()
except (ImportError, RuntimeError):  # pragma: no cover - needs the checkout
    agentless_session = None

DJANGO_LOG = """\
+ git apply -v -
Checking patch tests/auth_tests/test_validators.py...
Applied patch tests/auth_tests/test_validators.py cleanly.
+ ./tests/runtests.py --verbosity 2 auth_tests.test_validators
test_help_text (auth_tests.test_validators.UserAttributeSimilarityValidatorTest) ... ok
test_validate (auth_tests.test_validators.UserAttributeSimilarityValidatorTest) ... ok
test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests) ... FAIL
Ran 3 tests in 0.057s
"""


@unittest.skipIf(agentless_session is None, "Agentless checkout not available")
class AgentlessSessionTests(unittest.TestCase):
    def setUp(self):
        _, self.row = toy_repo()
        self.row = dict(self.row, problem_statement="ISSUE-WITH-CHAIN")
        self.client = ScriptedChatClient(["reply"])

    def test_stages_see_one_instance_with_forecasts_issue_text(self):
        with agentless_session(self.row, self.client, Path(tempfile.mkdtemp())) as m:
            for stage in ("localize", "repair", "generate_reproduction", "select_regression"):
                data = m[stage].load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
                self.assertEqual(["ISSUE-WITH-CHAIN"], [r["problem_statement"] for r in data])
            whole = m["run_regression"].load_dataset("princeton-nlp/SWE-bench_Lite")
            self.assertEqual([self.row["instance_id"]], whole["test"]["instance_id"])

    def test_every_stage_builds_its_model_through_forecast(self):
        with agentless_session(self.row, self.client, Path(tempfile.mkdtemp())) as m:
            for key in ("repair", "generate_reproduction", "select_regression", "model"):
                decoder = m[key].make_model("label", "anthropic", logging.getLogger(), temperature=0.8)
                out = decoder.codegen("PROMPT", num_samples=2)
                self.assertEqual(["reply", "reply"], [o["response"] for o in out])
        # Upstream's Anthropic request shape: the message as the single user turn.
        self.assertEqual([{"role": "user", "content": "PROMPT"}], self.client.requests[0]["messages"])
        self.assertEqual(0.8, self.client.requests[0]["temperature"])

    def test_every_step_may_answer_in_at_least_4096_tokens(self):
        with agentless_session(self.row, self.client, Path(tempfile.mkdtemp())) as m:
            for limit, expected in ((300, 4096), (1024, 4096), (2048, 4096), (8192, 8192)):
                decoder = m["model"].make_model("l", "anthropic", logging.getLogger(), max_tokens=limit)
                decoder.codegen("p")
                self.assertEqual(expected, self.client.requests[-1]["max_tokens"])

    def test_session_restores_upstream_on_exit(self):
        import agentless.util.model as model

        original = model.make_model
        with agentless_session(self.row, self.client, Path(tempfile.mkdtemp())):
            self.assertIsNot(original, model.make_model)
        self.assertIs(original, model.make_model)

    def test_decoder_retries_like_upstream(self):
        calls = {"n": 0}

        class Flaky(ScriptedChatClient):
            def chat(self, *args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("transient")
                return super().chat(*args, **kwargs)

        with agentless_session(self.row, Flaky(["ok"]), Path(tempfile.mkdtemp())) as m:
            decoder = m["model"].make_model("l", "anthropic", logging.getLogger(), temperature=0.0)
            self.assertEqual("ok", decoder.codegen("p")[0]["response"])
        self.assertEqual(2, calls["n"])

    def test_prompts_sampled_repeatedly_are_cached_as_upstream_does(self):
        client = ScriptedChatClient(["a", "b", "c"])
        with agentless_session(self.row, client, Path(tempfile.mkdtemp())) as m:
            decoder = m["model"].make_model("l", "anthropic", logging.getLogger(), temperature=0.8)
            decoder.codegen("p", num_samples=2, prompt_cache=True)
            decoder.codegen("p", num_samples=1)
        cached, plain = client.requests[0]["messages"], client.requests[2]["messages"]
        self.assertEqual({"type": "ephemeral"}, cached[0]["content"][0]["cache_control"])
        self.assertEqual("p", cached[0]["content"][0]["text"])
        self.assertEqual("p", plain[0]["content"])

    def test_docker_client_waits_longer_and_lists_instance_images_once(self):
        from unittest import mock

        from forecast.agents.agentless import compat

        compat._LISTING = None
        with agentless_session(self.row, self.client, Path(tempfile.mkdtemp())) as m:
            with mock.patch("docker.from_env") as from_env:
                client = m["run_tests"].docker.from_env()
                first = client.images.list(all=True)
                self.assertIs(first, m["run_tests"].docker.from_env().images.list(all=True))
                client.containers.create("image")
        from_env.assert_called_with(timeout=compat.DOCKER_TIMEOUT)
        images = from_env.return_value.images
        images.list.assert_called_once_with(filters={"reference": compat.INSTANCE_IMAGES})
        from_env.return_value.containers.create.assert_called_once_with("image")
        compat._LISTING = None

    def test_without_a_reproduction_test_rerank_falls_back_to_regression_tests(self):
        import json

        from forecast.agents.agentless import _main, _results_without_reproduction

        iid = self.row["instance_id"]
        work = Path(tempfile.mkdtemp())
        # Sample 1 fails a regression test, sample 2 passes all of them.
        diff = "--- a/calc.py\n+++ b/calc.py\n-    return a - b\n+    return {}\n"
        for folder, patch, failing in (("repair_sample_1", diff.format("a * b"), ["t1"]),
                                       ("repair_sample_2", diff.format("a + b"), [])):
            (work / folder).mkdir()
            entry = {"instance_id": iid, "model_patch": patch, "normalized_patch": patch}
            for name in ("processed", "normalized"):
                (work / folder / f"output_0_{name}.jsonl").write_text(json.dumps(entry) + "\n")
            (work / folder / "output_0_regression_test_results.jsonl").write_text(
                json.dumps(dict(entry, regression=failing)) + "\n")
            _results_without_reproduction(work / folder, 0)
        with agentless_session(self.row, self.client, work) as m:
            _main(m["rerank"], ["--patch_folder", "repair_sample_1/,repair_sample_2/", "--num_samples", "2",
                                "--deduplicate", "--regression", "--reproduction",
                                "--output_file", "all_preds.jsonl"])
        chosen = json.loads((work / "all_preds.jsonl").read_text().splitlines()[0])
        self.assertEqual(diff.format("a + b"), chosen["model_patch"])

    def test_old_log_semantics_parse_logs_without_markers(self):
        import swebench.harness.grading as grading

        from forecast.agents.agentless import compat

        compat.install()
        path = Path(tempfile.mkdtemp()) / "django__django-11099" / "test_output.txt"
        path.parent.mkdir()
        path.write_text(DJANGO_LOG)
        status, ok = grading.get_logs_eval(str(path))
        self.assertTrue(ok)
        self.assertEqual("PASSED", status["test_validate (auth_tests.test_validators.UserAttributeSimilarityValidatorTest)"])
        self.assertEqual("FAILED", status["test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)"])
        path.write_text(DJANGO_LOG.replace("Applied patch", "Skipped patch"))
        self.assertEqual(({}, False), grading.get_logs_eval(str(path)))


if __name__ == "__main__":
    unittest.main()
