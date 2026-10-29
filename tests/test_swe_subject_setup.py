import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from forecast.subject import SWESubject


class _FakeSpec:
    instance_image_key = "sweb.eval.x86_64.pylint-dev__pylint-7114:latest"
    repo_script_list = [
        "git clone -o origin --single-branch https://github.com/pylint-dev/pylint /testbed",
        "python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
    ]


class SWESubjectSetupTests(unittest.TestCase):
    def _make_subject(self) -> SWESubject:
        subject = SWESubject.__new__(SWESubject)
        subject.client = object()
        subject.path = Path("/testbed")
        subject.repo = "pylint-dev/pylint"
        subject.instance_id = "pylint-dev__pylint-7114"
        subject.base_commit = "deadbeef"
        subject.patch = ""
        subject.test_patch = ""
        subject.problem_statement = ""
        subject.hints_text = ""
        subject.created_at = ""
        subject.version = ""
        subject.fail_to_pass = []
        subject.pass_to_pass = []
        subject.environment_setup_commit = ""
        subject.golden_code_patch = ""
        subject.golden_test_patch = ""
        subject.conda_env = "testbed"
        subject.image = ""
        subject.modification = None
        subject._base_image = ""
        subject.issue = ""
        return subject

    def test_setup_raises_runtime_error_with_build_failure_details(self):
        subject = self._make_subject()
        captured_specs = []

        swebench_mod = types.ModuleType("swebench")

        def _fake_build_instance_images(_client, _specs):
            captured_specs.extend(_specs)
            return [], [
                RuntimeError(
                    "Error building image sweb.eval.x86_64.pylint-dev__pylint-7114:latest: "
                    "Check (logs/build_images/instances/sweb.eval.x86_64.pylint-dev__pylint-7114__latest/build_image.log) "
                    "for more information."
                )
            ]

        swebench_mod.build_instance_images = _fake_build_instance_images

        harness_mod = types.ModuleType("swebench.harness")
        test_spec_pkg = types.ModuleType("swebench.harness.test_spec")
        test_spec_mod = types.ModuleType("swebench.harness.test_spec.test_spec")
        test_spec_mod.make_test_spec = lambda *_args, **_kwargs: _FakeSpec()

        with (
            patch.dict(
                sys.modules,
                {
                    "swebench": swebench_mod,
                    "swebench.harness": harness_mod,
                    "swebench.harness.test_spec": test_spec_pkg,
                    "swebench.harness.test_spec.test_spec": test_spec_mod,
                },
            ),
            patch("forecast.subject.platform.machine", return_value="x86_64"),
            self.assertRaises(RuntimeError) as ctx,
        ):
            subject.setup()

        message = str(ctx.exception)
        self.assertIn("pylint-dev__pylint-7114", message)
        self.assertIn("0 successful image(s) and 1 failure(s)", message)
        self.assertIn("build_image.log", message)
        self.assertNotIn("Expected exactly one image", message)
        self.assertTrue(captured_specs)
        self.assertIn(
            "if ! python -m pip install -v --no-build-isolation -e .; then "
            "python -m pip install -v --no-build-isolation .; fi",
            captured_specs[0].repo_script_list,
        )
        self.assertFalse(
            any(
                "--no-use-pep517" in line
                for line in captured_specs[0].repo_script_list
            )
        )


if __name__ == "__main__":
    unittest.main()
