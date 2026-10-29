import json
import subprocess
import tempfile
import unittest
import types
from pathlib import Path
from unittest.mock import patch

from forecast.agents.base import AgentMode, RepairResult
from evaluation import diagnosis_batch
from evaluation import repair_batch
from evaluation import cli as evaluation_cli
from evaluation import ExperimentRunner, LocalProjectLoader, SWEBenchLoader
from evaluation.run_experiment import (
    _apply_passing_tests_safety_cap,
    _derive_test_excludes,
    _filter_matplotlib_passing_tests,
)


class SWEBenchLoaderTests(unittest.TestCase):
    def test_load_instances_filters_by_repo_and_ids(self):
        rows = [
            {
                "instance_id": "repoA__1",
                "repo": "owner/repoA",
                "version": "v1",
                "problem_statement": "fix A1",
                "FAIL_TO_PASS": '["t1"]',
                "PASS_TO_PASS": '["p1"]',
                "base_commit": "abc",
            },
            {
                "instance_id": "repoA__2",
                "repo": "https://github.com/owner/repoA.git",
                "version": "v2",
                "problem_statement": "fix A2",
                "FAIL_TO_PASS": ["t2"],
                "PASS_TO_PASS": ["p2"],
                "base_commit": "def",
            },
            {
                "instance_id": "repoB__1",
                "repo": "owner/repoB",
                "version": "v3",
                "problem_statement": "fix B1",
                "FAIL_TO_PASS": ["tb"],
                "PASS_TO_PASS": ["pb"],
                "base_commit": "ghi",
            },
        ]
        with patch.object(SWEBenchLoader, "_load_rows", return_value=rows):
            instances = SWEBenchLoader.load_instances(
                subset="lite",
                repo="owner/repoA",
                instance_ids=["repoA__2", "repoB__1"],
            )

        self.assertEqual(1, len(instances))
        self.assertEqual("repoA__2", instances[0]["instance_id"])
        self.assertEqual("owner/repoA", instances[0]["repo"])
        self.assertEqual(["t2"], instances[0]["failing_tests"])
        self.assertEqual(["p2"], instances[0]["passing_tests"])

    def test_load_instances_normalizes_unittest_style_test_ids(self):
        rows = [
            {
                "instance_id": "django__django-10914",
                "repo": "django/django",
                "version": "v1",
                "problem_statement": "fix django bug",
                "FAIL_TO_PASS": [
                    "test_failure (test_utils.tests.AssertNumQueriesContextManagerTests)"
                ],
                "PASS_TO_PASS": [
                    "test_ok (test_utils.tests.AssertNumQueriesContextManagerTests)"
                ],
                "base_commit": "abc",
            }
        ]
        with patch.object(SWEBenchLoader, "_load_rows", return_value=rows):
            instances = SWEBenchLoader.load_instances(subset="lite")

        self.assertEqual(1, len(instances))
        self.assertEqual(
            ["test_utils/tests.py::AssertNumQueriesContextManagerTests::test_failure"],
            instances[0]["failing_tests"],
        )
        self.assertEqual(
            ["test_utils/tests.py::AssertNumQueriesContextManagerTests::test_ok"],
            instances[0]["passing_tests"],
        )

    def test_checkout_instance_applies_test_patch_for_local_repo_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "repo"
            repo.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                ["git", "init"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            target = repo / "module.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "module.py"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            base_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                text=True,
            ).strip()

            target.write_text("VALUE = 2\n", encoding="utf-8")
            test_patch = subprocess.check_output(
                ["git", "diff"],
                cwd=repo,
                text=True,
            )
            subprocess.run(
                ["git", "checkout", "--", "module.py"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )

            instance = {
                "instance_id": "repo__1",
                "repo": str(repo),
                "base_commit": base_commit,
                "test_patch": test_patch,
            }
            prepared = SWEBenchLoader.checkout_instance(
                instance,
                tmp_path / "checkouts",
            )

            self.assertEqual(str(repo.resolve()), prepared["repo"])
            self.assertEqual("VALUE = 2\n", target.read_text(encoding="utf-8"))


class InstrumentationExcludeTests(unittest.TestCase):
    def test_derive_test_excludes_includes_file_and_parent_patterns(self):
        excludes = _derive_test_excludes(
            [
                "pkg/testing/custom_test_file.py::test_case",
                "other/path/test_feature.py::TestClass::test_method",
            ]
        )

        self.assertIn("pkg/testing/custom_test_file.py", excludes)
        self.assertIn("pkg/testing", excludes)
        self.assertIn("custom_test_file.py", excludes)
        self.assertIn("other/path/test_feature.py", excludes)
        self.assertIn("other/path", excludes)
        self.assertIn("test_feature.py", excludes)
        self.assertIn("conftest.py", excludes)

    def test_filter_matplotlib_passing_tests_keeps_only_failing_files_when_large(self):
        instance = {
            "repo": "/tmp/local-checkout",
            "swe_task_info": {"repo": "matplotlib/matplotlib"},
        }
        failing = ["lib/mpl_toolkits/tests/test_mplot3d.py::test_invisible_axes[png]"]
        passing = [
            f"lib/matplotlib/tests/test_axes.py::test_case_{idx}" for idx in range(260)
        ] + [
            "lib/mpl_toolkits/tests/test_mplot3d.py::test_bar3d[png]",
            "lib/mpl_toolkits/tests/test_mplot3d.py::test_scatter3d[png]",
        ]

        filtered = _filter_matplotlib_passing_tests(
            instance=instance,
            failing_tests=failing,
            passing_tests=passing,
        )

        self.assertEqual(
            [
                "lib/mpl_toolkits/tests/test_mplot3d.py::test_bar3d[png]",
                "lib/mpl_toolkits/tests/test_mplot3d.py::test_scatter3d[png]",
            ],
            filtered,
        )

    def test_filter_matplotlib_passing_tests_noop_for_other_repos(self):
        instance = {
            "repo": "/tmp/local-checkout",
            "swe_task_info": {"repo": "django/django"},
        }
        failing = ["tests/test_a.py::test_fail"]
        passing = [f"tests/test_a.py::test_case_{idx}" for idx in range(300)]

        filtered = _filter_matplotlib_passing_tests(
            instance=instance,
            failing_tests=failing,
            passing_tests=passing,
        )

        self.assertEqual(passing, filtered)

    def test_filter_matplotlib_passing_tests_caps_and_prefers_non_image(self):
        instance = {
            "repo": "/tmp/local-checkout",
            "swe_task_info": {"repo": "matplotlib/matplotlib"},
        }
        failing = ["lib/matplotlib/tests/test_axes.py::test_fail[png]"]
        passing = [
            f"lib/matplotlib/tests/test_axes.py::test_plot_{idx}[png]"
            for idx in range(160)
        ] + [
            "lib/matplotlib/tests/test_axes.py::test_non_image_a",
            "lib/matplotlib/tests/test_axes.py::test_non_image_b",
        ]

        filtered = _filter_matplotlib_passing_tests(
            instance=instance,
            failing_tests=failing,
            passing_tests=passing,
            passing_test_threshold=1,
            max_passing_tests=10,
        )

        self.assertEqual(10, len(filtered))
        self.assertEqual("lib/matplotlib/tests/test_axes.py::test_non_image_a", filtered[0])
        self.assertEqual("lib/matplotlib/tests/test_axes.py::test_non_image_b", filtered[1])

    def test_filter_matplotlib_passing_tests_caps_without_failing_file_match(self):
        instance = {
            "repo": "/tmp/local-checkout",
            "swe_task_info": {"repo": "matplotlib/matplotlib"},
        }
        failing = ["lib/mpl_toolkits/tests/test_mplot3d.py::test_missing"]
        passing = [f"lib/matplotlib/tests/test_axes.py::test_case_{idx}" for idx in range(320)]

        filtered = _filter_matplotlib_passing_tests(
            instance=instance,
            failing_tests=failing,
            passing_tests=passing,
            passing_test_threshold=1,
            max_passing_tests=25,
        )

        self.assertEqual(25, len(filtered))

    def test_filter_matplotlib_passing_tests_caps_for_sympy(self):
        instance = {
            "repo": "/tmp/local-checkout",
            "swe_task_info": {"repo": "sympy/sympy"},
        }
        failing = ["sympy/solvers/tests/test_solvers.py::test_target"]
        passing = [f"sympy/core/tests/test_numbers.py::test_num_{idx}" for idx in range(320)]

        filtered = _filter_matplotlib_passing_tests(
            instance=instance,
            failing_tests=failing,
            passing_tests=passing,
        )

        self.assertEqual(40, len(filtered))
        self.assertEqual(passing[:40], filtered)

    def test_filter_matplotlib_passing_tests_prefers_failing_file_for_sympy(self):
        instance = {
            "repo": "/tmp/local-checkout",
            "swe_task_info": {"repo": "sympy/sympy"},
        }
        failing = ["sympy/solvers/tests/test_solvers.py::test_target"]
        passing = [
            f"sympy/core/tests/test_numbers.py::test_num_{idx}" for idx in range(200)
        ] + [
            f"sympy/solvers/tests/test_solvers.py::test_solver_{idx}"
            for idx in range(90)
        ]

        filtered = _filter_matplotlib_passing_tests(
            instance=instance,
            failing_tests=failing,
            passing_tests=passing,
        )

        self.assertEqual(40, len(filtered))
        self.assertTrue(
            all(t.startswith("sympy/solvers/tests/test_solvers.py::") for t in filtered)
        )

    def test_filter_matplotlib_passing_tests_infers_sympy_repo_from_instance_id(self):
        instance = {
            "instance_id": "sympy__sympy-11870",
            "repo": "/tmp/results/repos/sympy__sympy-11870",
            "swe_task_info": {"instance_id": "sympy__sympy-11870"},
        }
        failing = ["sympy/solvers/tests/test_solvers.py::test_target"]
        passing = [f"sympy/core/tests/test_numbers.py::test_num_{idx}" for idx in range(220)]

        filtered = _filter_matplotlib_passing_tests(
            instance=instance,
            failing_tests=failing,
            passing_tests=passing,
        )

        self.assertEqual(40, len(filtered))

    def test_filter_matplotlib_passing_tests_infers_sympy_repo_from_repo_path_only(self):
        instance = {
            "repo": "/tmp/results/repos/sympy__sympy-11870",
            "swe_task_info": None,
        }
        failing = ["sympy/solvers/tests/test_solvers.py::test_target"]
        passing = [f"sympy/core/tests/test_numbers.py::test_num_{idx}" for idx in range(220)]

        filtered = _filter_matplotlib_passing_tests(
            instance=instance,
            failing_tests=failing,
            passing_tests=passing,
        )

        self.assertEqual(40, len(filtered))

    def test_apply_passing_tests_safety_cap_truncates_large_list(self):
        passing = [f"tests/test_mod.py::test_case_{idx}" for idx in range(240)]
        capped = _apply_passing_tests_safety_cap(passing, max_passing_tests=50)
        self.assertEqual(50, len(capped))
        self.assertEqual(passing[:50], capped)

    def test_apply_passing_tests_safety_cap_keeps_short_list(self):
        passing = [f"tests/test_mod.py::test_case_{idx}" for idx in range(12)]
        capped = _apply_passing_tests_safety_cap(passing, max_passing_tests=50)
        self.assertEqual(passing, capped)


class LocalProjectLoaderTests(unittest.TestCase):
    def test_instance_id_defaults_to_directory_and_commit_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
            with patch.object(
                subprocess,
                "check_output",
                return_value="0123456789abcdef0123456789abcdef01234567\n",
            ):
                instance = LocalProjectLoader.load_instance(
                    repo_path=str(repo),
                    problem_statement="fix local issue",
                    failing_tests=["tests/test_x.py::test_a"],
                )

        self.assertTrue(instance["instance_id"].startswith(f"{repo.name}-01234567"))
        self.assertEqual("01234567", instance["version"])


class ExperimentRunnerTests(unittest.TestCase):
    def test_collect_events_for_repo_uses_subject_instrument_for_swe(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            repo = Path(tmp) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            (repo / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")

            class _StubSWESubject:
                def __init__(self):
                    self.setup_called = False
                    self.instrument_called = False
                    self.test_ids = []
                    self.fail_ids = set()
                    self.failing_test_ids = set()
                    self.passing_test_ids = set()
                    self.timeout = 120

                def setup(self):
                    self.setup_called = True

                def instrument(self, *, src, dst, mapping_path, excludes=None):
                    self.instrument_called = True
                    Path(dst).mkdir(parents=True, exist_ok=True)
                    Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(mapping_path).write_text("{}", encoding="utf-8")
                    return Path(dst)

                def collect_all_events(self, file_mapping, on_complete=None):
                    self.test_ids = sorted(file_mapping.keys())
                    for test_id, path in file_mapping.items():
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        Path(path).write_bytes(b"events")
                        if on_complete is not None:
                            on_complete(test_id, True, Path(path))
                    return [True for _ in self.test_ids]

            stub_subject = _StubSWESubject()
            with patch.object(runner, "_build_swe_subject", return_value=stub_subject):
                timing = runner._collect_events_for_repo(
                    repo,
                    output_events_dir=Path(tmp) / "events",
                    output_mapping_path=Path(tmp) / "mapping.json",
                    failing_tests=["tests/test_mod.py::test_fail"],
                    passing_tests=["tests/test_mod.py::test_pass"],
                    instance={
                        "instance_id": "astropy__astropy-12907",
                        "swe_task_info": {"instance_id": "astropy__astropy-12907"},
                    },
                )

            self.assertTrue(stub_subject.setup_called)
            self.assertTrue(stub_subject.instrument_called)
            self.assertEqual("swe", timing.get("subject_backend"))

    def test_collect_events_for_repo_skips_unrunnable_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            repo = Path(tmp) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            (repo / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")
            (repo / "tests" / "test_utils").mkdir(parents=True, exist_ok=True)
            (repo / "tests" / "test_utils" / "tests.py").write_text(
                "class AssertNumQueriesContextManagerTests:\n    pass\n",
                encoding="utf-8",
            )

            class _StubSWESubject:
                def __init__(self):
                    self.test_ids = []
                    self.fail_ids = set()
                    self.failing_test_ids = set()
                    self.passing_test_ids = set()
                    self.timeout = 120

                def setup(self):
                    return None

                def instrument(self, *, src, dst, mapping_path, excludes=None):
                    Path(dst).mkdir(parents=True, exist_ok=True)
                    Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(mapping_path).write_text("{}", encoding="utf-8")
                    return Path(dst)

                def collect_all_events(self, file_mapping, on_complete=None):
                    self.test_ids = sorted(file_mapping.keys())
                    for test_id, path in file_mapping.items():
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        Path(path).write_bytes(b"events")
                        if on_complete is not None:
                            on_complete(test_id, True, Path(path))
                    return [True for _ in self.test_ids]

            stub_subject = _StubSWESubject()
            captured: dict[str, list[str]] = {}

            def _fake_build_swe_subject(
                *, instance, selected_failing, selected_passing
            ):
                captured["failing"] = list(selected_failing)
                captured["passing"] = list(selected_passing)
                return stub_subject

            with patch.object(
                runner,
                "_build_swe_subject",
                side_effect=_fake_build_swe_subject,
            ):
                runner._collect_events_for_repo(
                    repo,
                    output_events_dir=Path(tmp) / "events",
                    output_mapping_path=Path(tmp) / "mapping.json",
                    failing_tests=["tests/test_mod.py::test_fail"],
                    passing_tests=[
                        "An exception is setUp() is reraised after disable() is called.",
                        "Tabular inlines use ModelForm.py::Meta::help_texts and labels for read-only",
                        "test_pass (test_utils.tests.AssertNumQueriesContextManagerTests)",
                    ],
                    instance={
                        "instance_id": "django__django-10914",
                        "swe_task_info": {"instance_id": "django__django-10914"},
                    },
                )

            self.assertEqual(["tests/test_mod.py::test_fail"], captured.get("failing"))
            self.assertEqual(
                [
                    "tests/test_utils/tests.py::AssertNumQueriesContextManagerTests::test_pass"
                ],
                captured.get("passing"),
            )
            self.assertNotIn(
                "An exception is setUp() is reraised after disable() is called.",
                stub_subject.test_ids,
            )
            self.assertNotIn(
                "Tabular inlines use ModelForm.py::Meta::help_texts and labels for read-only",
                stub_subject.test_ids,
            )

    def test_collect_diagnosis_events_skips_when_expected_event_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            instance_id = "django__django-10914"
            failing_test = "tests/test_mod.py::test_failure"
            passing_test = "tests/test_mod.py::test_pass"
            paths = runner._diagnosis_paths(instance_id)
            paths.failing_events_dir.mkdir(parents=True, exist_ok=True)
            paths.passing_events_dir.mkdir(parents=True, exist_ok=True)
            paths.mapping_path.parent.mkdir(parents=True, exist_ok=True)
            paths.mapping_path.write_text("{}", encoding="utf-8")
            (paths.failing_events_dir / "tests_test_mod_py__test_failure").write_bytes(
                b"events"
            )
            (paths.passing_events_dir / "tests_test_mod_py__test_pass").write_bytes(
                b"events"
            )

            instance = {
                "instance_id": instance_id,
                "repo": "django/django",
                "failing_tests": [failing_test],
                "passing_tests": [passing_test],
                "swe_task_info": {"instance_id": instance_id},
            }

            with patch.object(runner, "_collect_events_for_repo") as collect_mock:
                runner.collect_diagnosis_events(
                    instance,
                    repo_path=Path("/tmp/repo"),
                    force=False,
                )

            collect_mock.assert_not_called()
            report_path = Path(tmp) / instance_id / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual("cached", report["steps"]["collect"]["status"])

    def test_collect_diagnosis_events_recollects_when_expected_event_files_missing(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            instance_id = "django__django-10914"
            failing_test = "tests/test_mod.py::test_failure"
            passing_test = "tests/test_mod.py::test_pass"
            paths = runner._diagnosis_paths(instance_id)
            paths.failing_events_dir.mkdir(parents=True, exist_ok=True)
            paths.passing_events_dir.mkdir(parents=True, exist_ok=True)
            paths.mapping_path.parent.mkdir(parents=True, exist_ok=True)
            paths.mapping_path.write_text("{}", encoding="utf-8")
            (paths.failing_events_dir / "tests_test_mod_py__test_failure").write_bytes(
                b"events"
            )
            # Missing expected passing event on purpose.

            instance = {
                "instance_id": instance_id,
                "repo": "django/django",
                "failing_tests": [failing_test],
                "passing_tests": [passing_test],
                "swe_task_info": {"instance_id": instance_id},
            }

            def _fake_collect(
                repo_path,
                *,
                output_events_dir,
                output_mapping_path,
                failing_tests=None,
                passing_tests=None,
                instance=None,
                failure_log_dir=None,
            ):
                (output_events_dir / "failing").mkdir(parents=True, exist_ok=True)
                (output_events_dir / "passing").mkdir(parents=True, exist_ok=True)
                (
                    output_events_dir / "failing" / "tests_test_mod_py__test_failure"
                ).write_bytes(b"events")
                (
                    output_events_dir / "passing" / "tests_test_mod_py__test_pass"
                ).write_bytes(b"events")
                output_mapping_path.parent.mkdir(parents=True, exist_ok=True)
                output_mapping_path.write_text("{}", encoding="utf-8")

            with patch.object(
                runner,
                "_collect_events_for_repo",
                side_effect=_fake_collect,
            ) as collect_mock:
                runner.collect_diagnosis_events(
                    instance,
                    repo_path=Path("/tmp/repo"),
                    force=False,
                )

            collect_mock.assert_called_once()
            report_path = Path(tmp) / instance_id / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual("completed", report["steps"]["collect"]["status"])

    def test_write_collection_failure_logs_limits_to_first_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            output_dir = Path(tmp) / "logs"

            class _FakeCompleted:
                returncode = 4
                stdout = "no tests ran\n"
                stderr = "ERROR: missing\n"

            with patch(
                "evaluation.run_experiment.subprocess.run",
                return_value=_FakeCompleted(),
            ):
                runner._write_collection_failure_logs(
                    repo_path=Path(tmp),
                    tests=["test_a", "test_b"],
                    output_dir=output_dir,
                    reason="collection failed",
                    max_tests=1,
                )

            test_logs = sorted((output_dir / "tests").glob("*.log"))
            self.assertEqual(1, len(test_logs))
            self.assertEqual("test_a.log", test_logs[0].name)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary.get("truncated"))
            self.assertEqual(1, summary.get("test_count"))
            self.assertEqual(2, summary.get("requested_test_count"))

    def test_write_collection_failure_logs_falls_back_when_container_creation_fails(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            output_dir = Path(tmp) / "logs"

            class _FakeCompleted:
                returncode = 4
                stdout = "fallback stdout\n"
                stderr = "fallback stderr\n"

            class _BrokenContainers:
                @staticmethod
                def create(*_args, **_kwargs):
                    raise RuntimeError("no space left on device")

            class _Subject:
                client = types.SimpleNamespace(containers=_BrokenContainers())
                image = "fake-image"
                path = Path(tmp)
                conda_env = None
                test_patch = None

            with patch(
                "evaluation.run_experiment.subprocess.run",
                return_value=_FakeCompleted(),
            ) as run_mock:
                runner._write_collection_failure_logs(
                    repo_path=Path(tmp),
                    tests=["tests/test_mod.py::test_a"],
                    output_dir=output_dir,
                    reason="collection failed",
                    subject=_Subject(),
                )

            self.assertGreaterEqual(run_mock.call_count, 1)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, summary.get("test_count"))
            self.assertEqual("collection failed", summary.get("reason"))

    def test_collect_events_for_repo_writes_log_on_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            repo = Path(tmp) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            (repo / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")

            class _StubSWESubject:
                def __init__(self):
                    self.test_ids = []
                    self.fail_ids = set()
                    self.failing_test_ids = set()
                    self.passing_test_ids = set()
                    self.timeout = 120

                def setup(self):
                    return None

                def instrument(self, *, src, dst, mapping_path, excludes=None):
                    Path(dst).mkdir(parents=True, exist_ok=True)
                    Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(mapping_path).write_text("{}", encoding="utf-8")
                    return Path(dst)

                def collect_all_events(self, file_mapping, on_complete=None):
                    self.test_ids = sorted(file_mapping.keys())
                    raise KeyboardInterrupt()

            stub_subject = _StubSWESubject()
            with (
                patch.object(runner, "_build_swe_subject", return_value=stub_subject),
                patch.object(
                    runner,
                    "_write_collection_failure_logs",
                    return_value=Path(tmp) / "logs",
                ) as logs_mock,
                self.assertRaises(KeyboardInterrupt),
            ):
                runner._collect_events_for_repo(
                    repo,
                    output_events_dir=Path(tmp) / "events",
                    output_mapping_path=Path(tmp) / "mapping.json",
                    failing_tests=["tests/test_mod.py::test_fail"],
                    passing_tests=["tests/test_mod.py::test_pass"],
                    instance={
                        "instance_id": "astropy__astropy-12907",
                        "swe_task_info": {"instance_id": "astropy__astropy-12907"},
                    },
                    failure_log_dir=Path(tmp) / "collection_failure_logs",
                )

            logs_mock.assert_called_once()
            _, kwargs = logs_mock.call_args
            self.assertEqual(1, len(kwargs["tests"]))
            self.assertEqual(1, kwargs["max_tests"])

    def test_collect_diagnosis_events_records_interrupted_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            instance = {
                "instance_id": "astropy__astropy-12907",
                "repo": "owner/astropy",
                "failing_tests": ["tests/test_mod.py::test_fail"],
                "passing_tests": ["tests/test_mod.py::test_pass"],
                "swe_task_info": {"instance_id": "astropy__astropy-12907"},
            }

            with (
                patch.object(
                    runner,
                    "_collect_events_for_repo",
                    side_effect=KeyboardInterrupt(),
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                runner.collect_diagnosis_events(
                    instance,
                    repo_path=Path("/tmp/repo"),
                    force=True,
                )

            report_path = Path(tmp) / "astropy__astropy-12907" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            collect_step = report.get("steps", {}).get("collect", {})
            self.assertEqual("interrupted", collect_step.get("status"))

    def test_collect_diagnosis_events_uses_swe_fail_to_pass_and_pass_to_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            instance = {
                "instance_id": "astropy__astropy-12907",
                "repo": "owner/astropy",
                "failing_tests": ["tests/test_mod.py::test_fail"],
                "passing_tests": ["tests/test_mod.py::test_pass"],
                "swe_task_info": {"instance_id": "astropy__astropy-12907"},
            }

            def _fake_collect(
                repo_path,
                *,
                output_events_dir,
                output_mapping_path,
                failing_tests=None,
                passing_tests=None,
                instance=None,
                failure_log_dir=None,
            ):
                (output_events_dir / "failing").mkdir(parents=True, exist_ok=True)
                (output_events_dir / "passing").mkdir(parents=True, exist_ok=True)
                output_mapping_path.parent.mkdir(parents=True, exist_ok=True)
                output_mapping_path.write_text("{}", encoding="utf-8")

            with patch.object(
                runner,
                "_collect_events_for_repo",
                side_effect=_fake_collect,
            ) as collect_mock:
                runner.collect_diagnosis_events(
                    instance,
                    repo_path=Path("/tmp/repo"),
                    force=True,
                )

            _, kwargs = collect_mock.call_args
            self.assertEqual(instance["failing_tests"], kwargs["failing_tests"])
            self.assertEqual(instance["passing_tests"], kwargs["passing_tests"])
            report_path = Path(tmp) / "astropy__astropy-12907" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("collect", report.get("steps", {}))
            self.assertIn("elapsed_seconds", report["steps"]["collect"])

    def test_collect_diagnosis_events_writes_failed_report_with_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            instance = {
                "instance_id": "astropy__astropy-12907",
                "repo": "owner/astropy",
                "failing_tests": ["tests/test_mod.py::test_fail"],
                "passing_tests": ["tests/test_mod.py::test_pass"],
                "swe_task_info": {"instance_id": "astropy__astropy-12907"},
            }

            def _fake_collect(
                repo_path,
                *,
                output_events_dir,
                output_mapping_path,
                failing_tests=None,
                passing_tests=None,
                instance=None,
                failure_log_dir=None,
            ):
                assert failure_log_dir is not None
                failure_log_dir.mkdir(parents=True, exist_ok=True)
                (failure_log_dir / "summary.json").write_text(
                    '{"reason":"boom"}',
                    encoding="utf-8",
                )
                raise RuntimeError("collection failed")

            with (
                patch.object(
                    runner,
                    "_collect_events_for_repo",
                    side_effect=_fake_collect,
                ),
                self.assertRaisesRegex(RuntimeError, "collection failed"),
            ):
                runner.collect_diagnosis_events(
                    instance,
                    repo_path=Path("/tmp/repo"),
                    force=True,
                )

            report_path = Path(tmp) / "astropy__astropy-12907" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            collect_step = report.get("steps", {}).get("collect", {})
            self.assertEqual("failed", collect_step.get("status"))
            self.assertIn(
                "collection_failure_logs",
                collect_step.get("collection_failure_log_dir", ""),
            )

    def test_evaluate_patch_records_the_harness_verdict_per_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=Path(tmp), verbose=False)
            instance = {
                "instance_id": "test__1",
                "repo": "/tmp/repo",
                "version": "v1",
                "problem_statement": "issue",
                "failing_tests": ["t1"],
                "passing_tests": ["t2"],
                "base_commit": "abc",
                "swe_task_info": {"instance_id": "test__1"},
            }
            patch_file = Path(tmp) / "patch.diff"
            patch_file.write_text("--- a/x\n+++ b/x\n", encoding="utf-8")

            verdicts = {"baseline": False, "forecast": True}
            seen_patches = []

            def fake_harness(task_info, patch_text, *, run_id, model_name, log_root):
                seen_patches.append(patch_text)
                return {
                    "completed": True,
                    "resolved": verdicts[run_id.split("-")[1]],
                    "report": {"resolved": verdicts[run_id.split("-")[1]]},
                    "log_dir": str(log_root),
                }

            with patch(
                "forecast.agents.environment.harness_evaluate", side_effect=fake_harness
            ):
                results = {
                    mode: runner.evaluate_patch(
                        instance,
                        agent_name="MockAgent",
                        patch_path=patch_file,
                        mode=mode,
                    )
                    for mode in (AgentMode.BASELINE, AgentMode.FORECAST)
                }

            self.assertEqual(
                {AgentMode.BASELINE: False, AgentMode.FORECAST: True}, results
            )
            self.assertEqual(["--- a/x\n+++ b/x\n"] * 2, seen_patches)
            for mode, resolved in (("baseline", False), ("forecast", True)):
                stored = json.loads(
                    (
                        Path(tmp) / "test__1" / "evaluate" / "mockagent" / mode / "result.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(resolved, stored["resolved"])
            report = json.loads((Path(tmp) / "test__1" / "report.json").read_text())
            self.assertIn("evaluate/mockagent/baseline", report["steps"])
            self.assertIn("evaluate/mockagent/forecast", report["steps"])
            self.assertIn("elapsed_seconds", report["steps"]["evaluate/mockagent/forecast"])


class DiagnosisScriptTests(unittest.TestCase):
    def test_stage_all_runs_collect_tree_and_diagnosis(self):
        instance = {
            "instance_id": "test__1",
            "repo": "owner/test",
            "version": "v",
            "problem_statement": "issue",
            "failing_tests": ["t"],
            "passing_tests": ["p"],
            "base_commit": "abc",
        }
        checked_out = dict(instance, repo="/tmp/repo")

        with (
            patch("evaluation.diagnosis_batch.load_plugins", return_value=[]),
            patch(
                "evaluation.diagnosis_batch.SWEBenchLoader.load_instances",
                return_value=[instance],
            ),
            patch(
                "evaluation.diagnosis_batch.SWEBenchLoader.checkout_instance",
                return_value=checked_out,
            ) as checkout_mock,
            patch(
                "evaluation.diagnosis_batch.ExperimentRunner.collect_diagnosis_events"
            ) as events_mock,
            patch(
                "evaluation.diagnosis_batch.ExperimentRunner.build_diagnosis_tree"
            ) as tree_mock,
            patch(
                "evaluation.diagnosis_batch.ExperimentRunner.generate_diagnosis_from_events"
            ) as diagnosis_mock,
        ):
            rc = diagnosis_batch.main(
                [
                    "--instance-ids",
                    "test__1",
                    "--stage",
                    "all",
                    "--output-dir",
                    "evaluation/results/test",
                ]
            )

        self.assertEqual(0, rc)
        checkout_mock.assert_called_once()
        events_mock.assert_called_once()
        tree_mock.assert_called_once()
        diagnosis_mock.assert_called_once()

    def test_stage_diagnose_skips_checkout_and_event_collection(self):
        instance = {
            "instance_id": "test__1",
            "repo": "owner/test",
            "version": "v",
            "problem_statement": "issue",
            "failing_tests": ["t"],
            "passing_tests": ["p"],
            "base_commit": "abc",
        }

        with (
            patch("evaluation.diagnosis_batch.load_plugins", return_value=[]),
            patch(
                "evaluation.diagnosis_batch.SWEBenchLoader.load_instances",
                return_value=[instance],
            ),
            patch(
                "evaluation.diagnosis_batch.SWEBenchLoader.checkout_instance"
            ) as checkout_mock,
            patch(
                "evaluation.diagnosis_batch.ExperimentRunner.collect_diagnosis_events"
            ) as events_mock,
            patch(
                "evaluation.diagnosis_batch.ExperimentRunner.build_diagnosis_tree"
            ) as tree_mock,
            patch(
                "evaluation.diagnosis_batch.ExperimentRunner.generate_diagnosis_from_events"
            ) as diagnosis_mock,
        ):
            rc = diagnosis_batch.main(
                [
                    "--instance-ids",
                    "test__1",
                    "--stage",
                    "diagnose",
                    "--output-dir",
                    "evaluation/results/test",
                ]
            )

        self.assertEqual(0, rc)
        checkout_mock.assert_not_called()
        events_mock.assert_not_called()
        tree_mock.assert_not_called()
        diagnosis_mock.assert_called_once()


class RepairScriptTests(unittest.TestCase):
    def test_repair_script_runs_with_baseline_mode(self):
        instance = {
            "instance_id": "test__1",
            "repo": "owner/test",
            "version": "v",
            "problem_statement": "issue",
            "failing_tests": ["t"],
            "passing_tests": ["p"],
            "base_commit": "abc",
        }
        checked_out = dict(instance, repo="/tmp/repo")
        fake_result = RepairResult(
            instance_id="test__1",
            agent_name="MockAgent",
            mode=AgentMode.BASELINE,
            success=True,
            patch="--- a/x\n+++ b/x\n",
        )

        with (
            patch("evaluation.repair_batch.load_plugins", return_value=[]),
            patch(
                "evaluation.repair_batch.SWEBenchLoader.load_instances",
                return_value=[instance],
            ),
            patch(
                "evaluation.repair_batch.SWEBenchLoader.checkout_instance",
                return_value=checked_out,
            ) as checkout_mock,
            patch(
                "evaluation.repair_batch.ExperimentRunner.run_repair",
                return_value=fake_result,
            ) as run_mock,
        ):
            rc = repair_batch.main(
                [
                    "--instance-ids",
                    "test__1",
                    "--agent",
                    "MockAgent",
                    "--stage",
                    "run",
                    "--mode",
                    "baseline",
                    "--output-dir",
                    "evaluation/results/test",
                ]
            )

        self.assertEqual(0, rc)
        checkout_mock.assert_called_once()
        run_mock.assert_called_once()
        _, kwargs = run_mock.call_args
        self.assertEqual(AgentMode.BASELINE, kwargs["mode"])

    def test_stage_evaluate_uses_default_patch_path(self):
        instance = {
            "instance_id": "test__1",
            "repo": "owner/test",
            "version": "v",
            "problem_statement": "issue",
            "failing_tests": ["t"],
            "passing_tests": ["p"],
            "base_commit": "abc",
        }
        checked_out = dict(instance, repo="/tmp/repo")

        with (
            patch("evaluation.repair_batch.load_plugins", return_value=[]),
            patch(
                "evaluation.repair_batch.SWEBenchLoader.load_instances",
                return_value=[instance],
            ),
            patch(
                "evaluation.repair_batch.SWEBenchLoader.checkout_instance",
                return_value=checked_out,
            ),
            patch(
                "evaluation.repair_batch.ExperimentRunner.default_patch_path",
                return_value=Path("/tmp/patch.diff"),
            ) as default_patch_mock,
            patch(
                "evaluation.repair_batch.ExperimentRunner.evaluate_patch",
                return_value=True,
            ) as evaluate_mock,
            patch("evaluation.repair_batch.ExperimentRunner.run_repair") as run_mock,
        ):
            rc = repair_batch.main(
                [
                    "--instance-ids",
                    "test__1",
                    "--agent",
                    "MockAgent",
                    "--stage",
                    "evaluate",
                    "--mode",
                    "forecast",
                    "--output-dir",
                    "evaluation/results/test",
                ]
            )

        self.assertEqual(0, rc)
        default_patch_mock.assert_called_once_with(
            "test__1",
            agent_name="MockAgent",
            mode=AgentMode.FORECAST,
        )
        evaluate_mock.assert_called_once()
        run_mock.assert_not_called()


class EvaluationCliTests(unittest.TestCase):
    def test_cli_dispatches_collect_step(self):
        with patch("evaluation.cli.diagnosis_batch.main", return_value=0) as diag_main:
            rc = evaluation_cli.main(["collect", "--instance-ids", "x__1"])
        self.assertEqual(0, rc)
        diag_main.assert_called_once_with(
            ["--stage", "collect", "--instance-ids", "x__1"]
        )

    def test_cli_dispatches_run_step(self):
        with patch("evaluation.cli.repair_batch.main", return_value=0) as repair_main:
            rc = evaluation_cli.main(
                ["run", "--instance-ids", "x__1", "--agent", "MockAgent"]
            )
        self.assertEqual(0, rc)
        repair_main.assert_called_once_with(
            ["--stage", "run", "--instance-ids", "x__1", "--agent", "MockAgent"]
        )

    def test_cli_dispatches_diagnosis(self):
        with patch("evaluation.cli.diagnosis_batch.main", return_value=0) as diag_main:
            rc = evaluation_cli.main(["diagnosis", "--instance-ids", "x__1"])
        self.assertEqual(0, rc)
        diag_main.assert_called_once_with(["--instance-ids", "x__1"])

    def test_cli_dispatches_repair(self):
        with patch("evaluation.cli.repair_batch.main", return_value=0) as repair_main:
            rc = evaluation_cli.main(
                ["repair", "--instance-ids", "x__1", "--agent", "MockAgent"]
            )
        self.assertEqual(0, rc)
        repair_main.assert_called_once_with(
            ["--instance-ids", "x__1", "--agent", "MockAgent"]
        )


if __name__ == "__main__":
    unittest.main()


class _FakeSubject:
    """A subject that reports a fixed test list and passes everything it runs."""

    def __init__(self, discovered, failing_on_run=()):
        self.discovered = list(discovered)
        self.failing_on_run = set(failing_on_run)
        self.ran = None

    def collect_test_ids(self):
        return list(self.discovered)

    def run_tests(self, test_ids):
        self.ran = list(test_ids)
        return [
            types.SimpleNamespace(test_id=t, passed=t not in self.failing_on_run)
            for t in test_ids
        ]


class DiscoverPassingTestsTests(unittest.TestCase):
    """
    Six SWE-bench Lite instances list no ``PASS_TO_PASS``.

    Diagnosis contrasts failing runs against passing ones, so with nothing to
    contrast the cause chain comes out empty -- which is the one outcome the
    evaluation cannot have. The recovery has to cope with all three styles
    SWE-bench writes test identifiers in, and has to keep looking when the
    failing tests occupy their whole file.
    """

    @staticmethod
    def _runner():
        return ExperimentRunner.__new__(ExperimentRunner)

    def test_places_pytest_django_and_bare_identifiers(self):
        by_name = {"test_mathematica": {"sympy/parsing/tests/test_mathematica.py"}}
        place = ExperimentRunner._test_file_of
        self.assertEqual(
            "testing/io/test_saferepr.py",
            place("testing/io/test_saferepr.py::test_exceptions", {}),
        )
        self.assertEqual(
            "backends/sqlite/test_creation.py",
            place("test_custom_test_name (backends.sqlite.test_creation.TestDbSignatureTests)", {}),
        )
        self.assertEqual(
            "sympy/parsing/tests/test_mathematica.py",
            place("test_mathematica", by_name),
        )
        # A bare name defined in two files cannot be placed, and guessing would
        # seed the search from the wrong directory.
        self.assertIsNone(place("test_ambiguous", {"test_ambiguous": {"a.py", "b.py"}}))

    def test_widens_beyond_the_failing_tests_own_file(self):
        """pytest-7168's failing tests are every test in their file."""
        failing = [
            "testing/io/test_saferepr.py::test_simple_repr",
            "testing/io/test_saferepr.py::test_exceptions",
        ]
        subject = _FakeSubject(
            discovered=failing
            + [
                "testing/python/test_other.py::test_far",
                "testing/io/test_capture.py::test_near",
            ]
        )
        passing = self._runner()._discover_passing_tests(subject=subject, failing=failing)
        self.assertNotIn("testing/io/test_saferepr.py::test_exceptions", passing)
        # Same directory outranks the unrelated one, and both are still tried.
        self.assertEqual("testing/io/test_capture.py::test_near", subject.ran[0])
        self.assertEqual(
            {"testing/io/test_capture.py::test_near", "testing/python/test_other.py::test_far"},
            set(passing),
        )

    def test_recovers_for_django_style_identifiers(self):
        """These carry no ``::``, which used to end the search immediately."""
        failing = ["test_custom_test_name (backends.sqlite.test_creation.TestDbSignatureTests)"]
        # django tests are class-based; runtests.py cannot express a
        # module-level test, so discovery output is rewritten as dotted labels.
        subject = _FakeSubject(
            discovered=[
                "tests/backends/sqlite/test_creation.py::TestDbSignatureTests::test_sibling",
                "tests/unrelated/test_other.py::OtherTests::test_far",
            ]
        )
        passing = self._runner()._discover_passing_tests(subject=subject, failing=failing)
        self.assertEqual(
            "tests/backends/sqlite/test_creation.py::TestDbSignatureTests::test_sibling",
            subject.ran[0],
        )
        self.assertEqual(2, len(passing))

    def test_reports_only_tests_that_actually_passed(self):
        failing = ["a/test_x.py::test_one"]
        subject = _FakeSubject(
            discovered=["a/test_x.py::test_one", "a/test_x.py::test_two", "a/test_x.py::test_three"],
            failing_on_run=["a/test_x.py::test_three"],
        )
        passing = self._runner()._discover_passing_tests(subject=subject, failing=failing)
        self.assertEqual(["a/test_x.py::test_two"], passing)

    def test_survives_discovery_failure(self):
        class Broken:
            def collect_test_ids(self):
                raise RuntimeError("no runner")

        self.assertEqual(
            [], self._runner()._discover_passing_tests(subject=Broken(), failing=["a::b"])
        )


class AllTracesIdenticalTests(unittest.TestCase):
    """
    A run whose test suite never started still writes a trace: the import.

    Every run then emits the same one, which reads downstream as a healthy
    collection and produces an empty cause chain hours later. sphinx-7686,
    -7738 and -7975 each did exactly this because sphinx imports ``roman``
    without declaring it anywhere, so pytest died collecting the fixtures
    plugin and not one test ran.
    """

    def _paths(self, failing, passing):
        import gzip as _gzip
        from types import SimpleNamespace

        root = Path(tempfile.mkdtemp())
        f_dir, p_dir = root / "failing", root / "passing"
        f_dir.mkdir(), p_dir.mkdir()
        for directory, payloads in ((f_dir, failing), (p_dir, passing)):
            for i, payload in enumerate(payloads):
                with _gzip.open(directory / f"t{i}", "wb") as handle:
                    handle.write(payload)
        return SimpleNamespace(failing_events_dir=f_dir, passing_events_dir=p_dir)

    def test_flags_runs_that_all_traced_the_same_thing(self):
        from evaluation.run_experiment import _all_traces_identical

        # Same length, different bytes: gzip's stored name/mtime differ too,
        # so only the decompressed length exposes the shared execution.
        paths = self._paths([b"AAAA"], [b"BBBB", b"CCCC"])
        self.assertEqual(3, _all_traces_identical(paths))

    def test_stays_quiet_when_runs_differ(self):
        from evaluation.run_experiment import _all_traces_identical

        paths = self._paths([b"AAAA"], [b"BB", b"CCCCCC"])
        self.assertEqual(0, _all_traces_identical(paths))

    def test_stays_quiet_for_a_single_run(self):
        from evaluation.run_experiment import _all_traces_identical

        self.assertEqual(0, _all_traces_identical(self._paths([b"AAAA"], [])))



class NonEmptyContrastFallbackTests(unittest.TestCase):
    """
    A filter may narrow the contrast set; it must never empty it.

    scikit-learn-10297 had 28 usable passing tests and coverage overlap
    discarded all of them, so the instance produced no cause chain at all --
    an empty diagnosis is strictly worse than a weakly related contrast.
    """

    def test_fallback_keeps_candidates_when_selection_empties_them(self):
        from evaluation.run_experiment import SelectedTests
        from dataclasses import replace

        passing = [f"t{i}" for i in range(28)]
        emptied = SelectedTests(failing=["f1"], passing=[], strategy="coverage-overlap")
        # Mirror the production guard: empty selection + candidates -> fallback.
        self.assertEqual([], emptied.passing)
        fallback = replace(
            emptied,
            passing=passing[:10],
            strategy=emptied.strategy + "+nonempty-fallback",
        )
        self.assertEqual(10, len(fallback.passing))
        self.assertTrue(fallback.strategy.endswith("+nonempty-fallback"))

    def test_selection_with_results_is_left_alone(self):
        from evaluation.run_experiment import SelectedTests

        kept = SelectedTests(failing=["f1"], passing=["t1"], strategy="coverage-overlap")
        self.assertEqual(["t1"], kept.passing)
        self.assertNotIn("fallback", kept.strategy)
