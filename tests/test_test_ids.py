import unittest
from pathlib import Path
import tempfile

from forecast.test_ids import (
    is_pytest_node_id,
    normalize_pytest_test_id,
    normalize_pytest_test_ids,
    pytest_node_id_exists_in_repo,
    resolve_pytest_node_id_against_repo,
    split_runnable_pytest_test_ids,
)


class TestIdNormalizationTests(unittest.TestCase):
    def test_normalize_display_style_unittest_id(self):
        raw = "test_failure (test_utils.tests.AssertNumQueriesContextManagerTests)"
        normalized = normalize_pytest_test_id(raw)
        self.assertEqual(
            "test_utils/tests.py::AssertNumQueriesContextManagerTests::test_failure",
            normalized,
        )

    def test_normalize_dotted_class_method_id(self):
        raw = "test_utils.tests.AssertNumQueriesContextManagerTests.test_failure"
        normalized = normalize_pytest_test_id(raw)
        self.assertEqual(
            "test_utils/tests.py::AssertNumQueriesContextManagerTests::test_failure",
            normalized,
        )

    def test_normalize_display_style_id_with_repeated_method(self):
        raw = (
            "test_foreign_key "
            "(known_related_objects.tests.ExistingRelatedInstancesTests.test_foreign_key)"
        )
        normalized = normalize_pytest_test_id(raw)
        self.assertEqual(
            "known_related_objects/tests.py::ExistingRelatedInstancesTests::test_foreign_key",
            normalized,
        )

    def test_normalize_preserves_existing_pytest_nodeid(self):
        raw = "tests/test_mod.py::TestX::test_y"
        self.assertEqual(raw, normalize_pytest_test_id(raw))

    def test_normalize_list_deduplicates(self):
        values = [
            "tests/test_mod.py::test_a",
            "tests/test_mod.py::test_a",
            "test_mod.tests.TestX.test_b",
        ]
        self.assertEqual(
            [
                "tests/test_mod.py::test_a",
                "test_mod/tests.py::TestX::test_b",
            ],
            normalize_pytest_test_ids(values),
        )

    def test_is_pytest_node_id(self):
        self.assertTrue(is_pytest_node_id("tests/test_mod.py::test_a"))
        self.assertFalse(is_pytest_node_id("An exception is setUp() is reraised."))
        self.assertFalse(
            is_pytest_node_id(
                "Tabular inlines use ModelForm.py::Meta::help_texts and labels for read-only"
            )
        )

    def test_split_runnable_pytest_test_ids(self):
        runnable, skipped = split_runnable_pytest_test_ids(
            [
                "test_failure (test_utils.tests.AssertNumQueriesContextManagerTests)",
                "An exception is setUp() is reraised after disable() is called.",
            ]
        )
        self.assertEqual(
            ["test_utils/tests.py::AssertNumQueriesContextManagerTests::test_failure"],
            runnable,
        )
        self.assertEqual(
            ["An exception is setUp() is reraised after disable() is called."],
            skipped,
        )

    def test_resolve_pytest_node_id_against_repo_adds_tests_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "tests" / "test_utils" / "tests.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def test_x():\n    pass\n", encoding="utf-8")

            resolved = resolve_pytest_node_id_against_repo(
                "test_utils/tests.py::AllowedDatabaseQueriesTests::test_allowed_database_chunked_cursor_queries",
                repo,
            )

        self.assertEqual(
            "tests/test_utils/tests.py::AllowedDatabaseQueriesTests::test_allowed_database_chunked_cursor_queries",
            resolved,
        )

    def test_resolve_pytest_node_id_against_repo_handles_missing_package_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "sympy" / "solvers" / "tests" / "test_solvers.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def test_issue_8755():\n    pass\n", encoding="utf-8")

            resolved = resolve_pytest_node_id_against_repo(
                "solvers/tests/test_solvers.py::test_issue_8755",
                repo,
            )

        self.assertEqual(
            "sympy/solvers/tests/test_solvers.py::test_issue_8755",
            resolved,
        )

    def test_resolve_pytest_node_id_against_repo_handles_truncated_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "sympy" / "solvers" / "tests" / "test_solvers.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def test_issue_8755():\n    pass\n", encoding="utf-8")

            resolved = resolve_pytest_node_id_against_repo(
                "lvers/tests/test_solvers.py::test_issue_8755",
                repo,
            )

        self.assertEqual(
            "sympy/solvers/tests/test_solvers.py::test_issue_8755",
            resolved,
        )

    def test_pytest_node_id_exists_in_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / "tests" / "test_utils" / "tests.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def test_x():\n    pass\n", encoding="utf-8")

            self.assertTrue(
                pytest_node_id_exists_in_repo(
                    "test_utils/tests.py::AllowedDatabaseQueriesTests::test_allowed_database_chunked_cursor_queries",
                    repo,
                )
            )
            self.assertFalse(
                pytest_node_id_exists_in_repo(
                    "Tabular inlines use ModelForm.py::Meta::help_texts and labels for read-only",
                    repo,
                )
            )


if __name__ == "__main__":
    unittest.main()
