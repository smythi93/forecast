import unittest

from forecast.localization.spectra import TestResult
from forecast.selection import (
    changed_regions,
    failing_footprint,
    select_passing_tests,
    select_tests_by_patch,
)


def result(test_id: str, passed: bool, coverage: dict[str, list[int]]) -> TestResult:
    return TestResult(
        test_id=test_id,
        passed=passed,
        line_coverage={file: set(lines) for file, lines in coverage.items()},
    )


class CoverageOverlapSelectionTests(unittest.TestCase):
    def setUp(self):
        self.failing = [result("t_fail", False, {"/repo/app.py": [10, 11, 12]})]

    def test_drops_tests_that_share_no_code_with_the_failure(self):
        passing = [
            result("t_overlap", True, {"/repo/app.py": [10, 11]}),
            result("t_elsewhere", True, {"/repo/other.py": [1, 2, 3]}),
        ]

        selection = select_passing_tests(
            failing_results=self.failing, passing_results=passing
        )

        self.assertEqual(["t_overlap"], selection.passing)
        self.assertEqual(["t_elsewhere"], selection.dropped)
        self.assertEqual(["t_fail"], selection.failing)
        self.assertFalse(selection.uses_oracle)
        self.assertEqual(1, selection.statistics["dropped_without_overlap"])

    def test_ranks_by_shared_code_then_by_focus(self):
        passing = [
            # Least shared code.
            result("t_low", True, {"/repo/app.py": [10]}),
            # Same shared code as t_tight but drags in much more besides.
            result(
                "t_broad",
                True,
                {"/repo/app.py": [10, 11, 12], "/repo/other.py": list(range(100))},
            ),
            result("t_tight", True, {"/repo/app.py": [10, 11, 12]}),
        ]

        selection = select_passing_tests(
            failing_results=self.failing, passing_results=passing
        )

        self.assertEqual(["t_tight", "t_broad", "t_low"], selection.passing)

    def test_respects_the_passing_test_limit(self):
        passing = [
            result("t_a", True, {"/repo/app.py": [10, 11, 12]}),
            result("t_b", True, {"/repo/app.py": [10, 11]}),
            result("t_c", True, {"/repo/app.py": [10]}),
        ]

        selection = select_passing_tests(
            failing_results=self.failing,
            passing_results=passing,
            max_passing_tests=2,
        )

        self.assertEqual(["t_a", "t_b"], selection.passing)
        self.assertEqual(["t_c"], selection.dropped)
        self.assertEqual(1, selection.statistics["dropped_over_limit"])

    def test_keeps_everything_when_the_failure_has_no_coverage(self):
        failing = [result("t_fail", False, {})]
        passing = [result("t_a", True, {"/repo/app.py": [1]})]

        selection = select_passing_tests(
            failing_results=failing, passing_results=passing
        )

        self.assertEqual(["t_a"], selection.passing)
        self.assertEqual([], selection.dropped)
        self.assertEqual("all", selection.strategy)

    def test_failing_footprint_is_the_union_over_failing_tests(self):
        footprint = failing_footprint(
            [
                result("a", False, {"/repo/app.py": [1, 2]}),
                result("b", False, {"/repo/app.py": [2, 3]}),
            ]
        )
        self.assertEqual(
            {("/repo/app.py", 1), ("/repo/app.py", 2), ("/repo/app.py", 3)}, footprint
        )


class ChangedRegionTests(unittest.TestCase):
    PATCH = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -8,7 +8,7 @@ def compute(values):
     total = 0
     for value in values:
-        total += value
+        total -= value
     return total
"""

    def test_parses_changed_files_and_lines(self):
        regions = changed_regions(self.PATCH)
        self.assertEqual({"src/app.py"}, set(regions))
        # The replaced line sits at post-image line 10.
        self.assertIn(10, regions["src/app.py"])

    def test_ignores_created_and_deleted_files(self):
        self.assertEqual({}, changed_regions("+++ /dev/null\n@@ -1,2 +1,2 @@\n-x\n"))

    def test_handles_empty_input(self):
        self.assertEqual({}, changed_regions(""))


class PatchSelectionTests(unittest.TestCase):
    def test_keeps_only_tests_reaching_the_changed_code(self):
        failing = [result("t_fail", False, {"/repo/src/app.py": [10]})]
        passing = [
            result("t_near", True, {"/repo/src/app.py": [10]}),
            result("t_far", True, {"/repo/src/app.py": [40]}),
        ]

        selection = select_tests_by_patch(
            patch=ChangedRegionTests.PATCH,
            failing_results=failing,
            passing_results=passing,
        )

        self.assertEqual(["t_near"], selection.passing)
        self.assertEqual(["t_far"], selection.dropped)
        self.assertTrue(selection.uses_oracle)
        self.assertEqual("reference-patch", selection.strategy)

    def test_context_widens_the_changed_region(self):
        failing = [result("t_fail", False, {"/repo/src/app.py": [10]})]
        passing = [result("t_near", True, {"/repo/src/app.py": [13]})]

        strict = select_tests_by_patch(
            patch=ChangedRegionTests.PATCH,
            failing_results=failing,
            passing_results=passing,
        )
        loose = select_tests_by_patch(
            patch=ChangedRegionTests.PATCH,
            failing_results=failing,
            passing_results=passing,
            context=5,
        )

        self.assertEqual([], strict.passing)
        self.assertEqual(["t_near"], loose.passing)

    def test_falls_back_to_coverage_overlap_without_a_usable_patch(self):
        failing = [result("t_fail", False, {"/repo/src/app.py": [10]})]
        passing = [result("t_near", True, {"/repo/src/app.py": [10]})]

        selection = select_tests_by_patch(
            patch="", failing_results=failing, passing_results=passing
        )

        self.assertEqual("coverage-overlap", selection.strategy)
        self.assertFalse(selection.uses_oracle)


if __name__ == "__main__":
    unittest.main()
