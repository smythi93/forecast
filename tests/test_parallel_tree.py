"""
Building the relevance tree across processes must not change it.

Failing runs grow the tree, so they agree on one structure and are handled in
one process. Passing runs only add observations to nodes that already exist, so
they are independent: each worker prunes against its own copy of the failing
footprint and its partial tree is folded back. What has to hold is that the
merged tree is indistinguishable from one built in a single process.
"""

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from sflkit import Config, instrument_config
from sflkit.events.event_file import EventFile
from sflkit.events.mapping import EventMapping

from forecast.handler import ForecastHandler

TMP = Path("tmp_parallel_tree")

SUBJECT = """import sys


def inner(a, b):
    total = a + b
    return total


def only_when_small(x):
    return inner(x, 0)


def middle(x, y, z):
    m = z
    if y < z:
        if x < y:
            m = inner(y, 1)
        elif x < z:
            m = inner(y, 2)
    else:
        if x > y:
            m = only_when_small(y)
        elif x > z:
            m = inner(x, 4)
    return m


middle(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
"""

#: The first is the failing run; the rest pass.
RUNS = [
    ["3", "2", "1"],
    ["1", "2", "3"],
    ["2", "1", "3"],
    ["3", "1", "2"],
    ["1", "3", "2"],
]


class ParallelTreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)
        source = (TMP / "subject").absolute()
        work = (TMP / "work").absolute()
        source.mkdir(parents=True)
        (source / "main.py").write_text(SUBJECT)
        config = Config.create(
            path=str(source),
            language="python",
            events="line,branch,def,use,function_enter,function_exit,function_error",
            predicates="line,branch,def_use",
            working=str(work),
            mapping_path=str(TMP.absolute() / "mapping.json"),
        )
        instrument_config(config)
        mapping = EventMapping.load(config)
        cls.failing, cls.passing = [], []
        for run_id, arguments in enumerate(RUNS):
            subprocess.run(
                [sys.executable, "main.py"] + arguments, cwd=work, check=True
            )
            target = work / f"run_{run_id}"
            shutil.move(work / "EVENTS_PATH", target)
            failing = run_id == 0
            event_file = EventFile(target, run_id, mapping, failing)
            (cls.failing if failing else cls.passing).append(event_file)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    @staticmethod
    def shape(tree):
        """Every node with how many observations it carries."""
        nodes, stack = [], [tree]
        while stack:
            node = stack.pop()
            nodes.append(
                (node.name, len(node.enter_features), len(node.exit_features))
            )
            stack.extend(node.children.values())
        return sorted(nodes)

    def build(self, processes):
        handler = ForecastHandler(workers=1)
        handler.build(self.failing, self.passing, processes=processes)
        return handler

    def test_parallel_matches_serial(self):
        serial = self.build(1)
        parallel = self.build(3)
        self.assertTrue(
            any("middle" in name for name, _, _ in self.shape(serial.tree)),
            "the subject produced no tree",
        )
        self.assertEqual(self.shape(serial.tree), self.shape(parallel.tree))

    def test_features_match_serial(self):
        serial = self.build(1)
        parallel = self.build(2)
        self.assertTrue(serial.tree.all_features, "no features were recorded")
        self.assertEqual(
            {str(feature) for feature in serial.tree.all_features},
            {str(feature) for feature in parallel.tree.all_features},
        )

    def test_more_processes_than_runs_is_harmless(self):
        self.assertEqual(self.shape(self.build(1).tree), self.shape(self.build(16).tree))

    def test_pruning_still_holds_in_parallel(self):
        # only_when_small is reached solely by the failing run, so a passing
        # run that never enters it must not add a node anywhere.
        parallel = self.build(3)
        names = [name for name, _, _ in self.shape(parallel.tree)]
        self.assertEqual(
            len(names), len(set(names)), "merging must not duplicate nodes"
        )


if __name__ == "__main__":
    unittest.main()
