import os.path
import sys
import tempfile
import unittest
from pathlib import Path

from forecast import subject
from forecast.localization.locations import Line, Function
from utils import MIDDLE_DIR, EPSILON


class SubjectsTests(unittest.TestCase):

    def test_local_subject(self):
        s = subject.LocalSubject(path=MIDDLE_DIR, test_base="tests.py")
        s.load()
        s.save()
        self.assertTrue(s.cache_path.exists())
        self.assertEqual(s.path, MIDDLE_DIR)
        self.assertTrue(s.path.exists())
        self.assertEqual(s.python, sys.executable)

    def test_repo_subject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            s = subject.RepoSubject(
                path=temp_dir,
                test_base="tests",
                repo="https://github.com/smythi93/middle.git",
                commit="eed99fa2741bd28744231dfcac0ea34679532bf9",
                should_install=False,
            )
            s.load()
            s.setup()
            s.save()
            self.assertTrue(s.cache_path.exists())
            self.assertEqual(s.path, Path(temp_dir))
            self.assertTrue(s.path.exists())
            self.assertEqual(s.python, sys.executable)

    def test_docker_subject(self):
        s = subject.DockerSubject(
            path="/app",
            test_base="tests",
            image="mariussmytzek/middle:latest",
        )
        s.load()
        s.setup()
        s.save()
        self.assertTrue(s.cache_path.exists())
        self.assertEqual(s.path, Path("/", "app"))
        # Check that the image is pulled by looking for it in the local Docker images
        import docker

        client = docker.from_env()
        images = client.images.list()
        image_names = [tag for image in images for tag in image.tags]
        self.assertIn("mariussmytzek/middle:latest", image_names)
        # check that we can run a command in the container
        result = s.run(
            ["echo", "hello world"],
        )
        self.assertEqual(result.strip(), "hello world")

    def test_swe_subject(self):
        s = subject.SWESubject.from_instance_id(instance_id="astropy__astropy-12907")
        self.assertEqual(s.repo, "astropy/astropy")
        self.assertEqual(s.instance_id, "astropy__astropy-12907")
        self.assertEqual(s.base_commit, "d16bfe05a744909de4b27f5875fe0d4ed41ce607")
        s.setup()
        s.save()
        self.assertTrue(s.cache_path.exists())
        result = s.run(["echo", "hello world"])
        self.assertEqual(result.strip(), "hello world")


class LocalizationTests(unittest.TestCase):
    def test_local_subject_line_localization(self):
        s = subject.LocalSubject(path=MIDDLE_DIR, test_base="tests.py")
        s.load()
        lines = s.sfl_line()
        s.save()
        self.assertEqual(10, len(lines))
        print(lines)
        self.assertEqual(Line(file="middle.py", line=6), lines[0])
        self.assertAlmostEqual(
            0.8333333333333334, lines[0].suspiciousness, delta=EPSILON
        )
        self.assertEqual(Line(file="middle.py", line=5), lines[1])
        self.assertAlmostEqual(
            0.7142857142857143, lines[1].suspiciousness, delta=EPSILON
        )
        self.assertEqual(Line(file="middle.py", line=3), lines[2])
        self.assertAlmostEqual(0.625, lines[2].suspiciousness, delta=EPSILON)

    def test_local_subject_function_localization(self):
        s = subject.LocalSubject(path=MIDDLE_DIR, test_base="tests.py")
        s.load()
        functions = s.sfl_function()
        s.save()
        self.assertEqual(1, len(functions))
        self.assertEqual(
            Function(
                file="middle.py",
                line=1,
                name="middle",
            ),
            functions[0],
        )
        self.assertAlmostEqual(0.5, functions[0].suspiciousness, delta=EPSILON)

    def test_repo_subject_line_localization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            s = subject.RepoSubject(
                path=temp_dir,
                test_base="tests",
                repo="https://github.com/smythi93/middle.git",
                commit="eed99fa2741bd28744231dfcac0ea34679532bf9",
            )
            s.load()
            s.setup()
            lines = s.sfl_line()
            s.save()
            self.assertEqual(10, len(lines))
            self.assertEqual(
                Line(file=os.path.join("src", "middle", "__init__.py"), line=6),
                lines[0],
            )
            self.assertAlmostEqual(
                0.8333333333333334, lines[0].suspiciousness, delta=EPSILON
            )
            self.assertEqual(
                Line(file=os.path.join("src", "middle", "__init__.py"), line=5),
                lines[1],
            )
            self.assertAlmostEqual(
                0.7142857142857143, lines[1].suspiciousness, delta=EPSILON
            )
            self.assertEqual(
                Line(file=os.path.join("src", "middle", "__init__.py"), line=3),
                lines[2],
            )
            self.assertAlmostEqual(0.625, lines[2].suspiciousness, delta=EPSILON)

    def test_repo_subject_function_localization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            s = subject.RepoSubject(
                path=temp_dir,
                test_base="tests",
                repo="https://github.com/smythi93/middle.git",
                commit="eed99fa2741bd28744231dfcac0ea34679532bf9",
            )
            s.load()
            s.setup()
            functions = s.sfl_function()
            s.save()
            self.assertEqual(1, len(functions))
            self.assertEqual(
                Function(
                    file=os.path.join("src", "middle", "__init__.py"),
                    line=1,
                    name="middle",
                ),
                functions[0],
            )
            self.assertAlmostEqual(0.5, functions[0].suspiciousness, delta=EPSILON)

    def test_docker_subject_line_localization(self):
        s = subject.DockerSubject(
            path="/app",
            test_base="tests",
            image="mariussmytzek/middle:latest",
        )
        s.load()
        s.setup()
        lines = s.sfl_line()
        s.save()
        self.assertEqual(10, len(lines))
        self.assertEqual(
            Line(file=os.path.join("src", "middle", "__init__.py"), line=6), lines[0]
        )
        self.assertAlmostEqual(
            0.8333333333333334, lines[0].suspiciousness, delta=EPSILON
        )
        self.assertEqual(
            Line(file=os.path.join("src", "middle", "__init__.py"), line=5), lines[1]
        )
        self.assertAlmostEqual(
            0.7142857142857143, lines[1].suspiciousness, delta=EPSILON
        )
        self.assertEqual(
            Line(file=os.path.join("src", "middle", "__init__.py"), line=3), lines[2]
        )
        self.assertAlmostEqual(0.625, lines[2].suspiciousness, delta=EPSILON)

    def test_docker_subject_function_localization(self):
        s = subject.DockerSubject(
            path="/app",
            test_base="tests",
            image="mariussmytzek/middle:latest",
        )
        s.load()
        s.setup()
        functions = s.sfl_function()
        s.save()
        self.assertEqual(1, len(functions))
        self.assertEqual(
            Function(
                file=os.path.join("src", "middle", "__init__.py"),
                line=1,
                name="middle",
            ),
            functions[0],
        )
        self.assertAlmostEqual(0.5, functions[0].suspiciousness, delta=EPSILON)

    def test_swe_subject_line_localization(self):
        s = subject.SWESubject.from_instance_id(instance_id="astropy__astropy-12907")
        s.load()
        s.setup()
        lines = s.sfl_line()
        s.save()
        highest_lines = [
            line
            for line in lines
            if line.suspiciousness > lines[0].suspiciousness - EPSILON
        ]
        self.assertEqual(13, len(highest_lines))
        for line in highest_lines:
            self.assertAlmostEqual(0.65, line.suspiciousness, delta=EPSILON)
            self.assertEqual("astropy/modeling/separable.py", line.file)
            self.assertIn(
                line.line, [60, 61, 62, 63, 121, 125, 239, 240, 244, 245, 307, 308, 309]
            )

    def test_swe_subject_function_localization(self):
        s = subject.SWESubject.from_instance_id(instance_id="astropy__astropy-12907")
        s.load()
        s.setup()
        functions = s.sfl_function()
        s.save()
        highest_functions = [
            func
            for func in functions
            if func.suspiciousness >= functions[0].suspiciousness - EPSILON
        ]
        self.assertEqual(5, len(highest_functions))
        for function in highest_functions:
            self.assertAlmostEqual(
                0.6190476190476191, function.suspiciousness, delta=EPSILON
            )
        self.assertEqual(
            {
                Function(
                    file="astropy/modeling/core.py",
                    line=808,
                    suspiciousness=0.6190476190476191,
                    name="_calculate_separability_matrix",
                    class_name="Model",
                    end_line=815,
                ),
                Function(
                    file="astropy/modeling/separable.py",
                    line=27,
                    suspiciousness=0.6190476190476191,
                    name="is_separable",
                    class_name=None,
                    end_line=63,
                ),
                Function(
                    file="astropy/modeling/separable.py",
                    line=290,
                    suspiciousness=0.6190476190476191,
                    name="_separable",
                    class_name=None,
                    end_line=311,
                ),
                Function(
                    file="astropy/modeling/separable.py",
                    line=219,
                    suspiciousness=0.6190476190476191,
                    name="_cstack",
                    class_name=None,
                    end_line=247,
                ),
                Function(
                    file="astropy/modeling/separable.py",
                    line=105,
                    suspiciousness=0.6190476190476191,
                    name="_compute_n_outputs",
                    class_name=None,
                    end_line=127,
                ),
            },
            set(highest_functions),
        )


class SystemTestTests(unittest.TestCase):
    def test_repo_subject_system_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            s = subject.RepoSystemTestSubject(
                path=temp_dir,
                access=os.path.join("src", "middle", "cli.py"),
                repo="https://github.com/smythi93/middle.git",
                commit="15350ede14501997656f2fd49bd5b9af2c8d2582",
                passing_tests=[
                    ["3", "3", "5"],
                    ["1", "2", "3"],
                    ["3", "2", "1"],
                    ["5", "5", "5"],
                    ["5", "3", "4"],
                ],
                failing_tests=[["2", "1", "3"]],
                should_install=False,
            )
            s.load()
            s.setup()
            s.save()
            self.assertTrue(s.cache_path.exists())
            self.assertEqual(s.path, Path(temp_dir))
            self.assertTrue(s.path.exists())
            self.assertEqual(s.python, sys.executable)

    def test_repo_subject_system_test_localization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            s = subject.RepoSystemTestSubject(
                path=temp_dir,
                access=os.path.join("src", "middle", "cli.py"),
                repo="https://github.com/smythi93/middle.git",
                commit="15350ede14501997656f2fd49bd5b9af2c8d2582",
                passing_tests=[
                    ["3", "3", "5"],
                    ["1", "2", "3"],
                    ["3", "2", "1"],
                    ["5", "5", "5"],
                    ["5", "3", "4"],
                ],
                failing_tests=[["2", "1", "3"]],
            )
            s.load()
            s.setup()
            s.save()
            lines = s.sfl_line()
            s.save()
            self.assertEqual(23, len(lines))
            self.assertEqual(
                Line(file=os.path.join("src", "middle", "__init__.py"), line=7),
                lines[0],
            )
            self.assertAlmostEqual(
                0.8333333333333334, lines[0].suspiciousness, delta=EPSILON
            )
            self.assertEqual(
                Line(file=os.path.join("src", "middle", "__init__.py"), line=6),
                lines[1],
            )
            self.assertAlmostEqual(
                0.7142857142857143, lines[1].suspiciousness, delta=EPSILON
            )
            self.assertEqual(
                Line(file=os.path.join("src", "middle", "__init__.py"), line=4),
                lines[2],
            )
            self.assertAlmostEqual(0.625, lines[2].suspiciousness, delta=EPSILON)
