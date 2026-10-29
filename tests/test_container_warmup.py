import unittest
from unittest.mock import patch

from forecast import subject


class _FakeContainer:
    def __init__(self):
        self.exec_calls: list[list[str]] = []

    def exec_run(self, cmd, workdir=None, environment=None):
        self.exec_calls.append(list(cmd))
        return 0, b""

    @property
    def pytest_calls(self) -> list[list[str]]:
        return [cmd for cmd in self.exec_calls if "pytest" in cmd]


class ContainerWarmupTests(unittest.TestCase):
    """
    A fresh container builds its caches on the first test that runs in it, and
    under instrumentation that work is traced. Warming the image once at setup
    keeps it out of every trace.
    """

    def _subject(self, **kwargs) -> subject.DockerSubject:
        with patch("docker.from_env"):
            s = subject.DockerSubject(
                path="/app", test_base="tests", image="example:latest", **kwargs
            )
        s.test_ids = ["tests/test_a.py::test_one", "tests/test_a.py::test_two"]
        return s

    def test_warmup_runs_one_test_in_the_container(self):
        container = _FakeContainer()
        self._subject()._warm_container(container)

        calls = container.pytest_calls
        self.assertEqual(1, len(calls))
        self.assertIn("tests/test_a.py::test_one", calls[0])
        self.assertNotIn("tests/test_a.py::test_two", calls[0])

    def test_warmup_is_bounded_by_the_subject_timeout(self):
        container = _FakeContainer()
        s = self._subject()
        s.timeout = 42
        s._warm_container(container)

        self.assertEqual(["timeout", "42s"], container.pytest_calls[0][:2])

    def test_warmup_uses_the_conda_environment(self):
        container = _FakeContainer()
        s = self._subject()
        s.conda_env = "testbed"
        s._warm_container(container)

        self.assertIn("conda", container.pytest_calls[0])
        self.assertIn("testbed", container.pytest_calls[0])

    def test_warmup_can_be_disabled(self):
        container = _FakeContainer()
        self._subject(warmup=False)._warm_container(container)

        self.assertEqual([], container.exec_calls)

    def test_warmup_is_skipped_without_known_tests(self):
        container = _FakeContainer()
        s = self._subject()
        s.test_ids = []
        s._warm_container(container)

        self.assertEqual([], container.exec_calls)

    def test_a_failing_warmup_does_not_break_setup(self):
        class _Exploding:
            def exec_run(self, *args, **kwargs):
                raise RuntimeError("container went away")

        # The warm-up exists for its side effect on the file system; a subject
        # whose tests error must still be set up.
        self._subject()._warm_container(_Exploding())

    def test_warmup_setting_survives_serialization(self):
        s = self._subject(warmup=False)
        self.assertIs(False, s.serialize()["warmup"])

        other = self._subject()
        other.load_from_data({"warmup": False})
        self.assertIs(False, other.warmup)


if __name__ == "__main__":
    unittest.main()
