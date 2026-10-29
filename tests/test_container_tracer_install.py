import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forecast.runner.utils import (
    MIN_SFLKITLIB_VERSION,
    docker_install_sflkit,
    local_sflkitlib_source,
)


class _FakeContainer:
    def __init__(self, failing_installs: tuple[str, ...] = ()):
        self.exec_calls: list[list[str]] = []
        self.archives: list[tuple[str, bytes]] = []
        # Requirement strings whose install should report failure, standing in
        # for a pin that no published release satisfies.
        self.failing_installs = failing_installs

    def exec_run(self, cmd, workdir=None, environment=None):
        self.exec_calls.append(list(cmd))
        if any(bad in cmd for bad in self.failing_installs):
            return 1, b"ERROR: No matching distribution found"
        return 0, b""

    def put_archive(self, path, data):
        self.archives.append((path, data.read() if hasattr(data, "read") else data))
        return True

    @property
    def pip_installs(self) -> list[list[str]]:
        return [cmd for cmd in self.exec_calls if "pip" in cmd]


class LocalSourceDetectionTests(unittest.TestCase):
    def test_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"FORECAST_SFLKITLIB": tmp}):
                self.assertEqual(Path(tmp), local_sflkitlib_source())

    def test_missing_override_falls_back_to_the_published_package(self):
        with patch.dict("os.environ", {"FORECAST_SFLKITLIB": "/does/not/exist"}):
            self.assertIsNone(local_sflkitlib_source())


class ContainerInstallTests(unittest.TestCase):
    def test_published_package_is_pinned_to_a_budget_aware_version(self):
        container = _FakeContainer()

        with patch(
            "forecast.runner.utils.local_sflkitlib_source", return_value=None
        ):
            docker_install_sflkit(container)

        installs = container.pip_installs
        self.assertEqual(1, len(installs))
        self.assertIn(f"sflkitlib>={MIN_SFLKITLIB_VERSION}", installs[0])

    def test_unsatisfiable_pin_falls_back_to_the_latest_release(self):
        # The budgets live in a version that is not on PyPI yet. Refusing to
        # install would make every container unusable; tracing without the
        # budgets still works, so fall back and warn.
        container = _FakeContainer(
            failing_installs=(f"sflkitlib>={MIN_SFLKITLIB_VERSION}",)
        )

        with patch(
            "forecast.runner.utils.local_sflkitlib_source", return_value=None
        ):
            docker_install_sflkit(container)

        installs = container.pip_installs
        self.assertEqual(2, len(installs))
        self.assertIn(f"sflkitlib>={MIN_SFLKITLIB_VERSION}", installs[0])
        self.assertIn("sflkitlib", installs[1])

    def test_local_checkout_is_copied_into_the_container(self):
        container = _FakeContainer()

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "pyproject.toml").write_text("[project]\nname='sflkitlib'\n")
            package = source / "src" / "sflkitlib"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("__version__ = '0.2.3'\n")
            # Should not be shipped into the container.
            (source / "build").mkdir()
            (source / "build" / "junk.bin").write_bytes(b"\x00" * 32)

            with patch(
                "forecast.runner.utils.local_sflkitlib_source", return_value=source
            ):
                docker_install_sflkit(container)

            self.assertEqual(1, len(container.archives))
            target, payload = container.archives[0]
            self.assertEqual("/tmp/forecast_sflkitlib", target)

            with tarfile.open(fileobj=__import__("io").BytesIO(payload)) as tar:
                names = set(tar.getnames())
            self.assertIn("pyproject.toml", names)
            self.assertTrue(any(n.startswith("src") for n in names))
            self.assertFalse(any(n.startswith("build") for n in names))

        installs = container.pip_installs
        self.assertEqual(1, len(installs))
        self.assertIn("/tmp/forecast_sflkitlib", installs[0])


if __name__ == "__main__":
    unittest.main()
