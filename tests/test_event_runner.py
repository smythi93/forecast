import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from forecast.runner.event_runner import docker_run_test_and_collect_events
from forecast.runner.trace_budget import TraceBudget


def _tar_chunks(name: str, payload: bytes, chunk_size: int = 64):
    """Render *payload* as the chunked tar stream the Docker API returns."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    data = buffer.getvalue()
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


class _FakeContainer:
    """
    Container double whose trace only materialises after a configured number
    of runs, so import-failure retries can be exercised.
    """

    def __init__(self, payload: bytes | None = b"events", available_after: int = 0):
        self.started = False
        self.stopped = False
        self.removed = False
        self.exec_calls: list[dict[str, object]] = []
        self.archive_calls: list[str] = []
        self._payload = payload
        self._available_after = available_after
        self._runs = 0
        self.trace_present = False

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def remove(self):
        self.removed = True

    # -- command execution -------------------------------------------------
    def exec_run(self, cmd, workdir=None, environment=None, demux=None):
        cmd = list(cmd)
        self.exec_calls.append(
            {
                "cmd": cmd,
                "workdir": workdir,
                "environment": dict(environment or {}),
            }
        )
        if cmd[:2] == ["rm", "-f"]:
            self.trace_present = False
        elif "pytest" in cmd:
            self._runs += 1
            if self._runs > self._available_after:
                self.trace_present = True
        return 0, b""

    @property
    def test_runs(self) -> list[dict[str, object]]:
        return [call for call in self.exec_calls if "pytest" in call["cmd"]]

    @property
    def removals(self) -> list[dict[str, object]]:
        return [call for call in self.exec_calls if call["cmd"][:2] == ["rm", "-f"]]

    # -- file transfer -----------------------------------------------------
    def get_archive(self, path):
        self.archive_calls.append(path)
        if not self.trace_present or self._payload is None:
            raise FileNotFoundError(path)
        return iter(_tar_chunks("forecast_events", self._payload)), {}


class _FakeContainers:
    def __init__(self, container):
        self._container = container

    def create(self, *_args, **_kwargs):
        return self._container


class _FakeClient:
    def __init__(self, container):
        self.containers = _FakeContainers(container)


def _collect(container, out: Path, **kwargs):
    docker_run_test_and_collect_events(
        client=_FakeClient(container),
        image="example:latest",
        test_id="tests/test_mod.py::test_a",
        src_root="/testbed",
        event_file_path=out,
        timeout=10,
        patch=None,
        **kwargs,
    )


class EventRunnerDockerTests(unittest.TestCase):
    def test_retries_with_pythonpath_when_no_trace_is_produced(self):
        container = _FakeContainer(available_after=1)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "events.bin"
            _collect(container, out, conda_env=None)

            self.assertTrue(out.exists())
            self.assertEqual(b"events", out.read_bytes())
            self.assertEqual(2, len(container.test_runs))
            second_env = container.test_runs[1]["environment"]
            self.assertEqual("/testbed", second_env.get("PYTHONPATH"))

    def test_single_run_when_trace_exists(self):
        container = _FakeContainer()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "events.bin"
            _collect(container, out, conda_env=None)

            self.assertTrue(out.exists())
            self.assertEqual(1, len(container.test_runs))

    def test_trace_is_removed_from_container_after_every_attempt(self):
        container = _FakeContainer(available_after=1)

        with tempfile.TemporaryDirectory() as tmp:
            _collect(container, Path(tmp) / "events.bin", conda_env=None)

        # One removal per attempt, so a container never holds two traces.
        self.assertEqual(len(container.test_runs), len(container.removals))
        self.assertFalse(container.trace_present)

    def test_empty_trace_is_not_retried(self):
        container = _FakeContainer(payload=b"")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "events.bin"
            with self.assertRaises(FileNotFoundError):
                _collect(container, out, conda_env=None)

        # An empty trace means the run happened but recorded nothing; re-running
        # it would only reproduce the same result.
        self.assertEqual(1, len(container.test_runs))

    def test_large_trace_is_streamed_to_disk(self):
        payload = bytes(range(256)) * 8192  # 2 MiB
        container = _FakeContainer(payload=payload)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "events.bin"
            _collect(container, out, conda_env=None)

            self.assertEqual(len(payload), out.stat().st_size)
            self.assertEqual(payload, out.read_bytes())

    def test_trace_budget_is_passed_to_the_tracer(self):
        container = _FakeContainer()
        budget = TraceBudget(
            max_value_bytes=64,
            max_loop_hits=3,
            max_bytes=1234,
            max_events=99,
            compress="gzip",
        )

        with tempfile.TemporaryDirectory() as tmp:
            _collect(container, Path(tmp) / "events.bin", conda_env=None, budget=budget)

        env = container.test_runs[0]["environment"]
        self.assertEqual("64", env["EVENTS_MAX_VALUE_BYTES"])
        self.assertEqual("3", env["EVENTS_MAX_LOOP_HITS"])
        self.assertEqual("1234", env["EVENTS_MAX_BYTES"])
        self.assertEqual("99", env["EVENTS_MAX_EVENTS"])
        self.assertEqual("gzip", env["EVENTS_COMPRESS"])

    def test_conda_injects_env_via_command(self):
        container = _FakeContainer()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "events.bin"
            _collect(container, out, conda_env="testbed")

            self.assertEqual(1, len(container.test_runs))
            call = container.test_runs[0]
            cmd = call["cmd"]
            self.assertIn("conda", cmd)
            self.assertIn("run", cmd)
            self.assertIn("-n", cmd)
            self.assertIn("testbed", cmd)
            self.assertIn("env", cmd)
            self.assertTrue(any(str(part).startswith("EVENTS_PATH=") for part in cmd))
            self.assertTrue(
                any(str(part).startswith("EVENTS_MAX_BYTES=") for part in cmd)
            )
            self.assertIn("python", cmd)
            self.assertIn("-m", cmd)
            self.assertIn("pytest", cmd)
            self.assertEqual({}, call["environment"])


if __name__ == "__main__":
    unittest.main()
