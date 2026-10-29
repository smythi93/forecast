"""
Integration coverage for trace collection inside a container.

The trace budgets are enforced by the tracer, which runs *inside* the
subject's environment. Unit tests can only check that FORECAST asks for the
right thing; whether the container ends up with a tracer that understands the
request, and whether the resulting trace survives the trip back to the host,
is only observable against a real Docker daemon.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from forecast import subject
from forecast.diagnosis import Forecast
from forecast.handler import ForecastHandler
from forecast.runner.event_file import (
    GZIP_MAGIC,
    make_event_file,
    resolve_event_path,
)
from forecast.runner.trace_budget import TraceBudget
from forecast.runner.utils import MIN_SFLKITLIB_VERSION

IMAGE = "mariussmytzek/middle:latest"


def _docker_is_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(_docker_is_available(), "requires a running Docker daemon")
class DockerTraceCollectionTests(unittest.TestCase):
    """Collect traces from a container and turn them into a diagnosis."""

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="forecast-docker-traces-"))
        cls.subject = subject.DockerSubject(
            path="/app", test_base="tests", image=IMAGE
        )
        cls.subject.load()
        cls.subject.setup()
        cls.base_image = cls.subject.image
        cls.mapping_path = cls.work / "mapping.json"
        # Instrumentation is what installs the tracer into the image.
        cls.subject.instrument(
            src="/app",
            dst=cls.work / "instrumented",
            mapping_path=cls.mapping_path,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.subject.image != cls.base_image:
            try:
                import docker

                docker.from_env().images.remove(cls.subject.image, force=True)
            except Exception:
                pass
        shutil.rmtree(cls.work, ignore_errors=True)

    def _in_container(self, code: str, environment: dict[str, str] | None = None):
        import docker

        client = docker.from_env()
        container = client.containers.run(
            self.subject.image, ["sleep", "infinity"], detach=True
        )
        try:
            _, output = container.exec_run(
                ["python", "-c", code], environment=environment
            )
            return output.decode("utf-8", errors="replace").strip()
        finally:
            container.stop()
            container.remove()

    def test_container_tracer_understands_the_budgets(self):
        version = self._in_container(
            "import sflkitlib; print(sflkitlib.__version__)"
        ).splitlines()[-1]
        self.assertGreaterEqual(
            tuple(int(p) for p in version.split(".")),
            tuple(int(p) for p in MIN_SFLKITLIB_VERSION.split(".")),
            f"container has sflkitlib {version}, which ignores the trace budgets",
        )

    def test_budgets_are_applied_inside_the_container(self):
        observed = self._in_container(
            "import sflkitlib.lib as lib; "
            "print(lib._max_value_bytes, lib._max_loop_hits, lib._max_bytes)",
            environment={
                "EVENTS_PATH": "/tmp/probe_events",
                "EVENTS_MAX_VALUE_BYTES": "64",
                "EVENTS_MAX_LOOP_HITS": "3",
                "EVENTS_MAX_BYTES": "12345",
            },
        ).splitlines()[-1]
        self.assertEqual("64 3 12345", observed)

    def test_compressed_traces_survive_the_trip_and_yield_a_diagnosis(self):
        from sflkit.events.mapping import EventMapping

        results = self.subject.run_tests(self.subject.collect_test_ids())
        failing = sorted(r.test_id for r in results if not r.passed)
        passing = sorted(r.test_id for r in results if r.passed)
        self.assertTrue(failing, "expected the middle subject to have a failing test")
        self.assertTrue(passing)

        self.subject.trace_budget = TraceBudget(compress="gzip")
        events_dir = self.work / "events"
        events_dir.mkdir(parents=True, exist_ok=True)

        selected = [(t, True) for t in failing[:1]] + [
            (t, False) for t in passing[:2]
        ]
        mapping = EventMapping.load_from_file(self.mapping_path)
        handler = ForecastHandler()

        for run_id, (test_id, is_failing) in enumerate(selected):
            target = events_dir / str(run_id)
            self.assertTrue(
                self.subject.collect_events_single_uncached(test_id, target),
                f"collection failed for {test_id}",
            )
            path = resolve_event_path(target)
            self.assertTrue(path.exists(), f"no trace written for {test_id}")
            with open(path, "rb") as f:
                self.assertEqual(
                    GZIP_MAGIC, f.read(2), f"trace for {test_id} is not compressed"
                )

            with make_event_file(path, run_id, mapping, failing=is_failing) as ef:
                self.assertTrue(
                    any(True for _ in ef.load()), f"no events decoded for {test_id}"
                )
            handler.handle(
                make_event_file(path, run_id, mapping, failing=is_failing)
            )

        diagnosis = Forecast(root=handler.tree).derive_diagnosis()
        self.assertTrue(diagnosis.to_text().strip())


if __name__ == "__main__":
    unittest.main()
