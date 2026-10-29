import io
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from forecast.subject import DockerSubject, LocalSubject, Subject


class SubjectInstrumentationTests(unittest.TestCase):
    def test_local_subject_instrument_updates_path_and_forwards_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "instrumented"
            mapping = Path(tmp) / "mapping.json"
            src.mkdir(parents=True, exist_ok=True)
            (src / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")

            calls: dict[str, object] = {}

            def _fake_instrument(*, src, dst, mapping_path, excludes=None):
                calls["src"] = Path(src)
                calls["dst"] = Path(dst)
                calls["mapping_path"] = Path(mapping_path)
                calls["excludes"] = list(excludes or [])
                Path(dst).mkdir(parents=True, exist_ok=True)
                Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)
                Path(mapping_path).write_text("{}", encoding="utf-8")

            fake_module = types.SimpleNamespace(instrument=_fake_instrument)
            subject = LocalSubject(path=src)
            with patch.dict(sys.modules, {"forecast.instrument": fake_module}):
                instrumented = subject.instrument(
                    src=src,
                    dst=dst,
                    mapping_path=mapping,
                    excludes=["tests"],
                )

            self.assertEqual(dst.resolve(), instrumented)
            self.assertEqual(dst.resolve(), subject.path)
            self.assertEqual(src.resolve(), calls["src"])
            self.assertEqual(dst.resolve(), calls["dst"])
            self.assertEqual(mapping.resolve(), calls["mapping_path"])
            self.assertEqual(["tests"], calls["excludes"])
            self.assertTrue(mapping.exists())

    def test_docker_subject_instrument_transfers_source_and_commits_new_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "instrumented"
            mapping = Path(tmp) / "mapping.json"
            base_image = "example:latest"
            container_source = Path(tmp) / "container-source" / "app"
            container_source.mkdir(parents=True, exist_ok=True)
            (container_source / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")

            source_archive = io.BytesIO()
            with tarfile.open(fileobj=source_archive, mode="w") as tar:
                tar.add(container_source, arcname="app")
            source_archive_bytes = source_archive.getvalue()

            class _FakeContainer:
                def __init__(self, archive: bytes):
                    self._archive = archive
                    self.commits: list[tuple[str, str | None]] = []
                    self.get_archive_calls: list[str] = []
                    self.put_archive_calls: list[str] = []
                    self.put_archive_payload: bytes | None = None
                    self.started = False
                    self.stopped = False
                    self.removed = False

                def start(self):
                    self.started = True

                def get_archive(self, path):
                    self.get_archive_calls.append(path)
                    return [self._archive], {}

                def put_archive(self, path, data):
                    self.put_archive_calls.append(path)
                    self.put_archive_payload = data.read()
                    return True

                def commit(self, repository, tag=None):
                    self.commits.append((repository, tag))

                def stop(self):
                    self.stopped = True

                def remove(self):
                    self.removed = True

            class _FakeContainers:
                def __init__(self, container):
                    self._container = container

                def run(self, *args, **kwargs):
                    return self._container

            class _FakeClient:
                def __init__(self, container):
                    self.containers = _FakeContainers(container)

            fake_container = _FakeContainer(source_archive_bytes)
            fake_client = _FakeClient(fake_container)

            subject = DockerSubject.__new__(DockerSubject)
            subject.path = Path("/app")
            subject.image = base_image
            subject.client = fake_client
            subject.conda_env = None
            subject.modification = None
            subject._base_image = base_image

            calls: dict[str, object] = {}

            def _fake_subject_instrument(
                _self, *, src, dst, mapping_path, excludes=None
            ) -> Path:
                src_path = Path(src)
                dst_path = Path(dst)
                mapping_path = Path(mapping_path)
                calls["src"] = src_path
                calls["dst"] = dst_path
                calls["mapping_path"] = mapping_path
                calls["excludes"] = list(excludes or [])
                self.assertTrue((src_path / "mod.py").exists())
                dst_path.mkdir(parents=True, exist_ok=True)
                (dst_path / "mod.py").write_text("VALUE = 2\n", encoding="utf-8")
                mapping_path.parent.mkdir(parents=True, exist_ok=True)
                mapping_path.write_text("{}", encoding="utf-8")
                return dst_path.resolve()

            with (
                patch("forecast.subject.docker_install_all") as install_all_mock,
                patch.object(
                    Subject,
                    "instrument",
                    autospec=True,
                    side_effect=_fake_subject_instrument,
                ) as instrument_mock,
            ):
                instrumented = subject.instrument(
                    src=Path(tmp) / "unused-src",
                    dst=dst,
                    mapping_path=mapping,
                    excludes=["tests"],
                )

            self.assertEqual(dst.resolve(), instrumented)
            self.assertTrue(mapping.exists())
            self.assertEqual("{}", mapping.read_text(encoding="utf-8"))
            install_all_mock.assert_called_once()
            instrument_mock.assert_called_once()
            self.assertTrue(fake_container.started)
            self.assertTrue(fake_container.stopped)
            self.assertTrue(fake_container.removed)
            self.assertEqual(["/app"], fake_container.get_archive_calls)
            self.assertEqual(["/app"], fake_container.put_archive_calls)
            self.assertEqual(1, len(fake_container.commits))
            self.assertNotEqual(base_image, subject.image)
            self.assertEqual(["tests"], calls["excludes"])
            self.assertEqual(mapping.resolve(), calls["mapping_path"])

            self.assertIsNotNone(fake_container.put_archive_payload)
            archive = io.BytesIO(fake_container.put_archive_payload or b"")
            with tarfile.open(fileobj=archive, mode="r") as tar:
                self.assertIn("mod.py", tar.getnames())
                content = tar.extractfile("mod.py").read().decode("utf-8")
            self.assertEqual("VALUE = 2\n", content)


if __name__ == "__main__":
    unittest.main()
