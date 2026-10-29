import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from forecast.instrument import (
    _discover_instrumentation_error_excludes,
    _discover_syntax_error_excludes,
    instrument,
)


class InstrumentModuleTests(unittest.TestCase):
    def test_discover_syntax_error_excludes_detects_invalid_python_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir(parents=True, exist_ok=True)
            (src / "ok.py").write_text("VALUE = 1\n", encoding="utf-8")
            (src / "bad.py").write_text(
                "CONFIG = {'keepEnglishLetter': constants.False,}\n",
                encoding="utf-8",
            )

            excludes = _discover_syntax_error_excludes(src, excludes=[])

        self.assertEqual(["bad.py"], excludes)

    def test_discover_syntax_error_excludes_respects_existing_excludes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir(parents=True, exist_ok=True)
            (src / "bad.py").write_text(
                "CONFIG = {'keepEnglishLetter': constants.False,}\n",
                encoding="utf-8",
            )

            excludes = _discover_syntax_error_excludes(src, excludes=["bad.py"])

        self.assertEqual([], excludes)

    def test_instrument_appends_invalid_python_files_to_sflkit_excludes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            mapping = Path(tmp) / "mapping.json"
            src.mkdir(parents=True, exist_ok=True)
            (src / "ok.py").write_text("VALUE = 1\n", encoding="utf-8")
            (src / "bad.py").write_text(
                "CONFIG = {'keepEnglishLetter': constants.False,}\n",
                encoding="utf-8",
            )

            captured: dict[str, object] = {}

            class _FakeConfig:
                @staticmethod
                def create(**kwargs):
                    captured["config"] = kwargs
                    return kwargs

            def _fake_instrument_config(config):
                captured["instrument_called"] = True
                captured["instrument_config_arg"] = config

            fake_sflkit = types.SimpleNamespace(
                Config=_FakeConfig,
                instrument_config=_fake_instrument_config,
            )
            with patch.dict(sys.modules, {"sflkit": fake_sflkit}):
                instrument(
                    src=src,
                    dst=dst,
                    mapping_path=mapping,
                    excludes=[],
                )

            self.assertTrue(captured.get("instrument_called"))
            config = captured.get("config")
            self.assertIsInstance(config, dict)
            exclude_value = str(config.get("exclude", ""))
            self.assertIn("bad.py", exclude_value)

    def test_discover_instrumentation_error_excludes_detects_broken_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir(parents=True, exist_ok=True)
            (src / "ok.py").write_text("VALUE = 1\n", encoding="utf-8")
            (src / "broken.py").write_text("VALUE = 2\n", encoding="utf-8")

            class _FakeInstrumenter:
                def __init__(self, _events=None):
                    return None

                def instrument_source(self, _source, _dst, file):
                    if str(file) == "broken.py":
                        raise IndexError("list index out of range")

            with patch("forecast.instrument.Instrumenter", _FakeInstrumenter):
                excludes = _discover_instrumentation_error_excludes(
                    src=src,
                    excludes=[],
                    events=None,
                )

            self.assertEqual(["broken.py"], excludes)

    def test_instrument_retries_with_instrumentation_error_excludes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            mapping = Path(tmp) / "mapping.json"
            src.mkdir(parents=True, exist_ok=True)
            (src / "ok.py").write_text("VALUE = 1\n", encoding="utf-8")
            (src / "broken.py").write_text("VALUE = 2\n", encoding="utf-8")

            configs: list[dict[str, object]] = []
            calls = {"count": 0}

            class _FakeConfig:
                @staticmethod
                def create(**kwargs):
                    configs.append(kwargs)
                    return kwargs

            def _fake_instrument_config(_config):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise IndexError("list index out of range")

            fake_sflkit = types.SimpleNamespace(
                Config=_FakeConfig,
                instrument_config=_fake_instrument_config,
            )
            with (
                patch.dict(sys.modules, {"sflkit": fake_sflkit}),
                patch(
                    "forecast.instrument._discover_instrumentation_error_excludes",
                    return_value=["broken.py"],
                ),
            ):
                instrument(
                    src=src,
                    dst=dst,
                    mapping_path=mapping,
                    excludes=[],
                )

            self.assertEqual(2, calls["count"])
            self.assertGreaterEqual(len(configs), 2)
            first_exclude = str(configs[0].get("exclude", ""))
            second_exclude = str(configs[1].get("exclude", ""))
            self.assertNotIn("broken.py", first_exclude)
            self.assertIn("broken.py", second_exclude)


if __name__ == "__main__":
    unittest.main()
