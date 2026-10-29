import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from forecast import cli


class _NullLogger:
    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class CliCollectTests(unittest.TestCase):
    def test_handle_collect_local_uses_subject_for_event_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "repo"
            src.mkdir(parents=True, exist_ok=True)
            output = tmp_path / "out"
            failing = "tests/test_mod.py::test_fail"
            passing = "tests/test_mod.py::test_pass"
            args = Namespace(
                log_dir=None,
                output=str(output),
                swe=False,
                src=str(src),
                dev=False,
                checkout_dir=None,
                workers=3,
                failing=failing,
                passing=passing,
            )

            subject_instance = MagicMock()
            with (
                patch("forecast.logger.get_logger", return_value=_NullLogger()),
                patch("forecast.instrument.instrument") as instrument_mock,
                patch(
                    "forecast.subject.LocalSubject",
                    return_value=subject_instance,
                ) as subject_cls,
            ):
                cli._handle_collect(args)

            instrument_mock.assert_called_once_with(
                src=src.resolve(),
                dst=output / "instrumented",
                mapping_path=output / "mapping.json",
            )
            subject_cls.assert_called_once_with(
                path=output / "instrumented",
                workers=3,
                test_ids=sorted([failing, passing]),
                fail_ids={failing},
                failing_test_ids={failing},
                passing_test_ids={passing},
            )
            subject_instance.collect_all_events.assert_called_once()
            (mapping,), _ = subject_instance.collect_all_events.call_args
            self.assertIn(failing, mapping)
            self.assertIn(passing, mapping)
            self.assertEqual(output / "events" / "failing", Path(mapping[failing]).parent)
            self.assertEqual(output / "events" / "passing", Path(mapping[passing]).parent)

    def test_handle_collect_swe_exits_when_instance_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                log_dir=None,
                output=str(Path(tmp) / "out"),
                swe=True,
                src="missing__instance",
                dev=False,
                checkout_dir=None,
                workers=4,
                failing=None,
                passing=None,
            )
            with (
                patch("forecast.logger.get_logger", return_value=_NullLogger()),
                patch("evaluation.SWEBenchLoader.load_instances", return_value=[]),
            ):
                with self.assertRaises(SystemExit) as raised:
                    cli._handle_collect(args)
        self.assertEqual(1, raised.exception.code)

    def test_handle_collect_swe_defaults_to_fail_to_pass_and_pass_to_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "out"
            checked_out_repo = tmp_path / "repo"
            checked_out_repo.mkdir(parents=True, exist_ok=True)
            failing = "tests/test_mod.py::test_fail"
            passing = "tests/test_mod.py::test_pass"
            args = Namespace(
                log_dir=None,
                output=str(output),
                swe=True,
                src="astropy__astropy-12907",
                dev=False,
                checkout_dir=None,
                workers=2,
                failing=None,
                passing=None,
            )

            subject_instance = MagicMock()
            swe_instance = {
                "instance_id": "astropy__astropy-12907",
                "repo": str(checked_out_repo),
                "failing_tests": [failing],
                "passing_tests": [passing],
            }
            with (
                patch("forecast.logger.get_logger", return_value=_NullLogger()),
                patch("evaluation.SWEBenchLoader.load_instances", return_value=[swe_instance]),
                patch("forecast.cli._checkout_swe", return_value=swe_instance),
                patch("forecast.instrument.instrument") as instrument_mock,
                patch(
                    "forecast.subject.LocalSubject",
                    return_value=subject_instance,
                ) as subject_cls,
            ):
                cli._handle_collect(args)

            instrument_mock.assert_called_once_with(
                src=checked_out_repo,
                dst=output / "instrumented",
                mapping_path=output / "mapping.json",
            )
            subject_cls.assert_called_once_with(
                path=output / "instrumented",
                workers=2,
                test_ids=sorted([failing, passing]),
                fail_ids={failing},
                failing_test_ids={failing},
                passing_test_ids={passing},
            )


class CliCheckoutTests(unittest.TestCase):
    def test_checkout_swe_uses_default_checkout_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            args = Namespace(output=str(output), checkout_dir=None)
            instance = {"instance_id": "astropy__astropy-12907"}
            expected_root = output / "repos"
            with patch(
                "evaluation.SWEBenchLoader.checkout_instance",
                return_value={"repo": "/tmp/repo"},
            ) as checkout_mock:
                result = cli._checkout_swe(instance, args, log=None)

        checkout_mock.assert_called_once_with(instance, expected_root, log=None)
        self.assertEqual("/tmp/repo", result["repo"])


if __name__ == "__main__":
    unittest.main()
