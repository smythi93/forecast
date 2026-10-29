"""Dates and times written at run time are removed from a run's files; the repository's are kept."""

import json
import tempfile
import unittest
from pathlib import Path

from forecast.run_dates import has_run_dates, remove_run_dates, strip_text


class StripTextTests(unittest.TestCase):
    def test_log_line_stamps_go_with_their_separator(self):
        text = (
            "2026-10-17 09:19:33,395 - INFO - Creating container\n"
            "2026-10-18T17:34:01  INFO      forecast.run  Agentless: localization\n"
            "[2026-10-18 17:34:01] started\n"
        )
        self.assertEqual(
            "INFO - Creating container\nINFO      forecast.run  Agentless: localization\nstarted\n",
            strip_text(text),
        )

    def test_dates_inside_lines_and_version_suffixes_go(self):
        text = "Running tests with Astropy version 5.3.dev134+g4331a8dde.d20261007.\n\nDate: 2026-10-07T18:22:18\n"
        self.assertEqual(
            "Running tests with Astropy version 5.3.dev134+g4331a8dde.\n\nDate: \n", strip_text(text)
        )

    def test_repository_dates_stay(self):
        text = "'%Y-%m-%d %H:%M:%S',     # '2006-10-25 14:30:59'\n+ [[ 2009-07-29 05:51:03 +0000 > 2019-02-27 ]]\n"
        self.assertEqual(text, strip_text(text))

    def test_a_line_holding_only_a_stamp_keeps_its_line_break(self):
        self.assertEqual("\nnext\n", strip_text("2026-10-07T18:22:18\nnext\n"))


class RemoveRunDatesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_time_stamps_leave_json_trajectories(self):
        traj = self.root / "mini.traj.json"
        traj.write_text(json.dumps({"messages": [
            {"role": "assistant", "content": "ok", "extra": {"cost": 0.1, "timestamp": 1792184864.94}},
            {"role": "user", "content": "x", "timestamp": "2026-10-20T23:26:13.422803+00:00"},
        ], "info": {"timestamp": 3}}, indent=2) + "\n")
        calls = self.root / "calls.jsonl"
        calls.write_text(json.dumps({"timestamp": "2026-10-18T17:34:23Z", "model": "m"}) + "\n")

        self.assertEqual(2, remove_run_dates([self.root]))

        data = json.loads(traj.read_text())
        self.assertEqual({"cost": 0.1}, data["messages"][0]["extra"])
        self.assertNotIn("timestamp", data["messages"][1])
        self.assertEqual({"timestamp": 3}, data["info"])  # not a time stamp
        self.assertTrue(traj.read_text().endswith("}\n"))
        self.assertEqual([{"model": "m"}], [json.loads(l) for l in calls.read_text().splitlines()])

    def test_patches_binaries_and_unchanged_files_are_left_alone(self):
        patch = self.root / "patch.diff"
        patch.write_text("+# 2026-10-07\n")
        binary = self.root / "django.mo"
        binary.write_bytes(b"\0\x01 2026-10-07 12:00")
        plain = self.root / "notes.txt"
        plain.write_text("nothing dated\n")
        undated = self.root / "result.json"
        undated.write_text('{"a":1,  "b": "\\u00e9"}\n')

        self.assertEqual(0, remove_run_dates([self.root, self.root / "missing"]))
        self.assertEqual("+# 2026-10-07\n", patch.read_text())
        self.assertEqual(b"\0\x01 2026-10-07 12:00", binary.read_bytes())
        self.assertEqual('{"a":1,  "b": "\\u00e9"}\n', undated.read_text())

    def test_contents_are_checked_without_a_file(self):
        self.assertTrue(has_run_dates("run.log", b"2026-10-17 09:19:33,395 - INFO - x\n"))
        self.assertTrue(has_run_dates("calls.jsonl", b'{"timestamp": 1792184864.94}\n'))
        self.assertFalse(has_run_dates("calls.jsonl", b'{"timestamp": 3}\n'))
        self.assertFalse(has_run_dates("patch.diff", b"+# 2026-10-07\n"))
        self.assertFalse(has_run_dates("run_info.json", b'{"model": "claude-haiku-4-5-20251001"}\n'))

    def test_a_dry_run_counts_without_writing(self):
        log = self.root / "run.log"
        log.write_text("2026-10-17 09:19:33,395 - INFO - x\n")
        self.assertEqual(1, remove_run_dates([log], apply=False))
        self.assertIn("2026", log.read_text())


if __name__ == "__main__":
    unittest.main()
