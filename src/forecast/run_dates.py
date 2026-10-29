"""
Remove the dates and times a run writes into its outputs.

FORECAST writes none; the tools it runs do: log lines of Agentless, SWE-agent
and the SWE-bench harness start with a time stamp, mini-swe-agent stamps every
model reply, and test runs print the date or put it into version strings.
These say when a run happened, nothing about what it did, so they are removed
from every file a run leaves.

A date counts as written at run time when its year is 2025 or later: the
repositories of SWE-bench end earlier, so dates in their code and data are
kept. Patches are never changed, and binary files are skipped.

Usage::

    python -m forecast.run_dates [--apply] PATH...
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

_YEAR = r"20(?:2[5-9]|[3-9]\d)"
_TIME = r"\d\d:\d\d(?::\d\d(?:[.,]\d+)?)?(?:Z|[+-]\d\d:?\d\d)?"
_DATE = rf"{_YEAR}-\d\d-\d\d(?:[ T]{_TIME})?"

#: A time stamp opening a log line, with the separator after it.
_LINE_STAMP = re.compile(rf"^\[?{_DATE}\]?(?:[ \t]+-[ \t]+|[ \t]+|$)", re.M)
#: Any other date or date and time.
_ANY_DATE = re.compile(_DATE)
#: setuptools-scm's date suffix of a dirty checkout's version, ``+g<hash>.d20261007``.
_VERSION_DATE = re.compile(rf"\.d{_YEAR}\d{{4}}\b")
#: Keys whose value is a time stamp.
_TIME_KEYS = {"timestamp", "created_at_utc", "updated_at_utc", "started_utc"}
#: Seconds since 1970 of 2025 through 2099.
_EPOCH = (1.7e9, 4.1e9)

_NEVER = {".diff", ".patch"}


def _is_stamp(value) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _EPOCH[0] <= value <= _EPOCH[1]
    return isinstance(value, str) and bool(re.fullmatch(_DATE, value.strip()))


def _drop_stamps(data):
    if isinstance(data, dict):
        return {
            k: _drop_stamps(v)
            for k, v in data.items()
            if not (k in _TIME_KEYS and _is_stamp(v))
        }
    if isinstance(data, list):
        return [_drop_stamps(v) for v in data]
    return data


def strip_text(text: str) -> str:
    """*text* without the dates and times written at run time."""
    text = _LINE_STAMP.sub("", text)
    text = _VERSION_DATE.sub("", text)
    return _ANY_DATE.sub("", text)


def _strip_json(text: str, lines: bool) -> str:
    """*text* without time-stamp keys; unchanged, byte for byte, when it has none."""
    if not any(f'"{key}"' in text for key in _TIME_KEYS):
        return text
    if lines:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        kept = [_drop_stamps(r) for r in records]
        if kept == records:
            return text
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    data = json.loads(text)
    kept = _drop_stamps(data)
    if kept == data:
        return text
    indent = 2 if "\n" in text.strip() else None
    out = json.dumps(kept, indent=indent, ensure_ascii=False)
    return out + ("\n" if text.endswith("\n") else "")


def _text(name: str, raw: bytes) -> str | None:
    """The text of a file named *name*, or ``None`` for a patch or a binary file."""
    if Path(name).suffix in _NEVER or b"\0" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _strip(name: str, text: str) -> str:
    """*text*, the content of a file named *name*, without the run's dates."""
    suffix = Path(name).suffix
    if suffix in {".json", ".jsonl", ".traj"}:
        try:
            text = _strip_json(text, lines=suffix == ".jsonl")
        except ValueError:
            pass
    return strip_text(text)


def has_run_dates(name: str, raw: bytes) -> bool:
    """
    Whether a file holds dates written at run time.

    :param name: The file's name; patches are never counted.
    :param raw: The file's content.
    :returns: Whether :func:`strip_file` would change the file.
    """
    text = _text(name, raw)
    return text is not None and _strip(name, text) != text


def strip_file(path: Path, apply: bool = True) -> bool:
    """
    Remove the run's dates from *path*.

    :param path: A file a run wrote.
    :param apply: Write the result; ``False`` only reports whether it would change.
    :returns: Whether the file has (or had) dates to remove.
    """
    if path.is_symlink():
        return False
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    text = _text(path.name, raw)
    if text is None:
        return False
    new = _strip(path.name, text)
    if new == text:
        return False
    if apply:
        path.write_text(new, encoding="utf-8")
    return True


def remove_run_dates(paths: Iterable[Path], apply: bool = True) -> int:
    """
    Remove the run's dates from every file under *paths*.

    :param paths: Files or directories; missing ones are skipped.
    :param apply: Write the results; ``False`` only counts.
    :returns: The number of files changed (or that would change).
    """
    changed = 0
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        for path in files:
            changed += strip_file(path, apply=apply)
    return changed


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    apply = "--apply" in args
    paths = [Path(a) for a in args if a != "--apply"]
    n = remove_run_dates(paths, apply=apply)
    print(f"{'changed' if apply else 'would change'} {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
