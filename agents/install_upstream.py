"""
Fetch the published repair agents FORECAST integrates, at their pinned commits.

SWE-agent and Agentless run from source checkouts -- SWE-agent needs its
``config/`` and ``tools/`` directories next to the package, Agentless is not a
package at all -- so they are cloned into ``agents/repos/<name>`` at the commit
recorded in ``agents/<name>.json``. SWE-agent is then installed in editable
mode; Agentless is imported from its checkout. mini-swe-agent is a pinned
PyPI dependency (``pip install -e ".[agents]"``) and needs nothing here.

AutoCodeRover and RepairAgent are re-implemented in FORECAST and need no
checkout either.

Usage::

    python agents/install_upstream.py            # clone/check out and install
    python agents/install_upstream.py --check    # only verify the pins
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent
#: Agents run from a checkout, and whether to pip-install it.
CHECKOUTS = {"swe-agent": True, "agentless": False}


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def ensure(name: str, install: bool, check_only: bool) -> bool:
    pin = json.loads((AGENTS_DIR / f"{name}.json").read_text())
    url, commit = pin["github"], pin["local"]["repo_commit"]
    path = AGENTS_DIR / "repos" / name
    if not (path / ".git").exists():
        if check_only:
            print(f"{name}: missing checkout at {path}")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", url, str(path))
    head = _git("rev-parse", "HEAD", cwd=path)
    if head != commit:
        if check_only:
            print(f"{name}: at {head[:10]}, pinned {commit[:10]}")
            return False
        _git("fetch", "--quiet", "origin", cwd=path)
        _git("checkout", "--quiet", commit, cwd=path)
    print(f"{name}: {path} at {commit[:10]}")
    if install and not check_only:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(path)], check=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="only verify the pinned checkouts")
    args = parser.parse_args()
    ok = all(ensure(name, install, args.check) for name, install in CHECKOUTS.items())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
