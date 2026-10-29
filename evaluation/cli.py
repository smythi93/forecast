"""
Standalone CLI entrypoint for evaluation workflows.

Examples:
    python -m evaluation collect --instance-ids astropy__astropy-12907
    python -m evaluation diagnose --instance-ids astropy__astropy-12907
    python -m evaluation prompt --repo astropy/astropy --agent SWEAgent --mode forecast
"""

from __future__ import annotations

import argparse
from typing import Sequence

from evaluation import diagnosis_batch, repair_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Standalone evaluation CLI (not part of distributed forecast package).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect", help="Instrument code and collect execution traces (Step 1).")
    sub.add_parser(
        "tree",
        help="Build a TracedRelevanceTree from collected event data (Step 1b).",
    )
    sub.add_parser("diagnose", help="Derive a FORECAST diagnosis from collected traces (Step 2).")
    sub.add_parser(
        "prompt",
        help="Extract the agent prompt, optionally injecting the diagnosis (Step 3).",
    )
    sub.add_parser(
        "run",
        help="Run the repair agent: LLM call, save trajectory, extract patch (Step 4).",
    )
    sub.add_parser("evaluate", help="Validate a patch against the test suite (Step 5).")
    # Backward-compatible entrypoints.
    sub.add_parser("diagnosis", help="Run diagnosis batch pipeline.")
    sub.add_parser("repair", help="Run repair batch pipeline.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)

    if args.command in {"collect", "tree", "diagnose"}:
        return diagnosis_batch.main(["--stage", args.command, *rest])
    if args.command in {"prompt", "run", "evaluate"}:
        return repair_batch.main(["--stage", args.command, *rest])
    if args.command == "diagnosis":
        return diagnosis_batch.main(rest)
    if args.command == "repair":
        return repair_batch.main(rest)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
