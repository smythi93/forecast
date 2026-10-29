"""
Batch repair script for SWE-bench instances.

Instances can be selected by explicit ID list and/or repository slug.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from forecast.agents.base import AgentMode
from forecast.agents.register import get_registered_agent
from evaluation.run_experiment import ExperimentRunner, SWEBenchLoader
from forecast.plugin_loader import load_plugins
from forecast.run_dates import remove_run_dates


def _parse_instance_ids(
    raw_values: Sequence[str] | None,
    file_path: str | None,
) -> list[str]:
    values: list[str] = []
    for raw in raw_values or []:
        values.extend([part.strip() for part in str(raw).split(",") if part.strip()])
    if file_path:
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(line)
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch repair execution for SWE-bench instances.",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="*",
        default=None,
        help="Instance IDs (space-separated and/or comma-separated).",
    )
    parser.add_argument(
        "--instance-ids-file",
        type=str,
        default=None,
        help="Path to a file with one instance ID per line.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repository slug filter (e.g. 'astropy/astropy').",
    )
    parser.add_argument(
        "--subset",
        choices=["lite", "full"],
        default="lite",
        help="SWE-bench subset.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Use the SWE-bench dev split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on loaded instances after filtering.",
    )
    parser.add_argument(
        "--agent",
        required=True,
        type=str,
        help="Registered repair agent name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Model name forwarded to the selected repair agent.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "forecast"],
        default="forecast",
        help="Repair mode for prompt/run stage execution.",
    )
    parser.add_argument(
        "--stage",
        choices=["prompt", "run", "repair", "evaluate", "all"],
        default="run",
        help="Repair pipeline stage to execute.",
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=None,
        help="Patch file to evaluate. Defaults to the patch from the run stage output.",
    )
    parser.add_argument(
        "--force-diagnosis",
        action="store_true",
        default=False,
        help="Regenerate diagnosis before forecast-mode repairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--checkout-dir",
        type=Path,
        default=None,
        help="Directory for SWE repository checkouts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_plugins()

    parser = build_parser()
    args = parser.parse_args(argv)

    instance_ids = _parse_instance_ids(args.instance_ids, args.instance_ids_file)
    if not instance_ids and not args.repo:
        parser.error("Provide --instance-ids / --instance-ids-file or --repo.")

    instances = SWEBenchLoader.load_instances(
        subset=args.subset,
        limit=args.limit,
        instance_ids=instance_ids or None,
        repo=args.repo,
        dev=args.dev,
    )
    if not instances:
        print("No instances matched the provided filters.")
        return 1

    mode = AgentMode(args.mode)
    stage = "run" if args.stage == "repair" else args.stage
    runner = ExperimentRunner(output_dir=args.output_dir, verbose=True)
    checkout_root = (args.checkout_dir or (args.output_dir / "repos")).resolve()

    # Agents running in the instance image need no host checkout; diagnosing
    # from scratch still does. Unknown agents keep the checkout and fail later,
    # where the runner builds them.
    agent_class = get_registered_agent(args.agent)
    needs_checkout = bool(getattr(agent_class, "needs_checkout", True)) or args.force_diagnosis

    total = len(instances)
    run_successes = 0
    evaluate_successes = 0
    for idx, instance in enumerate(instances, start=1):
        prepared = (
            SWEBenchLoader.checkout_instance(instance, checkout_root)
            if needs_checkout
            else dict(instance)
        )
        instance_id = str(prepared["instance_id"])
        print(f"[{idx}/{total}] {instance_id}")

        if stage in {"prompt", "all"}:
            runner.generate_prompt(
                prepared,
                agent_name=args.agent,
                model=args.model,
                mode=mode,
                force_diagnosis=args.force_diagnosis,
            )
            print("  - prompt: done")

        if stage in {"run", "all"}:
            result = runner.run_repair(
                prepared,
                agent_name=args.agent,
                model=args.model,
                mode=mode,
                force_diagnosis=args.force_diagnosis,
            )
            if result.success:
                run_successes += 1
            print(
                f"  - run: {'success' if result.success else 'failed'}"
                f"  (patch={'yes' if bool(result.patch) else 'no'})"
            )

        if stage in {"evaluate", "all"}:
            patch_path = args.patch or runner.default_patch_path(
                instance_id,
                agent_name=args.agent,
                mode=mode,
            )
            passed = runner.evaluate_patch(
                prepared,
                agent_name=args.agent,
                patch_path=patch_path,
                mode=mode,
                model=args.model,
            )
            if passed:
                evaluate_successes += 1
            print(f"  - evaluate: {'pass' if passed else 'fail'}")

        remove_run_dates(
            args.output_dir / instance_id / step / args.agent.lower() / mode.value
            for step in ("prompt", "run", "evaluate")
        )

    if stage in {"run", "all"}:
        print(f"Run summary: {run_successes}/{total} successful repairs.")
    if stage in {"evaluate", "all"}:
        print(f"Evaluate summary: {evaluate_successes}/{total} passing patches.")
    if stage == "prompt":
        print(f"Prompt summary: {total}/{total} prompts generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
