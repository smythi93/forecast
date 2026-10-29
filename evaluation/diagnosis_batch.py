"""
Batch diagnosis script for SWE-bench instances.

Supported stages:
* ``collect`` (alias: ``events``): instrument + collect traces,
* ``tree``: build and persist relevance tree from collected traces,
* ``diagnose``: derive diagnosis from traces/tree,
* ``all``: collect + tree + diagnose.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import docker

from evaluation.run_experiment import ExperimentRunner, SWEBenchLoader
from forecast.logger import LOGGER
from forecast.plugin_loader import load_plugins

LOG = LOGGER.getChild("diagnosis_batch")


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
    # Preserve order while deduplicating.
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch diagnosis extraction for SWE-bench instances.",
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
        "--stage",
        choices=["collect", "events", "tree", "diagnose", "diagnosis", "sbfl", "all"],
        default="all",
        help="Diagnosis stage to execute.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Recompute stage artifacts even if they already exist.",
    )
    parser.add_argument(
        "--no-prune-tree",
        action="store_true",
        default=False,
        help=(
            "Build the full relevance tree instead of pruning passing runs to "
            "the failing footprint. Pruning is on by default and is lossless "
            "for diagnosis; disable it only to reproduce the full tree."
        ),
    )
    parser.add_argument(
        "--no-compress-tree",
        action="store_true",
        default=False,
        help=(
            "Write the tree pickle uncompressed. Compression (gzip) is on by "
            "default; the reader detects the codec automatically."
        ),
    )
    parser.add_argument(
        "--tree-workers",
        type=int,
        default=1,
        help=(
            "Processes for the parallel tree build and the SBFL ranking "
            "(default 1 = sequential). "
            ">1 splits passing files across cores and merges; produces an "
            "identical tree while using idle cores to speed up large instances."
        ),
    )
    parser.add_argument(
        "--passing-cap",
        type=int,
        default=30,
        help=(
            "Keep at most this many passing runs per instance (default 30), the "
            "first of the coverage-overlap-selected set. Bounds tree size on "
            "high-run-count instances (~2x smaller, diagnosis-stable). Use 0 to "
            "keep all passing runs."
        ),
    )
    parser.add_argument(
        "--tree-mem-budget",
        type=float,
        default=50.0,
        help=(
            "Per-tree memory budget in GB (default 50). On top of --passing-cap, "
            "stop adding passing runs once the in-memory tree crosses this; only "
            "binds on a few few-run/huge-per-run outliers. Use 0 to disable."
        ),
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

    runner = ExperimentRunner(output_dir=args.output_dir, verbose=True)
    checkout_root = (args.checkout_dir or (args.output_dir / "repos")).resolve()
    stage = args.stage
    if stage == "events":
        stage = "collect"
    elif stage == "diagnosis":
        stage = "diagnose"

    total = len(instances)
    for idx, instance in enumerate(instances, start=1):
        instance_id = str(instance["instance_id"])
        print(f"[{idx}/{total}] {instance_id}")
        prepared = instance
        if stage in {"collect", "all"}:
            prepared = SWEBenchLoader.checkout_instance(
                instance,
                checkout_root,
            )
            runner.collect_diagnosis_events(
                prepared,
                repo_path=Path(str(prepared["repo"])),
                force=args.force,
            )
            print("  - collect: done")

        if stage in {"tree", "all"}:
            runner.build_diagnosis_tree(
                prepared,
                force=args.force,
                prune=not args.no_prune_tree,
                compress=not args.no_compress_tree,
                workers=args.tree_workers,
                passing_cap=args.passing_cap or None,
                mem_budget_gb=args.tree_mem_budget or None,
            )
            print("  - tree: done")

        if stage in {"diagnose", "all"}:
            runner.generate_diagnosis_from_events(
                prepared,
                force=args.force,
                prune=not args.no_prune_tree,
                compress=not args.no_compress_tree,
            )
            print("  - diagnose: done")

        if stage in {"sbfl", "all"}:
            ranking = runner.compute_sbfl(
                prepared, force=args.force, processes=args.tree_workers
            )
            print(f"  - sbfl: {'done' if ranking is not None else 'no traces'}")

    # Clean up Docker system resources after all instances are processed
    try:
        # Same longer timeout as the collection path: a busy daemon must not
        # turn into a spurious instance failure.
        from forecast.subject import docker_client

        client = docker_client()
        prune_result = client.containers.prune()
        removed_count = len(prune_result.get("ContainersDeleted") or [])
        space_freed = prune_result.get("SpaceReclaimed", 0)
        if removed_count > 0:
            print(
                f"\nPruned {removed_count} stopped containers, freed {space_freed // (1024*1024)} MB"
            )
            LOG.info(
                "Final Docker container prune",
                extra={
                    "containers_removed": removed_count,
                    "space_freed_mb": space_freed // (1024 * 1024),
                },
            )
    except Exception as exc:
        LOG.warning(
            "Failed to prune Docker containers after batch", extra={"error": str(exc)}
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
