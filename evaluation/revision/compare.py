"""Compare the runs of a variant or repetition folder with the evaluated baseline and FORECAST.

Reads the verdicts of <dir>/<model>/<id>/evaluate/<agent>/<mode>/result.json and compares
them, instance by instance, with the evaluated runs in results.json (the paper's numbers).
Only instances whose run has finished in <dir> are compared, on both sides, so the numbers
are meaningful while a run is still going; a run without a verdict counts as unresolved,
as in the paper. Where variants.py found that only some instances get a different prompt
(changed.txt), the others take the evaluated outcome of the run whose prompt they share:
the FORECAST run, or the baseline run for those in as_baseline.txt. The row then covers
all 300 instances once the changed ones have finished, and a second row gives the
changed ones alone. For each mode found in <dir>, it prints a Markdown row: finished runs,
resolved runs, and the paired difference to the evaluated baseline and FORECAST mode
(change in resolved runs, gained / lost instances, exact McNemar p). A folder with both
modes (repeat.sh) also gets the gain of its own FORECAST over its own baseline.

Usage: python evaluation/revision/compare.py <dir> --model qwen --agent mini-swe-agent
           [--results evaluation/results/results.json] [--label NAME]
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTANCES = HERE.parent / "scripts" / "instances.txt"


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def verdicts(folder, model, agent, mode, ids):
    """Instance -> resolved, for the runs of *mode* that have finished in *folder*."""
    out = {}
    for i in ids:
        verdict = folder / model / i / "evaluate" / agent / mode / "result.json"
        run = folder / model / i / "run" / agent / mode / "result.json"
        if verdict.exists():
            out[i] = bool(json.loads(verdict.read_text()).get("resolved"))
        elif run.exists():
            out[i] = False
    return out


def versus(mine, theirs):
    common = [i for i in mine if i in theirs]
    gained = sum(mine[i] and not theirs[i] for i in common)
    lost = sum(theirs[i] and not mine[i] for i in common)
    p = mcnemar(gained, lost)
    return f"{gained - lost:+d} ({gained}/{lost}, p={p:.2g})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", type=Path)
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--results", type=Path, default=HERE.parent / "results" / "results.json")
    parser.add_argument("--label")
    args = parser.parse_args()

    ids = INSTANCES.read_text().split()
    reference = json.loads(args.results.read_text())["repairs"][args.model][args.agent]
    evaluated = {mode: {i: bool(r["resolved"]) for i, r in reference[mode].items()}
                 for mode in ("baseline", "forecast")}
    label = args.label or args.dir.name
    runs = {mode: verdicts(args.dir, args.model, args.agent, mode, ids)
            for mode in ("baseline", "forecast")}
    changed_only = None
    if (args.dir / "changed.txt").exists() and runs["forecast"]:
        changed = set((args.dir / "changed.txt").read_text().split())
        as_baseline = set((args.dir / "as_baseline.txt").read_text().split()) \
            if (args.dir / "as_baseline.txt").exists() else set()
        changed_only = {i: r for i, r in runs["forecast"].items() if i in changed}
        reused = {i: evaluated["baseline" if i in as_baseline else "forecast"][i]
                  for i in ids if i not in changed}
        if len(changed_only) == len(changed):
            runs["forecast"] = {**changed_only, **reused}

    print(f"| {args.model} {args.agent} | finished | resolved | vs. evaluated baseline"
          " | vs. evaluated FORECAST | own gain |")
    print("|---|---:|---:|---|---|---|")
    for mode, mine in runs.items():
        if not mine:
            continue
        own = versus(runs["forecast"], runs["baseline"]) if mode == "forecast" and runs["baseline"] else ""
        resolved = sum(mine.values())
        print(f"| {label} ({mode}) | {len(mine)}/{len(ids)} | {resolved}"
              f" ({100 * resolved / len(mine):.1f}%) | {versus(mine, evaluated['baseline'])}"
              f" | {versus(mine, evaluated['forecast'])} | {own} |")
    if changed_only:
        resolved = sum(changed_only.values())
        print(f"| {label} (changed only) | {len(changed_only)}/{len(changed)} | {resolved}"
              f" ({100 * resolved / len(changed_only):.1f}%)"
              f" | {versus(changed_only, evaluated['baseline'])}"
              f" | {versus(changed_only, evaluated['forecast'])} | |")
    if not any(runs.values()):
        print(f"no finished {args.agent} runs in {args.dir / args.model}")
    for mode, ref in evaluated.items():
        print(f"\nevaluated {mode}: {sum(ref.values())}/{len(ref)} resolved", end="")
    print()


if __name__ == "__main__":
    main()
