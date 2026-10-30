"""Analyses for the revision that need no new repair run and no LLM.

  agentless  Agentless on the instances where neither mode failed with an internal error
  bootstrap  95% confidence intervals of the gains: paired bootstrap over instances per
             agent and model, and over instances and over projects for all agents together
  leakage    causes that state a condition the developer patch edits, or a value only
             the patch introduces; candidates for a manual check
  lost       the instances an agent resolves only without the diagnosis, for a manual look
  sample     a random sample of resolved instances with issue, chain, developer patch, and
             the agent's patches, plus a sheet to judge them by hand
  lengths    chain lengths before the cut to ten causes, recomputed from the trees

Everything reads results.json and the diagnoses of the results archive (<archive>/
diagnoses/<id>/diagnosis.json, and <archive>/<model>/<mode>/<id>/run/<agent>/<mode>/
patch.diff for sample); leakage and sample also read SWE-bench Lite (from the `datasets`
cache when offline). lengths needs the traced relevance trees (--trees, the work dir of
evaluation/scripts/diagnose.sh). Output that is meant to be read goes to --out, as
Markdown or JSON; the numbers are also printed.

Usage: python evaluation/revision/offline.py <analysis> [--archive DIR] [--results FILE]
           [--out FILE|DIR] [--model qwen] [--agent mini-swe-agent] [--n 50] [--trees DIR]
"""
import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
sys.path.insert(0, str(HERE))

from compare import mcnemar  # noqa: E402

MODELS = ("haiku", "qwen")
AGENTS = ("mini-swe-agent", "swe-agent", "acr", "repair-agent", "agentless")
DATASET = "princeton-nlp/SWE-bench_Lite"
RESAMPLES = 10_000


def project(instance_id):
    return instance_id.rsplit("-", 1)[0]


def resolved(results, model, agent, mode):
    return {i: bool(r["resolved"]) for i, r in results["repairs"][model][agent][mode].items()}


def chain(archive, instance_id):
    path = archive / "diagnoses" / instance_id / "diagnosis.json"
    return json.loads(path.read_text()) if path.exists() else {}


def cause_texts(data):
    from forecast.integration import _cause_text
    return [_cause_text(c) or "" for c in data.get("causes", [])]


def lite():
    from datasets import load_dataset
    return {r["instance_id"]: r for r in load_dataset(DATASET, split="test")}


def agentless(args, results):
    print("| model | instances | baseline | FORECAST | gained/lost | p |")
    print("|---|---:|---:|---:|---|---|")
    for model in MODELS:
        runs = results["repairs"][model]["agentless"]
        ok = [i for i in runs["baseline"]
              if not runs["baseline"][i].get("error") and not runs["forecast"].get(i, {}).get("error")]
        b = {i: bool(runs["baseline"][i]["resolved"]) for i in ok}
        f = {i: bool(runs["forecast"][i]["resolved"]) for i in ok}
        base, fc = sum(b.values()), sum(f.values())
        gained = sum(f[i] and not b[i] for i in ok)
        lost = sum(b[i] and not f[i] for i in ok)
        print(f"| {model} | {len(ok)} | {base} ({100 * base / len(ok):.1f}%) | {fc} "
              f"({100 * fc / len(ok):.1f}%) | {gained}/{lost} | {mcnemar(gained, lost):.2g} |")


def interval(values):
    values = sorted(values)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def bootstrap(args, results):
    rng = random.Random(0)
    rows = []
    print("| model | agents | gain (pp) | 95% CI, instances | 95% CI, projects |")
    print("|---|---|---:|---|---|")
    for model in MODELS:
        diffs = {}  # agent -> instance -> forecast minus baseline
        for agent in AGENTS:
            base = resolved(results, model, agent, "baseline")
            fc = resolved(results, model, agent, "forecast")
            diffs[agent] = {i: int(fc[i]) - int(base[i]) for i in base}
        ids = sorted(diffs[AGENTS[0]])
        for agent in AGENTS:
            d = [diffs[agent][i] for i in ids]
            samples = [100 * statistics.fmean(rng.choices(d, k=len(d))) for _ in range(RESAMPLES)]
            low, high = interval(samples)
            gain = 100 * statistics.fmean(d)
            print(f"| {model} | {agent} | {gain:+.1f} | [{low:+.1f}, {high:+.1f}] | |")
            rows.append({"model": model, "agent": agent, "gain": gain, "ci_instances": [low, high]})
        # All agents together: the same instance appears once per agent, so instances (or
        # whole projects) are resampled with all their agents.
        per_instance = {i: sum(diffs[a][i] for a in AGENTS) for i in ids}
        gain = 100 * sum(per_instance.values()) / (len(ids) * len(AGENTS))
        samples = []
        for _ in range(RESAMPLES):
            pick = rng.choices(ids, k=len(ids))
            samples.append(100 * sum(per_instance[i] for i in pick) / (len(pick) * len(AGENTS)))
        by_instance = interval(samples)
        projects = defaultdict(list)
        for i in ids:
            projects[project(i)].append(i)
        names = sorted(projects)
        samples = []
        for _ in range(RESAMPLES):
            pick = [i for p in rng.choices(names, k=len(names)) for i in projects[p]]
            samples.append(100 * sum(per_instance[i] for i in pick) / (len(pick) * len(AGENTS)))
        by_project = interval(samples)
        print(f"| {model} | all | {gain:+.1f} | [{by_instance[0]:+.1f}, {by_instance[1]:+.1f}]"
              f" | [{by_project[0]:+.1f}, {by_project[1]:+.1f}] |")
        rows.append({"model": model, "agent": "all", "gain": gain,
                     "ci_instances": list(by_instance), "ci_projects": list(by_project)})
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))


#: The expression of a condition cause, e.g. Condition "token == '+'" at ...
CONDITION = re.compile(r'^Condition "(.+)" at ')
#: Characters a condition needs, without whitespace, to count as found in an edited line.
MIN_EXPRESSION = 8
#: Literal values in a cause: quoted strings and numbers other than 0, 1 and -1.
LITERALS = re.compile(r"""'([^'\n]{2,})'|(?<![\w.])(-?\d+\.\d+|-?\d{2,})(?![\w.])""")


def changed_lines(patch):
    added, removed = [], []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    return added, removed


def normalize(code):
    return re.sub(r"\s+", "", code)


def leakage(args, results):
    dataset = lite()
    conditions, values, examples = set(), set(), []
    for i in sorted(results["diagnoses"]):
        added, removed = changed_lines(dataset[i]["patch"])
        edited = [normalize(line) for line in added + removed]
        before = "\n".join(removed)
        for text in cause_texts(chain(args.archive, i)):
            line = text.split(" (discriminability")[0]
            # Short expressions such as "other" occur in many lines by chance.
            expression = normalize(m.group(1)) if (m := CONDITION.match(line)) else ""
            if len(expression) >= MIN_EXPRESSION and any(expression in e for e in edited):
                conditions.add(i)
                examples.append((i, "condition on an edited line", line))
            for m in LITERALS.finditer(line):
                value = m.group(1) or m.group(2)
                if any(value in a for a in added) and value not in before:
                    values.add(i)
                    examples.append((i, f"value {value!r} only in added lines", line))
    print(f"instances with a cause whose condition the patch edits: {len(conditions)}")
    print(f"instances with a cause stating a value only the patch adds: {len(values)}")
    print(f"either: {len(conditions | values)} of {len(results['diagnoses'])}")
    if args.out:
        rows = "\n".join(f"| {i} | {kind} | `{line}` | |" for i, kind, line in examples)
        args.out.write_text(
            "# Leakage candidates\n\nJudge each: does the cause state (part of) the fix?\n\n"
            "| instance | match | cause | states the fix? |\n|---|---|---|---|\n" + rows + "\n")


def lost(args, results):
    lines = ["# Instances lost with the diagnosis", "",
             "| model | agent | instance | strongest | names fixed file | why lost? |",
             "|---|---|---|---:|---|---|"]
    for model in MODELS:
        for agent in AGENTS:
            base = resolved(results, model, agent, "baseline")
            fc = resolved(results, model, agent, "forecast")
            for i in sorted(i for i in base if base[i] and not fc[i]):
                d = results["diagnoses"][i]
                lines.append(f"| {model} | {agent} | {i} | {d['discriminability']:.2f} | "
                             f"{'yes' if d['names_fixed_file'] else 'no'} | |")
    print(f"{len(lines) - 4} lost runs")
    if args.out:
        args.out.write_text("\n".join(lines) + "\n")


def sample(args, results):
    dataset = lite()
    base = resolved(results, args.model, args.agent, "baseline")
    fc = resolved(results, args.model, args.agent, "forecast")
    pool = sorted(i for i in fc if fc[i])
    picked = sorted(random.Random(0).sample(pool, min(args.n, len(pool))))
    args.out.mkdir(parents=True, exist_ok=True)
    sheet = [f"# Patch check: {args.model} {args.agent}", "",
             "Per instance: is the FORECAST patch semantically correct (same behavior as the "
             "developer patch), plausible but wrong (passes the tests, misses the intent), or "
             "unrelated? Does a cause of the chain lie on the way to the fault?", "",
             "| instance | baseline resolved | FORECAST patch | baseline patch | chain on fault path? |",
             "|---|---|---|---|---|"]
    for i in picked:
        target = args.out / i
        target.mkdir(exist_ok=True)
        (target / "issue.md").write_text(dataset[i]["problem_statement"])
        (target / "developer.diff").write_text(dataset[i]["patch"])
        (target / "diagnosis.txt").write_text("\n".join(cause_texts(chain(args.archive, i))))
        for mode in ("forecast", "baseline"):
            patch = args.archive / args.model / mode / i / "run" / args.agent / mode / "patch.diff"
            if patch.exists():
                (target / f"{mode}.diff").write_text(patch.read_text())
        sheet.append(f"| {i} | {'yes' if base[i] else 'no'} | | | |")
    (args.out / "sheet.md").write_text("\n".join(sheet) + "\n")
    print(f"{len(picked)} of {len(pool)} resolved instances in {args.out}")


def lengths(args, results):
    from forecast.diagnosis import Forecast
    from forecast.tree import load_tree

    out = {}
    for i in sorted(results["diagnoses"]):
        tree = args.trees / i / "diagnosis" / "tree.pkl"
        if not tree.exists():
            continue
        forecast = Forecast(load_tree(tree), max_chain_length=10 ** 9)
        untruncated = len(forecast.derive_diagnosis().causes)
        out[i] = {"initial": len(forecast._extract_cause_chain(forecast.root)),
                  "untruncated": untruncated}
        print(i, out[i])
    for key in ("initial", "untruncated"):
        values = sorted(v[key] for v in out.values())
        if values:
            q = statistics.quantiles(values, n=4)
            print(f"{key}: median {statistics.median(values)}, quartiles {q[0]} and {q[2]}, "
                  f"max {values[-1]}, more than ten in {sum(v > 10 for v in values)} of {len(values)}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("analysis", choices=["agentless", "bootstrap", "leakage", "lost", "sample", "lengths"])
    parser.add_argument("--archive", type=Path, default=Path.home() / "Downloads" / "forecast-results")
    parser.add_argument("--results", type=Path, default=HERE.parent / "results" / "results.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--agent", default="mini-swe-agent")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--trees", type=Path)
    args = parser.parse_args()
    if args.analysis == "sample" and not args.out:
        parser.error("sample writes a folder: pass --out")
    if args.analysis == "lengths" and not args.trees:
        parser.error("lengths recomputes the chains: pass --trees")
    globals()[args.analysis](args, json.loads(args.results.read_text()))


if __name__ == "__main__":
    main()
