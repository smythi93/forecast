"""Build the diagnoses of an ablation variant as a runs folder the repair runs read.

A FORECAST-mode run reads <runs dir>/<model>/<id>/diagnosis/diagnosis.json and injects
its text; ACR and RepairAgent also read sbfl.json next to it. This script writes such
folders for a variant of the diagnosis, so a variant runs with the unchanged agents:

  bash evaluation/revision/run_variant.sh <out dir> <model> <agent> [workers]

Variants made from the stored diagnoses (<runs dir> holds the evaluated ones):
  full       the evaluated diagnosis, unchanged (for repeated runs)
  foreign    the chain of another instance of the same project (negative control)
  unordered  the same causes in a random order, as a list without LEADS TO
  locations  only the files, lines and functions each cause names, with its score
  sbfl       the ten methods spectrum-based fault localization ranks highest
  testinfo   the failing tests and their output on the unchanged code
             (--failures, written by failures.py)
  wording    the same chain, introduced and joined in neutral terms

Variants recomputed from the traced relevance trees, for the instances whose evaluated
chain has what the variant takes away; the others keep their evaluated diagnosis. The
chains come from recompute.py, run where the trees are (--recomputed, its out dir):
  nodiv      without divergences, for the chains with a divergence
  atoms      without conjunctions and disjunctions, for the chains with one

Variants recomputed from the traced relevance trees here (--trees, the work dir of
evaluation/scripts/diagnose.sh, with <trees>/<id>/diagnosis/tree.pkl):
  recomputed the default chain, to check that the trees reproduce the stored chains
  k<N>       at most N causes instead of 10, e.g. k5, k20
  t<X>       discriminability threshold X instead of 0.6, e.g. t0.4, t0.8

Variants that inject something other than a cause chain also write preamble.txt, the
sentence that introduces the block; run_variant.sh passes it to the agents. Every out
dir gets variant.json, which records how it was made, and changed.txt, the instances
to run: every instance with a diagnosis block, an empty chain included, and for nodiv,
atoms and the variants recomputed here, only those whose chain the variant changes.
run_variant.sh runs them alone, and compare.py takes the evaluated outcome for the
others, whose prompt is the same as in the evaluated FORECAST run, or, for the instances
in as_baseline.txt, where no diagnosis is left to inject, as in the evaluated baseline
run. foreign also writes donors.json.

Usage: python evaluation/revision/variants.py <variant> <runs dir> <out dir>
           [--model qwen] [--trees DIR] [--repos DIR] [--recomputed DIR] [--failures FILE]
           [--seed 0]
"""
import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from forecast.integration import _cause_text, diagnosis_text_from_json  # noqa: E402

INSTANCES = HERE.parent / "scripts" / "instances.txt"
SEPARATOR = "\n LEADS TO \n"
#: A file and a line or function, as cause texts name them: expr/parse.py:66,
#: expr/arithmetic.py:Sub.evaluate.
LOCATION = re.compile(r"[\w./-]+\.py:[\w<>]+(?:\.[\w<>]+)*")
PREAMBLES = {
    "sbfl": "The following methods are ranked most suspicious by spectrum-based fault "
    "localization (Ochiai) over the failing and passing tests:",
    "testinfo": "The following tests fail on the current code, with their output:",
    "wording": "The following observations distinguish the failing from the passing test "
    "runs, in the order in which they occur during the execution:",
}
#: Characters of test output kept per instance, about the size of a long chain.
TESTINFO_CHARS = 4000


def score(cause):
    value = cause.get("discriminability")
    return f" (discriminability: {value:.2f})" if isinstance(value, (int, float)) else ""


def project(instance_id):
    return instance_id.rsplit("-", 1)[0]


def stored(runs, model, instance_id):
    path = runs / model / instance_id / "diagnosis" / "diagnosis.json"
    return json.loads(path.read_text()) if path.exists() else None


def text_of(text):
    """A stored diagnosis with *text*; with ``None``, one that leaves the issue unchanged."""
    if text is None:
        return {"type": "none"}
    return {"type": "cause_chain", "text": text, "causes": []}


def foreign(ids, diagnoses):
    """Each instance's donor: the instance half a project away, or the next one overall."""
    usable = [i for i in ids if diagnosis_text_from_json(diagnoses[i])]
    donors = {}
    for i in ids:
        group = [j for j in usable if project(j) == project(i) and j != i]
        if group:
            group = sorted(group + [i])
            donors[i] = group[(group.index(i) + max(1, len(group) // 2)) % len(group)]
        else:
            others = [j for j in usable if project(j) != project(i)]
            donors[i] = others[(sorted(ids).index(i)) % len(others)]
    return donors


def unordered(i, data, seed):
    causes = [c for c in data.get("causes", []) if _cause_text(c)]
    order = list(range(len(causes)))
    rng = random.Random(f"{seed}:{i}")
    for _ in range(100):
        rng.shuffle(order)
        if len(order) < 2 or order != sorted(order):
            break
    return "\n".join(f"- {_cause_text(causes[k])}{score(causes[k])}" for k in order)


def locations(data):
    lines = []
    for index, cause in enumerate(data.get("causes", []), start=1):
        found = list(dict.fromkeys(LOCATION.findall(_cause_text(cause) or "")))
        if found:
            lines.append(f"{index}. {', '.join(found)}{score(cause)}")
    return SEPARATOR.join(lines)


#: The cause types a recomputed variant takes away.
REMOVES = {"nodiv": {"divergence"}, "atoms": {"conjunction", "disjunction"}}


def recomputed(variant, i, data, folder):
    """The chain recompute.py derived for *i* if the evaluated one has what *variant* removes."""
    if not REMOVES[variant] & {c.get("type") for c in data.get("causes", [])}:
        return data
    path = folder / variant / i / "diagnosis.json"
    if not path.exists():
        sys.exit(f"{variant}: no recomputed diagnosis for {i} in {folder}")
    return json.loads(path.read_text())


def sbfl(path, top=10):
    if not path.exists():
        return None
    ranking = json.loads(path.read_text())[:top]
    return "\n".join(
        f"{k}. {m['file']}:{m['method']} (line {m['line']}), suspiciousness {m['suspiciousness']:.2f}"
        for k, m in enumerate(ranking, start=1)
    )


def testinfo(entry):
    if not entry:
        return None
    tests = "\n".join(f"- {t}" for t in entry["tests"])
    output = entry["output"]
    if len(output) > TESTINFO_CHARS:
        output = output[:TESTINFO_CHARS] + "\n[... output truncated ...]"
    return f"Failing tests:\n{tests}\n\nOutput:\n```\n{output}\n```"


def recompute(i, trees, repos, **options):
    from forecast.diagnosis import Forecast, FunctionNames
    from forecast.integration import DiagnosisSerializer
    from forecast.tree import load_tree

    root = trees / i / "diagnosis"
    chain = Forecast(load_tree(root / "tree.pkl"), **options).derive_diagnosis()
    mapping, repo = root / "events" / "mapping.json", repos / i
    if repo.is_dir() and mapping.exists():
        chain.qualify_functions(FunctionNames.from_mapping(repo, mapping))
    return DiagnosisSerializer.to_json(chain)


def options_of(variant):
    if variant == "recomputed":
        return {}
    if m := re.fullmatch(r"k(\d+)", variant):
        return {"max_chain_length": int(m.group(1))}
    if m := re.fullmatch(r"t(0?\.\d+)", variant):
        return {"min_discriminability_threshold": float(m.group(1))}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("variant")
    parser.add_argument("runs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--trees", type=Path)
    parser.add_argument("--repos", type=Path, help="checkouts, default <trees>/repos")
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--recomputed", type=Path, help="out dir of recompute.py")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ids = INSTANCES.read_text().split()
    diagnoses = {i: stored(args.runs, args.model, i) for i in ids}
    missing = [i for i, d in diagnoses.items() if d is None]
    if missing:
        sys.exit(f"{len(missing)} instances have no stored diagnosis, e.g. {missing[0]}")
    options = options_of(args.variant)
    if options is not None and not args.trees:
        sys.exit(f"{args.variant} is recomputed from the trees: pass --trees")
    if args.variant in REMOVES and not args.recomputed:
        sys.exit(f"{args.variant} takes the chains of recompute.py: pass --recomputed")
    if args.variant == "testinfo" and not args.failures:
        sys.exit("testinfo needs --failures (see failures.py)")
    failures = json.loads(args.failures.read_text()) if args.failures else {}
    donors = foreign(ids, diagnoses) if args.variant == "foreign" else {}

    same, empty, changed = 0, [], []
    for i in ids:
        data = diagnoses[i]
        if args.variant == "full":
            new = data
        elif args.variant == "foreign":
            new = text_of(diagnosis_text_from_json(diagnoses[donors[i]]))
            new["discriminability"] = diagnoses[donors[i]].get("discriminability")
        elif args.variant == "unordered":
            new = text_of(unordered(i, data, args.seed))
        elif args.variant == "locations":
            new = text_of(locations(data))
        elif args.variant == "sbfl":
            new = text_of(sbfl(args.runs / args.model / i / "diagnosis" / "sbfl.json"))
        elif args.variant == "testinfo":
            new = text_of(testinfo(failures.get(i)))
        elif args.variant in REMOVES:
            new = recomputed(args.variant, i, data, args.recomputed)
        elif args.variant == "wording":
            new = text_of(diagnosis_text_from_json(data).replace(SEPARATOR, "\n THEN \n"))
        elif options is not None:
            new = recompute(i, args.trees, args.repos or args.trees / "repos", **options)
            same += new.get("text") == diagnosis_text_from_json(data)
        else:
            sys.exit(f"unknown variant {args.variant}")
        if new.get("type") == "none":
            empty.append(i)
        elif (args.variant not in REMOVES and options is None
              or diagnosis_text_from_json(new) != diagnosis_text_from_json(data)):
            changed.append(i)

        source, target = args.runs / args.model / i / "diagnosis", args.out / args.model / i / "diagnosis"
        target.mkdir(parents=True, exist_ok=True)
        (target / "diagnosis.json").write_text(json.dumps(new, indent=2))
        (target / "diagnosis.txt").write_text(diagnosis_text_from_json(new) or "")
        if (source / "sbfl.json").exists():
            shutil.copy2(source / "sbfl.json", target / "sbfl.json")

    if args.variant in PREAMBLES:
        (args.out / "preamble.txt").write_text(PREAMBLES[args.variant])
    if donors:
        (args.out / "donors.json").write_text(json.dumps(donors, indent=2))
    (args.out / "changed.txt").write_text("".join(f"{i}\n" for i in changed))
    (args.out / "as_baseline.txt").write_text("".join(f"{i}\n" for i in empty))
    (args.out / "variant.json").write_text(json.dumps(
        {"variant": args.variant, "model": args.model, "source": str(args.runs.resolve()),
         "seed": args.seed, "options": options, "preamble": PREAMBLES.get(args.variant),
         "without_text": empty}, indent=2))
    print(f"{args.variant}: {len(ids)} diagnoses in {args.out / args.model}, "
          f"{len(changed)} to run (changed.txt)")
    if empty:
        print(f"  {len(empty)} without a block, prompted as in the baseline (as_baseline.txt),"
              f" e.g. {empty[:3]}")
    if options is not None:
        print(f"  {same}/{len(ids)} texts identical to the stored chains")


if __name__ == "__main__":
    main()
