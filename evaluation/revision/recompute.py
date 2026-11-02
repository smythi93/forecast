"""Recompute diagnoses from the traced relevance trees, where the trees are.

The trees of the evaluation stay on the machines that built them (the work dir of
evaluation/scripts/diagnose.sh, <trees>/<id>/diagnosis/tree.pkl, with the checkouts in
<trees>/repos/<id>). This script loads each tree once and derives the chain without
divergences (nodiv), without conjunctions and disjunctions (atoms), with the defaults
(recomputed), and with other parameters (k<N>, t<X>, see variants.py), into
<out dir>/<variant>/<id>/diagnosis.json. variants.py --recomputed then takes them for the
instances a variant changes. With --lengths, it also writes the chain lengths before the
cut to ten causes into <out dir>/lengths/<id>.json (offline.py lengths). No LLM is
involved, and the trees are only read.

Usage: python evaluation/revision/recompute.py <trees dir> <ids file> <out dir>
           [--repos DIR] [--variants nodiv,atoms,recomputed] [--lengths]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
sys.path.insert(0, str(HERE))

from forecast.diagnosis import Forecast, FunctionNames  # noqa: E402
from forecast.integration import DiagnosisSerializer  # noqa: E402
from forecast.tree import load_tree  # noqa: E402
from variants import options_of  # noqa: E402

OPTIONS = {"nodiv": {"divergences": False}, "atoms": {"combine": False}, "recomputed": {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trees", type=Path)
    parser.add_argument("ids", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repos", type=Path, help="checkouts, default <trees>/repos")
    parser.add_argument("--variants", default="nodiv,atoms,recomputed")
    parser.add_argument("--lengths", action="store_true")
    args = parser.parse_args()
    variants = args.variants.split(",")
    options = {v: OPTIONS[v] if v in OPTIONS else options_of(v) for v in variants}
    unknown = [v for v, o in options.items() if o is None]
    if unknown:
        sys.exit(f"unknown variants {unknown}")
    repos = args.repos or args.trees / "repos"

    for i in args.ids.read_text().split():
        root = args.trees / i / "diagnosis"
        targets = {v: args.out / v / i / "diagnosis.json" for v in variants}
        lengths = args.out / "lengths" / f"{i}.json"
        if not (root / "tree.pkl").exists() or (all(t.exists() for t in targets.values())
                                                and (lengths.exists() or not args.lengths)):
            continue
        start = time.perf_counter()
        tree = load_tree(root / "tree.pkl")
        mapping, repo = root / "events" / "mapping.json", repos / i
        names = FunctionNames.from_mapping(repo, mapping) if repo.is_dir() and mapping.exists() else None
        for variant, target in targets.items():
            chain = Forecast(tree, **options[variant]).derive_diagnosis()
            if names:
                chain.qualify_functions(names)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(DiagnosisSerializer.to_json(chain), indent=2))
        if args.lengths:
            forecast = Forecast(tree, max_chain_length=10 ** 9)
            lengths.parent.mkdir(parents=True, exist_ok=True)
            lengths.write_text(json.dumps({
                "initial": len(forecast._extract_cause_chain(forecast.root)),
                "untruncated": len(forecast.derive_diagnosis().causes)}))
        print(f"{i}: {time.perf_counter() - start:.0f}s{'' if names else ' (names unqualified)'}",
              flush=True)


if __name__ == "__main__":
    main()
