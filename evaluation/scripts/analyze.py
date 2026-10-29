"""Answer the research questions of the repair evaluation from results.json.

Sections (all by default, or the ones named):

  overview      per model, agent and mode: resolved, errors, calls, tokens, cost, runtime
  subjects      per project: instances, traced tests, diagnosis length, discriminability and
                time, and the share of resolved repair attempts per model and mode
  diagnoses     distribution of discriminability and cause-chain length
  rq1           (RQ1, RQ2) resolved instances per agent and over all agents: absolute and
                relative change, gained and lost instances, exact two-sided McNemar test;
                instances at least one agent resolves
  rq3           (RQ3) fix quality: patches produced, applicability, patches that resolve,
                changed lines and files
  rq4           (RQ4) efficiency: diagnosis overhead against repair tokens, cost and time
  causes        cause types (atoms, conjunctions, disjunctions, divergences) over all chains
  localization  chains that name a file or line the developer patch changes, and the gain
                with and without such a chain
  selection     how the passing tests to trace were selected
  patches       (RQ3) patches that add files the agent created for itself, and the size of
                the changes to existing files
  exits         how SWE-agent and mini-SWE-agent runs ended (submitted, cost limit, ...)
  truncation    answers cut off at their token limit, per agent

Every run counts; a run without a harness verdict counts as unresolved. Costs use
Anthropic's list prices for Claude Haiku 4.5, never a price a provider or proxy reports:
"cost" is the per-call cost FORECAST computed with prompt caching ($1/M input, $5/M
output, $1.25/M cache writes, $0.10/M cache reads), "no cache" prices every prompt token
as uncached input. Qwen runs locally and has no cost (null in results.json); it is
compared by tokens. Distributions are given as
median (first quartile-third quartile) and mean.

--from DIR first rebuilds results.json from the extracted Zenodo archive (its top folder
forecast-results/) and SWE-bench Lite, as pack.py made it.

Usage: python3 analyze.py [section ...] [--results results.json] [--from DIR]
"""
import argparse, json, math, os, statistics, sys
from collections import Counter, defaultdict

MODELS = ("haiku", "qwen")
MODES = ("baseline", "forecast")
AGENTS = ("mini-swe-agent", "swe-agent", "acr", "repair-agent", "agentless")
NAMES = {"mini-swe-agent": "mini-swe-agent", "swe-agent": "SWE-agent", "acr": "ACR",
         "repair-agent": "RepairAgent", "agentless": "Agentless",
         "haiku": "Claude Haiku 4.5", "qwen": "Qwen3.5 27B"}
SECTIONS = ("overview", "subjects", "diagnoses", "rq1", "rq3", "rq4",
            "causes", "localization", "selection", "patches", "exits", "truncation")
#: Claude Haiku 4.5 list price per token (input, output).
PRICE_IN, PRICE_OUT = 1e-6, 5e-6
#: The diagnosis steps whose times add up to a diagnosis's overhead.
STEPS = ("collect_seconds", "tree_seconds", "diagnose_seconds")


def rebuild(src, out):
    """Write results.json from the extracted archive."""
    from pack import diagnosis, load, repair, swe_bench_lite, write_results
    ids = open(f"{src}/diagnoses/instances.txt").read().split()
    lite = swe_bench_lite()
    diagnoses = {i: diagnosis(f"{src}/diagnoses/{i}", load(f"{src}/diagnoses/{i}/pipeline.json") or {}, lite[i])
                 for i in ids}
    repairs = {}
    for model in MODELS:
        for mode in MODES:
            for agent in AGENTS:
                listing = f"{src}/{model}/{mode}/{agent}.txt"
                if os.path.exists(listing):
                    repairs[model, agent, mode] = {i: repair(f"{src}/{model}/{mode}", agent, mode, i)
                                                   for i in open(listing).read().split()}
    write_results(diagnoses, repairs, out)


def dist(values, fmt="{:,.0f}"):
    """Median (Q1-Q3), mean of the non-missing values."""
    v = [x for x in values if x is not None]
    if not v:
        return "-"
    q = statistics.quantiles(v, n=4, method="inclusive") if len(v) > 1 else [v[0]] * 3
    return f"{fmt.format(q[1])} ({fmt.format(q[0])}-{fmt.format(q[2])}), mean {fmt.format(statistics.mean(v))}"


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def resolved(runs):
    return {i for i, r in runs.items() if r["resolved"]}


def overhead(d):
    """Seconds to compute one diagnosis: collection (with instrumentation), tree, diagnosis."""
    return sum(d[k] or 0 for k in STEPS)


def repo(i):
    """The repository of an instance id, e.g. flask for pallets__flask-4992."""
    return i.split("__")[1].rsplit("-", 1)[0]


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def overview(res):
    print("## Overview\n")
    for m in res["repairs"]:
        print(f"### {NAMES[m]}\n")
        rows = []
        for mode in MODES:
            total = defaultdict(float)
            for agent in AGENTS:
                runs = res["repairs"][m][agent][mode].values()
                s = {k: sum(r[k] or 0 for r in runs)
                     for k in ("resolved", "error", "calls", "prompt_tokens", "completion_tokens", "cost", "seconds")}
                s["list"] = s["prompt_tokens"] * PRICE_IN + s["completion_tokens"] * PRICE_OUT
                s["n"] = len(runs)
                s["priced"] = priced(runs)
                for k, v in s.items():
                    total[k] += v
                rows.append(_overview_row(mode, NAMES[agent], s))
            total["priced"] = all(priced(res["repairs"][m][a][mode].values()) for a in AGENTS)
            rows.append(_overview_row(mode, "**All agents**", total))
        table(("Mode", "Agent", "Resolved", "Errors", "Calls", "Prompt tok", "Completion tok", "Cost",
               "Cost (no cache)", "$/run", "Runtime", "s/run"), rows)


def _overview_row(mode, agent, s):
    n = s["n"]
    return (mode, agent, f"{s['resolved']:.0f}/{n:.0f} ({pct(s['resolved'], n)})", f"{s['error']:.0f}",
            f"{s['calls']:,.0f}", f"{s['prompt_tokens'] / 1e6:,.1f}M", f"{s['completion_tokens'] / 1e6:,.1f}M",
            *((f"${s['cost']:,.2f}", f"${s['list']:,.2f}", f"${s['cost'] / n:,.3f}") if s["priced"] else ("-",) * 3),
            f"{s['seconds'] / 3600:,.1f} h", f"{s['seconds'] / n:,.0f}")


def priced(runs):
    """Whether the runs have a cost, i.e. ran on a model that charges per token."""
    return all(r["cost"] is not None for r in runs)


def tokens(r):
    return r["prompt_tokens"] + r["completion_tokens"]


def subjects(res):
    print("## Subjects\n")
    print("Resolved: share of the repair attempts of all five agents on the project's instances.\n")
    projects = defaultdict(list)
    for i in res["diagnoses"]:
        projects[repo(i)].append(i)
    rows = []
    for p, ids in sorted(projects.items()) + [("**All**", list(res["diagnoses"]))]:
        ds = [res["diagnoses"][i] for i in ids]
        rates = [pct(sum(bool(res["repairs"][m][a][mode][i]["resolved"]) for a in AGENTS for i in ids),
                     len(AGENTS) * len(ids))
                 for m in res["repairs"] for mode in MODES]
        rows.append((p, len(ds), sum(d["failing_tests"] or 0 for d in ds), sum(d["passing_tests"] or 0 for d in ds),
                     dist([d["causes"] for d in ds], "{:.0f}"),
                     dist([d["discriminability"] for d in ds], "{:.2f}"),
                     sum((d["discriminability"] or 0) >= 1 for d in ds),
                     sum(d["causes"] == 0 for d in ds),
                     dist([overhead(d) / 60 for d in ds], "{:.0f}"), *rates))
    table(("Project", "Instances", "Traced failing tests", "Traced passing tests", "Causes",
           "Discriminability", "Discriminability = 1.00", "Empty diagnoses", "Diagnosis time (min)")
          + tuple(f"Resolved {NAMES[m]} {mode}" for m in res["repairs"] for mode in MODES), rows)


def diagnoses(res):
    print("## Diagnoses\n")
    ds = list(res["diagnoses"].values())
    n = len(ds)
    disc = [d["discriminability"] for d in ds if d["causes"]]
    table(("Measure", "Value"), [
        ("Diagnoses", n),
        ("Empty (no cause)", sum(d["causes"] == 0 for d in ds)),
        ("Causes per chain", dist([d["causes"] for d in ds], "{:.0f}")),
        ("Discriminability (non-empty)", dist(disc, "{:.2f}")),
        ("Mean discriminability along the chain", dist([d["mean_discriminability"] for d in ds if d["causes"]], "{:.2f}")),
        ("Discriminability = 1.00", f"{sum(x >= 1 for x in disc)} ({pct(sum(x >= 1 for x in disc), n)})"),
        ("Discriminability >= 0.50", f"{sum(x >= 0.5 for x in disc)} ({pct(sum(x >= 0.5 for x in disc), n)})"),
        # The agents get these with a caveat (RepairAgent.WEAK_DISCRIMINABILITY).
        ("Weak (non-empty, discriminability <= 0.10)", sum(x <= 0.1 for x in disc)),
    ])


def rq1(res):
    print("## RQ1/RQ2: Resolved instances\n")
    tests = 0
    for m in res["repairs"]:
        print(f"### {NAMES[m]}\n")
        rows = []
        for agent in AGENTS:
            runs = res["repairs"][m][agent]
            b, f = resolved(runs["baseline"]), resolved(runs["forecast"])
            n = len(runs["baseline"])
            tests += 1
            name = NAMES[agent]
            rows.append(_rq1_row(name, len(b), len(f), n) +
                        (len(b & f), len(f - b), len(b - f), f"{mcnemar(len(f - b), len(b - f)):.2g}"))
        for label, agents in (("All agents", AGENTS),):
            s = [0] * 6  # baseline, FORECAST, runs, both, only FORECAST, only baseline
            for agent in agents:
                runs = res["repairs"][m][agent]
                b, f = resolved(runs["baseline"]), resolved(runs["forecast"])
                for k, v in enumerate((len(b), len(f), len(runs["baseline"]), len(b & f), len(f - b), len(b - f))):
                    s[k] += v
            rows.append(_rq1_row(f"**{label}**", *s[:3]) + (*s[3:], ""))
        table(("Agent", "Baseline", "FORECAST", "Change", "Change (pp)", "Relative change", "Both",
               "Only FORECAST", "Only baseline", "p (McNemar)"), rows)
        ids = list(res["diagnoses"])
        anyb, anyf = ({i for i in ids if any(res["repairs"][m][a][mode][i]["resolved"] for a in AGENTS)}
                      for mode in MODES)
        print(f"Resolved by at least one agent: baseline {len(anyb)} ({pct(len(anyb), len(ids))}), "
              f"FORECAST {len(anyf)} ({pct(len(anyf), len(ids))}); by some agent only with FORECAST "
              f"{len(anyf - anyb)}, only without {len(anyb - anyf)}\n")
    print(f"Bonferroni-corrected significance level over the {tests} McNemar tests: {0.05 / tests:.3g}\n")


def _rq1_row(name, b, f, n):
    return (name, f"{b}/{n} ({pct(b, n)})", f"{f}/{n} ({pct(f, n)})", f"{f - b:+d}",
            f"{100 * (f - b) / n:+.1f}", f"{100 * (f - b) / b:+.1f}%" if b else "-")


def rq3(res):
    print("## RQ3: Fix quality\n")
    print("Changed lines are added plus removed lines; a patch applies if the SWE-bench harness "
          "applied it cleanly (a run without verdict counts as not applied). Section `patches` "
          "separates the files agents create for themselves.\n")
    for m in res["repairs"]:
        print(f"### {NAMES[m]}: all patches\n")
        rows = []
        for agent in AGENTS:
            for mode in MODES:
                runs = res["repairs"][m][agent][mode].values()
                patches = [r for r in runs if r["patch"]]
                applied = sum(bool(r["applied"]) for r in patches)
                resolving = sum(bool(r["resolved"]) for r in patches)
                rows.append((NAMES[agent], mode, f"{len(patches)}/{len(runs)} ({pct(len(patches), len(runs))})",
                             f"{applied}/{len(patches)} ({pct(applied, len(patches))})",
                             f"{resolving}/{len(patches)} ({pct(resolving, len(patches))})",
                             dist([r["lines_added"] + r["lines_removed"] for r in patches]),
                             dist([r["files"] for r in patches], "{:.1f}")))
        table(("Agent", "Mode", "Patches", "Applied", "Resolving", "Changed lines", "Files"), rows)
        print(f"### {NAMES[m]}: instances both modes resolve\n")
        rows = []
        for agent in AGENTS:
            runs = res["repairs"][m][agent]
            both = sorted(resolved(runs["baseline"]) & resolved(runs["forecast"]))
            row = [NAMES[agent], len(both)]
            for mode in MODES:
                rs = [runs[mode][i] for i in both]
                row += [dist([r["lines_added"] + r["lines_removed"] for r in rs]),
                        dist([r["files"] for r in rs], "{:.1f}")]
            rows.append(row)
        table(("Agent", "Instances", "Changed lines (baseline)", "Files (baseline)",
               "Changed lines (FORECAST)", "Files (FORECAST)"), rows)


def rq4(res):
    print("## RQ4: Efficiency\n")
    overheads = {i: overhead(d) for i, d in res["diagnoses"].items()}
    ds = list(res["diagnoses"].values())
    total = list(overheads.values())
    print("### Diagnosis overhead per instance (seconds; computed once, shared by all agents and models)\n")
    table(("Step", "Per instance", "Max", "Sum", "Share"), [
        (name, dist(v), f"{max(x or 0 for x in v):,.0f}", f"{sum(x or 0 for x in v) / 3600:,.1f} h",
         pct(sum(x or 0 for x in v), sum(total)) if name != "Instrumentation" else "-")
        for name, v in (("Instrumentation", [d["instrument_seconds"] for d in ds]),
                        ("Collection (incl. instrumentation)", [d["collect_seconds"] for d in ds]),
                        ("Tree building", [d["tree_seconds"] for d in ds]),
                        ("Diagnosis", [d["diagnose_seconds"] for d in ds]),
                        ("**Total**", total))])
    print(f"Diagnoses taking under 1 h: {sum(t < 3600 for t in total)}, over 3 h: "
          f"{sum(t > 3 * 3600 for t in total)} (of {len(total)}); per project in section `subjects`\n")
    for m in res["repairs"]:
        print(f"### {NAMES[m]}: repair cost per instance\n")
        rows = []
        for agent in AGENTS:
            for mode in MODES:
                runs = res["repairs"][m][agent][mode]
                ok = sum(bool(r["resolved"]) for r in runs.values())
                has_cost = priced(runs.values())
                # Each FORECAST run is charged its own instance's diagnosis; runs without a
                # recorded runtime (no result.json) are left out.
                timed = [r["seconds"] + (overheads[i] if mode == "forecast" else 0)
                         for i, r in runs.items() if r["seconds"] is not None]
                rows.append((NAMES[agent], mode,
                             dist([tokens(r) for r in runs.values()]),
                             dist([r["cost"] for r in runs.values()], "${:,.3f}"),
                             dist([r["seconds"] for r in runs.values()]),
                             dist(timed),
                             f"{sum(tokens(r) for r in runs.values()) / ok:,.0f}" if ok else "-",
                             f"${sum(r['cost'] for r in runs.values()) / ok:,.3f}" if ok and has_cost else "-",
                             f"{sum(timed) / ok:,.0f}" if ok else "-"))
        table(("Agent", "Mode", "Tokens", "Cost", "Repair time (s)", "Time incl. diagnosis (s)",
               "Tokens per resolved", "Cost per resolved", "Time per resolved incl. diagnosis (s)"), rows)
    print("### LLM tokens and cost over all agents\n")
    rows = []
    for m in res["repairs"]:
        for label, agents in (("All agents", AGENTS),):
            runs = {mode: [r for a in agents for r in res["repairs"][m][a][mode].values()] for mode in MODES}
            row = [NAMES[m], label]
            for measure in ("tokens", "cost"):
                if measure == "cost" and not all(priced(rs) for rs in runs.values()):
                    row += ["-"] * 6
                    continue
                total = {mode: sum(tokens(r) if measure == "tokens" else r["cost"] for r in rs)
                         for mode, rs in runs.items()}
                per = {mode: total[mode] / max(1, sum(bool(r["resolved"]) for r in rs)) for mode, rs in runs.items()}
                fmt = "{:,.1f}M" if measure == "tokens" else "${:,.2f}"
                scale = 1e6 if measure == "tokens" else 1
                for values in (total, per):
                    row += [fmt.format(values["baseline"] / scale), fmt.format(values["forecast"] / scale),
                            f"{100 * (values['forecast'] / values['baseline'] - 1):+.1f}%"]
            rows.append(row)
    table(("Model", "Agents") + tuple(f"{what} ({col})" for what in
                                      ("Tokens", "Tokens per resolved", "Cost", "Cost per resolved")
                                      for col in ("baseline", "FORECAST", "change")), rows)


def causes(res):
    print("## Cause types\n")
    kinds = Counter()
    for d in res["diagnoses"].values():
        kinds.update(d["cause_types"])
    n = sum(kinds.values())
    chains = [d for d in res["diagnoses"].values() if d["causes"]]
    divergent = sum(bool(d["cause_types"].get("divergence")) for d in chains)
    table(("Cause type", "Causes", "Share"),
          [(k, v, pct(v, n)) for k, v in kinds.most_common()] + [("**All**", n, pct(n, n))])
    print(f"Chains with at least one divergence: {divergent} of {len(chains)} non-empty chains\n")


def localization(res):
    print("## Localization: diagnoses against the developer patch\n")
    print("A chain names a fixed file if a cause mentions a file the SWE-bench Lite patch changes, "
          "and a fixed line if it names a line of such a file within one of the patch's hunks "
          "(context lines included). FORECAST never sees the patch.\n")
    ds = res["diagnoses"]
    files, lines = (sum(d[k] for d in ds.values()) for k in ("names_fixed_file", "names_fixed_line"))
    print(f"Chains naming a fixed file: {files} ({pct(files, len(ds))}); "
          f"a fixed line: {lines} ({pct(lines, len(ds))}) of {len(ds)}\n")
    rows = []
    for m in res["repairs"]:
        for label, agents in (("All agents", AGENTS),):
            for group, named in (("names a fixed file", True), ("names no fixed file", False)):
                ids = [i for i, d in ds.items() if d["names_fixed_file"] == named]
                b, f = (sum(bool(res["repairs"][m][a][mode][i]["resolved"]) for a in agents for i in ids)
                        for mode in MODES)
                runs = len(agents) * len(ids)
                rows.append((NAMES[m], label, group, len(ids), pct(b, runs), pct(f, runs),
                             f"{100 * (f - b) / runs:+.1f}"))
    table(("Model", "Agents", "Chain", "Instances", "Resolved (baseline)", "Resolved (FORECAST)",
           "Change (pp)"), rows)


def selection(res):
    print("## Test selection\n")
    ds = res["diagnoses"].values()
    strategies = Counter((d["selection"], d["selection_reason"] or "-") for d in ds)
    table(("Strategy", "Reason", "Instances"), [(s, r, n) for (s, r), n in strategies.most_common()])
    print(f"Instances listing no PASS_TO_PASS test (passing tests recovered from the repository): "
          f"{sum(d['listed_passing_tests'] == 0 for d in ds)}; selections that consulted the developer "
          f"patch: {sum(d['selection_uses_patch'] for d in ds)}; trees built with "
          f"{dict(Counter(d['tree_workers'] for d in ds))} worker(s)\n")


def patches(res):
    print("## Patches: new files and changes to existing files\n")
    print("New files are files a patch adds, which the agents create for themselves (reproduction "
          "and debugging scripts, notes); changed lines and files here count only existing files.\n")
    for m in res["repairs"]:
        print(f"### {NAMES[m]}\n")
        rows = []
        for agent in AGENTS:
            runs = res["repairs"][m][agent]
            both = resolved(runs["baseline"]) & resolved(runs["forecast"])
            for mode in MODES:
                ps = {i: r for i, r in runs[mode].items() if r["patch"]}
                rows.append((NAMES[agent], mode, len(ps), sum(r["new_files"] > 0 for r in ps.values()),
                             dist([r["existing_lines"] for r in ps.values()]),
                             dist([r["existing_files"] for r in ps.values()], "{:.1f}"),
                             dist([r["lines_added"] + r["lines_removed"] for r in ps.values()]),
                             dist([ps[i]["existing_lines"] for i in both if i in ps])))
        table(("Agent", "Mode", "Patches", "With new files", "Changed lines (existing files)",
               "Files (existing)", "Changed lines (all files)", "Changed lines (existing), both resolve"), rows)


def exits(res):
    print("## How SWE-agent and mini-SWE-agent runs end\n")
    print("SWE-agent submits the changes made so far when a run reaches its cost limit "
          "(`submitted (exit_cost)`); mini-SWE-agent submits nothing (`LimitsExceeded`).\n")
    for m in res["repairs"]:
        rows = []
        for agent in ("swe-agent", "mini-swe-agent"):
            for mode in MODES:
                runs = res["repairs"][m][agent][mode].values()
                for status, n in Counter(r["exit_status"] for r in runs).most_common():
                    rs = [r for r in runs if r["exit_status"] == status]
                    rows.append((NAMES[agent], mode, status, n, sum(bool(r["resolved"]) for r in rs),
                                 sum(r["new_files"] > 0 for r in rs)))
        print(f"### {NAMES[m]}\n")
        table(("Agent", "Mode", "Exit status", "Runs", "Resolved", "Patches with new files"), rows)


def truncation(res):
    print("## Answers cut off at their token limit\n")
    print("Per answer length an agent requested and the one the model got: answers, and those that "
          "stopped at that length. Agentless requests 300 tokens for localization and 1024 for "
          "repair and tests; FORECAST raises both to 4096 (in the Haiku baseline, not yet).\n")
    for agent in AGENTS:
        rows = []
        for m in res["repairs"]:
            for mode in MODES:
                runs = res["repairs"][m][agent][mode].values()
                limits = defaultdict(lambda: [0, 0])
                for r in runs:
                    for limit, (n, cut) in r["max_tokens"].items():
                        limits[limit][0] += n
                        limits[limit][1] += cut
                stopped = sum(any(cut for _, cut in r["max_tokens"].values()) for r in runs)
                for limit, (n, cut) in sorted(limits.items()):
                    rows.append((NAMES[m], mode, limit, n, f"{cut} ({pct(cut, n)})", stopped))
        print(f"### {NAMES[agent]}\n")
        table(("Model", "Mode", "Limit (requested->given)", "Answers", "Cut off", "Runs with a cut-off answer"), rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sections", nargs="*", metavar="section")
    p.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "results", "results.json"))
    p.add_argument("--from", dest="src", metavar="DIR")
    a = p.parse_args()
    for s in a.sections:
        if s not in SECTIONS:
            p.error(f"unknown section {s!r}; choose from {', '.join(SECTIONS)}")
    if a.src:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        rebuild(a.src, a.results)
    with open(a.results) as f:
        res = json.load(f)
    for s in a.sections or SECTIONS:
        globals()[s](res)


if __name__ == "__main__":
    main()
