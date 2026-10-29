"""Pack the repair-evaluation results into one .tar.zst archive.

It needs only the evaluation's instance folders, <runs dir>/<model>/<instance>/, and only
reads them. Files are archived byte for byte, unless --rewrite OLD=NEW asks to replace a
string in them; their tar headers carry no owner, no time stamp and a fixed mode, so
packing the same results twice gives the same archive. Everything lies below one top
folder, <name>/, in the archive. The output directory gets

  <name>.tar.zst  the archive: results.json, the files given with --add, the diagnoses,
                  and every run of every model, mode and agent
  results.json    per instance the diagnosis (discriminability, causes and their types,
                  whether they name code the developer patch changes, test selection,
                  pipeline times) and per model, agent, mode and instance the repair
                  (verdict, error, patch size with and without the files the agent added,
                  exit status, calls, tokens, answers cut off at their token limit, cost,
                  runtime); see README.md
  manifest.json   per part of the archive: files, raw bytes, rewritten occurrences,
                  forbidden hits
  SHA256SUMS      the archive's checksum
  parts/          the archive's parts, one per model, mode and agent plus the diagnoses

Each part is a run of tar entries compressed as its own zstd frame, so the parts are
packed in parallel; the archive is their concatenation followed by the tar end marker,
which zstd decompresses as one stream and tar reads as one archive.

--forbid STRING and --forbid-env NAME (the value of an environment variable, which is
never printed) count occurrences of a string in every archived file after any rewrite.
Every archived file is also checked for dates written at run time, by the same rules
forecast.run_dates removed them with after each run (patches are exempt). Any hit is
reported (dated files are listed in manifest.json) and makes the exit status 1. Parts that already exist are kept, so an
interrupted run resumes; --force packs everything again.

The developer patches and test lists come from the SWE-bench Lite dataset (--dataset,
loaded with the `datasets` package, from its cache when offline); FORECAST never used the
patches, results.json only compares the diagnoses with them.

Usage: python3 pack.py <runs dir> <out dir> [--name NAME] [--add FILE]... [--jobs N]
           [--rewrite OLD=NEW]... [--forbid STRING]... [--forbid-env NAME]... [--force]
           [--dataset NAME]
"""
import argparse, hashlib, json, os, re, subprocess, sys, tarfile
from collections import Counter
from multiprocessing import Pool

# Inside the repository, use its FORECAST; elsewhere the installed one.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if os.path.isdir(os.path.join(_SRC, "forecast")):
    sys.path.insert(0, _SRC)
from forecast.run_dates import has_run_dates

MODELS = ("haiku", "qwen")
MODES = ("baseline", "forecast")
AGENTS = ("mini-swe-agent", "swe-agent", "acr", "repair-agent", "agentless")
#: The diagnoses (and the report with their pipeline times) are the same for both models;
#: they are taken from this model's folder.
DIAGNOSES = "haiku"
#: The report steps that produce a diagnosis.
STEPS = ("collect", "tree", "diagnose")
ZSTD = ("zstd", "-19", "--long=27", "-T8", "-q", "-f")
BLOCK, RECORD = 512, 10240
DATASET = "princeton-nlp/SWE-bench_Lite"


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def patch_size(patch):
    """Added lines, removed lines and files of a unified diff."""
    added = removed = files = 0
    for line in patch.splitlines():
        if line.startswith("+++ "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed, files


def diff_files(patch):
    """(new file, added lines, removed lines) per file of a unified diff."""
    for chunk in re.split(r"^(?=diff --git )", patch, flags=re.M):
        if chunk.startswith("diff --git"):
            lines = chunk.splitlines()
            yield (any(l.startswith("new file mode") or l == "--- /dev/null" for l in lines),
                   sum(l.startswith("+") and not l.startswith("+++") for l in lines),
                   sum(l.startswith("-") and not l.startswith("---") for l in lines))


def hunks(patch):
    """File -> lines of the original file that a unified diff's hunks span (context included)."""
    files, current = {}, None
    for line in patch.splitlines():
        if m := re.match(r"^--- a/(\S+)", line):
            current = files.setdefault(m.group(1), set())
        elif (m := re.match(r"^@@ -(\d+)(?:,(\d+))? ", line)) and current is not None:
            start = int(m.group(1))
            current.update(range(start, start + max(int(m.group(2) or 1), 1)))
    return files


def swe_bench_lite(name=DATASET):
    """Instance id -> SWE-bench Lite test instance."""
    from datasets import load_dataset
    return {r["instance_id"]: r for r in load_dataset(name, split="test")}


def repair(src, agent, mode, i):
    """The results.json entry of one repair run; `src` holds `<i>/run` and `<i>/evaluate`.

    Files a patch adds are the agent's own (reproduction and debugging scripts, notes), so the
    patch's size is also given for the files that existed before. `exit_status` is how the
    agent's run ended, where the agent reports it (SWE-agent, mini-SWE-agent); `max_tokens`
    gives, per answer length the agent requested and the one the model got, the answers and
    how many of them stopped at that length. `cost` is null for a model served locally: its
    calls carry another model's prices (run_info.json's price_as) only so that the agents'
    budgets work, and cost nothing."""
    verdict = load(f"{src}/{i}/evaluate/{agent}/{mode}/result.json") or {}
    result = load(f"{src}/{i}/run/{agent}/{mode}/result.json") or {}
    try:
        with open(f"{src}/{i}/run/{agent}/{mode}/patch.diff", errors="replace") as f:
            patch = f.read()
    except OSError:
        patch = ""
    added, removed, files = patch_size(patch)
    changed = list(diff_files(patch))
    existing = [a + d for new, a, d in changed if not new]
    outcome = load(f"{src}/{i}/run/{agent}/{mode}/trajs/{i}/outcome.json") or {}
    report = verdict.get("report") or {}
    r = dict(resolved=bool(verdict["resolved"]) if "resolved" in verdict else None,
             error=result.get("error") is not None,
             patch=bool(patch.strip()),
             applied=report.get("patch_successfully_applied"),
             lines_added=added, lines_removed=removed, files=files,
             new_files=sum(new for new, _, _ in changed),
             existing_lines=sum(existing), existing_files=len(existing),
             exit_status=(outcome.get("info") or {}).get("exit_status"),
             calls=0, prompt_tokens=0, completion_tokens=0, cost=0.0, max_tokens={},
             seconds=result.get("elapsed_seconds"))
    calls = f"{src}/{i}/run/{agent}/{mode}/trajs/{i}/calls.jsonl"
    if os.path.exists(calls):
        for line in open(calls):
            c = json.loads(line)
            r["calls"] += 1
            r["prompt_tokens"] += c.get("prompt_tokens") or 0
            r["completion_tokens"] += c.get("completion_tokens") or 0
            r["cost"] += c.get("cost") or 0
            answers = r["max_tokens"].setdefault(f"{c.get('requested_max_tokens')}->{c.get('max_tokens')}", [0, 0])
            answers[0] += 1
            answers[1] += c.get("finish_reason") == "length"
    info = load(f"{src}/{i}/run/{agent}/{mode}/trajs/{i}/run_info.json") or {}
    priced = info.get("price_as") in (None, info.get("model"))
    r["cost"] = round(r["cost"], 6) if priced else None
    return r


def pipeline(report):
    """The diagnosis steps of an instance's report.json."""
    return {k: report["steps"][k] for k in STEPS if k in (report or {}).get("steps", {})}


def diagnosis(folder, steps, instance):
    """The results.json entry of one diagnosis; `folder` holds diagnosis.json, `instance` is
    the SWE-bench Lite instance.

    A chain names a fixed file if a cause mentions a file the developer patch changes, and a
    fixed line if it names a line of such a file within one of the patch's hunks."""
    d = load(f"{folder}/diagnosis.json") or {}
    causes = d.get("causes") or []
    text = json.dumps(causes)
    fixed = hunks(instance["patch"])
    collect = steps.get("collect", {})
    timing = collect.get("timing", {})
    reason = (timing.get("selection_statistics") or {}).get("reason")
    passing = instance["PASS_TO_PASS"]
    return dict(discriminability=d.get("discriminability"),
                mean_discriminability=d.get("mean_discriminability"),
                causes=len(causes),
                cause_types=dict(Counter(c.get("type") for c in causes)),
                names_fixed_file=any(f in text for f in fixed),
                names_fixed_line=any(int(line) in lines for f, lines in fixed.items()
                                     for line in re.findall(re.escape(f) + r":(\d+)", text)),
                failing_tests=timing.get("selected_failing_count"),
                passing_tests=timing.get("selected_passing_count"),
                listed_passing_tests=len(json.loads(passing) if isinstance(passing, str) else passing),
                selection=timing.get("selection_strategy"),
                # Coverage failures carry the Docker error; keep only that coverage failed.
                selection_reason="coverage failed" if (reason or "").startswith("coverage failed") else reason,
                selection_uses_patch=bool(timing.get("selection_uses_oracle")),
                tree_workers=steps.get("tree", {}).get("workers"),
                instrument_seconds=timing.get("instrument_seconds"),
                collect_seconds=collect.get("elapsed_seconds"),
                tree_seconds=steps.get("tree", {}).get("elapsed_seconds"),
                diagnose_seconds=steps.get("diagnose", {}).get("elapsed_seconds"))


def results_json(diagnoses, repairs):
    """`repairs` maps (model, agent, mode) to {instance: entry}."""
    out = {"diagnoses": diagnoses, "repairs": {}}
    for (model, agent, mode), instances in repairs.items():
        out["repairs"].setdefault(model, {}).setdefault(agent, {})[mode] = instances
    return json.dumps(out, indent=1) + "\n"


def write_results(diagnoses, repairs, path):
    with open(path, "w") as f:
        f.write(results_json(diagnoses, repairs))


def files(path):
    """Every regular file below `path`, in a fixed order."""
    if os.path.isfile(path):
        yield path
        return
    for d, dirs, names in os.walk(path):
        dirs.sort()
        for n in sorted(names):
            if os.path.isfile(os.path.join(d, n)) and not os.path.islink(os.path.join(d, n)):
                yield os.path.join(d, n)


def entry(name, data, mode=0o644):
    """One tar entry: header blocks, then the data padded to whole blocks."""
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, 0
    return info.tobuf(tarfile.GNU_FORMAT, "utf-8", "surrogateescape") + data + b"\0" * (-len(data) % BLOCK)


def pack(job):
    """Write one part: tar entries without an end marker, as one zstd frame.

    `members` are (path, name in the archive) pairs; `extra` are (name, bytes) pairs."""
    part, members, extra, out, rewrites, forbidden = job
    tmp = f"{out}/parts/{part}.zst.tmp"
    stats = dict(files=0, raw_bytes=0, tar_bytes=0, rewritten=0, forbidden_hits={k: 0 for k in forbidden},
                 dated_files=[])
    zstd = subprocess.Popen(ZSTD + ("-o", tmp), stdin=subprocess.PIPE)

    def add(arcname, data, mode=0o644):
        for old, new in rewrites:
            stats["rewritten"] += data.count(old)
            data = data.replace(old, new)
        for k, v in forbidden.items():
            stats["forbidden_hits"][k] += data.count(v)
        if has_run_dates(arcname, data):
            stats["dated_files"].append(arcname)
        block = entry(arcname, data, mode)
        zstd.stdin.write(block)
        stats["files"] += 1
        stats["raw_bytes"] += len(data)
        stats["tar_bytes"] += len(block)

    for arcname, data in extra:
        add(arcname, data)
    for path, arcname in members:
        for f in files(path):
            with open(f, "rb") as h:
                data = h.read()
            add(arcname + f[len(path):], data, 0o755 if os.access(f, os.X_OK) else 0o644)
    zstd.stdin.close()
    if zstd.wait():
        raise RuntimeError(f"zstd failed for {part}")
    os.replace(tmp, f"{out}/parts/{part}.zst")
    return part, stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs_dir")
    p.add_argument("out_dir")
    p.add_argument("--name", default="forecast-results")
    p.add_argument("--add", action="append", default=[], metavar="FILE",
                   help="put FILE into the archive's top folder")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--rewrite", action="append", default=[], metavar="OLD=NEW")
    p.add_argument("--forbid", action="append", default=[], metavar="STRING")
    p.add_argument("--forbid-env", action="append", default=[], metavar="NAME")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dataset", default=DATASET, help="SWE-bench Lite, for the developer patches")
    a = p.parse_args()
    root = os.path.abspath(os.path.expanduser(a.runs_dir))
    out = os.path.abspath(os.path.expanduser(a.out_dir))
    os.makedirs(f"{out}/parts", exist_ok=True)
    top = a.name
    rewrites = [tuple(s.encode() for s in r.split("=", 1)) for r in a.rewrite]
    forbidden = {s: s.encode() for s in a.forbid}
    for n in a.forbid_env:
        if not os.environ.get(n):
            sys.exit(f"--forbid-env {n}: the variable is not set")
        forbidden[f"${n}"] = os.environ[n].encode()
    missing = [m for m in MODELS if not os.path.isdir(f"{root}/{m}")]
    if missing:
        hint = f"; did you mean {root}/runs?" if os.path.isdir(f"{root}/runs") else ""
        sys.exit(f"{root} is not a runs folder: it has no {', '.join(missing)}/ folder{hint}")
    # The instances are the <id> folders (SWE-bench ids contain "__"); both models must have the same.
    folders = {m: sorted(n for n in os.listdir(f"{root}/{m}") if "__" in n and os.path.isdir(f"{root}/{m}/{n}"))
               for m in MODELS}
    if len({tuple(v) for v in folders.values()}) != 1:
        sys.exit("the models' runs folders hold different instances")
    ids = folders[MODELS[0]]
    listing = "".join(f"{i}\n" for i in ids).encode()

    def todo(part):
        return a.force or not os.path.exists(f"{out}/parts/{part}.zst")

    parts, jobs, diagnoses, repairs = [], [], {}, {}
    steps = {i: pipeline(load(f"{root}/{DIAGNOSES}/{i}/report.json")) for i in ids}
    lite = swe_bench_lite(a.dataset)
    for i in ids:
        diagnoses[i] = diagnosis(f"{root}/{DIAGNOSES}/{i}/diagnosis", steps[i], lite[i])
    parts.append("diagnoses")
    if todo("diagnoses"):
        members = [(f"{root}/{DIAGNOSES}/{i}/diagnosis", f"{top}/diagnoses/{i}") for i in ids]
        extra = [(f"{top}/diagnoses/instances.txt", listing)]
        extra += [(f"{top}/diagnoses/{i}/pipeline.json", (json.dumps(steps[i], indent=2) + "\n").encode())
                  for i in ids]
        jobs.append(("diagnoses", members, extra, out, rewrites, forbidden))
    for model in MODELS:
        for mode in MODES:
            for agent in AGENTS:
                src = f"{root}/{model}"
                repairs[model, agent, mode] = {i: repair(src, agent, mode, i) for i in ids}
                part = f"{model}-{mode}-{agent}"
                parts.append(part)
                if not todo(part):
                    continue
                prefix = f"{top}/{model}/{mode}"
                members = []
                for i in ids:
                    members.append((f"{src}/{i}/run/{agent}/{mode}", f"{prefix}/{i}/run/{agent}/{mode}"))
                    verdict = f"{src}/{i}/evaluate/{agent}/{mode}/result.json"
                    if os.path.exists(verdict):
                        members.append((verdict, f"{prefix}/{i}/evaluate/{agent}/{mode}/result.json"))
                extra = [(f"{prefix}/{agent}.txt", listing)]
                jobs.append((part, members, extra, out, rewrites, forbidden))
    # The top part (results.json and the added files) is small and always packed again.
    results = results_json(diagnoses, repairs)
    with open(f"{out}/results.json", "w") as f:
        f.write(results)
    extra = [(f"{top}/results.json", results.encode())]
    for path in a.add:
        with open(path, "rb") as f:
            extra.append((f"{top}/{os.path.basename(path)}", f.read()))
    parts.insert(0, "top")
    jobs.insert(0, ("top", [], extra, out, rewrites, forbidden))

    print(f"{len(parts)} parts, {len(jobs)} to pack", flush=True)
    manifest = load(f"{out}/manifest.json") or {}
    with Pool(a.jobs) as pool:
        for part, stats in pool.imap_unordered(pack, jobs):
            manifest[part] = stats
            hits = {k: v for k, v in stats["forbidden_hits"].items() if v}
            dated = len(stats["dated_files"])
            print(f"packed {part}: {stats['files']} files, {stats['raw_bytes'] / 2**30:.2f} GiB raw, "
                  f"{stats['rewritten']} rewritten" + (f", FORBIDDEN {hits}" if hits else "")
                  + (f", {dated} FILES WITH RUN DATES" if dated else ""), flush=True)
    with open(f"{out}/manifest.json", "w") as f:
        json.dump({n: manifest[n] for n in parts}, f, indent=1)
        f.write("\n")

    # The end marker: two zero blocks, padded to a whole tar record.
    size = sum(manifest[n]["tar_bytes"] for n in parts) + 2 * BLOCK
    end = subprocess.run(ZSTD + ("-c",), input=b"\0" * (2 * BLOCK + -size % RECORD),
                         stdout=subprocess.PIPE, check=True).stdout
    archive = f"{out}/{a.name}.tar.zst"
    h = hashlib.sha256()
    with open(archive + ".tmp", "wb") as z:
        for n in parts:
            with open(f"{out}/parts/{n}.zst", "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    z.write(block)
                    h.update(block)
        z.write(end)
        h.update(end)
    os.replace(archive + ".tmp", archive)
    with open(f"{out}/SHA256SUMS", "w") as f:
        f.write(f"{h.hexdigest()}  {a.name}.tar.zst\n")
    bad = [n for n in parts if any(manifest[n]["forbidden_hits"].values())]
    dated = [n for n in parts if manifest[n].get("dated_files")]
    if bad or dated:
        sys.exit("".join([f"forbidden strings found in: {', '.join(bad)}\n" if bad else "",
                          f"run dates found in: {', '.join(dated)} (files listed in manifest.json)" if dated else ""]))
    print(f"done: {archive}")


if __name__ == "__main__":
    main()
