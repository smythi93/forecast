# FORECAST

[![PyPI](https://img.shields.io/pypi/v/forecast-repair.svg)](https://pypi.org/project/forecast-repair/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/smythi93/forecast/blob/main/LICENSE)
[![Benchmark: SWE-bench Lite](https://img.shields.io/badge/benchmark-SWE--bench%20Lite-orange.svg)](https://www.swebench.com/lite.html)
[![Agents: 5](https://img.shields.io/badge/agents-5-brightgreen.svg)](https://github.com/smythi93/forecast/tree/main/agents)

FORECAST is a debugging approach that generates diagnoses to boost repair agents.

Given a program with failing and passing tests, FORECAST traces their executions,
finds the execution features that separate the failing runs from the passing
ones, and links them into a **cause chain**: a short, ordered explanation of how
the failure comes about. Every step carries a *discriminability*, an MCC-based
score of how sharply it separates failing from passing runs. A repair agent gets
this diagnosis next to the issue it has to fix.

```
1. Variable "y" defined at middle.py:1 is used at middle.py:6 AND In middle.py:1, y < x (discriminability: 1.00)
```

FORECAST works in five steps:

| Step | Command | What it does |
|---|---|---|
| 1. Collect | `collect` | instruments the program and records an execution trace per test |
| 2. Tree | `tree` | builds the traced relevance tree from the failing and passing traces |
| 3. Diagnose | `diagnose` | derives the cause chain from the tree |
| 4. Repair | `prompt`, `run` | runs a repair agent on the issue, with or without the diagnosis |
| 5. Evaluate | `evaluate` | checks the agent's patch against the tests |

Contents:

- [Installation](#installation)
- [Quick start](#quick-start)
- [SWE-bench instances](#swe-bench-instances)
- [Repair agents](#repair-agents)
- [Models](#models)
- [Trace collection](#trace-collection)
- [Test selection](#test-selection)
- [Plugin system](#plugin-system)
- [Evaluation](#evaluation): reproducing the paper's results
- [License](#license)

## Installation

FORECAST needs Python 3.12 or newer, and Docker for SWE-bench instances and repair
agents.

```bash
pip install forecast-repair                       # diagnoses only
pip install "forecast-repair[agents,evaluation]"  # plus repair agents and the SWE-bench harness
```

The optional extra `zstd` adds zstd as a codec for traces (gzip is the default).

SWE-agent and Agentless run from checkouts at pinned commits, which the repository's
`agents/install_upstream.py` installs. To use them, to run the quick start below, or to
reproduce the evaluation, install from a clone instead:

```bash
git clone https://github.com/smythi93/forecast.git
cd forecast
pip install -e ".[agents,evaluation]"
python agents/install_upstream.py      # SWE-agent and Agentless at their pinned commits
```

## Quick start

`resources/middle` holds a small program, `middle(x, y, z)`, which should return
the middle of its three arguments, and six tests, one of which fails. The commands
below run from the repository root.

```bash
T=tests.py::TestMiddle
OUT=/tmp/middle

# 1. Instrument the program (into $OUT/instrumented) and trace the failing and passing tests
forecast collect -s resources/middle -o $OUT \
    -f $T::test_middle_213 \
    -p $T::test_middle_335,$T::test_middle_123,$T::test_middle_321,$T::test_middle_555,$T::test_middle_534

# 2. Build the traced relevance tree ($OUT/tree.pkl)
forecast tree -s resources/middle -d $OUT -o $OUT

# 3. Derive the diagnosis ($OUT/diagnosis.json, .txt and .md)
forecast diagnose -s resources/middle -d $OUT -o $OUT
cat $OUT/diagnosis.txt

# 4. Build an agent's prompt with the diagnosis ($OUT/prompt.txt)
forecast prompt -s resources/middle --agent mini-swe-agent --mode forecast \
    --diagnosis $OUT/diagnosis.json --problem-statement "middle(2, 1, 3) returns 1, not 2." \
    -f $T::test_middle_213 -o $OUT -v
```

The prompt is the issue followed by the diagnosis:

```
middle(2, 1, 3) returns 1, not 2.

## Failure Diagnosis

The following diagnosis shows the cause-effect chain leading to the test failure:

1. Variable "y" defined at middle.py:1 is used at middle.py:6 AND In middle.py:1, y < x (discriminability: 1.00)
```

`--mode baseline` gives the issue alone. `run` then runs the agent with a model
(`--model`, see [Models](#models)) and saves its trajectory and patch, and
`evaluate --patch <file>` checks a patch against the tests. Every command has
`--help`.

## SWE-bench instances

For SWE-bench Lite, the repository's `evaluation` package runs the steps per instance. It
collects the traces in the instance's SWE-bench Docker image, with the developer's
test patch applied, from the instance's `FAIL_TO_PASS` and `PASS_TO_PASS` tests:

```bash
# Diagnosis: collect, tree, diagnose, and SBFL (which ACR and RepairAgent use)
python -m evaluation.diagnosis_batch --subset lite --stage all \
    --instance-ids astropy__astropy-12907 --output-dir results

# Repair in both modes; each is evaluated by the SWE-bench harness
python -m evaluation.repair_batch --agent mini-swe-agent --mode baseline --stage all \
    --model claude-haiku-4-5-20251001 --instance-ids astropy__astropy-12907 --output-dir results
python -m evaluation.repair_batch --agent mini-swe-agent --mode forecast --stage all \
    --model claude-haiku-4-5-20251001 --instance-ids astropy__astropy-12907 --output-dir results
```

Results go to `results/<instance>/`:

- `diagnosis/`: the diagnosis and SBFL;
- `prompt/`, `run/` and `evaluate/<agent>/<mode>/`: the repair, its trajectory and the
  harness verdict;
- `report.json`: every step, with its time.

`evaluation.parallel_repair` runs one agent and mode over many instances in parallel.
The [Evaluation](#evaluation) uses it.

## Repair agents

FORECAST evaluates a diagnosis by what it does for repair agents. Every agent:

- runs inside the instance's SWE-bench image;
- sends every LLM call through FORECAST's model interface (`ModelClient.chat`,
  recorded per run in `calls.jsonl`);
- receives the diagnosis through one injection point, the issue text
  (`RepairAgent.issue_text`).

Patches are judged by the official SWE-bench harness.

| Agent | Integration |
|---|---|
| `swe-agent` | upstream SWE-agent 1.1, unmodified, with its SWE-bench config for Claude |
| `mini-swe-agent` | upstream mini-swe-agent 2.4.6, unmodified, with its SWE-bench config |
| `agentless` | upstream Agentless, its SWE-bench recipe replayed stage by stage (no embedding retrieval) |
| `acr` | AutoCodeRover, re-implemented from its paper; SBFL from FORECAST's traces |
| `repair-agent` | RepairAgent, ported from Java to Python from its paper and MIT prompts |

**AutoCodeRover.** AutoCodeRover's implementation is published under the SONAR
Source-Available License v1.0. That is not an open-source license, and it does not
permit our use. FORECAST therefore contains none of its code. The `acr` agent is a
complete re-implementation of the approach as the ISSTA'24 paper (Zhang et al.,
"AutoCodeRover: Autonomous Program Improvement") describes it, with the paper's
parameters:

- stratified context retrieval with the paper's seven search APIs;
- retrieval augmented by spectrum-based fault localization;
- validated patch generation.

The original code was consulted only to verify the re-implementation; nothing was
copied from it. Results for `acr` are results of this re-implementation, not of the
original tool.

## Models

`--model` picks the provider by the model's name:

| Name | Provider | Configuration |
|---|---|---|
| `claude-*` | Anthropic API | `ANTHROPIC_API_KEY` |
| `gpt*`, `o1*`, `o3*`, `o4*`, others | OpenAI API | `OPENAI_API_KEY` |
| `proxy/<alias>` | a LiteLLM proxy | `LITELLM_PROXY_API_BASE`, `LITELLM_PROXY_API_KEY`; `LITELLM_PROXY_CANONICAL_MODEL` names the model behind the alias |
| `ollama/<model>` | Ollama | `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`) |
| names containing `helmholtz` | Helmholtz Blablador | `HELMHOLTZ_API_KEY` |

FORECAST computes each call's cost from its tokens at the provider's list prices
(`forecast.agents.models.utils.PRICES`). Any price a provider or proxy reports is
ignored. `FORECAST_PRICE_AS=<model>` prices a model without a price, such as a local
one, as another model.

## Trace collection

Tracing records every observation a run makes, so a traced run can produce
orders of magnitude more data than a plain one and outlive its timeout.

### Warm images

The single biggest cost is usually not the test. Each test runs in its own
fresh container, and the first test to run in a container pays to build
whatever caches the project uses — matplotlib, for example, scans and parses
every system font. That work is small on its own, but it is paid once per
test, and under instrumentation it is *traced*.

Measured on `matplotlib__matplotlib-23314`, one test in a fresh container:

| | cold container | warm container |
| --- | --- | --- |
| time | 65.6 s | 4.1 s |
| trace | 1.1 GiB | 54.3 MiB |

`setup()` therefore runs one test before committing the image, so the caches
are built once, land in the image, and are never traced. Disable with
`warmup=False` on the subject if a subject's tests must not run during setup.

### Budgets

The tracer runs **inside** the subject's environment, so the limits below only
apply if that environment has `sflkitlib >= 0.2.3`. When `sflkitlib` is
installed from a working copy on the host, FORECAST copies that working copy
into the container automatically; set `FORECAST_SFLKITLIB` to a checkout,
wheel, or sdist to override.

Compression is the effective lever here, shrinking traces about 4x. The value
and loop caps matter only for subjects that genuinely move large buffers or run
very hot loops; on matplotlib they change trace size by about 4%.

Limits are tuned with host environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `FORECAST_TRACE_MAX_VALUE_BYTES` | `1024` | Longest string/bytes value stored in one event. Lengths are recorded separately and stay exact. |
| `FORECAST_TRACE_MAX_LOOP_HITS` | `2` | Iterations recorded per loop frame. Lossless: loop features only distinguish never / once / more than once. |
| `FORECAST_TRACE_MAX_BYTES` | `2147483648` | Trace size at which recording stops. Events are written whole, so a cut trace stays readable. |
| `FORECAST_TRACE_MAX_EVENTS` | `0` (off) | Alternative budget, counted in events. |
| `FORECAST_TRACE_COMPRESS` | `gzip` | Codec for the event stream (`gzip`, `zstd`, or empty). Readers detect it from the file's magic bytes. `zstd` needs the `zstandard` package on both sides. |

## Test selection

A diagnosis compares passing against failing runs only at functions both
reach, so a passing test sharing no executed code with the failure cannot
change the result — it only costs a traced run. `ExperimentRunner` therefore
picks which passing tests to trace from ordinary coverage, which is far cheaper
to collect than a trace:

- `coverage-overlap` (default) — keep tests that execute code the failing tests
  execute, most overlapping first; ties favour the test that executes least
  besides it.
- `reference-patch` — additionally require the test to reach code the
  developer patch changes. **Oracle-based**: it consults the fix, so results
  must be reported separately. The selection is recorded in
  `<instance>/diagnosis/test_selection.json` with `uses_oracle: true`.
- `name-heuristic` — the earlier per-repository name filter; needs no coverage run.
- `none` — trace every listed test.

## Plugin system

FORECAST loads plugins via Python entry points in the `forecast.plugins` group.

Plugins can register:
- repair agents
- model providers
- localization formulas
- localization techniques
- subject classes

Example plugin entry point:

```toml
[project.entry-points."forecast.plugins"]
MyForecastPlugin = "my_pkg.forecast_plugin"
```

Example plugin module:

```python
from forecast.plugin_loader import PluginContext


def register(context: PluginContext):
    @context.register_formula("MyFormula")
    def my_formula(ep, np, ef, nf):
        return (ef + 1) / (ep + np + ef + nf + 1)

    @context.register_localization_technique("my-localizer")
    def my_localizer(*, subject, location_type, formula="Tarantula", **kwargs):
        # Can call any subject API; technique is fully plugin-defined.
        return subject.sfl(location_type=location_type, formula=formula, technique="sfl")
```

# Evaluation

The paper evaluates FORECAST on all 300 instances of SWE-bench Lite (test split).
Five repair agents ran with two models, each agent in two modes:

- **baseline**: the agent gets the issue;
- **forecast**: the agent gets the issue plus FORECAST's diagnosis.

| Name | Model | Access |
|---|---|---|
| `haiku` | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Anthropic API via a LiteLLM proxy |
| `qwen` | Qwen3.5 27B (`qwen3.5:27b`) | Ollama, 8 servers on 8 GPUs |

The diagnoses were computed once per instance, before any repair, and both models and
all agents got the same one. Each instance had a limit of 6 hours per repair.

Everything needed to reproduce the results is in this repository:

| Path | Content |
|---|---|
| `evaluation/scripts/instances.txt` | the 300 instances |
| `evaluation/scripts/diagnose.sh` | computes the diagnoses |
| `evaluation/scripts/mirrors.sh` | local git mirrors of the 12 repositories, for Agentless |
| `evaluation/scripts/run_agent.sh` | runs one agent in one mode with one model over all instances |
| `evaluation/scripts/run_all.sh` | runs all 20 combinations (2 models × 2 modes × 5 agents), one after another |
| `evaluation/scripts/pack.py` | computes `results.json` from the runs and packs them into one archive |
| `evaluation/scripts/analyze.py` | answers the research questions from `results.json` |
| `evaluation/scripts/plot.py` | draws the paper's figures (diagnosis time, repair time) from `results.json` (needs matplotlib) |
| `evaluation/results/` | `results.json` of the paper's runs, and the archive's `manifest.json` and `SHA256SUMS` |

The runs themselves, about 54 GiB, are on Zenodo as one file,
`forecast-results.tar.zst` (DOI: _to be added_).

## Analyzing the results

`analyze.py` reads only `evaluation/results/results.json` and prints Markdown tables,
one section per research question:

```bash
python3 evaluation/scripts/analyze.py              # all sections
python3 evaluation/scripts/analyze.py rq1 rq3      # only these
```

| Section | Answers | Content |
|---|---|---|
| `overview` | | per model, agent and mode: resolved, errors, calls, tokens, cost, runtime |
| `subjects` | Subjects | per project: instances, traced tests, causes per chain, discriminability, empty diagnoses, diagnosis time, share of resolved repair attempts per model and mode |
| `diagnoses` | Subjects | distribution of discriminability and chain length over all diagnoses, weak chains |
| `causes` | Subjects | cause types (atoms, conjunctions, disjunctions, divergences), chains with a divergence |
| `selection` | Setup | how the passing tests to trace were selected, instances without `PASS_TO_PASS` |
| `rq1` | RQ1, RQ2 | resolved per agent and over all agents: change in instances and percentage points, relative change, instances gained and lost, paired exact McNemar test; instances some agent resolves |
| `localization` | RQ1 | chains naming a file or line the developer patch changes, and the change in resolved instances with and without such a chain |
| `exits` | RQ2, RQ3 | how SWE-agent and mini-SWE-agent runs ended (submitted, cost limit), with resolved instances and patches with new files |
| `rq3` | RQ3 | patches produced, patches the harness applied, patches that resolve, changed lines (added + removed) and files, for all patches and for instances both modes resolve |
| `patches` | RQ3 | patches that add files the agent created for itself; changed lines and files in existing files |
| `rq4` | RQ4 | diagnosis overhead (instrumentation, collection, tree, diagnosis) against tokens, cost and repair time per instance, time including the instance's own diagnosis, cost and time per resolved instance, LLM cost over all agents |
| `truncation` | Setup | answers cut off at their token limit, per agent, model, mode and limit |

Every number is computed from the per-instance entries in `results.json`:

- rates and sums count instances;
- distributions run over instances, given as median (first to third quartile) and mean;
- the McNemar tests pair each instance's baseline and FORECAST verdicts;
- each FORECAST run is charged the diagnosis time of its own instance.

A run without a harness verdict counts as unresolved.

## Reproducing the runs

Run the scripts from the repository root, after the [installation](#installation).
`<runs>` is the folder that will hold the results, `runs/<model>/<instance>/`.

1. **Diagnoses.** This collects traces for every instance, keeping them in `<work>`,
   which needs much disk. The paper's diagnoses took about 330 hours in total (the
   `rq4` section lists the time per step). Building a tree takes much memory for the
   large (sympy) instances, so keep the number of parallel instances low:

   ```bash
   bash evaluation/scripts/diagnose.sh <work> <runs> 4
   ```

2. **Models.**
   - For Haiku, run a LiteLLM proxy that serves `anthropic/claude-haiku-4-5-20251001`,
     and set `LITELLM_PROXY_API_BASE` and `LITELLM_PROXY_API_KEY`.
   - For Qwen, start one Ollama server per GPU serving `qwen3.5:27b`, and set
     `OLLAMA_ENDPOINTS` to their comma-separated URLs.

3. **Agentless mirrors.** Optional, but needed with many workers:

   ```bash
   bash evaluation/scripts/mirrors.sh <mirrors>
   export MIRRORS=<mirrors>
   ```

4. **Repairs.** These are all 20 combinations of model, agent and mode, or one at a
   time; finished instances are skipped:

   ```bash
   bash evaluation/scripts/run_all.sh <runs>                                  # everything
   bash evaluation/scripts/run_agent.sh <runs> haiku swe-agent forecast 10    # one model, agent and mode
   ```

   Each instance runs in its own process, so the number of workers does not affect
   the results.

5. **Results and archive.** This computes `results.json` from the per-instance files
   and packs everything into one archive:

   ```bash
   python3 evaluation/scripts/pack.py <runs> <out> --add README.md
   python3 evaluation/scripts/analyze.py --results <out>/results.json
   ```

## Results

`results.json` has one entry per instance and repair run. The comments below only
explain the fields; the file itself is plain JSON.

```json
{
  "diagnoses": {
    "<instance>": {
      "discriminability": 1.0,          // of the chain's sharpest step (MCC-based)
      "mean_discriminability": 1.0,     // over the chain
      "causes": 10,                     // length of the cause chain (0: empty diagnosis)
      "cause_types": {"atom": 8, "conjunction": 1, "disjunction": 1},  // and "divergence"
      "names_fixed_file": true,         // a cause lies in a file the developer patch changes
      "names_fixed_line": false,        // a cause names a line within one of its hunks
      "failing_tests": 3,               // traced tests
      "passing_tests": 19,
      "listed_passing_tests": 19,       // PASS_TO_PASS tests SWE-bench lists
      "selection": "all",               // how the passing tests were selected
      "selection_reason": "below selection threshold",  // why, where recorded; else null
      "selection_uses_patch": false,    // the selection consulted the developer patch
      "tree_workers": 1,                // processes that built the tree
      "instrument_seconds": 168.1,      // part of collect_seconds
      "collect_seconds": 315.3,
      "tree_seconds": 841.4,
      "diagnose_seconds": 47.3
    }
  },
  "repairs": {
    "<model>": {
      "<agent>": {
        "<mode>": {
          "<instance>": {
            "resolved": true,           // harness verdict; null: no verdict (counts as unresolved)
            "error": false,             // the agent ended with an error
            "patch": true,              // a non-empty patch was submitted
            "applied": true,            // the harness applied it cleanly; null: no verdict
            "lines_added": 2,
            "lines_removed": 2,
            "files": 1,
            "new_files": 0,             // files the patch adds (the agent's own scripts, notes)
            "existing_lines": 4,        // added + removed lines in files that existed before
            "existing_files": 1,
            "exit_status": "submitted", // how the run ended (SWE-agent, mini-SWE-agent); else null
            "calls": 4,                 // LLM calls, and their tokens and cost (USD)
            "prompt_tokens": 8633,
            "completion_tokens": 1924,
            "cost": 0.018253,           // null for Qwen, which runs locally (see Costs)
            "max_tokens": {"1024->1024": [4, 0]},  // per answer length requested->given:
                                        // answers, answers cut off at that length
            "seconds": 18.08            // the agent's runtime; null if not recorded
          }
        }
      }
    }
  }
}
```

### Costs

All costs use Anthropic's list prices for Claude Haiku 4.5. Prices reported by a
provider or proxy are never used, since they may include institutional discounts.

| Token type | Price |
|---|---|
| Input | $1 per million |
| Output | $5 per million |
| Cache writes | $1.25 per million |
| Cache reads | $0.10 per million |

The `overview` section also shows the cost without prompt caching: every prompt token
priced as input.

Qwen runs locally and costs nothing per token, so `results.json` has no cost for its
runs and `analyze.py` compares it by tokens. Its calls are nevertheless counted at
Haiku's prices (`FORECAST_PRICE_AS`), without caching, because SWE-agent's and
mini-SWE-agent's budgets ($3 per run) are given in dollars; for Qwen this amounts to
about three million tokens per run.

## The archive

`forecast-results.tar.zst` holds this README, `results.json`, the diagnoses and every
run. It is about 320 MB; extracted it takes about 54 GiB.

```bash
sha256sum -c SHA256SUMS                        # SHA256SUMS from evaluation/results/
tar --zstd -xf forecast-results.tar.zst        # or: zstd -dc forecast-results.tar.zst | tar -xf -
```

`tar --zstd` needs GNU tar and zstd. The archive was compressed with `--long=27`, which
zstd decompresses without extra flags.

```
forecast-results/
  README.md                        this file
  results.json                     the per-instance results
  diagnoses/
      instances.txt                the 300 instances
      <instance>/
          diagnosis.json           the diagnosis: cause chain with discriminability
          diagnosis.txt            the text the forecast mode appends to the issue
          diagnosis.md             the same text as Markdown
          sbfl.json                spectrum-based fault localization (used by ACR and RepairAgent)
          pipeline.json            FORECAST's report of the steps that built the diagnosis:
                                   collect (instrumentation, traced tests), tree, diagnose, with times
  <model>/<mode>/
      <agent>.txt                  the instances of this run
      <instance>/run/<agent>/<mode>/
          result.json              the agent's result: patch, elapsed_seconds, error, ...
          patch.diff               the submitted patch
          logs/run_<instance>.log  the agent's log
          trajs/<instance>/
              calls.jsonl          every LLM call: messages, response, tokens, cost
              outcome.json         per-run totals (calls, tokens, cost)
              run_info.json        model, prices, diagnosis and fault localization given
              <trajectory>         the upstream agent's own trajectory, see below
      <instance>/evaluate/<agent>/<mode>/result.json
                                   SWE-bench harness verdict: resolved, applied, per-test status
```

| Agent | Upstream trajectory in `trajs/<instance>/` |
|---|---|
| mini-swe-agent | `mini.traj.json` |
| SWE-agent | `<instance>.traj` |
| ACR | `acr_trajectory.json` |
| RepairAgent | `repairagent_trajectory.json` |
| Agentless | `agentless/` (localization, repair samples, regression and reproduction tests, reranking) |

The files are archived byte for byte as the evaluation wrote them, and the archive
stores no owner and no time stamps. `pack.py` checks every archived file:

- neither the API key nor the proxy's address appears (`--forbid-env`);
- no file holds a date written at run time. This uses the rules `forecast.run_dates`
  applied after each run: dates from 2025 on, epoch time stamps under time-stamp keys,
  and setuptools-scm date suffixes. Patches are exempt, since they are kept as the agents
  submitted them; dates in the repositories' own code are older and do not count.

The archive leaves out:

- the SWE-bench harness's test logs (the verdict file keeps the per-test outcome);
- FORECAST's execution traces and relevance trees;
- the schedulers' logs.

`results.json` also compares the diagnoses with the developer patches, which `pack.py`
reads from SWE-bench Lite (`--dataset`, through the `datasets` package); FORECAST never
used them. To rebuild `results.json` from the extracted archive and SWE-bench Lite:

```bash
python3 evaluation/scripts/analyze.py --results rebuilt.json --from forecast-results
```

The rebuilt file is byte-identical to `results.json`.

# License

FORECAST is licensed under the Apache License 2.0; see [LICENSE](https://github.com/smythi93/forecast/blob/main/LICENSE).

The repair agents FORECAST integrates are licensed as follows:

| Agent | License | How FORECAST uses it |
|---|---|---|
| SWE-agent | MIT | cloned at a pinned commit by `agents/install_upstream.py`, not part of this repository |
| mini-swe-agent | MIT | installed from PyPI as a pinned dependency |
| Agentless | MIT | cloned at a pinned commit by `agents/install_upstream.py`, not part of this repository |
| RepairAgent | MIT | ported to Python from its paper and prompts; its license, with the AutoGPT notice it carries, is in [`src/forecast/agents/repair_agent/LICENSE`](https://github.com/smythi93/forecast/blob/main/src/forecast/agents/repair_agent/LICENSE) |
| AutoCodeRover | Apache 2.0, as part of FORECAST | completely re-implemented from its paper; the original implementation is not used (see [Repair agents](#repair-agents)) |
