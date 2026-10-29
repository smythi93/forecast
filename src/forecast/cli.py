"""
FORECAST command-line interface.

Pipeline (each step can be run independently, outputs feed the next step):

  Step 1 : collect   – instrument code & collect execution traces
           → writes events/ and mapping.json to --output

  Step 1b: tree      – build TracedRelevanceTree from events (optional but faster)
           → reads  --data (collect output)
           → writes tree.pkl to --output
           → if skipped, 'diagnose' builds the tree itself from raw events

  Step 2 : diagnose  – derive a FORECAST diagnosis
           → reads  --data (collect / tree output; prefers tree.pkl if present)
           → writes diagnosis.{json,md,txt} to --output

  Step 3 : prompt    – extract the agent prompt and optionally inject diagnosis
           → reads  --diagnosis (from diagnose output, optional)
           → writes prompt.txt to --output

  Step 4 : run       – run the repair agent (LLM call, saves traj, extracts patch)
           → reads  --prompt (from prompt output, optional) and --diagnosis
           → writes traj.json and patch.diff to --output

  Step 5 : evaluate  – validate a patch (local pytest or SWE-bench Docker)
           → reads  --patch (from run output)
"""

import argparse
import json
import sys
from pathlib import Path

from forecast.plugin_loader import load_plugins

# ─────────────────────────────────────────────────────────────────────────────
# Shared argument helpers
# ─────────────────────────────────────────────────────────────────────────────


def add_source_argument(parser: argparse.ArgumentParser):
    """
    Add the shared ``-s`` / ``--src`` / ``--source`` argument to *parser*.

    This argument specifies the source directory of the subject project.
    When ``--swe`` is used, it instead takes a SWE-bench Lite instance ID
    (e.g. ``"astropy__astropy-12907"``) and the actual source directory is resolved
    by checking out the instance repository.

    :param parser: The subparser to augment.
    :return: None
    """
    parser.add_argument(
        "-s",
        "--src",
        "--source",
        type=str,
        required=True,
        help="Source directory of the subject project, or a SWE-bench lite "
        "instance ID when --swe is used.",
    )


def add_workers_argument(parser: argparse.ArgumentParser):
    """
    Add the shared ``-w`` / ``--workers`` argument to *parser*.

    This argument specifies the number of parallel workers to use for tree building and diagnosis steps.
    More workers can speed up processing of large event sets, but also increase memory usage.
    The default is 4.

    :param parser: The subparser to augment.
    :return: None
    """
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker PROCESSES to build the tree with. Threads "
        "cannot help here: building is Python-level work and the GIL "
        "serializes it. Failing runs are always handled in one process, "
        "because they are what defines the tree's structure; only the "
        "passing runs are spread out. Default 1 (no process pool).",
    )


def add_swe_argument(parser: argparse.ArgumentParser):
    """
    Add the shared ``--swe`` flag to *parser*.

    When this flag is set, the CLI operates in "SWE-bench Lite mode" where the ``--src`` argument is
    interpreted as a SWE-bench Lite instance ID (e.g. ``"astropy__astropy-12907"``) rather than a local directory.

    :param parser: The subparser to augment.
    :return: None
    """
    parser.add_argument(
        "--swe",
        action="store_true",
        help="Use a SWE-bench Lite subject.  --src is the SWE-bench instance ID.",
    )


def add_dev_argument(parser: argparse.ArgumentParser):
    """
    Add the shared ``--dev`` flag to *parser*.

    When this flag is set, the CLI loads from the SWE-bench Lite "dev" split instead of "test".
    This is intended for development and experimentation so that test-split results are not
    contaminated by iterative runs.

    :param parser: The subparser to augment.
    :return: None
    """
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help=(
            "Load from the SWE-bench Lite 'dev' split instead of 'test'. "
            "Use this during development so that test-split results are not "
            "contaminated by experimentation."
        ),
    )


def add_data_argument(parser: argparse.ArgumentParser, required: bool = True):
    """
    Add the shared ``-d`` / ``--data`` argument to *parser*.

    This argument specifies the directory containing collected execution-trace data, including the events/ subdirectory
    and mapping.json file produced by the 'collect' step.  The 'diagnose' and 'tree' steps read from this data.
    When *required* is ``False``, the 'diagnose' step can run without pre-collected data and will build the tree
    itself from raw events if needed, but providing pre-collected data is recommended for faster processing.

    :param parser: The subparser to augment.
    :param required: Whether this argument is required.  Defaults to True.
        Set to False for 'diagnose' to allow running without pre-collected data.
    :return: None
    """
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=required,
        help="Directory with collected execution-trace data.",
    )


def add_output_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )


def add_log_dir_argument(parser: argparse.ArgumentParser):
    """
    Add the shared ``--log-dir`` argument to *parser*.

    :param parser: The subparser to augment.
    """
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Directory for per-step log files.  Each step writes its own "
            "<step>[_<instance_id>].log file here.  Defaults to None "
            "(console only)."
        ),
    )


def add_checkout_dir_argument(parser: argparse.ArgumentParser):
    """
    Add the shared ``--checkout-dir`` argument to *parser*.

    :param parser: The subparser to augment.
    """
    parser.add_argument(
        "--checkout-dir",
        type=str,
        default=None,
        help=(
            "Directory where SWE-bench subject repositories are checked out. "
            "Defaults to <output>/repos when --swe is used.  Reuses an "
            "existing checkout if the base commit already matches."
        ),
    )


def add_agent_argument(parser: argparse.ArgumentParser):
    from forecast.agents.register import list_registered_agents

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=list_registered_agents(),
        help="Repair agent to use.",
    )


def add_model_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="LLM model name (passed to the underlying agent).",
    )


def add_mode_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--mode",
        choices=["baseline", "forecast"],
        default="baseline",
        help="baseline: plain agent prompt.  forecast: prompt + FORECAST diagnosis.",
    )


def add_diagnosis_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--diagnosis",
        type=str,
        default=None,
        help="Path to a diagnosis JSON file produced by 'forecast diagnose'.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step handlers
# ─────────────────────────────────────────────────────────────────────────────


def _handle_collect(args):
    """Step 1 – Instrument the project and collect execution traces."""
    from forecast.instrument import instrument
    from forecast.logger import get_logger
    from forecast.subject import LocalSubject, safe

    log_dir = Path(args.log_dir) if args.log_dir else None
    log = get_logger("collect", log_dir=log_dir)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # ── Resolve the source directory ──────────────────────────────────
    # For SWE subjects: check out the repo first so we have a real local
    # path before anything else touches the filesystem.
    swe_failing_tests: list[str] | None = None
    swe_passing_tests: list[str] | None = None
    if args.swe:
        from evaluation import SWEBenchLoader

        instances = SWEBenchLoader.load_instances(instance_ids=[args.src], dev=args.dev)
        if not instances:
            log.error("Instance %s not found in SWE-bench Lite.", args.src)
            sys.exit(1)
        inst = _checkout_swe(instances[0], args, log)
        src = Path(inst["repo"])
        swe_failing_tests = list(inst.get("failing_tests") or [])
        swe_passing_tests = list(inst.get("passing_tests") or [])
    else:
        src = Path(args.src).resolve()

    instrumented = output / "instrumented"
    mapping_path = output / "mapping.json"

    log.info("Instrumenting %s → %s", src, instrumented)
    instrument(src=src, dst=instrumented, mapping_path=mapping_path)
    log.info("Instrumentation done.")

    if args.failing:
        failing_tests = _parse_test_list(args.failing)
    elif swe_failing_tests is not None:
        failing_tests = list(swe_failing_tests)
    else:
        failing_tests = []
    if args.passing:
        passing_tests = _parse_test_list(args.passing)
    elif swe_passing_tests is not None:
        passing_tests = list(swe_passing_tests)
    else:
        passing_tests = []
    selected_tests = sorted(set(failing_tests) | set(passing_tests))

    subject = LocalSubject(
        path=instrumented,
        workers=args.workers,
        test_ids=selected_tests,
        fail_ids=set(failing_tests),
        failing_test_ids=set(failing_tests),
        passing_test_ids=set(passing_tests),
    )

    events_dir = output / "events"
    failing_dir = events_dir / "failing"
    passing_dir = events_dir / "passing"
    failing_dir.mkdir(parents=True, exist_ok=True)
    passing_dir.mkdir(parents=True, exist_ok=True)

    if selected_tests:
        failing_ids = set(failing_tests)
        passing_ids = set(passing_tests)
    else:
        failing_ids = set(subject.get_failing_tests())
        all_test_ids = set(subject.test_ids) or (
            set(subject.passing_test_ids) | set(subject.failing_test_ids)
        )
        if not all_test_ids:
            all_test_ids = set(subject.collect_test_ids())
            subject.test_ids = sorted(all_test_ids)
        passing_ids = set(subject.passing_test_ids) or (all_test_ids - failing_ids)

    event_file_mapping: dict[str, Path] = {}
    for test_id in sorted(failing_ids):
        event_file_mapping[test_id] = failing_dir / safe(test_id)
    for test_id in sorted(passing_ids):
        event_file_mapping[test_id] = passing_dir / safe(test_id)

    log.info(
        "Collecting events for %d tests (%d failing, %d passing) → %s",
        len(event_file_mapping),
        len(failing_ids),
        len(passing_ids),
        events_dir,
    )
    subject.collect_all_events(event_file_mapping)
    log.info("Event collection done.")


def _handle_tree(args):
    """Step 1b – Build TracedRelevanceTree from collected events and persist it."""
    import pickle
    from forecast.handler import ForecastHandler
    from forecast.tree import PrintTreeVisitor
    from forecast.logger import get_logger

    log_dir = Path(args.log_dir) if args.log_dir else None
    log = get_logger("tree", log_dir=log_dir)

    data_dir = Path(args.data)
    events_dir = data_dir / "events"
    mapping_path = data_dir / "mapping.json"

    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)
    if not events_dir.exists():
        log.error("Events directory not found: %s", events_dir)
        sys.exit(1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    failing, passing = _load_event_files(events_dir=events_dir, mapping_path=mapping_path)
    event_files = failing + passing

    if not event_files:
        log.error("No event files found.")
        sys.exit(1)

    log.info(
        "Building relevance tree from %d failing + %d passing event files…",
        len(failing),
        len(passing),
    )

    from forecast.tree import dump_tree

    handler = ForecastHandler(
        workers=args.workers, prune=not getattr(args, "no_prune_tree", False)
    )
    processes = getattr(args, "processes", 1)
    if processes > 1:
        log.info("Building passing runs across %d processes.", processes)
    # Failing runs first: a passing run is only recorded where a failing run
    # has already been, so the footprint has to exist before any of them.
    handler.build(failing, passing, processes=processes)

    tree = handler.tree
    tree_path = output / "tree.pkl"
    dump_tree(tree, tree_path, compress=not getattr(args, "no_compress_tree", False))
    log.info("Tree saved to %s", tree_path)

    n_nodes = _count_tree_nodes(tree)
    n_discriminative = _count_discriminative_nodes(tree)
    log.info(
        "Tree summary: nodes=%d  discriminative=%d  features=%d",
        n_nodes,
        n_discriminative,
        len(tree.all_features),
    )

    if args.verbose:
        PrintTreeVisitor().traverse(tree)


def _count_tree_nodes(tree) -> int:
    """Recursively count all nodes in the tree."""
    return 1 + sum(_count_tree_nodes(c) for c in tree.children.values())


def _count_discriminative_nodes(tree) -> int:
    """Recursively count discriminative nodes."""
    own = 1 if tree.is_discriminative else 0
    return own + sum(_count_discriminative_nodes(c) for c in tree.children.values())


def _handle_diagnose(args):
    """Step 2 – Derive a FORECAST diagnosis from a pre-built tree or collected events."""
    import pickle
    from forecast.diagnosis import Forecast
    from forecast.integration import save_diagnosis
    from forecast.logger import get_logger

    log_dir = Path(args.log_dir) if args.log_dir else None
    log = get_logger("diagnose", log_dir=log_dir)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)

    tree_path = data_dir / "tree.pkl"
    if tree_path.exists():
        from forecast.tree import load_tree

        log.info("Loading pre-built tree from %s", tree_path)
        tree = load_tree(tree_path)
    else:
        from forecast.handler import ForecastHandler

        events_dir = data_dir / "events"
        mapping_path = data_dir / "mapping.json"

        if not events_dir.exists():
            log.error("Events directory not found at %s", events_dir)
            sys.exit(1)

        log.info("Loading events from %s", events_dir)
        failing, passing = _load_event_files(events_dir=events_dir, mapping_path=mapping_path)
        event_files = failing + passing

        log.info("Processing %d event files…", len(event_files))
        handler = ForecastHandler()
        for ef in event_files:
            handler.handle(ef)
        tree = handler.tree

    forecast = Forecast(root=tree)
    diagnosis = forecast.derive_diagnosis()
    mapping_path = data_dir / "events" / "mapping.json"
    if Path(args.src).is_dir() and mapping_path.exists():
        from forecast.diagnosis import FunctionNames

        diagnosis.qualify_functions(FunctionNames.from_mapping(args.src, mapping_path))

    save_diagnosis(diagnosis, output / "diagnosis.json", file_format="json")
    save_diagnosis(diagnosis, output / "diagnosis.md", file_format="markdown")
    save_diagnosis(diagnosis, output / "diagnosis.txt", file_format="text")

    log.info("Diagnosis saved to %s/", output)
    print(diagnosis.to_text())


def _handle_prompt(args):
    """Step 3 – Extract the agent prompt, optionally injecting the diagnosis."""
    from forecast.agents.base import RepairContext, AgentMode
    from evaluation import LocalProjectLoader, SWEBenchLoader
    from forecast.logger import get_logger

    log_dir = Path(args.log_dir) if args.log_dir else None
    log = get_logger("prompt", log_dir=log_dir)

    if args.swe:
        instances = SWEBenchLoader.load_instances(instance_ids=[args.src], dev=args.dev)
        if not instances:
            log.error("Instance %s not found in SWE-bench Lite.", args.src)
            sys.exit(1)
        inst = _checkout_swe(instances[0], args, log)
    else:
        if args.problem_statement and Path(args.problem_statement).exists():
            problem_statement = Path(args.problem_statement).read_text()
        else:
            problem_statement = args.problem_statement or ""
        inst = LocalProjectLoader.load_instance(
            repo_path=args.src,
            problem_statement=problem_statement,
            failing_tests=_parse_test_list(args.failing) if args.failing else [],
            passing_tests=_parse_test_list(args.passing) if args.passing else None,
        )

    context = RepairContext(
        instance_id=inst["instance_id"],
        repo=inst["repo"],
        version=inst["version"],
        problem_statement=inst["problem_statement"],
        failing_tests=inst["failing_tests"],
        passing_tests=inst.get("passing_tests"),
        base_commit=inst.get("base_commit"),
        swe_task_info=inst.get("swe_task_info"),
        swe_setup_info=inst.get("swe_setup_info"),
    )

    if args.diagnosis:
        from forecast.integration import diagnosis_text_from_json, load_diagnosis_json

        diag_data = load_diagnosis_json(Path(args.diagnosis))
        context.diagnosis = diagnosis_text_from_json(diag_data)
        context.diagnosis_discriminability = diag_data.get("discriminability")

    agent = _build_agent(args.agent, args.mode, args.model, args.output)

    # Step 1: get user prompt (carrying the diagnosis in FORECAST mode) and
    # system prompt
    prompt = agent.get_prompt(context)
    system_prompt = agent.get_system_prompt(context)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    prompt_file = output / "prompt.txt"
    prompt_file.write_text(prompt)
    log.info("Prompt saved to %s  (%d chars)", prompt_file, len(prompt))

    if system_prompt is not None:
        system_prompt_file = output / "system_prompt.txt"
        system_prompt_file.write_text(system_prompt)
        log.info(
            "System prompt saved to %s  (%d chars)",
            system_prompt_file,
            len(system_prompt),
        )

    if args.verbose:
        if system_prompt is not None:
            print("=== SYSTEM PROMPT ===")
            print(system_prompt)
            print()
        print("=== USER PROMPT ===")
        print(prompt)


def _handle_run(args):
    """Step 4 – Run the repair agent (LLM call, save trajectory, extract patch)."""
    from forecast.agents.base import AgentMode, RepairContext
    from forecast.agents.models.utils import get_model_client
    from evaluation import LocalProjectLoader, SWEBenchLoader
    from forecast.integration import diagnosis_text_from_json, load_diagnosis_json
    from forecast.logger import get_logger

    log_dir = Path(args.log_dir) if args.log_dir else None
    log = get_logger("run", log_dir=log_dir)

    if args.swe:
        instances = SWEBenchLoader.load_instances(instance_ids=[args.src], dev=args.dev)
        if not instances:
            log.error("Instance %s not found.", args.src)
            sys.exit(1)
        inst = _checkout_swe(instances[0], args, log)
    else:
        problem_statement = (
            Path(args.problem_statement).read_text()
            if args.problem_statement and Path(args.problem_statement).exists()
            else (args.problem_statement or "")
        )
        inst = LocalProjectLoader.load_instance(
            repo_path=args.src,
            problem_statement=problem_statement,
            failing_tests=_parse_test_list(args.failing) if args.failing else [],
            passing_tests=_parse_test_list(args.passing) if args.passing else None,
        )

    context = RepairContext(
        instance_id=inst["instance_id"],
        repo=inst["repo"],
        version=inst["version"],
        problem_statement=inst["problem_statement"],
        failing_tests=inst["failing_tests"],
        passing_tests=inst.get("passing_tests"),
        base_commit=inst.get("base_commit"),
        swe_task_info=inst.get("swe_task_info"),
        swe_setup_info=inst.get("swe_setup_info"),
    )

    mode = AgentMode(args.mode)

    if args.diagnosis:
        raw = load_diagnosis_json(Path(args.diagnosis))
        context.diagnosis = diagnosis_text_from_json(raw)
        context.diagnosis_discriminability = raw.get("discriminability")
    elif mode == AgentMode.FORECAST:
        from evaluation import ExperimentRunner

        runner = ExperimentRunner(output_dir=Path(args.output))
        diag = runner._generate_diagnosis(inst, Path(inst["repo"]))
        if diag is not None:
            context.diagnosis = (
                diag.to_text() if hasattr(diag, "to_text") else str(diag)
            )
            context.diagnosis_discriminability = getattr(diag, "discriminability", None)
            from forecast.integration import save_diagnosis

            save_diagnosis(
                diag, Path(args.output) / "diagnosis.json", file_format="json"
            )

    agent = _build_agent(args.agent, args.mode, args.model, args.output)
    model_client = get_model_client(args.model)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Running %s in %s mode on %s", agent.name, args.mode, context.instance_id)
    result = agent.repair(
        context,
        model_client=model_client,
        traj_dir=out / "trajs",
        log_dir=log_dir,
    )

    result_file = out / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "instance_id": result.instance_id,
                "agent_name": result.agent_name,
                "mode": result.mode.value,
                "success": result.success,
                "patch": result.patch,
                "lines_changed": result.lines_changed,
                "files_modified": result.files_modified,
                "time_seconds": result.time_seconds,
                "tokens_used": result.tokens_used,
                "diagnosis_used": result.diagnosis_used,
                "diagnosis_discriminability": result.diagnosis_discriminability,
                "output_dir": result.output_dir,
                "error": result.error,
            },
            indent=2,
        )
    )
    if result.patch:
        (out / "patch.diff").write_text(result.patch)

    log.info(
        "Repair %s  patch=%s  lines=%d  tokens=%d",
        "SUCCESS" if result.success else "FAILED",
        "yes" if result.patch else "none",
        result.lines_changed,
        result.tokens_used,
    )
    log.info("Result saved to %s", result_file)


def _handle_evaluate(args):
    """Step 5 – Validate a patch against the test suite."""
    from forecast.agents.base import RepairContext
    from evaluation import LocalProjectLoader, SWEBenchLoader
    from forecast.logger import get_logger

    log_dir = Path(args.log_dir) if args.log_dir else None
    log = get_logger("evaluate", log_dir=log_dir)

    patch_path = Path(args.patch)
    if not patch_path.exists():
        log.error("Patch file not found: %s", patch_path)
        sys.exit(1)
    patch = patch_path.read_text()

    if args.swe:
        instances = SWEBenchLoader.load_instances(instance_ids=[args.src], dev=args.dev)
        if not instances:
            log.error("Instance %s not found.", args.src)
            sys.exit(1)
        inst = _checkout_swe(instances[0], args, log)
    else:
        inst = LocalProjectLoader.load_instance(
            repo_path=args.src,
            problem_statement="",
            failing_tests=_parse_test_list(args.failing) if args.failing else [],
            passing_tests=_parse_test_list(args.passing) if args.passing else None,
        )

    context = RepairContext(
        instance_id=inst["instance_id"],
        repo=inst["repo"],
        version=inst["version"],
        problem_statement=inst["problem_statement"],
        failing_tests=inst["failing_tests"],
        passing_tests=inst.get("passing_tests"),
        base_commit=inst.get("base_commit"),
        swe_task_info=inst.get("swe_task_info"),
        swe_setup_info=inst.get("swe_setup_info"),
    )

    agent = _build_agent(args.agent, "baseline", args.model, ".")
    passed = agent.validate_patch(context, patch)

    log.info("Validation: %s", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_test_list(value: str) -> list[str]:
    """Accept comma-separated list or path to a file with one test per line."""
    p = Path(value)
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return [t.strip() for t in value.split(",") if t.strip()]


def _load_event_files(
    *,
    events_dir: Path,
    mapping_path: Path,
) -> tuple[list, list]:
    """
    Load failing/passing event files from an existing events directory.
    """
    from sflkit.events.mapping import EventMapping

    from forecast.runner.event_file import make_event_file

    mapping = EventMapping.load_from_file(mapping_path)
    failing_paths = sorted(
        p for p in (events_dir / "failing").iterdir() if p.is_file()
    )
    passing_paths = sorted(
        p for p in (events_dir / "passing").iterdir() if p.is_file()
    )

    failing = [
        make_event_file(path, run_id, mapping, failing=True)
        for run_id, path in enumerate(failing_paths, start=0)
    ]
    passing = [
        make_event_file(path, run_id, mapping, failing=False)
        for run_id, path in enumerate(passing_paths, start=len(failing))
    ]
    return failing, passing


def _checkout_swe(inst: dict, args, log) -> dict:
    """
    Check out the SWE-bench instance repository to a local directory.

    Returns an updated instance dict with ``"repo"`` pointing at the checkout.
    The checkout root is taken from ``args.checkout_dir`` when provided, or
    defaults to ``<args.output>/repos``.

    :param inst: Instance dict from :class:`SWEBenchLoader`.
    :param args: Parsed argparse namespace (must have ``output`` and
        optionally ``checkout_dir``).
    :param log: Logger instance.
    :returns: Updated instance dict with local ``"repo"`` path.
    """
    from evaluation import SWEBenchLoader

    checkout_root = (
        Path(args.checkout_dir) if args.checkout_dir else Path(args.output) / "repos"
    )
    return SWEBenchLoader.checkout_instance(inst, checkout_root, log=log)


def _build_agent(agent_name: str, mode_str: str, model: str, output_dir: str):
    import inspect

    from forecast.agents.register import get_registered_agent
    from forecast.agents.base import AgentMode

    mode = AgentMode(mode_str)
    name = agent_name.lower()
    agent_cls = get_registered_agent(name)
    if agent_cls is None:
        raise ValueError(f"Unknown agent '{agent_name}'.")

    kwargs = {
        "name": agent_name,
        "mode": mode,
        "model": model,
        "output_dir": output_dir,
    }
    signature = inspect.signature(agent_cls.__init__)
    accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return agent_cls(**accepted)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="FORECAST: Improve repair agents with diagnoses",
    )
    command = parser.add_subparsers(dest="command", required=True)

    # ── collect ──────────────────────────────────────────────────────
    collect_parser = command.add_parser(
        "collect",
        help="Instrument code and collect execution traces (Step 1).",
    )
    add_source_argument(collect_parser)
    add_output_argument(collect_parser)
    add_swe_argument(collect_parser)
    add_dev_argument(collect_parser)
    add_workers_argument(collect_parser)
    add_log_dir_argument(collect_parser)
    add_checkout_dir_argument(collect_parser)
    collect_parser.add_argument(
        "-p",
        "--passing",
        type=str,
        help="Passing test cases (comma-separated or file path).",
    )
    collect_parser.add_argument(
        "-f",
        "--failing",
        type=str,
        help="Failing test cases (comma-separated or file path).",
    )

    # ── diagnose ─────────────────────────────────────────────────────
    diagnose_parser = command.add_parser(
        "diagnose",
        help="Derive a FORECAST diagnosis from collected traces (Step 2).",
    )
    add_source_argument(diagnose_parser)
    add_data_argument(diagnose_parser, required=True)
    add_output_argument(diagnose_parser)
    add_log_dir_argument(diagnose_parser)

    # ── prompt ───────────────────────────────────────────────────────
    prompt_parser = command.add_parser(
        "prompt",
        help="Extract the agent prompt, optionally injecting the diagnosis (Step 3).",
    )
    add_source_argument(prompt_parser)
    add_swe_argument(prompt_parser)
    add_dev_argument(prompt_parser)
    add_agent_argument(prompt_parser)
    add_model_argument(prompt_parser)
    add_mode_argument(prompt_parser)
    add_output_argument(prompt_parser)
    add_diagnosis_argument(prompt_parser)
    add_log_dir_argument(prompt_parser)
    add_checkout_dir_argument(prompt_parser)
    prompt_parser.add_argument(
        "--problem-statement",
        type=str,
        help="Problem statement text or path to a .txt file (non-SWE mode).",
    )
    prompt_parser.add_argument(
        "-f",
        "--failing",
        type=str,
        help="Failing test cases (non-SWE mode).",
    )
    prompt_parser.add_argument(
        "-p",
        "--passing",
        type=str,
        help="Passing test cases (non-SWE mode).",
    )
    prompt_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the prompt to stdout.",
    )

    # ── run ──────────────────────────────────────────────────────────
    run_parser = command.add_parser(
        "run",
        help="Run the repair agent: LLM call, save trajectory, extract patch (Step 4).",
    )
    add_source_argument(run_parser)
    add_swe_argument(run_parser)
    add_dev_argument(run_parser)
    add_agent_argument(run_parser)
    add_model_argument(run_parser)
    add_mode_argument(run_parser)
    add_output_argument(run_parser)
    add_diagnosis_argument(run_parser)
    add_log_dir_argument(run_parser)
    add_checkout_dir_argument(run_parser)
    run_parser.add_argument(
        "--problem-statement",
        type=str,
        help="Problem statement text or path (non-SWE mode).",
    )
    run_parser.add_argument(
        "-f",
        "--failing",
        type=str,
        help="Failing test cases (non-SWE mode).",
    )
    run_parser.add_argument(
        "-p",
        "--passing",
        type=str,
        help="Passing test cases (non-SWE mode).",
    )

    # ── evaluate ─────────────────────────────────────────────────────
    evaluate_parser = command.add_parser(
        "evaluate",
        help="Validate a patch against the test suite (Step 5).",
    )
    add_source_argument(evaluate_parser)
    add_swe_argument(evaluate_parser)
    add_dev_argument(evaluate_parser)
    add_agent_argument(evaluate_parser)
    add_model_argument(evaluate_parser)
    add_log_dir_argument(evaluate_parser)
    add_checkout_dir_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Output/working directory (used to derive default --checkout-dir).",
    )
    evaluate_parser.add_argument(
        "--patch",
        type=str,
        required=True,
        help="Path to the .diff / .patch file to validate.",
    )
    evaluate_parser.add_argument(
        "-f",
        "--failing",
        type=str,
        help="Failing test cases (non-SWE mode).",
    )
    evaluate_parser.add_argument(
        "-p",
        "--passing",
        type=str,
        help="Passing test cases (non-SWE mode).",
    )

    # ── tree (internal helper) ────────────────────────────────────────
    tree_parser = command.add_parser(
        "tree",
        help="Build a TracedRelevanceTree from collected event data (Step 1b).",
    )
    add_source_argument(tree_parser)
    add_swe_argument(tree_parser)
    add_dev_argument(tree_parser)
    add_data_argument(tree_parser, required=True)
    add_output_argument(tree_parser)
    add_workers_argument(tree_parser)
    add_log_dir_argument(tree_parser)
    add_checkout_dir_argument(tree_parser)
    tree_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the full tree structure to stdout.",
    )
    tree_parser.add_argument(
        "--no-prune-tree",
        action="store_true",
        default=False,
        help=(
            "Build the full tree instead of pruning passing runs to the failing "
            "footprint (pruning is on by default and lossless for diagnosis)."
        ),
    )
    tree_parser.add_argument(
        "--no-compress-tree",
        action="store_true",
        default=False,
        help="Write the tree pickle uncompressed (gzip is on by default).",
    )

    return parser.parse_args()


def main():
    load_plugins()

    args = parse_args()

    if args.command == "collect":
        _handle_collect(args)
    elif args.command == "diagnose":
        _handle_diagnose(args)
    elif args.command == "prompt":
        _handle_prompt(args)
    elif args.command == "run":
        _handle_run(args)
    elif args.command == "evaluate":
        _handle_evaluate(args)
    elif args.command == "tree":
        _handle_tree(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
