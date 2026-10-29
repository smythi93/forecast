"""
Run one agent of the repair evaluation in one mode with one model over
many instances in parallel.

Each instance runs in its own ``evaluation.repair_batch`` process (stage
``all``: prompt, run, harness evaluation), so agents that change process-wide
state (Agentless changes directory, the harness its log root) never interfere.
Runs resume: an instance whose harness verdict for this agent and mode already
exists is skipped, so an interrupted run is restarted with the same command.

Local models served by several Ollama servers (one per GPU) are spread over
them: worker ``i`` talks to endpoint ``i mod len(endpoints)``.

Example::

    python -m evaluation.parallel_repair --agent swe-agent --mode baseline \\
        --model proxy/claude-haiku-4-5-20251001 --instance-ids-file lite.txt \\
        --output-dir results --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from forecast.run_dates import remove_run_dates


def done(output_dir: Path, instance_id: str, agent: str, mode: str) -> bool:
    result = output_dir / instance_id / "evaluate" / agent.lower() / mode / "result.json"
    if not result.exists():
        return False
    try:
        return "resolved" in json.loads(result.read_text())
    except (json.JSONDecodeError, OSError):
        return False


def instance_dirs(output_dir: Path, instance_id: str, agent: str, mode: str) -> list[Path]:
    """The folders one agent and mode write for one instance."""
    return [output_dir / instance_id / stage / agent.lower() / mode
            for stage in ("prompt", "run", "evaluate")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agent", required=True)
    parser.add_argument("--mode", choices=["baseline", "forecast"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instance-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--ollama-endpoints",
        default="",
        help="Comma-separated Ollama URLs to spread workers over (local models).",
    )
    parser.add_argument("--timeout", type=int, default=6 * 3600, help="Seconds per instance.")
    args = parser.parse_args(argv)

    ids = [l.strip() for l in args.instance_ids_file.read_text().splitlines() if l.strip()]
    todo = [i for i in ids if not done(args.output_dir, i, args.agent, args.mode)]
    endpoints = [e for e in args.ollama_endpoints.split(",") if e]
    logs = args.output_dir / "_logs" / f"{args.agent.lower()}-{args.mode}"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"{args.agent}/{args.mode}/{args.model}: {len(todo)} to run, "
          f"{len(ids) - len(todo)} already done, {args.workers} workers", flush=True)

    slots = list(range(args.workers))
    slot_lock = threading.Lock()
    counts = {"resolved": 0, "unresolved": 0, "failed": 0}

    def run(instance_id: str) -> None:
        with slot_lock:
            slot = slots.pop()
        try:
            env = dict(os.environ)
            if endpoints:
                env["OLLAMA_BASE_URL"] = endpoints[slot % len(endpoints)]
            cmd = [sys.executable, "-m", "evaluation.repair_batch",
                   "--instance-ids", instance_id, "--agent", args.agent,
                   "--mode", args.mode, "--model", args.model, "--stage", "all",
                   "--output-dir", str(args.output_dir)]
            t0 = time.time()
            with open(logs / f"{instance_id}.log", "w") as log:
                try:
                    code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                          env=env, timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            remove_run_dates([logs / f"{instance_id}.log", *instance_dirs(
                args.output_dir, instance_id, args.agent, args.mode)])
            if done(args.output_dir, instance_id, args.agent, args.mode):
                verdict = json.loads((args.output_dir / instance_id / "evaluate" / args.agent.lower()
                                      / args.mode / "result.json").read_text())["resolved"]
                key = "resolved" if verdict else "unresolved"
            else:
                key = "failed"
            with slot_lock:
                counts[key] += 1
                print(f"{instance_id}: {key} (exit {code}, {time.time() - t0:.0f}s) "
                      f"| resolved {counts['resolved']} unresolved {counts['unresolved']} "
                      f"failed {counts['failed']}", flush=True)
        finally:
            with slot_lock:
                slots.append(slot)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, todo))
    print(f"done: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
