"""Record what the failing tests of each instance print on the unchanged code.

For the testinfo variant (variants.py), this runs the SWE-bench harness on the empty
patch, in the instance's image and with the developer's test patch, exactly as a patch is
evaluated, and keeps for every instance the FAIL_TO_PASS tests and the part of the test
output that reports their failure: pytest's FAILURES and ERRORS sections, and the FAIL and
ERROR blocks of Django's and SymPy's runners. Everything else, such as the setup commands
the harness logs, is cut, and an instance whose output shows no failure keeps the tail of
the test output. No LLM is involved.

Needs Docker and the instance images (FORECAST_IMAGE_NAMESPACE=swebench). Instances that
are already in <out json> are skipped, so the same command resumes.

Usage: python evaluation/revision/failures.py <out json> [--logs DIR] [--workers 8]
"""
import argparse
import json
import re
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
sys.path.insert(0, str(HERE.parent.parent))

INSTANCES = HERE.parent / "scripts" / "instances.txt"
#: The harness brackets the test run with these lines.
START, END = ">>>>> Start Test Output", ">>>>> End Test Output"
#: Where a report of a failure starts, per test runner.
FAILURE = re.compile(
    r"^(=+ (FAILURES|ERRORS) =+|_+ .+ _+|(FAIL|ERROR): \S.*|Traceback \(most recent call last\):)$",
    re.MULTILINE,
)
#: Where pytest's report of the failures ends.
SUMMARY = re.compile(r"^=+ (short test summary info|.*\b(passed|failed)\b.*) =+$", re.MULTILINE)
TAIL = 80


def failure_output(log):
    """The part of a harness test log that reports the failures."""
    if START in log:
        log = log.split(START, 1)[1].split(END, 1)[0]
    start = FAILURE.search(log)
    if not start:
        return "\n".join(log.strip().splitlines()[-TAIL:]), False
    report = log[start.start():]
    end = SUMMARY.search(report, 1)
    return (report[: end.start()] if end else report).strip(), True


def run(task_info, logs):
    from forecast.agents.environment import harness_evaluate

    instance_id = task_info["instance_id"]
    verdict = harness_evaluate(
        task_info, "", run_id=f"failures-{uuid.uuid4().hex[:8]}", model_name="empty",
        log_root=logs / instance_id,
    )
    log = Path(verdict["log_dir"]) / "test_output.txt"
    output, found = failure_output(log.read_text(errors="replace")) if log.exists() else ("", False)
    return instance_id, {
        "tests": json.loads(task_info["FAIL_TO_PASS"])
        if isinstance(task_info["FAIL_TO_PASS"], str) else list(task_info["FAIL_TO_PASS"]),
        "output": output,
        "failure_found": found,
        "completed": verdict["completed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--logs", type=Path, default=Path("failures-logs"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from evaluation.run_experiment import SWEBenchLoader

    done = json.loads(args.out.read_text()) if args.out.exists() else {}
    ids = [i for i in INSTANCES.read_text().split() if i not in done]
    instances = SWEBenchLoader.load_instances(subset="lite", instance_ids=ids) if ids else []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, inst["swe_task_info"], args.logs.resolve()) for inst in instances]
        for future in as_completed(futures):
            instance_id, entry = future.result()
            done[instance_id] = entry
            args.out.write_text(json.dumps(done, indent=2))
            print(f"{instance_id}: {len(entry['output'])} chars"
                  f"{'' if entry['failure_found'] else ', no failure found (tail kept)'}")
    missing = sum(not e["failure_found"] for e in done.values())
    print(f"{len(done)} instances in {args.out}, {missing} without a recognized failure report")


if __name__ == "__main__":
    main()
