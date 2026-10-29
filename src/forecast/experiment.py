"""
Experiment runners for SWE-bench and local projects.

This module provides:

* :class:`SWEBenchLoader` to load/filter SWE-bench instances,
* :class:`LocalProjectLoader` for local one-off instances, and
* :class:`ExperimentRunner` with step-wise pipeline helpers:
  1) collect,
  1b) tree,
  2) diagnose,
  3) prompt,
  4) run,
  5) evaluate.
"""

from __future__ import annotations

import inspect
import gzip
import json
import uuid
import multiprocessing as mp
import errno
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Any, Optional, Sequence

from forecast.agents.base import AgentMode, RepairContext, RepairResult
from forecast.agents.models.utils import get_model_client
from forecast.agents.register import get_registered_agent
from forecast.logger import LOGGER
from forecast.selection import (
    SelectedTests,
    select_passing_tests,
    select_tests_by_patch,
)
from forecast.test_ids import (
    _DISPLAY_STYLE_RE,
    normalize_pytest_test_ids,
    resolve_bare_test_ids_with_patch,
    resolve_pytest_node_id_against_repo,
    split_runnable_pytest_test_ids,
)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_DATASET_SUBSET = "lite"
_DEFAULT_DATASET_NAME = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}
_MATPLOTLIB_PASSING_FILTER_THRESHOLD = 250
_MATPLOTLIB_MAX_FAILING_FILES_FOR_FILTER = 6
_MATPLOTLIB_MAX_PASSING_TESTS = 80
_SYMPY_PASSING_FILTER_THRESHOLD = 120
_SYMPY_MAX_FAILING_FILES_FOR_FILTER = 12
_SYMPY_MAX_PASSING_TESTS = 40
_ABSOLUTE_MAX_PASSING_TESTS = 120
# Below this many candidate passing tests, a dedicated coverage run costs
# more than it saves, so the tests are traced as listed.
_SELECTION_THRESHOLD = 25
# Above this many candidate passing tests, cap what the coverage sweep measures.
# Coverage runs all candidates in one batched pytest process; a very large
# PASS_TO_PASS set (matplotlib instances reach 800+) overwhelms that single run
# and forces the slow per-test fallback. Tests in the failing tests' files are
# kept first, since those are the ones most likely to overlap the failure.
_COVERAGE_CANDIDATE_CAP = 200


def _prioritize_by_failing_files(
    failing: Sequence[str], passing: Sequence[str]
) -> list[str]:
    """Order *passing* so tests sharing a file with a failing test come first."""
    failing_files = {str(t).split("::", 1)[0] for t in failing}
    same = [t for t in passing if str(t).split("::", 1)[0] in failing_files]
    other = [t for t in passing if str(t).split("::", 1)[0] not in failing_files]
    return same + other

_HEAVY_REPO_PASSING_FILTERS: dict[str, dict[str, Any]] = {
    "matplotlib/matplotlib": {
        "passing_test_threshold": _MATPLOTLIB_PASSING_FILTER_THRESHOLD,
        "max_failing_test_files": _MATPLOTLIB_MAX_FAILING_FILES_FOR_FILTER,
        "max_passing_tests": _MATPLOTLIB_MAX_PASSING_TESTS,
        "prefer_non_image_variants": True,
    },
    "sympy/sympy": {
        "passing_test_threshold": _SYMPY_PASSING_FILTER_THRESHOLD,
        "max_failing_test_files": _SYMPY_MAX_FAILING_FILES_FOR_FILTER,
        "max_passing_tests": _SYMPY_MAX_PASSING_TESTS,
        "prefer_non_image_variants": False,
    },
}

LOG = LOGGER.getChild("evaluation")


def _process_rss_gb() -> float:
    """Resident set size of the current process, in GiB (0.0 if unavailable).

    Used as a cheap running estimate of the in-memory tree size for the tree
    build's memory budget (see :meth:`ExperimentRunner.build_diagnosis_tree`).
    """
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _parse_test_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return normalize_pytest_test_ids(str(v) for v in raw if str(v).strip())
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return normalize_pytest_test_ids(
                    str(v) for v in decoded if str(v).strip()
                )
        except json.JSONDecodeError:
            pass
        if "," in text:
            return normalize_pytest_test_ids(
                part.strip() for part in text.split(",") if part.strip()
            )
        return normalize_pytest_test_ids([text])
    return normalize_pytest_test_ids([str(raw)])


def _normalize_repo_slug(repo: Optional[str]) -> Optional[str]:
    if repo is None:
        return None
    slug = str(repo).strip()
    if not slug:
        return None
    if slug.startswith("https://github.com/"):
        slug = slug[len("https://github.com/") :]
    elif slug.startswith("http://github.com/"):
        slug = slug[len("http://github.com/") :]
    slug = slug.strip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    return slug or None




def _all_traces_identical(paths) -> int:
    """
    Number of runs when every run's trace is identical, else ``0``.

    Compared on decompressed length, not on the compressed bytes: gzip stores
    the source filename and mtime, so two byte-identical event streams still
    produce different files on disk. Length is a coarse proxy for "same
    execution", and a deliberately conservative one -- it only fires when
    *every* run matches, which no instance with a working test suite does.

    :param paths: Instance paths carrying the two event directories.
    :returns: The run count when all traces match, otherwise ``0``.
    """
    sizes = []
    for directory in (paths.failing_events_dir, paths.passing_events_dir):
        if not directory.is_dir():
            continue
        for trace in directory.iterdir():
            if not trace.is_file():
                continue
            try:
                with gzip.open(trace, "rb") as handle:
                    length = 0
                    while chunk := handle.read(1 << 20):
                        length += len(chunk)
            except OSError:
                return 0
            sizes.append(length)
    return len(sizes) if len(sizes) > 1 and len(set(sizes)) == 1 else 0


def _move_event_file_into_output(source: Path, destination: Path) -> bool:
    """
    Move one collected event file into the output directory.
    """
    if (
        not source.exists()
        or not source.is_file()
        or source.stat().st_size <= 0
    ):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.replace(destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)
        source.unlink(missing_ok=True)
    return True


def _event_file_move_worker(queue_obj) -> None:
    """
    Dedicated worker process that moves event files from tmp to output.
    """
    while True:
        item = queue_obj.get()
        if item is None:
            break
        source_str, destination_str = item
        source = Path(str(source_str))
        destination = Path(str(destination_str))
        try:
            _move_event_file_into_output(source, destination)
        except Exception:
            continue


def _extract_test_file_from_nodeid(test_id: str) -> Optional[str]:
    """
    Extract the pytest file path prefix from a node ID.
    """
    raw = str(test_id or "").strip()
    if not raw:
        return None
    file_part = raw.split("::", 1)[0].strip()
    if not file_part:
        return None
    if not file_part.endswith(".py"):
        return None
    return file_part.replace("\\", "/")


def _derive_test_excludes(test_ids: Sequence[str] | None) -> list[str]:
    """
    Build extra instrumentation exclude patterns from selected pytest node IDs.
    """
    excludes: set[str] = set()
    for test_id in test_ids or []:
        file_part = _extract_test_file_from_nodeid(str(test_id))
        if not file_part:
            continue
        path = Path(file_part)
        excludes.add(file_part)
        excludes.add(path.name)
        parent = str(path.parent).replace("\\", "/")
        if parent and parent != ".":
            excludes.add(parent)

    # Common test-file patterns that may live outside "tests/".
    excludes.update(
        {
            "conftest.py",
            "test_",
            "_test.py",
            "testing",
        }
    )
    return sorted(excludes)


def _instance_repo_slug(instance: Optional[dict[str, Any]]) -> Optional[str]:
    """
    Resolve the canonical repository slug for an instance payload.
    """
    def _infer_slug_from_instance_token(raw_value: Any) -> Optional[str]:
        raw = str(raw_value or "").strip()
        if not raw:
            return None
        token = Path(raw).name
        if "__" not in token or "-" not in token:
            return None
        repo_token, _ = token.rsplit("-", 1)
        if "__" not in repo_token:
            return None
        owner, repo = repo_token.split("__", 1)
        if not owner or not repo:
            return None
        return _normalize_repo_slug(f"{owner}/{repo}")

    if not isinstance(instance, dict):
        return None
    task_info = instance.get("swe_task_info")
    if isinstance(task_info, dict):
        nested = _normalize_repo_slug(task_info.get("repo"))
        if nested:
            return nested
        nested_inferred = _infer_slug_from_instance_token(task_info.get("instance_id"))
        if nested_inferred:
            return nested_inferred

    direct = _normalize_repo_slug(instance.get("repo"))
    if direct:
        inferred_direct = _infer_slug_from_instance_token(direct)
        if inferred_direct:
            return inferred_direct
        return direct
    inferred_instance = _infer_slug_from_instance_token(instance.get("instance_id"))
    if inferred_instance:
        return inferred_instance
    return _infer_slug_from_instance_token(instance.get("repo"))


def _filter_matplotlib_passing_tests(
    *,
    instance: Optional[dict[str, Any]],
    failing_tests: Sequence[str] | None,
    passing_tests: Sequence[str] | None,
    passing_test_threshold: int | None = None,
    max_failing_test_files: int | None = None,
    max_passing_tests: int | None = None,
) -> list[str]:
    """
    Trim PASS_TO_PASS sets for heavy repositories (e.g., matplotlib, sympy).

    Preference order:
    1) keep tests from failing test files when possible,
    2) cap total passing tests to ``max_passing_tests``,
    3) optionally prioritize non-image variants before image backend variants.
    """
    selected_passing = [
        str(test_id).strip()
        for test_id in (passing_tests or [])
        if str(test_id).strip()
    ]
    if not selected_passing:
        return selected_passing

    repo_slug = _instance_repo_slug(instance)
    policy = _HEAVY_REPO_PASSING_FILTERS.get(repo_slug or "")
    if policy is None:
        return selected_passing

    threshold = int(
        passing_test_threshold
        if passing_test_threshold is not None
        else policy["passing_test_threshold"]
    )
    max_failing_files = int(
        max_failing_test_files
        if max_failing_test_files is not None
        else policy["max_failing_test_files"]
    )
    max_allowed = int(
        max_passing_tests
        if max_passing_tests is not None
        else policy["max_passing_tests"]
    )
    prefer_non_image_variants = bool(policy.get("prefer_non_image_variants"))

    if len(selected_passing) < threshold:
        return selected_passing

    failing_files = {
        file_part
        for file_part in (
            _extract_test_file_from_nodeid(str(test_id))
            for test_id in (failing_tests or [])
        )
        if file_part
    }
    candidate = selected_passing
    if failing_files and len(failing_files) <= max_failing_files:
        same_file = [
            test_id
            for test_id in selected_passing
            if _extract_test_file_from_nodeid(test_id) in failing_files
        ]
        if same_file:
            candidate = same_file

    if max_allowed > 0 and len(candidate) > max_allowed:
        if prefer_non_image_variants:
            def _is_image_backend_variant(test_id: str) -> bool:
                lowered = test_id.lower()
                return any(
                    marker in lowered
                    for marker in ("[png", "[pdf", "[svg", "[ps", "[eps", "[agg")
                )

            non_image = [t for t in candidate if not _is_image_backend_variant(t)]
            image = [t for t in candidate if _is_image_backend_variant(t)]
            candidate = (non_image + image)[:max_allowed]
        else:
            candidate = candidate[:max_allowed]

    return candidate


def _apply_passing_tests_safety_cap(
    passing_tests: Sequence[str] | None,
    *,
    max_passing_tests: int = _ABSOLUTE_MAX_PASSING_TESTS,
) -> list[str]:
    """
    Apply a global hard cap for PASS_TO_PASS tests as a final safeguard.
    """
    selected = [str(test_id).strip() for test_id in (passing_tests or []) if str(test_id).strip()]
    max_allowed = int(max_passing_tests)
    if max_allowed <= 0:
        return selected
    if len(selected) <= max_allowed:
        return selected
    return selected[:max_allowed]


@dataclass(frozen=True)
class DiagnosisPaths:
    """
    Output paths for one instance diagnosis run.
    """

    root: Path
    events_dir: Path
    failing_events_dir: Path
    passing_events_dir: Path
    mapping_path: Path
    events_metadata_path: Path
    tree_path: Path
    diagnosis_json_path: Path
    diagnosis_md_path: Path
    diagnosis_txt_path: Path
    selection_path: Path


class SWEBenchLoader:
    """
    Load SWE-bench instances with optional filtering by ID and repository.
    """

    @staticmethod
    def _load_rows(
        *,
        subset: str = _DEFAULT_DATASET_SUBSET,
        dev: bool = False,
    ) -> list[dict[str, Any]]:
        dataset_name = _DEFAULT_DATASET_NAME.get(subset, _DEFAULT_DATASET_NAME["lite"])
        split = "dev" if dev else "test"
        try:
            from swebench.harness.utils import load_swebench_dataset

            rows = load_swebench_dataset(dataset_name, split=split)
            return [dict(row) for row in rows]
        except ImportError:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "Neither 'swebench' nor 'datasets' is installed. "
                    "Install one of them to load SWE-bench instances."
                ) from exc
            rows = load_dataset(dataset_name, split=split)
            return [dict(row) for row in rows]

    @classmethod
    def load_instances(
        cls,
        *,
        subset: str = _DEFAULT_DATASET_SUBSET,
        limit: Optional[int] = None,
        instance_ids: Optional[Sequence[str]] = None,
        repo: Optional[str] = None,
        dev: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Load SWE-bench instances and optionally filter by ids and/or repository.
        """
        selected_ids = {str(v).strip() for v in (instance_ids or []) if str(v).strip()}
        selected_repo = _normalize_repo_slug(repo)
        rows = cls._load_rows(subset=subset, dev=dev)
        instances: list[dict[str, Any]] = []

        for row in rows:
            iid = str(row.get("instance_id", "")).strip()
            if not iid:
                continue
            if selected_ids and iid not in selected_ids:
                continue

            row_repo = _normalize_repo_slug(row.get("repo"))
            if selected_repo and row_repo != selected_repo:
                continue

            failing = _parse_test_list(row.get("FAIL_TO_PASS", []))
            passing = _parse_test_list(row.get("PASS_TO_PASS", []))
            instances.append(
                {
                    "instance_id": iid,
                    "repo": row_repo or "",
                    "version": str(row.get("version", "")),
                    "problem_statement": str(row.get("problem_statement", "")),
                    "failing_tests": failing,
                    "passing_tests": passing,
                    "base_commit": str(row.get("base_commit", "")),
                    "test_patch": str(row.get("test_patch", "")),
                    "swe_task_info": dict(row),
                    "swe_setup_info": None,
                }
            )
            if limit is not None and len(instances) >= int(limit):
                break

        return instances

    @staticmethod
    def _resolve_test_patch(instance: dict[str, Any]) -> str:
        """
        Resolve test_patch text from normalized or raw SWE instance payload.
        """
        direct = str(instance.get("test_patch", "") or "")
        if direct.strip():
            return direct
        task_info = instance.get("swe_task_info")
        if isinstance(task_info, dict):
            nested = str(task_info.get("test_patch", "") or "")
            if nested.strip():
                return nested
        return ""

    @staticmethod
    def _apply_patch_file(repo_path: Path, patch_text: str) -> bool:
        """
        Apply *patch_text* with git. Returns True if newly applied, False if already applied.
        """
        if not patch_text.strip():
            return False

        with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as tmp:
            tmp.write(patch_text)
            patch_file = Path(tmp.name)
        try:
            check_apply = subprocess.run(
                ["git", "apply", "--check", str(patch_file)],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )
            if check_apply.returncode == 0:
                apply_result = subprocess.run(
                    ["git", "apply", str(patch_file)],
                    capture_output=True,
                    text=True,
                    cwd=repo_path,
                )
                if apply_result.returncode != 0:
                    raise RuntimeError(
                        f"git apply failed:\n{apply_result.stderr.strip()}"
                    )
                return True

            # If reverse-check succeeds, the patch has already been applied.
            check_reverse = subprocess.run(
                ["git", "apply", "--check", "-R", str(patch_file)],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )
            if check_reverse.returncode == 0:
                return False

            raise RuntimeError(
                "Unable to apply SWE test_patch cleanly.\n"
                f"git apply --check stderr:\n{check_apply.stderr.strip()}"
            )
        finally:
            patch_file.unlink(missing_ok=True)

    @classmethod
    def _ensure_test_patch_applied(
        cls,
        instance: dict[str, Any],
        checkout_path: Path,
        log=None,
    ) -> None:
        """
        Apply SWE ``test_patch`` on top of the checked-out base commit.
        """
        test_patch = cls._resolve_test_patch(instance)
        if not test_patch:
            return
        applied = cls._apply_patch_file(checkout_path, test_patch)
        if log:
            if applied:
                log.info("Applied SWE test_patch in %s", checkout_path)
            else:
                log.info("SWE test_patch already applied in %s", checkout_path)

    @staticmethod
    def checkout_instance(
        instance: dict[str, Any],
        base_dir: Path,
        log=None,
    ) -> dict[str, Any]:
        """
        Checkout one SWE-bench repository instance to ``base_dir/<instance_id>``.
        """
        repo_value = str(instance.get("repo", "")).strip()
        repo_path_candidate = Path(repo_value)
        if repo_value and repo_path_candidate.exists():
            resolved = str(repo_path_candidate.resolve())
            if log:
                log.info("Using existing local repo path: %s", resolved)
            updated = dict(instance, repo=resolved)
            SWEBenchLoader._ensure_test_patch_applied(updated, Path(resolved), log=log)
            return updated

        slug = _normalize_repo_slug(repo_value)
        if not slug:
            raise ValueError(f"Invalid repo value for instance: {instance.get('repo')}")

        base_commit = str(instance.get("base_commit", "")).strip()
        if not base_commit:
            raise ValueError(
                f"Instance {instance.get('instance_id')} has no base_commit."
            )

        instance_id = str(instance["instance_id"])
        checkout_path = Path(base_dir).resolve() / instance_id
        git_dir = checkout_path / ".git"

        if git_dir.exists():
            try:
                current = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=checkout_path,
                    text=True,
                ).strip()
                if current == base_commit:
                    if log:
                        log.info(
                            "Reusing checkout at %s (%s)",
                            checkout_path,
                            base_commit[:12],
                        )
                    updated = dict(instance, repo=str(checkout_path))
                    SWEBenchLoader._ensure_test_patch_applied(
                        updated,
                        checkout_path,
                        log=log,
                    )
                    return updated
                subprocess.run(
                    ["git", "fetch", "origin"],
                    capture_output=True,
                    text=True,
                    cwd=checkout_path,
                )
                checkout_existing = subprocess.run(
                    ["git", "checkout", base_commit],
                    capture_output=True,
                    text=True,
                    cwd=checkout_path,
                )
                if checkout_existing.returncode == 0:
                    if log:
                        log.info(
                            "Updated existing checkout at %s (%s)",
                            checkout_path,
                            base_commit[:12],
                        )
                    updated = dict(instance, repo=str(checkout_path))
                    SWEBenchLoader._ensure_test_patch_applied(
                        updated,
                        checkout_path,
                        log=log,
                    )
                    return updated
            except subprocess.CalledProcessError:
                pass
            # Repo exists but cannot be moved to requested commit: fresh clone.
            shutil.rmtree(checkout_path, ignore_errors=True)

        checkout_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{slug}"
        clone_result = subprocess.run(
            ["git", "clone", "-o", "origin", url, str(checkout_path)],
            capture_output=True,
            text=True,
        )
        if clone_result.returncode != 0:
            raise RuntimeError(
                f"git clone failed for {url}:\n{clone_result.stderr.strip()}"
            )

        checkout_result = subprocess.run(
            ["git", "checkout", base_commit],
            capture_output=True,
            text=True,
            cwd=checkout_path,
        )
        if checkout_result.returncode != 0:
            subprocess.run(
                ["git", "fetch", "origin"],
                capture_output=True,
                text=True,
                cwd=checkout_path,
            )
            checkout_result = subprocess.run(
                ["git", "checkout", base_commit],
                capture_output=True,
                text=True,
                cwd=checkout_path,
            )
        if checkout_result.returncode != 0:
            raise RuntimeError(
                f"git checkout {base_commit} failed:\n{checkout_result.stderr.strip()}"
            )

        if log:
            log.info("Checked out %s at %s", slug, base_commit[:12])
        updated = dict(instance, repo=str(checkout_path))
        SWEBenchLoader._ensure_test_patch_applied(updated, checkout_path, log=log)
        return updated


class LocalProjectLoader:
    """
    Build one local-project instance payload compatible with :class:`RepairContext`.
    """

    @staticmethod
    def load_instance(
        repo_path: str,
        problem_statement: str,
        failing_tests: list[str],
        passing_tests: Optional[list[str]] = None,
        instance_id: Optional[str] = None,
    ) -> dict[str, Any]:
        repo = str(Path(repo_path).resolve())
        try:
            base_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            base_commit = "HEAD"

        if instance_id is None:
            instance_id = f"{Path(repo).name}-{base_commit[:8]}"

        return {
            "instance_id": instance_id,
            "repo": repo,
            "version": base_commit[:8],
            "problem_statement": problem_statement,
            "failing_tests": list(failing_tests),
            "passing_tests": list(passing_tests or []),
            "base_commit": base_commit,
            "swe_task_info": None,
            "swe_setup_info": None,
        }


def _build_passing_partial(args):
    """
    Worker (multiprocessing): build a partial relevance tree from a chunk of
    passing event files, pruning against the given footprint *skeleton*.

    Returns the partial tree (the skeleton populated with this chunk's passing
    observations) for :func:`~forecast.tree.merge_into`. Module-level so it is
    picklable by the process pool. Returns ``None`` on failure so one bad chunk
    does not abort the whole build.

    :param args: ``(mapping_path, [(path, run_id), ...], skeleton)``.
    """
    mapping_path, specs, skeleton = args
    try:
        from sflkit.events.mapping import EventMapping

        from forecast.handler import ForecastHandler
        from forecast.runner.event_file import make_event_file

        mapping = EventMapping.load_from_file(Path(mapping_path))
        handler = ForecastHandler(prune=True, footprint=skeleton)
        for path, run_id in specs:
            handler.handle(make_event_file(path, run_id, mapping, failing=False))
        return handler.tree
    except Exception:  # pragma: no cover - defensive; logged by the parent
        LOGGER.exception("Parallel passing-partial worker failed")
        return None


class ExperimentRunner:
    """
    Batch helper to generate diagnoses and run repairs.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        verbose: bool = True,
        selection_strategy: str = "coverage-overlap",
        max_passing_tests: int = _ABSOLUTE_MAX_PASSING_TESTS,
        selection_threshold: int = _SELECTION_THRESHOLD,
    ):
        """
        :param output_dir: Directory holding per-instance artifacts.
        :param verbose: Whether to print progress to stdout.
        :param selection_strategy: How to choose which passing tests to trace —
            ``"coverage-overlap"`` (default) keeps the tests that execute code
            the failing tests also execute; ``"reference-patch"`` additionally
            restricts to tests reaching the reference fix and is therefore
            oracle-based; ``"name-heuristic"`` is the legacy per-repository
            name filter, which needs no coverage run; ``"none"`` traces every
            listed test.
        :param max_passing_tests: Upper bound on traced passing tests.  ``0``
            removes the bound.
        :param selection_threshold: Candidate count above which selection is
            worth its own coverage run.  Smaller test sets are traced as is.
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.selection_strategy = selection_strategy
        self.max_passing_tests = int(max_passing_tests)
        self.selection_threshold = int(selection_threshold)

    @staticmethod
    def _build_agent(
        *,
        agent_name: str,
        mode: AgentMode,
        model: str,
        output_dir: Path,
    ):
        agent_cls = get_registered_agent(agent_name)
        if agent_cls is None:
            raise ValueError(f"Unknown agent '{agent_name}'.")

        kwargs: dict[str, Any] = {
            "name": agent_name,
            "mode": mode,
            "model": model,
            "output_dir": str(output_dir),
        }
        signature = inspect.signature(agent_cls.__init__)
        accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return agent_cls(**accepted)

    def _instance_dir(self, instance_id: str) -> Path:
        return self.output_dir / str(instance_id)

    def _instance_report_path(self, instance_id: str) -> Path:
        return self._instance_dir(instance_id) / "report.json"

    @staticmethod
    def _safe_file_name(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "unknown"
        out = []
        for ch in raw:
            if ch.isalnum() or ch in {"-", "_", "."}:
                out.append(ch)
            else:
                out.append("_")
        return "".join(out)

    @staticmethod
    def _build_swe_subject(
        *,
        instance: dict[str, Any],
        selected_failing: Sequence[str],
        selected_passing: Sequence[str],
    ):
        """
        Build an SWESubject from normalized instance payload.
        """
        from forecast.subject import SWESubject

        row: dict[str, Any] = {}
        raw_task_info = instance.get("swe_task_info")
        if isinstance(raw_task_info, dict):
            row.update(raw_task_info)
        row.setdefault("instance_id", str(instance.get("instance_id", "")))
        row.setdefault("repo", str(instance.get("repo", "")))
        row.setdefault("base_commit", str(instance.get("base_commit", "")))
        row.setdefault("patch", str(instance.get("patch", "")))
        row.setdefault("test_patch", str(instance.get("test_patch", "")))
        row.setdefault("problem_statement", str(instance.get("problem_statement", "")))
        row.setdefault("version", str(instance.get("version", "")))
        row["FAIL_TO_PASS"] = list(selected_failing)
        row["PASS_TO_PASS"] = list(selected_passing)
        return SWESubject.from_dataset_row(
            row,
            workers=4,
            timeout=120,
        )

    def _update_instance_report(
        self,
        *,
        instance_id: str,
        step: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Merge one step payload into ``<instance>/report.json``.
        """
        report_path = self._instance_report_path(instance_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report: dict[str, Any]
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                report = {}
        else:
            report = {}

        report.setdefault("instance_id", instance_id)
        report.setdefault("steps", {})
        report["steps"][step] = payload

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _test_file_of(test_id: str, discovered_by_name: dict[str, set[str]]) -> str | None:
        """
        The test file a test identifier refers to, whatever style it is written in.

        SWE-bench writes identifiers three ways and only one of them carries the
        path: pytest node ids (``dir/test_x.py::test_y``), django's display style
        (``test_y (dir.test_x.Case)``), and bare function names (``test_y``,
        which sympy uses). Reading only the first style is what made the
        fallback give up on django and sympy instances before it had looked at
        anything.

        :param test_id: The identifier to place.
        :param discovered_by_name: Test function name -> files that define it,
            used to place identifiers that carry no path of their own.
        :returns: The file, or ``None`` when it cannot be placed unambiguously.
        """
        if "::" in test_id:
            return test_id.split("::", 1)[0]
        match = _DISPLAY_STYLE_RE.match(test_id)
        if match:
            # "test_y (dir.test_x.Case)" -- the context is a dotted module path
            # with the class appended; the module is everything but the class.
            parts = match.group("context").split(".")
            if len(parts) > 1:
                return "/".join(parts[:-1]) + ".py"
            return None
        # A bare function name: the only way to place it is to find where that
        # name is defined, which discovery already told us.
        files = discovered_by_name.get(test_id.strip())
        if files and len(files) == 1:
            return next(iter(files))
        return None

    def _discover_passing_tests(
        self,
        *,
        subject,
        failing: Sequence[str],
        limit: int = 200,
    ) -> list[str]:
        """
        Find passing tests for an instance whose ``PASS_TO_PASS`` is empty.

        Six of the 300 SWE-bench Lite instances list no ``PASS_TO_PASS`` at all.
        Diagnosis contrasts failing runs against passing ones, so those
        instances produce an empty cause chain by construction -- the benchmark
        simply never enumerated the tests that pass, and the repository is full
        of them.

        Candidates are ranked by how close they sit to the failing tests: the
        same file first, then the same directory, then the rest of the suite.
        Closeness is what makes a candidate useful -- a passing test that shares
        no function with the failure contributes no contrasting feature vector
        -- but an unhelpful contrast still beats no diagnosis, so the search
        widens rather than stopping. It has to: in ``pytest-7168`` the eleven
        failing tests *are* every test in their file, and restricting candidates
        to that file found nothing at all.

        Running them is what establishes that they pass: the benchmark makes no
        claim about them, so the claim has to be ours.

        :param subject: Subject used to discover and run tests.
        :param failing: The failing tests, whose location seeds the search.
        :param limit: Stop after this many candidates.
        :returns: Test identifiers that were discovered and observed to pass.
        """
        if not failing:
            return []
        try:
            discovered = list(subject.collect_test_ids())
        except Exception as exc:
            LOG.warning("Could not discover tests for a fallback passing set: %s", exc)
            return []
        if not discovered:
            LOG.warning("Test discovery returned nothing; no passing set to recover.")
            return []

        discovered_by_name: dict[str, set[str]] = {}
        for test_id in discovered:
            if "::" in test_id:
                path, _, name = test_id.partition("::")
                discovered_by_name.setdefault(name.split("::")[-1], set()).add(path)

        failing_files = {
            file
            for file in (
                self._test_file_of(test_id, discovered_by_name) for test_id in failing
            )
            if file
        }
        failing_dirs = {file.rsplit("/", 1)[0] for file in failing_files if "/" in file}

        failing_set = set(failing)
        # Compare on the trailing path so a candidate still matches when
        # discovery and the benchmark disagree about the leading directory.
        def _rank(test_id: str) -> int:
            file = self._test_file_of(test_id, discovered_by_name)
            if file is None:
                return 2
            if any(file.endswith(f) or f.endswith(file) for f in failing_files):
                return 0
            directory = file.rsplit("/", 1)[0] if "/" in file else ""
            if directory and any(
                directory.endswith(d) or d.endswith(directory) for d in failing_dirs
            ):
                return 1
            return 2

        candidates = sorted(
            (test_id for test_id in discovered if test_id not in failing_set),
            key=_rank,
        )[:limit]
        if not candidates:
            LOG.warning(
                "Discovery found no test outside the %d failing one(s); "
                "the diagnosis will have nothing to contrast against.",
                len(failing),
            )
            return []

        try:
            results = subject.run_tests(candidates)
        except Exception as exc:
            LOG.warning("Could not run candidate passing tests: %s", exc)
            return []

        passing = [
            result.test_id
            for result in results
            if result is not None and getattr(result, "passed", False)
        ]
        LOG.info(
            "PASS_TO_PASS was empty; recovered %d passing test(s) from %d "
            "candidate(s) ranked by closeness to the failing tests.",
            len(passing),
            len(candidates),
        )
        return passing

    def _select_diagnosis_tests(
        self,
        *,
        subject,
        instance: dict[str, Any],
        selected_failing: Sequence[str],
        selected_passing: Sequence[str],
    ) -> SelectedTests:
        """
        Decide which passing tests to trace for this instance.

        Tracing is far more expensive than plain execution, and a passing test
        that shares no code with the failing tests cannot influence the
        diagnosis, so coverage — which is cheap to collect — is used to drop
        those tests before the expensive stage.

        The coverage run itself only pays off for large candidate sets, so
        smaller ones are traced as listed.

        :param subject: Subject used to collect coverage, before instrumentation.
        :param instance: The SWE-bench instance payload.
        :param selected_failing: Failing tests, kept in full.
        :param selected_passing: Candidate passing tests.
        :returns: The resulting selection.
        """
        failing = list(selected_failing)
        passing = list(selected_passing)
        strategy = self.selection_strategy

        # PASS_TO_PASS is a curated regression list, not an inventory of the
        # tests that pass: six SWE-bench Lite instances carry an empty one. A
        # diagnosis contrasts failing runs against passing ones, so with no
        # passing candidate there is nothing to contrast and the cause chain
        # comes out empty by construction. The repository still has passing
        # tests, so find them rather than give up.
        if strategy != "none" and not passing and failing:
            passing = self._discover_passing_tests(subject=subject, failing=failing)

        if strategy == "none" or not passing:
            return SelectedTests(failing=failing, passing=passing, strategy="all")

        if strategy == "name-heuristic":
            # Legacy behaviour: trim known-heavy repositories by test-name
            # patterns alone. Needs no coverage run, so it stays available for
            # environments where collecting coverage is not practical.
            kept = _apply_passing_tests_safety_cap(
                _filter_matplotlib_passing_tests(
                    instance=instance,
                    failing_tests=failing,
                    passing_tests=passing,
                ),
                max_passing_tests=self.max_passing_tests,
            )
            keep = set(kept)
            return SelectedTests(
                failing=failing,
                passing=kept,
                dropped=[t for t in passing if t not in keep],
                strategy="name-heuristic",
                statistics={"candidates": len(passing), "selected": len(kept)},
            )

        if (
            len(passing) <= self.selection_threshold
            and not (self.max_passing_tests and len(passing) > self.max_passing_tests)
        ):
            return SelectedTests(
                failing=failing,
                passing=passing,
                strategy="all",
                statistics={"reason": "below selection threshold"},
            )

        # Bound the candidate set before the coverage sweep so a huge
        # PASS_TO_PASS list stays on the fast batched path instead of falling
        # back to per-test coverage.
        coverage_dropped: list[str] = []
        if _COVERAGE_CANDIDATE_CAP and len(passing) > _COVERAGE_CANDIDATE_CAP:
            prioritized = _prioritize_by_failing_files(failing, passing)
            passing = prioritized[:_COVERAGE_CANDIDATE_CAP]
            coverage_dropped = prioritized[_COVERAGE_CANDIDATE_CAP:]
            if self.verbose:
                print(
                    "  - collect info: capped coverage candidates to "
                    f"{len(passing)} of {len(passing) + len(coverage_dropped)} "
                    "passing tests before the coverage sweep."
                )

        try:
            results, _ = subject.collect_coverage(failing + passing)
        except Exception as exc:
            # Coverage is an optimisation; if it fails we still have to produce
            # a diagnosis, so fall back to tracing a bounded prefix.
            LOG.warning(
                "Coverage-based test selection failed for %s (%s); "
                "falling back to a plain cap.",
                instance.get("instance_id"),
                exc,
            )
            capped = (
                passing[: self.max_passing_tests]
                if self.max_passing_tests
                else passing
            )
            return SelectedTests(
                failing=failing,
                passing=capped,
                dropped=passing[len(capped):],
                strategy="cap",
                statistics={"reason": f"coverage failed: {exc}"},
            )

        by_id = {result.test_id: result for result in results if result is not None}
        failing_results = [by_id[t] for t in failing if t in by_id]
        passing_results = [by_id[t] for t in passing if t in by_id]

        if strategy == "reference-patch":
            selection = select_tests_by_patch(
                patch=str(instance.get("patch") or ""),
                failing_results=failing_results,
                passing_results=passing_results,
                max_passing_tests=self.max_passing_tests,
            )
        else:
            selection = select_passing_tests(
                failing_results=failing_results,
                passing_results=passing_results,
                max_passing_tests=self.max_passing_tests,
            )
        # A filter may narrow the contrast set; it must never empty it. With no
        # passing run there is nothing to contrast against and the cause chain
        # is empty by construction, which is a worse outcome than a weakly
        # related contrast: scikit-learn-10297 had 28 usable passing tests and
        # coverage overlap discarded every one, so the instance produced no
        # diagnosis at all. Fall back to a bounded prefix of the candidates.
        if passing and not selection.passing:
            fallback = (
                passing[: self.max_passing_tests]
                if self.max_passing_tests
                else passing
            )
            LOG.warning(
                "%s: coverage selection dropped all %d passing candidate(s); "
                "keeping %d unfiltered so the diagnosis has something to "
                "contrast against.",
                instance.get("instance_id"),
                len(passing),
                len(fallback),
            )
            selection = _dc_replace(
                selection,
                passing=fallback,
                dropped=[t for t in passing if t not in set(fallback)],
                strategy=selection.strategy + "+nonempty-fallback",
            )

        # Candidates dropped before the coverage sweep were never traced; record
        # them alongside the ones selection dropped.
        if coverage_dropped:
            selection = _dc_replace(
                selection, dropped=list(selection.dropped) + coverage_dropped
            )
        return selection

    @staticmethod
    def _write_selection(paths: DiagnosisPaths, selection: SelectedTests) -> None:
        """
        Persist a selection so later stages know which tests were traced.
        """
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.selection_path.write_text(
            json.dumps(
                {
                    "strategy": selection.strategy,
                    "uses_oracle": selection.uses_oracle,
                    "failing_tests": selection.failing,
                    "passing_tests": selection.passing,
                    "dropped_tests": selection.dropped,
                    "statistics": selection.statistics,
                },
                indent=2,
            )
        )

    @staticmethod
    def _read_selection(paths: DiagnosisPaths) -> Optional[SelectedTests]:
        """
        Load a previously persisted selection, or ``None`` when absent.
        """
        if not paths.selection_path.exists():
            return None
        try:
            raw = json.loads(paths.selection_path.read_text())
        except (OSError, ValueError):
            return None
        return SelectedTests(
            failing=list(raw.get("failing_tests") or []),
            passing=list(raw.get("passing_tests") or []),
            dropped=list(raw.get("dropped_tests") or []),
            strategy=str(raw.get("strategy") or "all"),
            uses_oracle=bool(raw.get("uses_oracle")),
            statistics=dict(raw.get("statistics") or {}),
        )

    def _diagnosis_paths(self, instance_id: str) -> DiagnosisPaths:
        root = self._instance_dir(instance_id) / "diagnosis"
        events_dir = root / "events"
        return DiagnosisPaths(
            root=root,
            events_dir=events_dir,
            failing_events_dir=events_dir / "failing",
            passing_events_dir=events_dir / "passing",
            mapping_path=events_dir / "mapping.json",
            events_metadata_path=events_dir / "metadata.json",
            tree_path=root / "tree.pkl",
            diagnosis_json_path=root / "diagnosis.json",
            diagnosis_md_path=root / "diagnosis.md",
            diagnosis_txt_path=root / "diagnosis.txt",
            selection_path=root / "test_selection.json",
        )

    @staticmethod
    def _has_complete_event_collection(
        paths: DiagnosisPaths,
        *,
        repo_path: Path,
        failing_tests: Sequence[str] | None,
        passing_tests: Sequence[str] | None,
    ) -> bool:
        """
        Return True when collected event artifacts are complete for expected tests.
        """
        if not paths.mapping_path.exists():
            return False
        if (
            not paths.failing_events_dir.exists()
            or not paths.passing_events_dir.exists()
        ):
            return False

        failing_ids, _ = split_runnable_pytest_test_ids(failing_tests or [])
        passing_ids, _ = split_runnable_pytest_test_ids(passing_tests or [])
        failing_ids = [
            resolve_pytest_node_id_against_repo(test_id, repo_path)
            for test_id in failing_ids
        ]
        passing_ids = [
            resolve_pytest_node_id_against_repo(test_id, repo_path)
            for test_id in passing_ids
        ]
        expected = failing_ids or passing_ids

        def _non_empty_names(directory: Path) -> set[str]:
            names: set[str] = set()
            for item in directory.iterdir():
                if item.is_file() and item.stat().st_size > 0:
                    names.add(item.name)
            return names

        existing_failing = _non_empty_names(paths.failing_events_dir)
        existing_passing = _non_empty_names(paths.passing_events_dir)
        if not expected:
            return bool(existing_failing or existing_passing)

        from forecast.subject import safe

        expected_failing = {safe(test_id) for test_id in failing_ids}
        expected_passing = {safe(test_id) for test_id in passing_ids}
        return (
            existing_failing == expected_failing
            and existing_passing == expected_passing
        )

    def _write_collection_failure_logs(
        self,
        *,
        repo_path: Path,
        tests: Sequence[str],
        output_dir: Path,
        reason: str,
        python: str = sys.executable,
        env: Optional[dict[str, str]] = None,
        timeout: int = 120,
        subject: Any | None = None,
        max_tests: Optional[int] = None,
        error: Optional[BaseException] = None,
        error_trace: Optional[str] = None,
    ) -> Path:
        """
        Persist pytest stdout/stderr logs when event collection fails.
        """
        selected_tests = list(tests)
        truncated = False
        if max_tests is not None and max_tests >= 0 and len(selected_tests) > max_tests:
            selected_tests = selected_tests[:max_tests]
            truncated = True

        output_dir.mkdir(parents=True, exist_ok=True)
        test_logs_dir = output_dir / "tests"
        test_logs_dir.mkdir(parents=True, exist_ok=True)

        def _run_in_subject(
            test_id: str,
        ) -> tuple[bool, Optional[int], str, str, Optional[list[str]]]:
            """
            Run one pytest test in the SWE/Docker subject context when available.
            Returns: (used_subject_context, returncode, stdout, stderr, command)
            """
            if subject is None:
                return False, None, "", "", None
            client = getattr(subject, "client", None)
            image = getattr(subject, "image", None)
            src_root = getattr(subject, "path", None)
            if client is None or not image or src_root is None:
                return False, None, "", "", None

            from forecast.runner.utils import write_file_to_container
            from forecast.test_ids import normalize_pytest_test_id

            normalized_test_id = normalize_pytest_test_id(str(test_id))
            container = None
            try:
                container = client.containers.create(
                    image, command="sleep infinity", tty=True
                )
                container.start()
                patch = getattr(subject, "test_patch", None)
                if patch:
                    patch_file = "/tmp/patch.diff"
                    write_file_to_container(container, patch_file, patch)
                    container.exec_run(
                        ["git", "apply", patch_file], workdir=str(src_root)
                    )

                cmd = ["timeout", f"{timeout}s"]
                conda_env = getattr(subject, "conda_env", None)
                if conda_env:
                    cmd += ["conda", "run", "-n", conda_env]
                cmd += [
                    "pytest",
                    normalized_test_id,
                    "-rA",
                    "--tb=short",
                    "-q",
                ]
                exit_code, output = container.exec_run(
                    cmd,
                    workdir=str(src_root),
                    demux=True,
                )
                if isinstance(output, tuple):
                    out_bytes = output[0] or b""
                    err_bytes = output[1] or b""
                else:
                    out_bytes = output or b""
                    err_bytes = b""
                return (
                    True,
                    int(exit_code),
                    out_bytes.decode("utf-8", errors="replace"),
                    err_bytes.decode("utf-8", errors="replace"),
                    cmd,
                )
            except Exception:
                return False, None, "", "", None
            finally:
                if container is not None:
                    try:
                        container.stop()
                    finally:
                        container.remove()

        entries: list[dict[str, Any]] = []
        for test_id in selected_tests:
            cmd = [
                python,
                "-m",
                "pytest",
                str(test_id),
                "-rA",
                "--tb=short",
                "-q",
            ]
            t0 = time.perf_counter()
            timeout_hit = False
            returncode: Optional[int] = None
            stdout = ""
            stderr = ""
            executed_cmd: list[str] = list(cmd)
            (
                used_subject_context,
                subject_code,
                subject_stdout,
                subject_stderr,
                subject_cmd,
            ) = _run_in_subject(str(test_id))
            if used_subject_context:
                returncode = subject_code
                stdout = subject_stdout
                stderr = subject_stderr
                timeout_hit = returncode == 124
                if subject_cmd:
                    executed_cmd = list(subject_cmd)
            else:
                try:
                    run = subprocess.run(
                        executed_cmd,
                        capture_output=True,
                        text=True,
                        cwd=repo_path,
                        env=dict(env) if env is not None else None,
                        timeout=timeout,
                    )
                    returncode = run.returncode
                    stdout = run.stdout or ""
                    stderr = run.stderr or ""
                except subprocess.TimeoutExpired as exc:
                    timeout_hit = True
                    stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                    stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""

            log_path = test_logs_dir / f"{self._safe_file_name(str(test_id))}.log"
            log_path.write_text(
                (
                    f"test_id: {test_id}\n"
                    f"reason: {reason}\n"
                    f"context: {'subject' if used_subject_context else 'local'}\n"
                    f"command: {executed_cmd}\n"
                    f"cwd: {repo_path}\n"
                    f"python: {python}\n"
                    f"timeout_seconds: {timeout}\n"
                    f"timeout: {timeout_hit}\n"
                    f"returncode: {returncode}\n\n"
                    "=== STDOUT ===\n"
                    f"{stdout}\n\n"
                    "=== STDERR ===\n"
                    f"{stderr}\n"
                ),
                encoding="utf-8",
            )
            entries.append(
                {
                    "test_id": str(test_id),
                    "log_path": str(log_path),
                    "returncode": returncode,
                    "timeout": timeout_hit,
                    "elapsed_seconds": time.perf_counter() - t0,
                }
            )

        summary: dict[str, Any] = {
            "reason": reason,
            "repo_path": str(repo_path),
            "test_count": len(selected_tests),
            "requested_test_count": len(tests),
            "truncated": truncated,
            "tests": entries,
        }
        if error is not None:
            summary["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        if error_trace:
            summary["traceback"] = error_trace

        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_dir

    def _collect_events_for_repo(
        self,
        repo_path: Path,
        *,
        output_events_dir: Path,
        output_mapping_path: Path,
        failing_tests: Optional[Sequence[str]] = None,
        passing_tests: Optional[Sequence[str]] = None,
        instance: Optional[dict[str, Any]] = None,
        failure_log_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Collect and persist events for one repository.
        """
        selected_failing = [
            str(test_id).strip()
            for test_id in (failing_tests or [])
            if str(test_id).strip()
        ]
        selected_passing = [
            str(test_id).strip()
            for test_id in (passing_tests or [])
            if str(test_id).strip()
        ]
        # Some projects (e.g. sympy) report tests as bare function names, and the
        # failing ones are added by the instance's test_patch, so they carry no
        # module and do not exist in the base checkout. Resolve them to node IDs
        # via the test_patch before the runnable/skip split drops them.
        _test_patch = str((instance or {}).get("test_patch") or "")
        if not _test_patch.strip():
            _task_info = (instance or {}).get("swe_task_info")
            if isinstance(_task_info, dict):
                _test_patch = str(_task_info.get("test_patch") or "")
        if _test_patch.strip():
            selected_failing = resolve_bare_test_ids_with_patch(
                selected_failing, _test_patch, repo_path
            )
            selected_passing = resolve_bare_test_ids_with_patch(
                selected_passing, _test_patch, repo_path
            )
        selected_failing, skipped_failing = split_runnable_pytest_test_ids(
            selected_failing
        )
        selected_passing, skipped_passing = split_runnable_pytest_test_ids(
            selected_passing
        )
        selected_failing = [
            resolve_pytest_node_id_against_repo(test_id, repo_path)
            for test_id in selected_failing
        ]
        selected_passing = [
            resolve_pytest_node_id_against_repo(test_id, repo_path)
            for test_id in selected_passing
        ]
        filter_repo_slug = _instance_repo_slug(instance) or "unknown"
        original_passing_count = len(selected_passing)
        # Which passing tests are actually worth tracing is decided further
        # below, once the subject exists and coverage can be collected.
        selection = SelectedTests(
            failing=list(selected_failing),
            passing=list(selected_passing),
            strategy="all",
        )
        skipped_unrunnable = skipped_failing + skipped_passing
        if skipped_unrunnable and self.verbose:
            sample = ", ".join(skipped_unrunnable[:5])
            print(
                "  - collect warning: skipped unrunnable test labels "
                f"({len(skipped_unrunnable)}): {sample}"
            )
        selected_tests = sorted(set(selected_failing) | set(selected_passing))
        if (failing_tests or passing_tests) and not selected_tests:
            raise RuntimeError(
                "No runnable pytest test IDs remained after normalization. "
                "The provided FAIL_TO_PASS/PASS_TO_PASS values appear to be "
                "non-pytest labels."
            )
        overall_t0 = time.perf_counter()
        instrument_seconds = 0.0
        collect_seconds = 0.0
        copy_seconds = 0.0

        with tempfile.TemporaryDirectory(prefix="forecast-diagnosis-") as tmp:
            tmp_root = Path(tmp)
            instrumented = tmp_root / "instrumented"
            tmp_mapping_path = tmp_root / "mapping.json"
            tmp_events_dir = tmp_root / "events"
            log_repo_path = repo_path

            try:
                from forecast.instrument import (
                    DEFAULT_EXCLUDES as _DEFAULT_INSTRUMENT_EXCLUDES,
                )
            except Exception:
                _DEFAULT_INSTRUMENT_EXCLUDES = []
            from forecast.subject import safe

            if not isinstance(instance, dict) or instance.get("swe_task_info") is None:
                raise ValueError(
                    "Evaluation event collection requires SWE instance payload "
                    "with 'swe_task_info'."
                )
            subject = self._build_swe_subject(
                instance=instance,
                selected_failing=selected_failing,
                selected_passing=selected_passing,
            )
            subject.setup()

            selection = self._select_diagnosis_tests(
                subject=subject,
                instance=instance,
                selected_failing=selected_failing,
                selected_passing=selected_passing,
            )
            if len(selection.passing) != original_passing_count and self.verbose:
                print(
                    f"  - collect info: {selection.strategy} selection kept "
                    f"{len(selection.passing)} of {original_passing_count} "
                    "passing tests."
                )
            selected_failing = list(selection.failing)
            selected_passing = list(selection.passing)
            selected_tests = sorted(set(selected_failing) | set(selected_passing))

            subject.test_ids = list(selected_tests)
            subject.fail_ids = set(selected_failing)
            subject.failing_test_ids = set(selected_failing)
            subject.passing_test_ids = set(selected_passing)

            t0 = time.perf_counter()
            extra_excludes = _derive_test_excludes(selected_tests)
            excludes = sorted(
                {
                    *[str(e) for e in (_DEFAULT_INSTRUMENT_EXCLUDES or [])],
                    *extra_excludes,
                }
            )
            subject.instrument(
                src=repo_path,
                dst=instrumented,
                mapping_path=tmp_mapping_path,
                excludes=excludes,
            )
            instrument_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            collection_logs_written = False
            event_file_mapping: dict[str, Path] = {}
            event_output_mapping: dict[str, Path] = {}
            ordered_test_ids: list[str] = list(selected_tests)
            try:
                output_events_dir.parent.mkdir(parents=True, exist_ok=True)
                if output_events_dir.exists():
                    shutil.rmtree(output_events_dir)
                output_failing_dir = output_events_dir / "failing"
                output_passing_dir = output_events_dir / "passing"
                output_failing_dir.mkdir(parents=True, exist_ok=True)
                output_passing_dir.mkdir(parents=True, exist_ok=True)

                if selected_tests:
                    failing_ids = set(selected_failing)
                    passing_ids = set(selected_passing)
                else:
                    failing_ids = set(subject.get_failing_tests())
                    all_test_ids = set(subject.test_ids) or (
                        set(subject.passing_test_ids) | set(subject.failing_test_ids)
                    )
                    if not all_test_ids:
                        all_test_ids = set(subject.collect_test_ids())
                        subject.test_ids = sorted(all_test_ids)
                    passing_ids = set(subject.passing_test_ids) or (
                        all_test_ids - failing_ids
                    )

                failing_dir = tmp_events_dir / "failing"
                passing_dir = tmp_events_dir / "passing"
                failing_dir.mkdir(parents=True, exist_ok=True)
                passing_dir.mkdir(parents=True, exist_ok=True)
                for test_id in sorted(failing_ids):
                    event_file_mapping[test_id] = failing_dir / safe(test_id)
                    event_output_mapping[test_id] = output_failing_dir / safe(test_id)
                for test_id in sorted(passing_ids):
                    event_file_mapping[test_id] = passing_dir / safe(test_id)
                    event_output_mapping[test_id] = output_passing_dir / safe(test_id)

                ordered_test_ids = list(subject.test_ids or selected_tests)
                move_ctx = mp.get_context("spawn")
                move_queue = move_ctx.Queue()
                mover_process = move_ctx.Process(
                    target=_event_file_move_worker,
                    args=(move_queue,),
                )
                mover_process.start()

                def _notify_event_completed(
                    test_id: str,
                    outcome_ok: bool,
                    event_path: Path,
                ) -> None:
                    if not outcome_ok:
                        return
                    dst = event_output_mapping.get(test_id)
                    if dst is None:
                        return
                    try:
                        move_queue.put((str(event_path), str(dst)), block=False)
                    except Exception:
                        return

                try:
                    outcomes = list(
                        subject.collect_all_events(
                            event_file_mapping,
                            on_complete=_notify_event_completed,
                        )
                        or []
                    )
                finally:
                    try:
                        move_queue.put(None)
                    except Exception:
                        pass
                    mover_process.join(timeout=120)
                    if mover_process.is_alive():
                        LOG.warning(
                            "Event move worker did not exit cleanly; terminating.",
                        )
                        mover_process.terminate()
                        mover_process.join(timeout=5)
                    try:
                        move_queue.close()
                        move_queue.join_thread()
                    except Exception:
                        pass

                for test_id, source_path in event_file_mapping.items():
                    destination_path = event_output_mapping.get(test_id)
                    if destination_path is None:
                        continue
                    try:
                        _move_event_file_into_output(source_path, destination_path)
                    except Exception:
                        continue

                # A failing run defines the tree's structure, so losing one is
                # fatal. A passing run that produced nothing contributes nothing
                # to the tree either, so dropping it costs nothing -- whereas
                # treating it as fatal discards every other run of the instance.
                # sympy-11897 aborted on a single passing test that exited 0 and
                # simply emitted no events, taking its other 95 runs with it.
                failed_tests: list[str] = []
                dropped_passing: list[str] = []
                for idx, test_id in enumerate(ordered_test_ids):
                    path = Path(event_output_mapping[test_id])
                    outcome_ok = bool(outcomes[idx]) if idx < len(outcomes) else False
                    has_events = (
                        path.exists() and path.is_file() and path.stat().st_size > 0
                    )
                    if outcome_ok and has_events:
                        continue
                    if test_id in failing_ids:
                        failed_tests.append(test_id)
                    else:
                        dropped_passing.append(test_id)

                if dropped_passing:
                    # Recorded, not silent: the run count per instance is part of
                    # what the diagnosis is derived from.
                    LOG.warning(
                        "%s: dropped %d passing run(s) with no usable trace: %s",
                        repo_path.name,
                        len(dropped_passing),
                        ", ".join(sorted(dropped_passing)[:5])
                        + (" ..." if len(dropped_passing) > 5 else ""),
                    )
                    for test_id in dropped_passing:
                        try:
                            Path(event_output_mapping[test_id]).unlink(missing_ok=True)
                        except OSError:
                            pass

                if failed_tests:
                    reason = (
                        "Event collection returned failures or empty event files "
                        f"for {len(failed_tests)} test(s)."
                    )
                    first_failed = sorted(failed_tests)[0]
                    if failure_log_dir is not None:
                        self._write_collection_failure_logs(
                            repo_path=log_repo_path,
                            tests=[first_failed],
                            output_dir=failure_log_dir,
                            reason=reason,
                            python=getattr(subject, "python", sys.executable),
                            env=getattr(subject, "env", None),
                            timeout=getattr(subject, "timeout", 120),
                            subject=subject,
                            max_tests=1,
                        )
                        collection_logs_written = True
                        first_log_path = (
                            failure_log_dir
                            / "tests"
                            / f"{self._safe_file_name(first_failed)}.log"
                        )
                        print(
                            f"  - collect failed; first failing test log: {first_log_path}"
                        )
                        if first_log_path.exists():
                            try:
                                print(first_log_path.read_text(encoding="utf-8"))
                            except OSError:
                                pass
                    raise RuntimeError(reason + f" First failing test: {first_failed}")
            except KeyboardInterrupt as exc:
                if (
                    selected_tests
                    and not collection_logs_written
                    and failure_log_dir is not None
                ):
                    interrupted_tests = []
                    for test_id in ordered_test_ids or list(selected_tests):
                        event_path = event_output_mapping.get(test_id) or event_file_mapping.get(
                            test_id
                        )
                        if event_path is None:
                            interrupted_tests.append(test_id)
                            continue
                        if (
                            not event_path.exists()
                            or not event_path.is_file()
                            or event_path.stat().st_size <= 0
                        ):
                            interrupted_tests.append(test_id)
                    candidate_tests = interrupted_tests or list(
                        ordered_test_ids or selected_tests
                    )
                    first_test = candidate_tests[0] if candidate_tests else None
                    if first_test:
                        self._write_collection_failure_logs(
                            repo_path=log_repo_path,
                            tests=[first_test],
                            output_dir=failure_log_dir,
                            reason="Interrupted during event collection (KeyboardInterrupt).",
                            python=getattr(subject, "python", sys.executable),
                            env=getattr(subject, "env", None),
                            timeout=getattr(subject, "timeout", 120),
                            subject=subject,
                            max_tests=1,
                            error=exc,
                        )
                        collection_logs_written = True
                        first_log_path = (
                            failure_log_dir
                            / "tests"
                            / f"{self._safe_file_name(first_test)}.log"
                        )
                        print(
                            f"  - collect interrupted; first test log: {first_log_path}"
                        )
                        if first_log_path.exists():
                            try:
                                print(first_log_path.read_text(encoding="utf-8"))
                            except OSError:
                                pass
                raise
            except Exception:
                if selected_tests:
                    if collection_logs_written:
                        raise
                    if failure_log_dir is not None:
                        self._write_collection_failure_logs(
                            repo_path=log_repo_path,
                            tests=selected_tests[:1],
                            output_dir=failure_log_dir,
                            reason="Exception raised while collecting events.",
                            subject=subject,
                            max_tests=1,
                            error=sys.exc_info()[1],
                            error_trace=traceback.format_exc(),
                        )
                        if selected_tests:
                            first_log_path = (
                                failure_log_dir
                                / "tests"
                                / f"{self._safe_file_name(selected_tests[0])}.log"
                            )
                            print(
                                "  - collect failed; first test log: "
                                f"{first_log_path}"
                            )
                            if first_log_path.exists():
                                try:
                                    print(first_log_path.read_text(encoding="utf-8"))
                                except OSError:
                                    pass
                    raise
                raise
            collect_seconds = time.perf_counter() - t0

            if not tmp_mapping_path.exists():
                raise RuntimeError(
                    f"No mapping.json generated for repository: {repo_path}"
                )

            t0 = time.perf_counter()
            output_mapping_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_mapping_path, output_mapping_path)
            copy_seconds = time.perf_counter() - t0
        
        # Clean up Docker containers and instrumented images after collection
        try:
            subject.cleanup_docker()
        except Exception as cleanup_exc:
            LOG.warning(
                "Failed to cleanup Docker resources after event collection",
                extra={"instance_id": instance.get("instance_id"), "error": str(cleanup_exc)},
            )
        
        return {
            "instrument_seconds": instrument_seconds,
            "collect_seconds": collect_seconds,
            "copy_seconds": copy_seconds,
            "elapsed_seconds": time.perf_counter() - overall_t0,
            "used_compat": False,
            "subject_backend": "swe",
            "collection_failure_log_dir": (
                str(failure_log_dir) if failure_log_dir is not None else None
            ),
            "selected_test_count": len(selected_tests),
            "selected_failing_count": len(selected_failing),
            "selected_passing_count": len(selected_passing),
            "selected_passing_before_filter_count": original_passing_count,
            "passing_filter_repo_slug": filter_repo_slug,
            "selection_strategy": selection.strategy,
            "selection_uses_oracle": selection.uses_oracle,
            "selection_statistics": selection.statistics,
            "selection": selection,
            "unrunnable_test_count": len(skipped_unrunnable),
            "unrunnable_tests_sample": skipped_unrunnable[:10],
        }

    def collect_diagnosis_events(
        self,
        instance: dict[str, Any],
        repo_path: Path,
        *,
        force: bool = False,
    ) -> DiagnosisPaths:
        """
        Stage 1: collect and persist events.
        """
        t0 = time.perf_counter()
        instance_id = str(instance["instance_id"])
        paths = self._diagnosis_paths(instance_id)
        paths.root.mkdir(parents=True, exist_ok=True)

        selected_failing: Optional[list[str]] = None
        selected_passing: Optional[list[str]] = None
        if instance.get("swe_task_info") is not None:
            raw_failing = list(instance.get("failing_tests") or [])
            raw_passing = list(instance.get("passing_tests") or [])
            # Some projects (e.g. sympy) report tests as bare function names that
            # the test_patch adds; resolve them to node IDs before the split
            # would otherwise discard them as unrunnable.
            _test_patch = str(instance.get("test_patch") or "")
            if not _test_patch.strip():
                _ti = instance.get("swe_task_info")
                if isinstance(_ti, dict):
                    _test_patch = str(_ti.get("test_patch") or "")
            if _test_patch.strip():
                raw_failing = resolve_bare_test_ids_with_patch(
                    raw_failing, _test_patch, repo_path
                )
                raw_passing = resolve_bare_test_ids_with_patch(
                    raw_passing, _test_patch, repo_path
                )
            selected_failing, skipped_failing = split_runnable_pytest_test_ids(
                raw_failing
            )
            selected_passing, skipped_passing = split_runnable_pytest_test_ids(
                raw_passing
            )
            skipped = skipped_failing + skipped_passing
            if skipped and self.verbose:
                sample = ", ".join(skipped[:5])
                print(
                    "  - collect warning: ignoring unrunnable test labels "
                    f"({len(skipped)}): {sample}"
                )
            # A previous run may already have narrowed the test set; the cache
            # is only complete with respect to the tests that were selected
            # then, so compare against those rather than the full listing.
            previous = self._read_selection(paths)
            if previous is not None and not force:
                selected_failing = list(previous.failing)
                selected_passing = list(previous.passing)

        has_events = self._has_complete_event_collection(
            paths,
            repo_path=Path(repo_path).resolve(),
            failing_tests=selected_failing,
            passing_tests=selected_passing,
        )
        if has_events and not force:
            self._update_instance_report(
                instance_id=instance_id,
                step="collect",
                payload={
                    "status": "cached",
                    "elapsed_seconds": time.perf_counter() - t0,
                    "events_dir": str(paths.events_dir),
                    "mapping_path": str(paths.mapping_path),
                },
            )
            return paths

        failure_log_dir = paths.events_dir / "collection_failure_logs"
        try:
            timing = (
                self._collect_events_for_repo(
                    Path(repo_path).resolve(),
                    output_events_dir=paths.events_dir,
                    output_mapping_path=paths.mapping_path,
                    failing_tests=selected_failing,
                    passing_tests=selected_passing,
                    instance=instance,
                    failure_log_dir=failure_log_dir,
                )
                or {}
            )
        except KeyboardInterrupt:
            payload: dict[str, Any] = {
                "status": "interrupted",
                "elapsed_seconds": time.perf_counter() - t0,
                "error": "KeyboardInterrupt",
                "events_dir": str(paths.events_dir),
                "mapping_path": str(paths.mapping_path),
            }
            if failure_log_dir.exists():
                payload["collection_failure_log_dir"] = str(failure_log_dir)
            self._update_instance_report(
                instance_id=instance_id,
                step="collect",
                payload=payload,
            )
            raise
        except Exception as exc:
            payload: dict[str, Any] = {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - t0,
                "error": str(exc),
                "events_dir": str(paths.events_dir),
                "mapping_path": str(paths.mapping_path),
            }
            if failure_log_dir.exists():
                payload["collection_failure_log_dir"] = str(failure_log_dir)
            self._update_instance_report(
                instance_id=instance_id,
                step="collect",
                payload=payload,
            )
            raise

        # Record which tests were traced, so a resumed run compares its cache
        # against the same set and later stages can report the strategy used.
        selection = timing.pop("selection", None)
        if isinstance(selection, SelectedTests):
            self._write_selection(paths, selection)

        failing_count = len(
            [p for p in paths.failing_events_dir.iterdir() if p.is_file()]
        )
        passing_count = len(
            [p for p in paths.passing_events_dir.iterdir() if p.is_file()]
        )
        # Every run emitting the identical trace means the test suite never
        # started: what got traced is the import, once per run. That is not a
        # weak diagnosis, it is no data at all, and it has to be said here --
        # where the collection error is still visible -- rather than surface
        # hours later as an empty cause chain. sphinx-7686/-7738/-7975 each
        # wrote 17/31/7 such traces because sphinx imports ``roman`` without
        # declaring it, so pytest died collecting sphinx.testing.fixtures.
        identical_runs = _all_traces_identical(paths)
        if identical_runs:
            LOG.error(
                "%s: all %d runs produced byte-identical traces -- the test "
                "suite did not start, so these traces carry no signal. Check "
                "the collection log for an import or usage error.",
                instance_id,
                identical_runs,
            )

        metadata = {
            "instance_id": instance_id,
            "repo": str(repo_path),
            "base_commit": instance.get("base_commit"),
            "stage": "collect",
            "failing_event_files": failing_count,
            "passing_event_files": passing_count,
            "identical_trace_runs": identical_runs,
            "elapsed_seconds": time.perf_counter() - t0,
            "timing": timing,
        }
        paths.events_metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._update_instance_report(
            instance_id=instance_id,
            step="collect",
            payload={
                "status": "completed",
                "elapsed_seconds": metadata["elapsed_seconds"],
                "timing": timing,
                "failing_event_files": failing_count,
                "passing_event_files": passing_count,
                "events_dir": str(paths.events_dir),
                "mapping_path": str(paths.mapping_path),
            },
        )
        return paths

    @staticmethod
    def _load_event_files(paths: DiagnosisPaths) -> list[Any]:
        """
        Load persisted event files into sflkit ``EventFile`` objects.
        """
        from sflkit.events.mapping import EventMapping

        from forecast.runner.event_file import make_event_file

        mapping = EventMapping.load_from_file(paths.mapping_path)
        failing_paths = sorted(
            p for p in paths.failing_events_dir.iterdir() if p.is_file()
        )
        passing_paths = sorted(
            p for p in paths.passing_events_dir.iterdir() if p.is_file()
        )

        failing = [
            make_event_file(path, idx, mapping, failing=True)
            for idx, path in enumerate(failing_paths, start=0)
        ]
        passing = [
            make_event_file(path, idx, mapping, failing=False)
            for idx, path in enumerate(passing_paths, start=len(failing))
        ]
        return failing + passing

    def build_diagnosis_tree(
        self,
        instance: dict[str, Any],
        *,
        force: bool = False,
        prune: bool = True,
        compress: bool = True,
        workers: int = 1,
        passing_cap: int | None = 30,
        mem_budget_gb: float | None = 50.0,
    ) -> Path:
        """
        Stage 1b: build and persist a traced relevance tree from collected events.

        :param prune: When ``True`` (default) passing runs are pruned to the
            failing footprint (off-footprint, passing-only subtrees are dropped),
            which greatly shrinks large trees at no cost to diagnosis. Set
            ``False`` to build the full tree.
        :param compress: When ``True`` (default) the pickled tree is gzip'd.
        :param workers: Number of processes for the parallel build. ``1``
            (default) builds sequentially. With ``>1`` and pruning, the failing
            footprint is built once and passing files are split across processes,
            each pruning against a skeleton of the footprint; the partial trees
            are merged. Produces a byte-identical tree to the sequential build
            (validated) but uses idle cores to speed up large instances.
        :param passing_cap: Keep at most this many passing runs (``30`` by
            default), the first ``passing_cap`` of the passing set — which is
            already ordered/limited by coverage overlap with the failing test at
            collection time. Tree size grows as ``nodes x cap x runs``, so
            high-run-count instances (matplotlib P2P=817, big sympy) build trees
            of 90-170 GB with all ~120 runs even at per-run cap 2; capping the
            runs shrinks them ~2x (validated diagnosis-stable) so they build in
            parallel and fit the diagnose stage. ``None`` keeps all passing runs.
        :param mem_budget_gb: Per-tree memory budget (``50`` GB by default). On
            top of ``passing_cap``, passing runs are added one at a time and the
            build stops once the in-memory tree crosses this budget. It only
            binds on a few "few-run / huge-per-run" outliers (e.g.
            matplotlib-25433: 55 runs but ~4 GB/run -> 125 GB at cap 30), keeping
            every tree buildable and loadable by the diagnose stage. The vast
            majority never reach it, so their trees are unaffected. ``None``
            disables the budget. Forces the sequential build (the parallel path
            cannot observe the running tree size).
        """
        from forecast.handler import ForecastHandler
        from forecast.tree import dump_tree

        t0 = time.perf_counter()
        instance_id = str(instance["instance_id"])
        paths = self._diagnosis_paths(instance_id)

        if paths.tree_path.exists() and not force:
            self._update_instance_report(
                instance_id=instance_id,
                step="tree",
                payload={
                    "status": "cached",
                    "elapsed_seconds": time.perf_counter() - t0,
                    "tree_path": str(paths.tree_path),
                },
            )
            return paths.tree_path
        if not paths.mapping_path.exists():
            raise FileNotFoundError(
                f"Diagnosis mapping not found for {instance_id}: {paths.mapping_path}"
            )
        if (
            not paths.failing_events_dir.exists()
            or not paths.passing_events_dir.exists()
        ):
            raise FileNotFoundError(
                f"Diagnosis events not found for {instance_id} under {paths.events_dir}"
            )

        event_files = self._load_event_files(paths)
        if not event_files:
            raise RuntimeError(f"No event files available for diagnosis: {instance_id}")

        # Cap the number of passing runs (failing-first order is preserved). The
        # passing set is already coverage-overlap-selected at collection; keeping
        # the first ``passing_cap`` bounds tree size on high-run-count instances
        # without changing the diagnosis (validated). See :param passing_cap:.
        if passing_cap is not None:
            failing_efs = [e for e in event_files if e.failing]
            passing_efs = [e for e in event_files if not e.failing]
            if len(passing_efs) > passing_cap:
                event_files = failing_efs + passing_efs[:passing_cap]

        t_build = time.perf_counter()
        failing_efs = [e for e in event_files if e.failing]
        passing_efs = [e for e in event_files if not e.failing]
        # The parallel builder cannot observe the running tree size, so the
        # memory budget forces the sequential path.
        if (
            prune
            and workers > 1
            and len(passing_efs) >= 2 * workers
            and mem_budget_gb is None
        ):
            tree = self._build_tree_parallel(
                event_files, str(paths.mapping_path), workers
            )
        else:
            # Failing files come first (see _load_event_files), so the failing
            # footprint is complete before any passing file is pruned against it.
            # Passing files are then added one at a time; once the in-memory tree
            # crosses ``mem_budget_gb`` we stop adding runs, so the few
            # few-run/huge-per-run outliers cannot build 100 GB+ trees.
            handler = ForecastHandler(prune=prune)
            for event_file in failing_efs:
                handler.handle(event_file)
            kept = 0
            for event_file in passing_efs:
                handler.handle(event_file)
                kept += 1
                if mem_budget_gb is not None and _process_rss_gb() >= mem_budget_gb:
                    break
            tree = handler.tree
            if kept < len(passing_efs):
                LOG.warning(
                    "%s: memory budget %.0f GB reached; kept %d/%d passing runs",
                    instance_id,
                    mem_budget_gb,
                    kept,
                    len(passing_efs),
                )
        build_seconds = time.perf_counter() - t_build

        t_write = time.perf_counter()
        paths.root.mkdir(parents=True, exist_ok=True)
        dump_tree(tree, paths.tree_path, compress=compress)
        write_seconds = time.perf_counter() - t_write

        self._update_instance_report(
            instance_id=instance_id,
            step="tree",
            payload={
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - t0,
                "build_seconds": build_seconds,
                "write_seconds": write_seconds,
                "event_file_count": len(event_files),
                "tree_path": str(paths.tree_path),
                "pruned": prune,
                "compressed": compress,
                "workers": workers,
            },
        )
        return paths.tree_path

    def _build_tree_parallel(self, event_files, mapping_path, workers):
        """
        Build the relevance tree using *workers* processes (see
        :meth:`build_diagnosis_tree`). Builds the failing footprint once, splits
        the passing files across processes (each pruning against a skeleton of
        the footprint), and merges the resulting partial trees. Produces a tree
        identical to the sequential build (validated).
        """
        import multiprocessing as mp

        from forecast.handler import ForecastHandler
        from forecast.tree import merge_into, structure_only_copy

        failing = [e for e in event_files if e.failing]
        passing = [e for e in event_files if not e.failing]

        # Phase 1: failing footprint (the pruning reference).
        footprint_handler = ForecastHandler(prune=True)
        for ef in failing:
            footprint_handler.handle(ef)
        footprint = footprint_handler.tree

        # Phase 2: passing files across workers, each against a footprint
        # skeleton (structure only), then merged back.
        skeleton = structure_only_copy(footprint)
        specs = [(str(ef.path), ef.run_id) for ef in passing]
        chunks = [specs[i::workers] for i in range(workers)]
        args = [(mapping_path, chunk, skeleton) for chunk in chunks if chunk]
        ctx = mp.get_context("fork")
        with ctx.Pool(len(args)) as pool:
            partials = pool.map(_build_passing_partial, args)

        # Phase 3: merge each worker's passing observations onto the footprint.
        for partial in partials:
            if partial is not None:
                merge_into(footprint, partial)
        return footprint

    def generate_diagnosis_from_events(
        self,
        instance: dict[str, Any],
        *,
        force: bool = False,
        prune: bool = True,
        compress: bool = True,
    ):
        """
        Stage 2: derive diagnosis from persisted events.

        :param prune: Forwarded to :meth:`build_diagnosis_tree` when the tree
            must be built here (no cached ``tree.pkl``).
        :param compress: Forwarded to :meth:`build_diagnosis_tree` likewise.
        """
        from forecast.diagnosis import Forecast
        from forecast.tree import load_tree

        t0 = time.perf_counter()
        instance_id = str(instance["instance_id"])
        paths = self._diagnosis_paths(instance_id)

        if (
            paths.diagnosis_json_path.exists()
            and paths.diagnosis_md_path.exists()
            and paths.diagnosis_txt_path.exists()
            and not force
        ):
            from forecast.integration import load_diagnosis_json

            raw = load_diagnosis_json(paths.diagnosis_json_path)
            self._update_instance_report(
                instance_id=instance_id,
                step="diagnose",
                payload={
                    "status": "cached",
                    "elapsed_seconds": time.perf_counter() - t0,
                    "diagnosis_json_path": str(paths.diagnosis_json_path),
                },
            )
            return raw

        tree_load_seconds = 0.0
        tree_source = "built"
        t_tree = time.perf_counter()
        if paths.tree_path.exists() and not force:
            tree = load_tree(paths.tree_path)
            tree_source = "cached"
        else:
            if not paths.mapping_path.exists():
                raise FileNotFoundError(
                    f"Diagnosis mapping not found for {instance_id}: {paths.mapping_path}"
                )
            if (
                not paths.failing_events_dir.exists()
                or not paths.passing_events_dir.exists()
            ):
                raise FileNotFoundError(
                    f"Diagnosis events not found for {instance_id} under {paths.events_dir}"
                )
            tree_path = self.build_diagnosis_tree(
                instance, force=force, prune=prune, compress=compress
            )
            tree = load_tree(tree_path)
        tree_load_seconds = time.perf_counter() - t_tree

        t_derive = time.perf_counter()
        forecast = Forecast(root=tree)
        diagnosis = forecast.derive_diagnosis()
        repo = Path(str(instance.get("repo", "")))
        if repo.is_dir() and paths.mapping_path.exists():
            from forecast.diagnosis import FunctionNames

            diagnosis.qualify_functions(FunctionNames.from_mapping(repo, paths.mapping_path))
        derive_seconds = time.perf_counter() - t_derive

        from forecast.integration import save_diagnosis

        t_save = time.perf_counter()
        paths.root.mkdir(parents=True, exist_ok=True)
        save_diagnosis(diagnosis, paths.diagnosis_json_path, file_format="json")
        save_diagnosis(diagnosis, paths.diagnosis_md_path, file_format="markdown")
        save_diagnosis(diagnosis, paths.diagnosis_txt_path, file_format="text")
        save_seconds = time.perf_counter() - t_save
        self._update_instance_report(
            instance_id=instance_id,
            step="diagnose",
            payload={
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - t0,
                "tree_load_seconds": tree_load_seconds,
                "derive_seconds": derive_seconds,
                "save_seconds": save_seconds,
                "tree_source": tree_source,
                "diagnosis_json_path": str(paths.diagnosis_json_path),
                "diagnosis_md_path": str(paths.diagnosis_md_path),
                "diagnosis_txt_path": str(paths.diagnosis_txt_path),
            },
        )
        return diagnosis

    def generate_diagnosis_features(
        self,
        instance: dict[str, Any],
        repo_path: Path,
        *,
        force: bool = False,
    ) -> DiagnosisPaths:
        """
        Backward-compatible alias: collect events.
        """
        return self.collect_diagnosis_events(instance, repo_path, force=force)

    def build_diagnosis_from_features(
        self,
        instance: dict[str, Any],
        *,
        force: bool = False,
    ):
        """
        Backward-compatible alias: generate diagnosis from events.
        """
        return self.generate_diagnosis_from_events(instance, force=force)

    def generate_diagnosis(
        self,
        instance: dict[str, Any],
        repo_path: Path,
        *,
        force_events: bool = False,
        force_diagnosis: bool = False,
    ):
        """
        Convenience wrapper: events -> diagnosis.
        """
        self.collect_diagnosis_events(
            instance,
            repo_path,
            force=force_events,
        )
        return self.generate_diagnosis_from_events(instance, force=force_diagnosis)

    def _generate_diagnosis(
        self,
        instance: dict[str, Any],
        repo_path: Path,
    ):
        """
        Backward-compatible helper used by existing CLI code paths.
        """
        try:
            return self.generate_diagnosis(
                instance,
                repo_path=repo_path,
                force_events=False,
                force_diagnosis=False,
            )
        except Exception:
            if self.verbose:
                print(
                    f"Diagnosis generation failed for {instance.get('instance_id')}.",
                )
            return None

    def _load_or_generate_diagnosis_text(
        self,
        instance: dict[str, Any],
        *,
        repo_path: Path,
        force_diagnosis: bool = False,
    ) -> tuple[Optional[str], Optional[float]]:
        paths = self._diagnosis_paths(str(instance["instance_id"]))
        if not paths.diagnosis_json_path.exists() or force_diagnosis:
            self.generate_diagnosis(
                instance,
                repo_path=repo_path,
                force_events=force_diagnosis,
                force_diagnosis=force_diagnosis,
            )
        if not paths.diagnosis_json_path.exists():
            return None, None
        from forecast.integration import diagnosis_text_from_json, load_diagnosis_json

        raw = load_diagnosis_json(paths.diagnosis_json_path)
        diagnosis_text = diagnosis_text_from_json(raw)
        disc = raw.get("discriminability") if isinstance(raw, dict) else None
        try:
            discriminability = float(disc) if disc is not None else None
        except (TypeError, ValueError):
            discriminability = None
        return diagnosis_text, discriminability

    def compute_sbfl(
        self, instance: dict[str, Any], *, force: bool = False, processes: int = 1
    ) -> Optional[list[dict]]:
        """
        Compute (or, unless *force*, load) the instance's SBFL ranking; see
        :meth:`_load_or_compute_sbfl`.
        """
        cache = self._diagnosis_paths(str(instance["instance_id"])).root / "sbfl.json"
        if force:
            cache.unlink(missing_ok=True)
        return self._load_or_compute_sbfl(instance, processes=processes)

    def _load_or_compute_sbfl(
        self, instance: dict[str, Any], processes: int = 1
    ) -> Optional[list[dict]]:
        """
        The method-level SBFL ranking (Ochiai) over the instance's traced
        executions, cached as ``diagnosis/sbfl.json`` next to the diagnosis.

        Computed from the stored traces, never by re-running tests, so it
        covers exactly the executions the diagnosis is built from. ``None``
        when neither a cached ranking nor traces exist.
        """
        from forecast.localization.traces import rank_functions

        paths = self._diagnosis_paths(str(instance["instance_id"]))
        cache = paths.root / "sbfl.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        if not (paths.mapping_path.exists() and paths.failing_events_dir.exists()):
            return None
        ranking = rank_functions(self._load_event_files(paths), processes=processes)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
        return ranking

    @staticmethod
    def _build_context(
        instance: dict[str, Any],
        *,
        mode: AgentMode,
    ) -> RepairContext:
        return RepairContext(
            instance_id=str(instance["instance_id"]),
            repo=str(instance["repo"]),
            version=str(instance.get("version", "")),
            problem_statement=str(instance.get("problem_statement", "")),
            failing_tests=list(instance.get("failing_tests", [])),
            passing_tests=list(instance.get("passing_tests", [])),
            base_commit=instance.get("base_commit"),
            swe_task_info=instance.get("swe_task_info"),
            swe_setup_info=instance.get("swe_setup_info"),
        )

    def _prompt_root(self, instance_id: str, agent_name: str, mode: AgentMode) -> Path:
        return (
            self._instance_dir(instance_id)
            / "prompt"
            / str(agent_name).lower()
            / mode.value
        )

    def _run_root(self, instance_id: str, agent_name: str, mode: AgentMode) -> Path:
        return (
            self._instance_dir(instance_id)
            / "run"
            / str(agent_name).lower()
            / mode.value
        )

    def _evaluate_root(
        self, instance_id: str, agent_name: str, mode: AgentMode
    ) -> Path:
        return (
            self._instance_dir(instance_id)
            / "evaluate"
            / str(agent_name).lower()
            / mode.value
        )

    def default_patch_path(
        self,
        instance_id: str,
        *,
        agent_name: str,
        mode: AgentMode,
    ) -> Path:
        return self._run_root(instance_id, agent_name, mode) / "patch.diff"

    def generate_prompt(
        self,
        instance: dict[str, Any],
        *,
        agent_name: str,
        model: str = _DEFAULT_MODEL,
        mode: AgentMode = AgentMode.FORECAST,
        force_diagnosis: bool = False,
    ) -> Path:
        """
        Stage 3: extract and persist prompts for one instance.
        """
        t0 = time.perf_counter()
        instance_id = str(instance["instance_id"])
        repo_path = Path(str(instance["repo"])).resolve()
        context = self._build_context(instance, mode=mode)
        diagnosis_prepare_seconds = 0.0
        t_diag = time.perf_counter()
        if mode == AgentMode.FORECAST:
            diagnosis_text, diagnosis_disc = self._load_or_generate_diagnosis_text(
                instance,
                repo_path=repo_path,
                force_diagnosis=force_diagnosis,
            )
            context.diagnosis = diagnosis_text
            context.diagnosis_discriminability = diagnosis_disc
        # A test-based signal for agents that use the tests, in both modes.
        context.fault_localization = self._load_or_compute_sbfl(instance)
        diagnosis_prepare_seconds = time.perf_counter() - t_diag

        prompt_root = self._prompt_root(instance_id, agent_name, mode)
        prompt_root.mkdir(parents=True, exist_ok=True)

        t_build = time.perf_counter()
        agent = self._build_agent(
            agent_name=agent_name,
            mode=mode,
            model=model,
            output_dir=prompt_root,
        )
        prompt = agent.get_prompt(context)
        system_prompt = agent.get_system_prompt(context)
        build_seconds = time.perf_counter() - t_build

        t_write = time.perf_counter()
        (prompt_root / "prompt.txt").write_text(prompt, encoding="utf-8")
        if system_prompt is not None:
            (prompt_root / "system_prompt.txt").write_text(
                system_prompt,
                encoding="utf-8",
            )
        write_seconds = time.perf_counter() - t_write

        result_payload = {
            "instance_id": instance_id,
            "agent_name": agent_name,
            "mode": mode.value,
            "model": model,
            "elapsed_seconds": time.perf_counter() - t0,
            "diagnosis_prepare_seconds": diagnosis_prepare_seconds,
            "build_seconds": build_seconds,
            "write_seconds": write_seconds,
            "prompt_chars": len(prompt),
            "system_prompt_chars": (
                len(system_prompt) if system_prompt is not None else 0
            ),
            "output_dir": str(prompt_root),
        }
        (prompt_root / "result.json").write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._update_instance_report(
            instance_id=instance_id,
            step="prompt",
            payload=result_payload,
        )
        return prompt_root

    def evaluate_patch(
        self,
        instance: dict[str, Any],
        *,
        agent_name: str,
        patch_path: Path,
        mode: AgentMode = AgentMode.FORECAST,
        model: str = _DEFAULT_MODEL,
    ) -> bool:
        """
        Stage 5: the official SWE-bench verdict on a patch.

        The patch is judged by the SWE-bench harness in the instance's image --
        ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` from the unmodified dataset row,
        with the developer's test patch -- never by an agent's own check, so
        "resolved" means what it means on the SWE-bench leaderboard. Results
        are kept per mode; the two arms of an agent never overwrite each other.
        """
        from forecast.agents.environment import harness_evaluate

        t0 = time.perf_counter()
        instance_id = str(instance["instance_id"])
        eval_root = self._evaluate_root(instance_id, agent_name, mode)
        eval_root.mkdir(parents=True, exist_ok=True)

        patch_file = Path(patch_path)
        base_payload = {
            "instance_id": instance_id,
            "agent_name": agent_name,
            "mode": mode.value,
            "model": model,
            "patch_path": str(patch_file),
        }
        task_info = instance.get("swe_task_info")
        if not patch_file.exists() or not isinstance(task_info, dict):
            error = (
                f"Patch file not found: {patch_file}"
                if not patch_file.exists()
                else "Instance carries no SWE-bench row (swe_task_info)."
            )
            result_payload = {
                **base_payload,
                "passed": False,
                "resolved": False,
                "error": error,
                "elapsed_seconds": time.perf_counter() - t0,
            }
            (eval_root / "result.json").write_text(
                json.dumps(result_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._update_instance_report(
                instance_id=instance_id,
                step=f"evaluate/{agent_name.lower()}/{mode.value}",
                payload=result_payload,
            )
            return False

        verdict = harness_evaluate(
            task_info,
            patch_file.read_text(encoding="utf-8"),
            # Unique per evaluation: the harness names its container after the
            # run id, and a stale container of the same name would block it.
            run_id=f"{agent_name.lower()}-{mode.value}-{uuid.uuid4().hex[:8]}",
            model_name=model,
            log_root=eval_root / "harness",
        )
        result_payload = {
            **base_payload,
            "passed": verdict["resolved"],
            "resolved": verdict["resolved"],
            "completed": verdict["completed"],
            "report": verdict["report"],
            "harness_log_dir": verdict["log_dir"],
            "elapsed_seconds": time.perf_counter() - t0,
        }
        (eval_root / "result.json").write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._update_instance_report(
            instance_id=instance_id,
            step=f"evaluate/{agent_name.lower()}/{mode.value}",
            payload=result_payload,
        )
        return bool(verdict["resolved"])

    def run_repair(
        self,
        instance: dict[str, Any],
        *,
        agent_name: str,
        model: str = _DEFAULT_MODEL,
        mode: AgentMode = AgentMode.FORECAST,
        force_diagnosis: bool = False,
    ) -> RepairResult:
        """
        Run one repair attempt for a single instance.
        """
        t0 = time.perf_counter()
        instance_id = str(instance["instance_id"])
        repo_path = Path(str(instance["repo"])).resolve()

        context = self._build_context(instance, mode=mode)
        diagnosis_prepare_seconds = 0.0
        t_diag = time.perf_counter()
        if mode == AgentMode.FORECAST:
            diagnosis_text, diagnosis_disc = self._load_or_generate_diagnosis_text(
                instance,
                repo_path=repo_path,
                force_diagnosis=force_diagnosis,
            )
            context.diagnosis = diagnosis_text
            context.diagnosis_discriminability = diagnosis_disc
        # A test-based signal for agents that use the tests, in both modes.
        context.fault_localization = self._load_or_compute_sbfl(instance)
        diagnosis_prepare_seconds = time.perf_counter() - t_diag

        run_root = self._run_root(instance_id, agent_name, mode)
        run_root.mkdir(parents=True, exist_ok=True)

        agent = self._build_agent(
            agent_name=agent_name,
            mode=mode,
            model=model,
            output_dir=run_root,
        )
        model_client = get_model_client(model)
        result = agent.repair(
            context=context,
            model_client=model_client,
            traj_dir=run_root / "trajs",
            log_dir=run_root / "logs",
        )

        result_json = run_root / "result.json"
        result_json.write_text(
            json.dumps(
                {
                    "instance_id": result.instance_id,
                    "agent_name": result.agent_name,
                    "mode": result.mode.value,
                    "success": result.success,
                    "patch": result.patch,
                    "lines_changed": result.lines_changed,
                    "files_modified": result.files_modified,
                    "test_pass_rate": result.test_pass_rate,
                    "has_regression": result.has_regression,
                    "attempts": result.attempts,
                    "time_seconds": result.time_seconds,
                    "tokens_used": result.tokens_used,
                    "diagnosis_used": result.diagnosis_used,
                    "diagnosis_discriminability": result.diagnosis_discriminability,
                    "output_dir": result.output_dir,
                    "error": result.error,
                    "diagnosis_prepare_seconds": diagnosis_prepare_seconds,
                    "elapsed_seconds": time.perf_counter() - t0,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if result.patch:
            (run_root / "patch.diff").write_text(result.patch, encoding="utf-8")
        self._update_instance_report(
            instance_id=instance_id,
            step="run",
            payload={
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - t0,
                "diagnosis_prepare_seconds": diagnosis_prepare_seconds,
                "agent_repair_seconds": result.time_seconds,
                "success": result.success,
                "patch_present": bool(result.patch),
                "tokens_used": result.tokens_used,
                "result_path": str(result_json),
            },
        )
        return result

    def run_repairs(
        self,
        instances: Sequence[dict[str, Any]],
        *,
        agent_name: str,
        model: str = _DEFAULT_MODEL,
        mode: AgentMode = AgentMode.FORECAST,
        force_diagnosis: bool = False,
    ) -> list[RepairResult]:
        """
        Run repair for each instance.
        """
        results: list[RepairResult] = []
        for instance in instances:
            results.append(
                self.run_repair(
                    instance,
                    agent_name=agent_name,
                    model=model,
                    mode=mode,
                    force_diagnosis=force_diagnosis,
                )
            )
        return results
