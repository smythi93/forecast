"""
Helpers for normalizing test identifiers into pytest-compatible node IDs.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

_DISPLAY_STYLE_RE = re.compile(
    r"^\s*(?P<method>[^\s()]+)\s+\((?P<context>[^()]+)\)\s*$"
)

_REPO_BASENAME_MATCH_CACHE: dict[tuple[str, str], list[str]] = {}


def _find_repo_files_by_basename(repo_root: Path, basename: str) -> list[str]:
    """
    Return repo-relative POSIX paths for files named *basename*.
    """
    key = (str(repo_root.resolve()), basename)
    cached = _REPO_BASENAME_MATCH_CACHE.get(key)
    if cached is not None:
        return cached
    matches = [
        found.relative_to(repo_root).as_posix()
        for found in repo_root.rglob(basename)
        if found.is_file()
    ]
    _REPO_BASENAME_MATCH_CACHE[key] = matches
    return matches


def normalize_pytest_test_id(test_id: str) -> str:
    """
    Normalize one test identifier to a pytest node ID when possible.

    Handles already-valid pytest node IDs unchanged. Also accepts common
    unittest/django display formats such as:

    * ``"test_foo (pkg.tests.TestCls)"``
    * ``"pkg.tests.TestCls.test_foo"``
    """
    raw = str(test_id or "").strip()
    if not raw:
        return raw
    if "::" in raw:
        return raw
    if "/" in raw or raw.endswith(".py"):
        return raw

    match = _DISPLAY_STYLE_RE.match(raw)
    if match:
        method = match.group("method").strip()
        context = match.group("context").strip()
        if context.endswith(f".{method}") or context.endswith(f"::{method}"):
            raw = context
        else:
            raw = f"{context}.{method}"

    # Some upstream tools already append the method name in the context,
    # which can leave duplicated trailing selectors like ``::test_x::test_x``.
    if "::" in raw:
        selectors = raw.split("::")
        while len(selectors) >= 3 and selectors[-1] == selectors[-2]:
            selectors.pop()
        raw = "::".join(selectors)

    parts = [part for part in raw.split(".") if part]
    if len(parts) < 2:
        return raw

    # module.Class.test_method -> module.py::Class::test_method
    if len(parts) >= 3:
        class_candidate = parts[-2]
        method_candidate = parts[-1]
        module_parts = parts[:-2]
        if module_parts and (
            method_candidate.startswith("test")
            or class_candidate.startswith("Test")
            or (class_candidate and class_candidate[0].isupper())
        ):
            return (
                f"{'/'.join(module_parts)}.py::"
                f"{class_candidate}::"
                f"{method_candidate}"
            )

    # module.test_function -> module.py::test_function
    function_candidate = parts[-1]
    module_parts = parts[:-1]
    if module_parts and function_candidate.startswith("test"):
        return f"{'/'.join(module_parts)}.py::{function_candidate}"

    return raw


def normalize_pytest_test_ids(test_ids: Iterable[str]) -> list[str]:
    """
    Normalize an iterable of test IDs and de-duplicate while preserving order.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for value in test_ids:
        converted = normalize_pytest_test_id(str(value))
        if not converted or converted in seen:
            continue
        seen.add(converted)
        normalized.append(converted)
    return normalized


def is_pytest_node_id(test_id: str) -> bool:
    """
    Return True when *test_id* looks like a pytest node ID selector.
    """
    raw = str(test_id or "").strip()
    if not raw:
        return False
    if "::" not in raw:
        return False
    file_part = raw.split("::", 1)[0].strip()
    if not file_part:
        return False
    if not file_part.endswith(".py"):
        return False
    if any(ch.isspace() for ch in file_part):
        return False
    return True


def pytest_node_id_exists_in_repo(
    test_id: str,
    repo_path: str | Path,
) -> bool:
    """
    Return True when *test_id* resolves to a file that exists in *repo_path*.
    """
    normalized = normalize_pytest_test_id(test_id)
    if not is_pytest_node_id(normalized):
        return False
    resolved = resolve_pytest_node_id_against_repo(normalized, repo_path)
    file_part = resolved.split("::", 1)[0].strip()
    if not file_part:
        return False
    return (Path(repo_path) / file_part).is_file()


def split_runnable_pytest_test_ids(
    test_ids: Iterable[str],
) -> tuple[list[str], list[str]]:
    """
    Normalize test IDs and split into runnable pytest node IDs vs skipped labels.
    """
    runnable: list[str] = []
    skipped: list[str] = []
    for test_id in normalize_pytest_test_ids(test_ids):
        if is_pytest_node_id(test_id):
            runnable.append(test_id)
        else:
            skipped.append(test_id)
    return runnable, skipped


def resolve_pytest_node_id_against_repo(
    test_id: str,
    repo_path: str | Path,
) -> str:
    """
    Resolve a pytest node ID against files present in *repo_path*.

    If the node's file part does not exist, common test-root prefixes are
    tried (e.g. ``tests/``).
    """
    normalized = normalize_pytest_test_id(test_id)
    if not is_pytest_node_id(normalized):
        return normalized

    repo_root = Path(repo_path)
    file_part, sep, rest = normalized.partition("::")
    candidates: list[str] = [file_part]

    if not file_part.startswith("tests/"):
        candidates.append(f"tests/{file_part}")
    if not file_part.startswith("test/"):
        candidates.append(f"test/{file_part}")

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (repo_root / candidate).is_file():
            return f"{candidate}{sep}{rest}" if sep else candidate

    # Fallback: resolve by matching files with the same basename and best suffix
    # overlap. This recovers common malformed relative IDs, e.g.
    # "solvers/tests/test_solvers.py::..." or truncated "lvers/tests/...".
    path_candidate = Path(file_part)
    basename = path_candidate.name
    if not basename:
        return normalized

    raw_parts = [
        part
        for part in file_part.replace("\\", "/").split("/")
        if part and part != "."
    ]
    if not raw_parts:
        return normalized

    best_match: tuple[int, str] | None = None
    best_score = 0
    ambiguous = False
    for rel in _find_repo_files_by_basename(repo_root, basename):
        rel_parts = rel.split("/")
        score = 0
        max_overlap = min(len(raw_parts), len(rel_parts))
        for idx in range(1, max_overlap + 1):
            if raw_parts[-idx] != rel_parts[-idx]:
                break
            score += 1
        if score <= 0:
            continue
        if score > best_score:
            best_score = score
            best_match = (score, rel)
            ambiguous = False
        elif score == best_score and best_match is not None and rel != best_match[1]:
            ambiguous = True

    if best_match is not None and not ambiguous:
        resolved_file = best_match[1]
        return f"{resolved_file}{sep}{rest}" if sep else resolved_file
    return normalized


def _test_files_in_patch(patch: str) -> list[str]:
    """Return the repo-relative files a unified diff modifies (its ``+++ b/`` targets)."""
    files: list[str] = []
    for line in (patch or "").splitlines():
        if line.startswith("+++ b/"):
            target = line[6:].strip()
            if target and target != "/dev/null" and target not in files:
                files.append(target)
    return files


_PATCH_DEF_RE = re.compile(r"^[+ ]\s*def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PATCH_CLASS_RE = re.compile(r"^[+ ]\s*class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")


def _functions_in_patch(patch: str) -> dict[str, str]:
    """
    Map each test function/class name a diff adds or touches to its file.

    Tracks the current ``+++ b/`` target and records ``def``/``class`` names on
    added or context lines, so a bare test name from SWE-bench (which carries no
    module) can be tied back to the file the test_patch places it in.
    """
    mapping: dict[str, str] = {}
    current: str | None = None
    for line in (patch or "").splitlines():
        if line.startswith("+++ b/"):
            target = line[6:].strip()
            current = target if target and target != "/dev/null" else None
        elif current:
            m = _PATCH_DEF_RE.match(line) or _PATCH_CLASS_RE.match(line)
            if m:
                mapping.setdefault(m.group("name"), current)
    return mapping


def _looks_like_docstring(label: str) -> bool:
    """
    Is this label a test's docstring rather than its name?

    Django's test runner prints a test's docstring in place of
    ``test_name (module.Class)`` whenever the test has one, and SWE-bench
    recorded whatever the runner printed. The result is entries like
    ``"Apps listed first in INSTALLED_APPS have precedence."`` -- prose, with
    no file, class or function anywhere in it.

    :param label: A raw SWE-bench test label.
    :returns: ``True`` when the label reads as prose.
    """
    if "::" in label or not label.strip():
        return False
    # `test_x (a.b.C)` is the runner's other form and resolves on its own.
    if _DISPLAY_STYLE_RE.match(label.strip()):
        return False
    return " " in label.strip()


def _locate_test_file(repo_root: Path, rel: str) -> Path | None:
    """
    Find the file a test label refers to.

    Django's labels are relative to its ``tests/`` directory, not the
    repository root -- ``admin_scripts.tests.StartApp`` becomes
    ``admin_scripts/tests.py`` while the file is at
    ``tests/admin_scripts/tests.py``. Try the label's own path first, then the
    conventional roots, then fall back to matching the tail of the path.

    :param repo_root: The checkout.
    :param rel: Repository-relative path derived from a label.
    :returns: The file, or ``None`` if it cannot be found.
    """
    if not rel:
        return None
    direct = repo_root / rel
    if direct.is_file():
        return direct
    for root in ("tests", "test", "testing"):
        candidate = repo_root / root / rel
        if candidate.is_file():
            return candidate
    # Last resort: a unique file whose path ends with the label's path.
    tail = Path(rel).name
    matches = [
        path
        for path in repo_root.rglob(tail)
        if path.is_file() and path.as_posix().endswith(rel)
    ]
    return matches[0] if len(matches) == 1 else None


def _docstring_index(repo_root: Path, rel_files: Iterable[str]) -> dict[str, str]:
    """
    Map the first line of each test's docstring to its pytest node ID.

    :param repo_root: The checkout to read test files from.
    :param rel_files: Repository-relative test files to index.
    :returns: First docstring line -> node ID.
    """
    index: dict[str, str] = {}
    for rel in dict.fromkeys(rel_files):
        path = _locate_test_file(repo_root, rel)
        if path is None:
            continue
        rel = path.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue

        def register(node, class_name=None):
            if not node.name.startswith("test"):
                return
            doc = ast.get_docstring(node)
            if not doc:
                return
            first = doc.strip().splitlines()[0].strip()
            if not first:
                return
            selector = f"{class_name}::{node.name}" if class_name else node.name
            # First definition wins: a docstring repeated across classes is
            # ambiguous, and guessing the later one is no better.
            index.setdefault(first, f"{rel}::{selector}")

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                register(node)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        register(child, node.name)
    return index


def resolve_bare_test_ids_with_patch(
    test_ids: Iterable[str],
    test_patch: str,
    repo_path: str | Path | None = None,
) -> list[str]:
    """
    Resolve bare SWE-bench test labels to pytest node IDs using the test_patch.

    Some projects (notably sympy) report ``FAIL_TO_PASS``/``PASS_TO_PASS`` as
    bare function names with no module, and the failing ones are *added* by the
    instance's ``test_patch``.  Such labels carry no file, so they cannot be run
    or resolved against the base checkout.  This ties each bare name to a file:

    1. a file the test_patch adds/touches the function in;
    2. the sole test file the patch modifies, when there is only one;
    3. a file under the patched test directories that defines it (preferring the
       patched files);

    Names that are already node IDs (or carry a module, e.g. django's
    ``test_x (a.b.C)``) are passed through :func:`normalize_pytest_test_id`
    unchanged.

    :param test_ids: Raw test labels.
    :param test_patch: The instance's test_patch (unified diff).
    :param repo_path: Optional base checkout, used only for the search fallback.
    :returns: Resolved identifiers, order preserved.
    """
    patch_files = _test_files_in_patch(test_patch or "")
    in_patch = _functions_in_patch(test_patch or "")
    patch_dirs = {str(Path(f).parent.as_posix()) for f in patch_files}

    test_ids = list(test_ids)
    # Labels that are a test's docstring carry no file, so they need one from
    # somewhere else. The rest of the same batch is the best source: django
    # reports some tests by name and others by docstring, and they sit in the
    # same files. Index those files once, plus whatever the patch touched.
    docstrings = [raw for raw in test_ids if _looks_like_docstring(str(raw))]
    doc_index: dict[str, str] = {}
    if docstrings and repo_path:
        sibling_files = [
            normalize_pytest_test_id(str(raw)).split("::", 1)[0]
            for raw in test_ids
            if is_pytest_node_id(normalize_pytest_test_id(str(raw)))
        ]
        doc_index = _docstring_index(
            Path(repo_path), [*sibling_files, *patch_files]
        )

    resolved: list[str] = []
    for raw in test_ids:
        text = str(raw).strip()
        if doc_index and _looks_like_docstring(text):
            hit = doc_index.get(text)
            if hit:
                resolved.append(hit)
                continue
            # Unresolved prose cannot be run; passing it through would only
            # produce a node ID pytest rejects. Keep it so the caller's
            # skipped-test accounting still sees it.
            resolved.append(text)
            continue
        norm = normalize_pytest_test_id(text)
        if is_pytest_node_id(norm):
            resolved.append(norm)
            continue
        # A bare "test_x" or, rarely, "Class.test_x".
        pieces = [p for p in norm.replace("::", ".").split(".") if p]
        func = pieces[-1] if pieces else norm
        cls = pieces[-2] if len(pieces) >= 2 and pieces[-2][:1].isupper() else None
        selector = f"{cls}::{func}" if cls else func

        target = in_patch.get(func) or (cls and in_patch.get(cls))
        if not target and len(patch_files) == 1:
            target = patch_files[0]
        if not target and repo_path and patch_dirs:
            target = _search_test_dirs_for_def(Path(repo_path), func, patch_dirs, patch_files)
        if not target and patch_files:
            target = patch_files[0]

        resolved.append(f"{target}::{selector}" if target else norm)
    return resolved


def _search_test_dirs_for_def(
    repo_root: Path, func: str, patch_dirs: set[str], patch_files: list[str]
) -> str | None:
    """Find a file under the patched test directories that defines ``def func``."""
    pattern = re.compile(rf"^\s*def\s+{re.escape(func)}\s*\(", re.MULTILINE)
    hits: list[str] = []
    for rel_dir in patch_dirs:
        directory = repo_root / rel_dir
        if not directory.is_dir():
            continue
        for path in directory.glob("*.py"):
            try:
                if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                    hits.append(path.relative_to(repo_root).as_posix())
            except OSError:
                continue
    if not hits:
        return None
    for preferred in patch_files:
        if preferred in hits:
            return preferred
    return sorted(hits)[0]
