"""
AutoCodeRover, re-implemented in FORECAST from its paper.

AutoCodeRover's code is published under a source-available licence that does
not permit our use, so this is a complete re-implementation of the approach as
the ISSTA'24 paper (Zhang et al., "AutoCodeRover: Autonomous Program
Improvement", Sections 4.1-4.5) describes it, with its parameters:

* **Stratified context retrieval** (Sections 4.2-4.3). The agent is given the
  issue and the seven retrieval APIs of Table 1 (:mod:`forecast.agents.acr.api`,
  AST search over the repository). In each stratum it selects the API calls it
  needs, they are executed, and their results are added to the context; it
  then judges whether the context suffices, and if so names the buggy
  locations. At most :attr:`ACRRepairAgent.STRATA` strata.
* **Analysis-augmented retrieval** (Section 4.4). With tests available, the
  methods spectrum-based fault localization ranks highest are shown before
  retrieval as "results from an external analysis tool". FORECAST computes
  the ranking (method-level Ochiai) over the same test executions it traces --
  in both modes, so the control has the same tests the diagnosis comes from.
* **Patch generation** (Section 4.5). A second agent receives the issue, the
  buggy locations with their exact code, and the retrieval history, and
  writes a patch in a fixed format. A patch that does not follow the format,
  does not apply, fails to parse, or -- tests being available -- fails the
  tests is fed back for another attempt, up to :attr:`PATCH_ATTEMPTS`.
* **Termination** (Section 5): retrieval and patch generation repeat until a
  patch is accepted or :attr:`REPEATS` rounds have run; the best patch so far
  is returned.
* **Decoding** (Section 5): temperature 0.2, at most 1024 tokens per reply.

The issue is :meth:`~forecast.agents.base.RepairAgent.issue_text` -- the issue
plus, in ``FORECAST`` mode, the diagnosis. Patches are applied, linted and
tested in the instance's SWE-bench image, with the developer's tests added.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from forecast.agents.acr.api import ContextRetrievalAPI
from forecast.agents.base import AgentMode, RepairContext
from forecast.agents.loop import LoopRepairAgent

# --------------------------------------------------------------------------
# Prompts. Written for this implementation from the paper's description.
# --------------------------------------------------------------------------

RETRIEVAL_SYSTEM = """\
You are a software developer maintaining a large project and you are fixing an \
issue reported against it. You cannot see the code yet; you explore it through \
search APIs. Work in steps: request the code you need, read what comes back, \
and stop once you understand the issue well enough to name the code that has to \
change."""

API_DESCRIPTION = """\
Search APIs you can call:
- search_class(class_name): the signature of a class (its header and the \
signatures of its members)
- search_class_in_file(class_name, file_path): the same, restricted to one file
- search_method(method_name): the implementation of a method or function
- search_method_in_class(method_name, class_name): a method of one class
- search_method_in_file(method_name, file_path): a method in one file
- search_code(code_str): a code snippet, with 3 lines of context around it
- search_code_in_file(code_str, file_path): the same, restricted to one file"""

RETRIEVAL_INSTRUCTION = """\
First analyse the issue and the code retrieved so far. Then decide:
- If you do not yet have enough context, request only the API calls that are \
necessary next.
- If you have enough context, name the buggy locations: the files, and within \
them the classes and methods, that must be modified.

End your reply with a JSON object in a ```json block:
{"api_calls": [{"name": "<api>", "args": {"<param>": "<value>"}}],
 "sufficient": <true|false>,
 "bug_locations": [{"file": "<path>", "class": "<class or empty>", "method": "<method or empty>"}]}
When "sufficient" is true, "api_calls" may be empty and "bug_locations" must not be."""

SBFL_INTRO = """\
An external analysis tool that runs the project's tests identified the \
following methods as suspicious (most suspicious first). They may help you \
decide where to look; cross-check them against the issue."""

PATCH_SYSTEM = """\
You are a software developer fixing an issue in a large project. You are given \
the issue, the code an earlier investigation found relevant, and the locations \
it identified as buggy. Write a patch that resolves the issue."""

PATCH_INSTRUCTION = """\
Write the patch as one or more modifications in exactly this format:

# modification 1
<file>path/to/file.py</file>
<original>
lines copied exactly from the current file, including indentation
</original>
<patched>
the replacement lines
</patched>

# modification 2
...

Each <original> must match the file exactly and be long enough to be unique. \
Change only what the issue requires; do not modify tests."""

_MODIFICATION_RE = re.compile(
    r"<file>(?P<file>.*?)</file>\s*<original>\n?(?P<original>.*?)</original>\s*"
    r"<patched>\n?(?P<patched>.*?)</patched>",
    re.DOTALL,
)
_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

API_PARAMETERS = {
    "search_class": ["class_name"],
    "search_class_in_file": ["class_name", "file_path"],
    "search_method": ["method_name"],
    "search_method_in_class": ["method_name", "class_name"],
    "search_method_in_file": ["method_name", "file_path"],
    "search_code": ["code_str"],
    "search_code_in_file": ["code_str", "file_path"],
}


@dataclass
class _Patch:
    diff: str
    applies: bool
    feedback: str = ""
    passed: bool = False

    @property
    def rank(self) -> tuple:
        return (self.passed, self.applies)


@dataclass
class _Retrieval:
    """What one retrieval phase found."""

    history: list[dict] = field(default_factory=list)
    bug_locations: list[dict] = field(default_factory=list)
    patches: list[dict] = field(default_factory=list)


def _last_json(text: str) -> Optional[dict]:
    blocks = _JSON_BLOCK_RE.findall(text or "")
    candidates = list(reversed(blocks))
    if not candidates:
        start, end = (text or "").rfind("{"), (text or "").rfind("}")
        if 0 <= start < end:
            candidates = [text[start : end + 1]]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _render(results, kind: str) -> str:
    rendered = []
    for result in results:
        if kind == "class":
            rendered.append(f"Found class {result.class_name} in {result.file_path}:\n{result.signatures}")
        elif kind == "method":
            rendered.append(f"Found {result.function_name} in {result.file_path}:\n{result.code}")
        else:
            rendered.append(f"Found in {result.file_path}:\n{result.code}")
    return "\n\n".join(rendered)


class ACRRepairAgent(LoopRepairAgent):
    """AutoCodeRover as its paper describes it, inside FORECAST."""

    #: Retrieval strata per phase.
    STRATA = 15
    #: Retrieval + patch-generation rounds before giving up (paper, Section 5).
    REPEATS = 10
    #: Patch attempts per round (paper, Section 4.5).
    PATCH_ATTEMPTS = 3
    TEMPERATURE = 0.2
    MAX_TOKENS = 1024
    #: Results shown per API call, and SBFL methods shown.
    MAX_RESULTS = 3
    SBFL_METHODS = 5
    TEST_TIMEOUT = 1800
    #: Retries of a failed model call (transient API errors).
    RETRIES = 5

    @classmethod
    def names(cls) -> list[str]:
        return ["AutoCodeRover", "ACR", "acr", "auto-code-rover", "autocoderover"]

    def __init__(
        self,
        name: str = "AutoCodeRover",
        mode: AgentMode = AgentMode.BASELINE,
        model: str = "claude-haiku-4-5-20251001",
        output_dir: str = ".",
    ):
        super().__init__(name=name, mode=mode, model=model, output_dir=output_dir)

    # -- model ---------------------------------------------------------------

    def _ask(self, client, messages: list[dict]) -> str:
        reply = client.chat(
            messages, temperature=self.TEMPERATURE, max_tokens=self.MAX_TOKENS,
            retries=self.RETRIES,
        )[0]
        messages.append({"role": "assistant", "content": reply.content})
        return reply.content

    # -- retrieval -------------------------------------------------------------

    def _issue_message(self, context: RepairContext) -> str:
        parts = [f"<issue>\n{self.get_prompt(context)}\n</issue>"]
        ranking = (context.fault_localization or [])[: self.SBFL_METHODS]
        if ranking:
            lines = [
                f"{i}. {loc.get('file')}"
                + (f", class {loc['class']}" if loc.get("class") else "")
                + f", method {loc.get('method')}"
                for i, loc in enumerate(ranking, start=1)
            ]
            parts.append(SBFL_INTRO + "\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _call_api(self, api: ContextRetrievalAPI, call: dict) -> str:
        name = str(call.get("name", "")).strip()
        args = call.get("args") or {}
        if name not in API_PARAMETERS or not isinstance(args, dict):
            return f"Unknown API call {name!r}."
        params = API_PARAMETERS[name]
        values = [str(args.get(p, "")).strip() for p in params]
        if not all(values):
            return f"{name} needs {', '.join(params)}."
        kind = "class" if "class" in name and "method" not in name else (
            "method" if "method" in name else "code"
        )
        try:
            results = getattr(api, name)(*values)[: self.MAX_RESULTS]
        except Exception as exc:
            return f"{name}({', '.join(values)}) failed: {exc}"
        if not results:
            return f"{name}({', '.join(repr(v) for v in values)}): nothing found."
        return f"{name}({', '.join(repr(v) for v in values)}):\n" + _render(results, kind)

    def _retrieve(self, client, context, api, log) -> _Retrieval:
        retrieval = _Retrieval()
        messages = [
            {"role": "system", "content": RETRIEVAL_SYSTEM},
            {"role": "user", "content": self._issue_message(context) + "\n\n"
             + API_DESCRIPTION + "\n\n" + RETRIEVAL_INSTRUCTION},
        ]
        for stratum in range(self.STRATA):
            reply = self._ask(client, messages)
            plan = _last_json(reply) or {}
            locations = [l for l in plan.get("bug_locations") or [] if isinstance(l, dict) and l.get("file")]
            calls = [c for c in plan.get("api_calls") or [] if isinstance(c, dict)]
            retrieval.history.append({"analysis": reply})
            if plan.get("sufficient") and locations:
                retrieval.bug_locations = locations
                log.info("ACR: stratum %d named %d buggy location(s)", stratum + 1, len(locations))
                break
            if not calls:
                messages.append({"role": "user", "content": "Your reply named no API calls and no "
                                 "buggy locations. " + RETRIEVAL_INSTRUCTION})
                continue
            results = [self._call_api(api, call) for call in calls]
            retrieval.history.append({"calls": calls, "results": results})
            messages.append({"role": "user", "content": "Results of your API calls:\n\n"
                             + "\n\n".join(results) + "\n\n" + RETRIEVAL_INSTRUCTION})
        return retrieval

    # -- patching --------------------------------------------------------------

    def _location_code(self, api: ContextRetrievalAPI, locations: list[dict]) -> str:
        blocks = []
        for loc in locations:
            file, cls, method = loc.get("file", ""), loc.get("class", ""), loc.get("method", "")
            if method and cls:
                results = api.search_method_in_class(method, cls)
            elif method:
                results = api.search_method_in_file(method, file) or api.search_method(method)
            elif cls:
                results = api.search_class_in_file(cls, file) or api.search_class(cls)
            else:
                results = []
            header = f"Location: {file}" + (f", class {cls}" if cls else "") + (f", method {method}" if method else "")
            body = _render(results[: self.MAX_RESULTS], "method" if method else "class") if results else "(code not found)"
            blocks.append(f"{header}\n{body}")
        return "\n\n".join(blocks)

    @staticmethod
    def _history_text(retrieval: _Retrieval) -> str:
        parts = []
        for step in retrieval.history:
            if "analysis" in step:
                parts.append(f"Analysis:\n{step['analysis']}")
            else:
                parts.append("Retrieved:\n" + "\n\n".join(step["results"]))
        return "\n\n".join(parts)

    def _build_patch(self, repo: Path, reply: str) -> tuple[Optional[str], str]:
        """Apply the reply's modifications to a scratch copy; diff or error."""
        modifications = list(_MODIFICATION_RE.finditer(reply or ""))
        if not modifications:
            return None, "Your reply contains no modification in the required format."
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=repo, capture_output=True)
        for mod in modifications:
            path = repo / mod.group("file").strip()
            if not path.is_file():
                return None, f"File {mod.group('file').strip()} does not exist."
            content = path.read_text(encoding="utf-8")
            original = mod.group("original").rstrip("\n")
            patched = mod.group("patched").rstrip("\n")
            if not original.strip() or content.count(original) == 0:
                return None, f"The <original> of {mod.group('file').strip()} does not match the file."
            if content.count(original) > 1:
                return None, f"The <original> of {mod.group('file').strip()} is not unique in the file."
            content = content.replace(original, patched, 1)
            if path.suffix == ".py":
                try:
                    ast.parse(content)
                except SyntaxError as exc:
                    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=repo, capture_output=True)
                    return None, f"The patched {mod.group('file').strip()} does not parse: {exc}"
            path.write_text(content, encoding="utf-8")
        diff = subprocess.run(["git", "diff", "--no-color"], cwd=repo, capture_output=True, text=True).stdout
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=repo, capture_output=True)
        if not diff.strip():
            return None, "The modifications do not change the code."
        return diff, ""

    def _validate(self, container, task_info: dict, diff: str) -> tuple[bool, str]:
        from forecast.agents.environment import test_verdict

        container.reset()
        container.apply_test_patch()
        applied = container.apply_patch(diff)
        if applied.returncode != 0:
            container.reset()
            return False, f"The patch does not apply: {applied.output[-1500:]}"
        run = container.run_tests(timeout=self.TEST_TIMEOUT)
        container.reset()
        verdict = test_verdict(task_info, run.output)
        if verdict["passed"]:
            return True, ""
        failing = verdict["failing_fail_to_pass"] + verdict["failing_pass_to_pass"]
        return False, (
            "The patch fails these tests:\n- " + "\n- ".join(failing[:20])
            + "\n\nTest output (end):\n" + run.output[-3000:]
        )

    def _generate_patch(self, client, context, api, repo, container, task_info, retrieval, log) -> Optional[_Patch]:
        messages = [
            {"role": "system", "content": PATCH_SYSTEM},
            {"role": "user", "content": f"<issue>\n{self.get_prompt(context)}\n</issue>\n\n"
             "Investigation so far:\n" + self._history_text(retrieval) + "\n\n"
             "Buggy locations and their code:\n" + self._location_code(api, retrieval.bug_locations)
             + "\n\n" + PATCH_INSTRUCTION},
        ]
        best: Optional[_Patch] = None
        for attempt in range(self.PATCH_ATTEMPTS):
            reply = self._ask(client, messages)
            diff, problem = self._build_patch(repo, reply)
            step = {"attempt": attempt + 1, "reply": reply, "diff": diff, "problem": problem}
            retrieval.patches.append(step)
            if diff is None:
                messages.append({"role": "user", "content": problem + "\n\n" + PATCH_INSTRUCTION})
                continue
            passed, feedback = self._validate(container, task_info, diff)
            step.update(passed=passed, feedback=feedback)
            candidate = _Patch(diff=diff, applies=True, feedback=feedback, passed=passed)
            if best is None or candidate.rank >= best.rank:
                best = candidate
            log.info("ACR: patch attempt %d %s", attempt + 1, "passes the tests" if passed else "fails")
            if passed:
                break
            messages.append({"role": "user", "content": feedback + "\n\nRevise the patch. " + PATCH_INSTRUCTION})
        return best

    # -- loop ------------------------------------------------------------------

    def make_container(self, task_info: dict):
        """The instance's container; overridable for tests."""
        from forecast.agents.environment import InstanceContainer

        return InstanceContainer(task_info)

    def run_loop(self, context, model_client, work_dir: Path, log: logging.Logger) -> tuple[str, dict[str, Any]]:
        task_info = self.task_info(context)
        scratch = Path(tempfile.mkdtemp(prefix="acr_", dir=work_dir))
        best: Optional[_Patch] = None
        rounds = 0
        trajectory: list[_Retrieval] = []
        try:
            with self.make_container(task_info) as container:
                repo = container.export(scratch)
                api = ContextRetrievalAPI(repo)
                for rounds in range(1, self.REPEATS + 1):
                    retrieval = self._retrieve(model_client, context, api, log)
                    trajectory.append(retrieval)
                    if not retrieval.bug_locations:
                        log.info("ACR: round %d found no buggy location", rounds)
                        continue
                    patch = self._generate_patch(
                        model_client, context, api, repo, container, task_info, retrieval, log
                    )
                    if patch is not None and (best is None or patch.rank >= best.rank):
                        best = patch
                    if best is not None and best.passed:
                        break
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            (work_dir / "acr_trajectory.json").write_text(json.dumps(
                [{"round": i, "retrieval": r.history, "bug_locations": r.bug_locations,
                  "patch_attempts": r.patches} for i, r in enumerate(trajectory, start=1)],
                indent=2, default=str))
        return (best.diff if best else ""), {
            "exit_status": "validated" if best and best.passed else ("unvalidated" if best else "no_patch"),
            "rounds": rounds,
        }


def register(context):
    """
    Plugin entry point for FORECAST's ``forecast.plugins`` group.
    """

    @context.register_agent(names=ACRRepairAgent.names(), replace=True)
    class _RegisteredACRRepairAgent(ACRRepairAgent):
        pass

    return _RegisteredACRRepairAgent


__all__ = ["ACRRepairAgent", "register"]
