"""
RepairAgent, ported from Java/Defects4J to Python/SWE-bench inside FORECAST.

RepairAgent (Bouzenia, Devanbu, Pradel, ICSE'25) is an autonomous agent whose
tools are Defects4J commands for Java projects; it cannot run on Python as
published. This port keeps its design (paper, Section III) and its prompts
(MIT-licensed, adapted from Java to Python), and re-targets its tools:

* **Dynamic prompt** (Table I): role, goals, guidelines and fix hints (static);
  current state, the state's tools, gathered information, last command and its
  result, and the remaining budget (dynamic). One LLM query per *cycle*.
* **State machine** (Figure 2): *collect information to understand the bug* ->
  ``express_hypothesis`` -> *collect information to fix the bug* ->
  ``write_fix`` -> *trying out candidate fixes*, with ``discard_hypothesis``
  and ``go_back_to_collect_more_info`` leading back; each state offers
  upstream's set of tools (``commands_by_state``).
* **Tools** (Table II): ``read_range``, ``get_classes_and_methods``,
  ``extract_method_code``, ``extract_test_code``, ``search_code_base``
  (sub-token keyword matching), ``extract_similar_functions_calls``,
  ``AI_generate_method_code`` (a separate LLM query), ``write_fix`` (line-based
  edits, tests run, reverted on failure), and the state commands.
* **Middleware** (Section III-E): tool and argument names are repaired by
  substring or edit-distance matching; a command identical to an earlier one
  is refused (``RESTRICT``); budget of 40 commands, with the reminder to
  suggest 4 fixes (``hyperparams.json``).

Two adaptations, both forced by the setting:

* **Fault localization.** RepairAgent's default gives the agent *perfect*
  fault localization -- the lines the developer changed -- and ``write_fix``
  checks fixes against them. On SWE-bench that leaks the fix, so the port uses
  the paper's other variant, an existing fault localization tool: the
  method-level SBFL ranking FORECAST computes over its traced executions
  (``RepairContext.fault_localization``), in both modes.
* **Tests.** The failing and passing tests are SWE-bench's ``FAIL_TO_PASS``
  and ``PASS_TO_PASS``, run with the project's own test command in the
  instance image with the developer's tests added -- as RepairAgent runs the
  Defects4J test suite. This is what makes it the test-access control.

The bug description RepairAgent gets is
:meth:`~forecast.agents.base.RepairAgent.issue_text` -- the issue plus, in
``FORECAST`` mode, the diagnosis.
"""

from __future__ import annotations

import ast
import difflib
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from forecast.agents.acr.api import ContextRetrievalAPI
from forecast.agents.base import AgentMode, RepairContext
from forecast.agents.loop import LoopRepairAgent

UNDERSTAND = "collect information to understand the bug"
COLLECT = "collect information to fix the bug"
TRY = "trying out candidate fixes"

# --------------------------------------------------------------------------
# Prompt text, adapted from RepairAgent (MIT, see LICENSE in this package):
# prepare_ai_settings.py, prompt_settings.yaml, hints.txt, states_description.json,
# construct_commands_descriptions.py, cycle_instruction_text.txt.
# --------------------------------------------------------------------------

ROLE = """\
You are an AI assistant specialized in fixing bugs in Python code.
You will be given a buggy project, and your objective is to autonomously understand and fix the bug.
You have three states, which each offer a unique set of commands,
 * 'collect information to understand the bug', where you gather information to understand a bug;
 * 'collect information to fix the bug', where you gather information to fix the bug;
 * 'trying out candidate fixes', where you suggest bug fixes that will be validated by a test suite."""

GOALS = """\
- Locate the Bug: Execute test cases and get info to systematically identify and isolate the bug within the project.
- Perform code Analysis: Analyze the lines of code associated with the bug to discern and comprehend the potentially faulty sections.
- Try simple Fixes: Attempt straightforward remedies, such as altering operators, changing identifiers, modifying numerical or boolean literals, adjusting function arguments, or refining conditional statements. Explore all plausible and elementary fixes relevant to the problematic code.
- Try complex Fixes: In instances where simple fixes prove ineffective, utilize the information gathered to propose more intricate solutions aimed at resolving the bug.
- Iterative Testing: Repeat the debugging process iteratively, incorporating the insights gained from each iteration, until all test cases pass successfully."""

GUIDELINES = """\
- Concrete Next Steps: Conclude your reasoning with a clear next step that can be translated into a command call, providing a tangible direction for implementation.
- Code Modification Comments: When modifying code within a file, insert comments above the altered code, elucidating the change and detailing the reasoning behind it.
- Understanding Bug Types: Recognize that bugs can vary, ranging from single-line issues to multiline challenges that may involve changing, removing, or adding lines.
- Operational Constraints: Execute only the commands listed in the commands section."""

HINTS = """\
- 1: Think of mutating comparison operators in a condition, e.g, < to >= and == to !=... Consider all possibilities and combinations even == to >, != to <= ...for example.
    Original: if x < 10:
    Fix: if x >= 10:
- 2: Think of replacing variables with explicit values (when possible) or with close variable names
    Original: result = add(x, y)
    Fix: result = add(5, 8)
- 3: Consider changing some tokens partially, e.g, minXXX to maxXXX startXXX to endXXX, vice versa and so on... (consider all possibilities)
    Original: start_elements_valid = True
    Fix: end_elements_valid = True
- 4: Think of making conditions more complex (by adding 'and' / 'or' operators and the corresponding part) or less complex (by removing some of 'and'/ 'or' parts)
    Original: if (x > 0 and y < 5):
    More complex condition: if (x > 0 and y < 5 and z == 10):
    Less complex condition: if (x > 0):
- 5: if your patch consists only of adding a comment or modifying a line into a comment, you should discard the patch and try to suggest mutation of the buggy line(s) of code instead
- 6: in some cases you will be required to insert new lines of code, you should consider adding more lines at given location. Usually, the code around the insertion location give you an idea about the lines to insert. Also the type of assertion failure.
- 7: pay attention to the comments and doc string above the buggy code, specifically if any important comments exist above or next to buggy lines.
- 8: the usual fix of some exceptions on None values is to add a condition on the variable(s) that can be None.
    Original: entity.create()
    Fix: if entity is not None:
             entity.create()
- 9: Do not modify a file by replacing lines with a comment. If you want to delete a line you can add its number to the list of lines to delete. Always suggest code and not comments only.
- 10: Pay attention the type of variables. The code before and after the buggy lines might give hints of what the type of a variable should be.
- 11: If the buggy line is a function signature, consider changing the signature of the function. Also, consider checking similar functions or neighboring functions by searching the code base."""

STATE_DESCRIPTIONS = {
    UNDERSTAND: "collect information to understand the bug: In this phase, the primary objective is to locate and understand the bug and the reason behind it. However, upon accumulating a satisfactory level of understanding, you should make your hypothesis about what is wrong and where.",
    COLLECT: "collect information to fix the bug: Within this state, the emphasis lies in collecting additional information necessary for addressing and resolving the identified bug. It is permissible to remain in this state for multiple cycles to gather the required information. However, once a sufficient amount of information has been gathered, it is advisable to transition to suggesting fixes by calling the command write_fix.",
    TRY: "trying out candidate fixes: While in this state, your objective is to propose solutions for the bug and assess the outcomes of the test cases executed after implementing the suggested fixes. If your solution effectively resolves the bug, you can consider the goal accomplished. Alternatively, you may choose to remain in this state and propose additional fixes and write them using the command 'write_fix', transition to the state of 'collect information to fix the bug' or shift to the state of 'collect information to understand the bug'.",
}

COMMAND_DESCRIPTIONS = {
    "write_fix": "write_fix: Use this command to implement the fix you came up with. The test cases are run automatically after writing the changes. The changes are reverted automatically if the the test cases fail. This command requires the following params: (changes_dicts: list[dict]) where changes_dicts is a list of dictionaries in the format defined in section '## The format of the fix'. The list should contain at least one non empty dictionary of changes. If you are not already in the state 'trying out candidate fixes', by calling this command you will automatically switch that state. [RESPECT LINES NUMBERS AS GIVEN IN THE LIST OF READ LINES SECTIONS]",
    "read_range": "read_range: Read a range of lines in a given file, params: (filepath: string, startline: int, endline: int)",
    "go_back_to_collect_more_info": "go_back_to_collect_more_info: This command allows you to go back to the state 'collect information to fix the bug'. Call this command after you have suggested many fixes but none of them worked, params: (reason_for_going_back: string)",
    "discard_hypothesis": "discard_hypothesis: This command allows you to discard the hypothesis that you made earlier about the bug and automatically return back again to the state 'collect information to understand the bug' where you can express a new hypothesis, params: (reason_for_discarding: string), calling this command will automatically change the state to 'collect information to understand the bug'",
    "goals_accomplished": "goals_accomplished: Call this function when you are sure you fixed the bug and all tests have passed and give the reason that made you believe that you fixed the bug successfully, params: (reason: string)",
    "search_code_base": "search_code_base: This utility function scans all Python files within the project for a given list of keywords. It generates a dictionary as output, organized by file names, classes, and method names. Within each method name, it provides a list of keywords that match the method's content. The resulting structure is as follows: { file_name: { class_name: { method_name: [...list of matched keywords...] } } }. This functionality proves beneficial for identifying pre-existing methods that may be reusable or for locating similar code to gain insights into implementing specific functionalities. It's important to note that this function does not return the actual code but rather the names of matched methods containing at least one of the specified keywords. It requires the following params: (key_words: list). Once the method names are obtained, the extract_method_code command can be used to retrieve their corresponding code snippets (only do it for the ones that are relevant)",
    "get_classes_and_methods": "get_classes_and_methods: This function allows you to get all classes and methods names within a file. It returns a dictionary where keys are classes names and values are list of methods names within each class. The required params are: (file_path: string)",
    "extract_similar_functions_calls": "extract_similar_functions_calls: For a provided buggy code snippet in 'code_snippet' within the file 'file_path', this function extracts similar function calls. This aids in understanding how functions are utilized in comparable code snippets, facilitating the determination of appropriate parameters to pass to a function, params: (file_path: string, code_snippet: string)",
    "extract_method_code": "extract_method_code: This command allows you to extract possible implementations of a given method name inside a file. The required params to call this command are: (filepath: string, method_name: string)",
    "AI_generate_method_code": "AI_generate_method_code: This function allows to use an AI Large Language model to generate the code of the buggy method. This helps see another implementation of that method given the context before it which would help in 'probably' infering a fix but no garantee. params: (filepath: str, method_name: str)",
    "extract_test_code": "extract_test_code: This function allows you to extract the code of the failing test cases which will help you understand the test case that led to failure for example by looking at the assertions and the given input and expected output, params: (test_file_path: string). You are allowed to execute this command for once only, unless it returns an error message, in which case you can try again with different arguments.",
    "express_hypothesis": "express_hypothesis: This command allows to express a hypothesis about what exactly is the bug. Call this command after you have collected enough information about the bug in the project, params: (hypothesis: string). By calling this command, you also automatically switch to the state 'collect information to fix the bug'. Before delving into fixing, you should always express a hypothesis.",
}

COMMANDS_BY_STATE = {
    TRY: ["write_fix", "read_range", "go_back_to_collect_more_info", "discard_hypothesis", "goals_accomplished"],
    COLLECT: ["search_code_base", "get_classes_and_methods", "extract_similar_functions_calls",
              "extract_method_code", "write_fix", "read_range", "AI_generate_method_code"],
    UNDERSTAND: ["extract_test_code", "express_hypothesis", "read_range"],
}

PARAMETERS = {
    "write_fix": ["changes_dicts"],
    "read_range": ["filepath", "startline", "endline"],
    "go_back_to_collect_more_info": ["reason_for_going_back"],
    "discard_hypothesis": ["reason_for_discarding"],
    "goals_accomplished": ["reason"],
    "search_code_base": ["key_words"],
    "get_classes_and_methods": ["file_path"],
    "extract_similar_functions_calls": ["file_path", "code_snippet"],
    "extract_method_code": ["filepath", "method_name"],
    "AI_generate_method_code": ["filepath", "method_name"],
    "extract_test_code": ["test_file_path"],
    "express_hypothesis": ["hypothesis"],
}

FIX_FORMAT = """\
## The format of the fix
The fix is a list of dictionaries, one per file to change:
[
  {
    "file_path": "path/to/file.py",
    "insertions": [{"line_number": 175, "new_lines": ["    new line 1\\n", "    new line 2\\n"]}],
    "deletions": [179, 183],
    "modifications": [{"line_number": 181, "modified_line": "    if dataset is None:\\n"}]
  }
]
Line numbers refer to the file as it is before the fix. Insertions go before the given line. \
Keep the indentation of the surrounding code."""

OUTPUT_FORMAT = """\
Determine exactly one command to use based on the given goals and the progress you have made so far, and respond using the JSON schema specified previously:
Respond strictly with JSON. The JSON should be compatible with the TypeScript type `Response` from the following:
```ts
interface Response {
// Express your thoughts based on the information that you have collected so far, the possible steps that you could do next and also your reasoning about fixing the bug in question"
thoughts: string;
command: {
name: string;
args: Record<string, any>;
};
}
```
**IMPORTANT NOTE TO THE ASSISTANT:** DO NOT OUTPUT ANY ENGLISH TEXT WITH YOUR JSON RESPONSE."""

GENERATE_METHOD_PROMPT = """\
Here is the code of a Python file up to the start of the function `{name}`:

```python
{context}
```

Write the complete implementation of `{name}`, starting with this signature:

```python
{signature}
```

Reply with the code only, in one ```python block."""


def _subtokens(word: str) -> list[str]:
    parts = re.split(r"[_.\s]+", word)
    tokens = []
    for part in parts:
        tokens += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
    return [t.lower() for t in tokens if len(t) > 1]


def _normalized_distance(a: str, b: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def _closest(name: str, candidates: list[str], threshold: float = 0.1) -> Optional[str]:
    """Upstream's middleware: substring match, then edit distance below *threshold*."""
    if name in candidates:
        return name
    matches = [c for c in candidates if name and (name in c or c in name)]
    if len(matches) == 1:
        return matches[0]
    close = [c for c in candidates if _normalized_distance(name, c) < threshold]
    return close[0] if len(close) == 1 else None


def _first_json(text: str) -> Optional[dict]:
    text = text or ""
    start = text.find("{")
    while start >= 0:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                escaped = (ch == "\\") and not escaped
                if ch == '"' and not escaped:
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


class RepairAgentRepairAgent(LoopRepairAgent):
    """RepairAgent's design, on Python and SWE-bench."""

    #: ``hyperparams.json``: commands_limit and the FULL-TRACK fix target.
    COMMANDS_LIMIT = 40
    TARGET_FIXES = 4
    TEMPERATURE = 0.0
    MAX_TOKENS = 4096
    #: ``AI_generate_method_code`` limits: 12k tokens of context, 4k generated.
    GENERATE_CONTEXT_CHARS = 48_000
    GENERATE_MAX_TOKENS = 4096
    #: SBFL methods shown as the fault localization result.
    LOCALIZATION_METHODS = 10
    OUTPUT_CHARS = 4000
    TEST_TIMEOUT = 1800
    #: Retries of a failed model call (transient API errors).
    RETRIES = 5

    @classmethod
    def names(cls) -> list[str]:
        return ["RepairAgent", "repair-agent", "repairagent"]

    def __init__(
        self,
        name: str = "RepairAgent",
        mode: AgentMode = AgentMode.BASELINE,
        model: str = "claude-haiku-4-5-20251001",
        output_dir: str = ".",
    ):
        super().__init__(name=name, mode=mode, model=model, output_dir=output_dir)

    def make_container(self, task_info: dict):
        """The instance's container; overridable for tests."""
        from forecast.agents.environment import InstanceContainer

        return InstanceContainer(task_info)

    # -- the run ---------------------------------------------------------------

    def run_loop(self, context, model_client, work_dir: Path, log: logging.Logger) -> tuple[str, dict[str, Any]]:
        task_info = self.task_info(context)
        scratch = Path(tempfile.mkdtemp(prefix="repairagent_", dir=work_dir))
        try:
            with self.make_container(task_info) as container:
                container.apply_test_patch()
                repo = container.export(scratch)
                # Commit the developer's tests in the exported copy, so that
                # resetting between fixes keeps them and every fix's diff is
                # the fix alone.
                for cmd in (["git", "add", "-A"],
                            ["git", "-c", "user.email=forecast@localhost", "-c", "user.name=forecast",
                             "commit", "-qm", "tests", "--no-verify"]):
                    subprocess.run(cmd, cwd=repo, capture_output=True)
                session = _Session(self, context, model_client, container, repo, task_info, log)
                try:
                    return session.run()
                finally:
                    (work_dir / "repairagent_trajectory.json").write_text(json.dumps(
                        {"initial_information": session.initial, "steps": session.steps,
                         "hypotheses": session.hypotheses,
                         "fixes": session.fixes}, indent=2, default=str))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


class _Session:
    """One RepairAgent run: the dynamic prompt, the middleware, the tools."""

    def __init__(self, agent, context, client, container, repo, task_info, log):
        self.agent = agent
        self.context = context
        self.client = client
        self.container = container
        self.repo = Path(repo)
        self.task_info = task_info
        self.log = log
        self.api = ContextRetrievalAPI(self.repo)
        self.state = UNDERSTAND
        self.cycle = 0
        self.gathered: dict[str, list[str]] = {}
        self.hypotheses: list[str] = []
        self.history: list[str] = []
        self.fixes: list[dict] = []
        self.plausible: Optional[str] = None
        self.last = "No command executed yet."
        self.tools: dict[str, Callable[..., str]] = {
            "write_fix": self.write_fix,
            "read_range": self.read_range,
            "go_back_to_collect_more_info": self.go_back,
            "discard_hypothesis": self.discard_hypothesis,
            "goals_accomplished": self.goals_accomplished,
            "search_code_base": self.search_code_base,
            "get_classes_and_methods": self.get_classes_and_methods,
            "extract_similar_functions_calls": self.extract_similar_calls,
            "extract_method_code": self.extract_method_code,
            "AI_generate_method_code": self.generate_method_code,
            "extract_test_code": self.extract_test_code,
            "express_hypothesis": self.express_hypothesis,
        }
        self.done = False
        #: Every cycle: state, the model's reply, the command, its full output.
        self.steps: list[dict] = []
        self.initial = self._initial_information()

    # -- the dynamic prompt ---------------------------------------------------

    def _initial_information(self) -> str:
        tests = json.loads(self.task_info["FAIL_TO_PASS"]) if isinstance(
            self.task_info["FAIL_TO_PASS"], str) else list(self.task_info["FAIL_TO_PASS"])
        run = self._run_tests()
        ranking = (self.context.fault_localization or [])[: self.agent.LOCALIZATION_METHODS]
        localization = "\n".join(
            f"{i}. {r['file']}, function {r['method']} (line {r['line']}), "
            f"suspiciousness {r['suspiciousness']:.2f}"
            for i, r in enumerate(ranking, start=1)
        ) or "(no fault localization available)"
        return (
            f"## Bug description\n{self.agent.get_prompt(self.context)}\n\n"
            f"## Failing test cases\n" + "\n".join(f"- {t}" for t in tests) + "\n\n"
            f"## Result of running the tests\n{run}\n\n"
            f"## Fault localization (spectrum-based, from an external tool)\n{localization}"
        )

    def prompt(self) -> list[dict]:
        commands = COMMANDS_BY_STATE[self.state]
        gathered = "\n\n".join(
            f"### {tool}\n" + "\n\n".join(outputs[-3:]) for tool, outputs in self.gathered.items()
        ) or "(nothing yet)"
        hypotheses = "\n".join(f"- {h}" for h in self.hypotheses) or "(none yet)"
        fixes = "\n".join(
            f"- fix {i}: {'passes all tests' if f['passed'] else 'fails: ' + f['summary']}"
            for i, f in enumerate(self.fixes, start=1)
        ) or "(none yet)"
        budget = (
            f"You have, so far, executed, {self.cycle} commands and suggested {len(self.fixes)} fixes. "
            f"You have {self.agent.COMMANDS_LIMIT - self.cycle} commands left. However, you need to suggest "
            f"{max(0, self.agent.TARGET_FIXES - len(self.fixes))} fixes before consuming all the left commands."
        )
        user = "\n\n".join([
            "## Goals\n" + GOALS,
            "## Guidelines\n" + GUIDELINES,
            "## Hints for fixing bugs\n" + HINTS,
            self.initial,
            "## Current state\n" + STATE_DESCRIPTIONS[self.state],
            "## Commands\n" + "\n".join(f"{i}. {COMMAND_DESCRIPTIONS[c]}" for i, c in enumerate(commands, start=1)),
            FIX_FORMAT,
            "## Hypothesis about the bug\n" + hypotheses,
            "## Fixes suggested so far\n" + fixes,
            "## Gathered information\n" + gathered,
            "## Last executed command and result\n" + self.last,
            budget,
            OUTPUT_FORMAT,
        ])
        return [{"role": "system", "content": ROLE}, {"role": "user", "content": user}]

    # -- the middleware ------------------------------------------------------

    def run(self) -> tuple[str, dict[str, Any]]:
        while self.cycle < self.agent.COMMANDS_LIMIT and not self.done:
            # The prompt reports the commands executed so far, then the cycle counts.
            messages = self.prompt()
            self.cycle += 1
            reply = self.client.chat(
                messages, temperature=self.agent.TEMPERATURE, max_tokens=self.agent.MAX_TOKENS,
                retries=self.agent.RETRIES,
            )[0].content
            step = {"cycle": self.cycle, "state": self.state, "reply": reply}
            self.steps.append(step)
            response = _first_json(reply)
            if not response or not isinstance(response.get("command"), dict):
                self.last = "Your response was not valid JSON in the required format."
                step["output"] = self.last
                continue
            command = response["command"]
            name = _closest(str(command.get("name", "")), COMMANDS_BY_STATE[self.state])
            if name is None:
                self.last = (f"The command {command.get('name')!r} is not available in the state "
                             f"'{self.state}'. Use one of: {', '.join(COMMANDS_BY_STATE[self.state])}.")
                step["output"] = self.last
                continue
            args = self._map_args(name, command.get("args") or {})
            key = json.dumps([name, args], sort_keys=True, default=str)
            if key in self.history:
                self.last = (f"You already executed {name} with these arguments. "
                             "Repeating a command is not allowed; choose a different step.")
                step["output"] = self.last
                continue
            self.history.append(key)
            try:
                output = self.tools[name](**args)
            except TypeError as exc:
                output = f"Invalid arguments for {name}: {exc}"
            except Exception as exc:  # a failing tool is reported to the agent
                output = f"{name} failed: {exc}"
            step.update(command=name, args=args, output=output, state_after=self.state)
            self.last = f"Command: {name}({json.dumps(args, default=str)[:1500]})\nResult:\n{output}"
            self.log.info("RepairAgent cycle %d: %s (state: %s)", self.cycle, name, self.state)
        status = "plausible_fix" if self.plausible else "no_plausible_fix"
        return self.plausible or "", {"exit_status": status, "cycles": self.cycle, "fixes": len(self.fixes)}

    @staticmethod
    def _map_args(name: str, args: dict) -> dict:
        expected = PARAMETERS[name]
        mapped = {}
        for key, value in args.items():
            target = _closest(str(key), expected)
            if target is not None:
                mapped[target] = value
        return mapped

    def _remember(self, tool: str, output: str) -> str:
        self.gathered.setdefault(tool, []).append(output)
        return output

    def _clip(self, text: str) -> str:
        limit = self.agent.OUTPUT_CHARS
        return text if len(text) <= limit else text[:limit] + "\n...[output truncated]..."

    def _file(self, path: str) -> Path:
        candidate = (self.repo / str(path).strip().lstrip("/")).resolve()
        if self.repo.resolve() not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(f"No file {path} in the project.")
        return candidate

    # -- tools -----------------------------------------------------------------

    def _run_tests(self) -> str:
        from forecast.agents.environment import test_verdict

        run = self.container.run_tests(timeout=self.agent.TEST_TIMEOUT)
        verdict = test_verdict(self.task_info, run.output)
        failing = verdict["failing_fail_to_pass"] + verdict["failing_pass_to_pass"]
        head = (f"{len(failing)} failing test(s)" + (":\n- " + "\n- ".join(failing[:20]) if failing else ""))
        return head + "\n\nTest output (end):\n" + run.output[-2500:]

    def read_range(self, filepath, startline, endline) -> str:
        lines = self._file(filepath).read_text(encoding="utf-8", errors="replace").splitlines()
        start, end = max(1, int(startline)), min(len(lines), int(endline))
        text = "\n".join(f"Line {i}:{lines[i - 1]}" for i in range(start, end + 1))
        return self._remember("read_range", f"{filepath} lines {start}-{end}:\n{self._clip(text)}")

    def get_classes_and_methods(self, file_path) -> str:
        tree = ast.parse(self._file(file_path).read_text(encoding="utf-8", errors="replace"))
        result: dict[str, list[str]] = {"(module level)": []}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                result[node.name] = [
                    f.name for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result["(module level)"].append(node.name)
        return self._remember("get_classes_and_methods", f"{file_path}: {json.dumps(result)}")

    def extract_method_code(self, filepath, method_name) -> str:
        results = self.api.search_method_in_file(str(method_name), str(filepath).lstrip("/"))
        if not results:
            return f"No method {method_name} found in {filepath}."
        body = "\n\n".join(r.code for r in results)
        return self._remember("extract_method_code", f"{filepath}, {method_name}:\n{self._clip(body)}")

    def extract_test_code(self, test_file_path) -> str:
        path = str(test_file_path).split("::")[0].lstrip("/")
        tests = json.loads(self.task_info["FAIL_TO_PASS"]) if isinstance(
            self.task_info["FAIL_TO_PASS"], str) else list(self.task_info["FAIL_TO_PASS"])
        names = {re.split(r"[\s(\[]", t.split("::")[-1])[0] for t in tests}
        blocks = []
        for name in sorted(names):
            for result in self.api.search_method_in_file(name, path):
                blocks.append(result.code)
        if not blocks:
            return f"No failing test found in {test_file_path}. Failing tests: {', '.join(tests)}"
        return self._remember("extract_test_code", self._clip("\n\n".join(blocks)))

    def search_code_base(self, key_words) -> str:
        if isinstance(key_words, str):
            key_words = [key_words]
        tokens = {t for word in key_words for t in _subtokens(str(word))} | {
            str(w).lower() for w in key_words if " " not in str(w)}
        result: dict[str, dict[str, dict[str, list[str]]]] = {}
        for path in sorted(self.repo.rglob("*.py")):
            rel = str(path.relative_to(self.repo))
            if any(part.startswith(".") for part in Path(rel).parts):
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                continue
            lines = source.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for func in node.body:
                    if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        text = "\n".join(lines[func.lineno - 1 : func.end_lineno]).lower()
                        hits = sorted(t for t in tokens if t in text)
                        if hits:
                            result.setdefault(rel, {}).setdefault(node.name, {})[func.name] = hits
            for func in tree.body:
                if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    text = "\n".join(lines[func.lineno - 1 : func.end_lineno]).lower()
                    hits = sorted(t for t in tokens if t in text)
                    if hits:
                        result.setdefault(rel, {}).setdefault("(module level)", {})[func.name] = hits
        ranked = sorted(result.items(), key=lambda kv: -sum(len(m) for c in kv[1].values() for m in c.values()))
        return self._remember("search_code_base", self._clip(json.dumps(dict(ranked[:30]), indent=1)))

    def extract_similar_calls(self, file_path, code_snippet) -> str:
        called = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(code_snippet))
        if not called:
            return "The snippet contains no function call."
        name = called[-1] if called[-1] not in {"if", "for", "while", "print"} else called[0]
        pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
        hits = []
        for path in sorted(self.repo.rglob("*.py")):
            rel = str(path.relative_to(self.repo))
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pattern.search(line) and not line.lstrip().startswith(("def ", "#")):
                        hits.append(f"{rel}:{number}: {line.strip()}")
            except OSError:
                continue
            if len(hits) >= 40:
                break
        return self._remember("extract_similar_functions_calls",
                              f"Calls to {name}:\n" + ("\n".join(hits) or "(none)"))

    def generate_method_code(self, filepath, method_name) -> str:
        source = self._file(filepath).read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                signature = "\n".join(lines[node.lineno - 1 : node.body[0].lineno - 1]) or lines[node.lineno - 1]
                before = "\n".join(lines[: node.lineno - 1])[-self.agent.GENERATE_CONTEXT_CHARS :]
                reply = self.client.chat(
                    [{"role": "user", "content": GENERATE_METHOD_PROMPT.format(
                        name=method_name, context=before, signature=signature)}],
                    temperature=self.agent.TEMPERATURE,
                    max_tokens=self.agent.GENERATE_MAX_TOKENS,
                    retries=self.agent.RETRIES,
                )[0].content
                return self._remember("AI_generate_method_code", self._clip(reply))
        return f"No method {method_name} in {filepath}."

    def express_hypothesis(self, hypothesis) -> str:
        self.hypotheses.append(str(hypothesis))
        self.state = COLLECT
        return ("Since you have a hypothesis about the bug, the current state have been changed from "
                "'collect information to understand the bug' to 'collect information to fix the bug'")

    def discard_hypothesis(self, reason_for_discarding) -> str:
        if self.hypotheses:
            self.hypotheses[-1] += f" [discarded: {reason_for_discarding}]"
        self.state = UNDERSTAND
        return "Hypothesis discarded! You are now back at the state 'collect information to understand the bug'"

    def go_back(self, reason_for_going_back) -> str:
        self.state = COLLECT
        return "You are now back at the state 'collect information to fix the bug'"

    def goals_accomplished(self, reason) -> str:
        if self.plausible:
            self.done = True
            return "Goal accomplished."
        return "No fix has passed all tests yet, so the goal is not accomplished."

    def _apply_changes(self, changes: list) -> str:
        """Apply line-based changes to the exported tree; returns the diff."""
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=self.repo, capture_output=True)
        for change in changes:
            path = self._file(change["file_path"])
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            deletions = {int(n) for n in change.get("deletions") or []}
            modifications = {int(m["line_number"]): m["modified_line"] for m in change.get("modifications") or []}
            insertions: dict[int, list[str]] = {}
            for ins in change.get("insertions") or []:
                insertions.setdefault(int(ins["line_number"]), []).extend(ins.get("new_lines") or [])
            out = []
            for number, line in enumerate(lines + [""], start=1):
                for new in insertions.get(number, []):
                    out.append(new if new.endswith("\n") else new + "\n")
                if number > len(lines) or number in deletions:
                    continue
                if number in modifications:
                    new = modifications[number]
                    out.append(new if new.endswith("\n") else new + "\n")
                else:
                    out.append(line)
            path.write_text("".join(out), encoding="utf-8")
        diff = subprocess.run(["git", "diff", "--no-color"], cwd=self.repo, capture_output=True, text=True).stdout
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=self.repo, capture_output=True)
        return diff

    def write_fix(self, changes_dicts) -> str:
        from forecast.agents.environment import test_verdict

        if isinstance(changes_dicts, str):
            changes_dicts = json.loads(changes_dicts)
        if isinstance(changes_dicts, dict):
            changes_dicts = [changes_dicts]
        if not changes_dicts:
            return "The fix you passed is empty. Please provide a non empty implementation of the fix."
        diff = self._apply_changes(changes_dicts)
        if not diff.strip():
            return "The fix does not change the code."
        # The container holds the tests; the source diff goes on top of them.
        applied = self.container.apply_patch(diff)
        if applied.returncode != 0:
            return f"The fix could not be applied: {applied.output[-1000:]}"
        run = self.container.run_tests(timeout=self.agent.TEST_TIMEOUT)
        self.container.reset()
        self.container.apply_test_patch()
        verdict = test_verdict(self.task_info, run.output)
        failing = verdict["failing_fail_to_pass"] + verdict["failing_pass_to_pass"]
        self.fixes.append({"diff": diff, "passed": verdict["passed"],
                           "summary": f"{len(failing)} failing test(s)"})
        self.state = TRY
        note = "\n **Note:** You are automatically switched to the state 'trying out candidate fixes'"
        if verdict["passed"]:
            self.plausible = self.plausible or diff
            return "The fix passed all test cases. 0 failing tests." + note
        return (f"The fix was applied, the tests ran and the changes were reverted. "
                f"{len(failing)} failing test(s):\n- " + "\n- ".join(failing[:20])
                + "\n\nTest output (end):\n" + run.output[-2000:] + note)


def register(context):
    """
    Plugin entry point for FORECAST's ``forecast.plugins`` group.
    """

    @context.register_agent(names=RepairAgentRepairAgent.names(), replace=True)
    class _RegisteredRepairAgentRepairAgent(RepairAgentRepairAgent):
        pass

    return _RegisteredRepairAgentRepairAgent


__all__ = ["RepairAgentRepairAgent", "register"]
