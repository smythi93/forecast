"""
Scripted stand-ins for an LLM and a SWE-bench instance, for agent tests.

:class:`ScriptedChatClient` replays fixed replies through FORECAST's
``ModelClient.chat`` contract -- including a litellm response object, which
upstream agents parse themselves -- and records every request, so a test can
check what the agent actually sent. :func:`toy_repo` builds a tiny repository
with one bug and a SWE-bench-shaped row describing it.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional, Union

from forecast.agents.models.utils import ChatResponse, ModelClient, ModelResponse

ISSUE = "add() subtracts instead of adding: add(2, 3) returns -1."
TEST_PATCH_MARK = "HIDDEN_ORACLE_TEST"
FIX_DIFF = (
    "diff --git a/calc.py b/calc.py\n"
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)

Reply = Union[str, dict]


def tool_call(command: str, name: str = "bash", **arguments) -> dict:
    """A reply consisting of one tool call (OpenAI format)."""
    args = {"command": command, **arguments} if command is not None else arguments
    return {
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _litellm_response(model: str, content: str, tool_calls: list[dict]):
    import litellm

    message = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return litellm.ModelResponse(
        model=model,
        choices=[
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


class ScriptedChatClient(ModelClient):
    """
    A FORECAST model client that replays scripted replies.

    :param script: Replies in order (the last repeats), or a function of the
        call index and the messages returning the reply.
    """

    LITELLM_PROVIDER = "anthropic"
    context_window = 200_000

    def __init__(self, script: Union[list[Reply], Callable[[int, list], Reply]], cost: float = 0.0):
        super().__init__(model="claude-haiku-4-5-20251001")
        self.script = script
        self.cost = cost
        self.requests: list[dict] = []

    def _next(self, messages) -> dict:
        index = len(self.requests)
        if callable(self.script):
            reply = self.script(index, messages)
        else:
            reply = self.script[min(index, len(self.script) - 1)]
        return {"content": reply, "tool_calls": []} if isinstance(reply, str) else reply

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None, n=1, retries=0, **kwargs):
        replies = []
        for _ in range(max(1, n)):
            reply = self._next(messages)
            self.requests.append(
                {"messages": json.loads(json.dumps(messages, default=str)), "tools": tools,
                 "temperature": temperature, "max_tokens": max_tokens, "n": n}
            )
            raw = _litellm_response(self.model, reply["content"], reply["tool_calls"])
            response = ChatResponse(
                content=reply["content"],
                model=self.model,
                tool_calls=reply["tool_calls"],
                prompt_tokens=10,
                completion_tokens=5,
                cost=self.cost,
                raw=raw,
            )
            replies.append(response)
            if self.recorder is not None:
                self.recorder(
                    {"messages": messages, "content": response.content,
                     "prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}
                )
        return replies

    def complete(self, prompt, system_prompt=None) -> ModelResponse:
        reply = self.chat([{"role": "user", "content": prompt}])[0]
        return ModelResponse(content=reply.content, model=self.model, total_tokens=15)

    def sent_text(self) -> str:
        """Everything the agent sent, flattened."""
        return json.dumps([r["messages"] for r in self.requests])


def toy_repo() -> tuple[Path, dict]:
    """
    A git repository with one bug, and a SWE-bench-shaped row for it.

    :returns: The repository path and the row (``swe_task_info``).
    """
    repo = Path(tempfile.mkdtemp(prefix="toyrepo_"))
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_zero():\n    assert add(0, 0) == 0\n"
    )
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    test_patch = (
        "diff --git a/tests/test_calc.py b/tests/test_calc.py\n"
        "--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n"
        "@@ -3,3 +3,7 @@ from calc import add\n \n def test_zero():\n"
        "     assert add(0, 0) == 0\n+\n+\n"
        f"+def test_add():  # {TEST_PATCH_MARK}\n+    assert add(2, 3) == 5\n"
    )
    row = {
        "instance_id": "toy__calc-1",
        "repo": "toy/calc",
        "version": "1.0",
        "base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        ).stdout.strip(),
        "problem_statement": ISSUE,
        "patch": FIX_DIFF,
        "test_patch": test_patch,
        "FAIL_TO_PASS": json.dumps(["tests/test_calc.py::test_add"]),
        "PASS_TO_PASS": json.dumps(["tests/test_calc.py::test_zero"]),
        "hints_text": "",
        "created_at": "2024-01-01T00:00:00Z",
        "environment_setup_commit": "",
    }
    return repo, row


def context_for(repo: Path, row: dict, diagnosis: Optional[str]):
    from forecast.agents.base import RepairContext

    return RepairContext(
        instance_id=row["instance_id"],
        repo=str(repo),
        version=row["version"],
        problem_statement=row["problem_statement"],
        failing_tests=json.loads(row["FAIL_TO_PASS"]),
        passing_tests=json.loads(row["PASS_TO_PASS"]),
        diagnosis=diagnosis,
        diagnosis_discriminability=0.9 if diagnosis else None,
        base_commit=row["base_commit"],
        swe_task_info=row,
    )
