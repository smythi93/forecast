"""
Mock repair agent for testing the evaluation framework.

All five pipeline steps are implemented with trivial deterministic logic so
that the framework can be exercised without a real LLM.
"""

import logging
import time
from typing import Optional

from forecast.agents.base import RepairAgent, RepairContext, RepairResult, AgentMode
from forecast.agents.register import register_agent


@register_agent
class MockRepairAgent(RepairAgent):
    """
    Mock agent for testing the evaluation framework.

    Simulates repair attempts with a configurable success rate without making
    any real LLM calls.

    :ivar name: Agent name (defaults to ``"MockAgent"``).
    :ivar mode: Operating mode (``BASELINE`` or ``FORECAST``).
    :ivar success_rate: Fraction of instances treated as successfully repaired
        (0.0–1.0).
    """

    @classmethod
    def names(cls):
        return ["MockAgent", "mock"]

    def __init__(
        self,
        name: str = "MockAgent",
        mode: AgentMode = AgentMode.BASELINE,
        success_rate: float = 0.3,
    ):
        """
        Initialize the mock agent.

        :param name: Agent name.  Defaults to ``"MockAgent"``.
        :param mode: Operating mode.  Defaults to :attr:`AgentMode.BASELINE`.
        :param success_rate: Fraction of repairs that succeed.  Defaults to
            0.3.
        """
        super().__init__(name, mode)
        self.success_rate = success_rate

    def get_prompt(self, context: RepairContext) -> str:
        """
        Return the issue text (no agent-specific formatting).

        :param context: Repair task context.
        :returns: The problem statement, with the diagnosis in FORECAST mode.
        """
        return self.issue_text(context)

    def get_patch(
        self,
        response,
        context: RepairContext,
        log: Optional[logging.Logger] = None,
    ) -> Optional[str]:
        """
        Return a deterministic mock patch based on the instance ID hash.

        :param response: The model response (content is ignored).
        :param context: Repair task context.
        :param log: Unused.  Defaults to ``None``.
        :returns: A minimal mock unified diff, or ``None`` when the
            deterministic hash would produce a failure.
        """
        boost = 0.15 if self.mode == AgentMode.FORECAST else 0.0
        if hash(context.instance_id) % 100 < (self.success_rate + boost) * 100:
            return self._generate_mock_patch()
        return None

    def validate_patch(self, context: RepairContext, patch: str) -> bool:
        """
        Mock validation — any non-empty patch is considered passing.

        :param context: Repair task context (unused).
        :param patch: Unified diff string to validate.
        :returns: ``True`` if *patch* is non-empty.
        """
        return bool(patch)

    def repair(
        self,
        context: RepairContext,
        model_client=None,
        traj_dir=None,
        log_dir=None,
    ) -> RepairResult:
        """
        Simulate a repair attempt without calling a real LLM.

        :param context: Repair task context.
        :param model_client: Ignored — no LLM call is made.
        :param traj_dir: Ignored.
        :param log_dir: Ignored.
        :returns: Deterministic mock :class:`~forecast.agents.base.RepairResult`.
        """
        start = time.time()
        boost = 0.15 if self.mode == AgentMode.FORECAST else 0.0
        success = hash(context.instance_id) % 100 < (self.success_rate + boost) * 100
        patch = self._generate_mock_patch() if success else None
        prompt = self.get_prompt(context)

        return RepairResult(
            instance_id=context.instance_id,
            agent_name=self.name,
            mode=self.mode,
            success=success,
            patch=patch,
            lines_changed=5 + (hash(context.instance_id) % 10) if success else 0,
            files_modified=1 if success else 0,
            test_pass_rate=1.0 if success else 0.0,
            has_regression=False,
            attempts=1,
            time_seconds=time.time() - start,
            tokens_used=len(prompt) // 4,
            diagnosis_used=(
                context.diagnosis if self.mode == AgentMode.FORECAST else None
            ),
            diagnosis_discriminability=(
                context.diagnosis_discriminability
                if self.mode == AgentMode.FORECAST
                else None
            ),
        )

    @staticmethod
    def _generate_mock_patch() -> str:
        """
        Generate a minimal but syntactically valid mock unified diff.

        :returns: A mock unified diff string.
        """
        return (
            "--- a/example.py\n"
            "+++ b/example.py\n"
            "@@ -10,7 +10,7 @@\n"
            " def example_function():\n"
            "-    return incorrect_value\n"
            "+    return correct_value\n"
        )
