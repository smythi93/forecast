import unittest

from forecast.agents.base import AgentMode, RepairAgent, RepairContext
from forecast.agents.mock_agent.mock_agent import MockRepairAgent
from forecast.integration import diagnosis_text_from_json

ISSUE = "add() subtracts instead of adding."


def _context(diagnosis):
    return RepairContext(
        instance_id="toy-1",
        repo="/tmp/toy",
        version="1",
        problem_statement=ISSUE,
        failing_tests=[],
        diagnosis=diagnosis,
    )


class _Dummy(RepairAgent):
    def repair(self, context): ...
    def get_patch(self, *a, **k): ...
    def get_prompt(self, *a, **k): ...

    @classmethod
    def names(cls):
        return ["dummy"]


class EmptyDiagnosisIsEvidenceTests(unittest.TestCase):
    """
    An empty cause chain is a finding, not a missing diagnosis.

    When no execution feature separates the failing run from the passing ones,
    the runs are behaviourally indistinguishable at the traced level. That
    rules out a divergent branch and points at inputs, configuration, or
    absent behaviour instead -- which an agent can act on. Emitting nothing
    would leave it unable to tell "we looked and they agree" from "we never
    looked". pallets/flask-4992 is the instance that made this concrete: 18
    passing and 1 failing run, all healthy traces, no discriminating feature.
    """

    def setUp(self):
        self.forecast = _Dummy("t", mode=AgentMode.FORECAST)
        self.baseline = _Dummy("t", mode=AgentMode.BASELINE)

    def test_empty_chain_reports_the_finding(self):
        out = self.forecast.diagnosis_block(_context(""))
        self.assertIn("Failure Diagnosis", out)
        self.assertIn("no execution feature that separates them", out)
        self.assertIn("evidence", out)
        # and it must steer somewhere actionable
        self.assertIn("configuration", out)
        self.assertIn("absent", out)

    def test_no_diagnosis_is_not_reported_as_agreement(self):
        """
        ``None`` means collection failed, so nothing was observed about how the
        runs compare. Telling the agent the runs are indistinguishable would
        assert a finding we never made, so the prompt is left unchanged.
        """
        self.assertEqual("", self.forecast.diagnosis_block(_context(None)))

    def test_baseline_still_gets_nothing(self):
        self.assertEqual("", self.baseline.diagnosis_block(_context("")))
        self.assertEqual(
            "", self.baseline.diagnosis_block(_context("1. x (discriminability: 1.00)"))
        )

    def test_weak_chain_is_reported_with_its_weakness(self):
        weak = "1. a (discriminability: 0.01)\n2. b (discriminability: 0.00)"
        out = self.forecast.diagnosis_block(_context(weak))
        self.assertIn("weakly", out)
        self.assertIn("0.01", out)
        self.assertIn("starting point", out)

    def test_strong_chain_is_not_undermined(self):
        out = self.forecast.diagnosis_block(_context("1. a (discriminability: 1.00)"))
        self.assertNotIn("weakly", out)
        self.assertTrue(out.endswith("(discriminability: 1.00)"))

    def test_block_ends_with_the_chain(self):
        chain = "1. a (discriminability: 1.00)"
        out = self.forecast.diagnosis_block(_context(chain))
        self.assertTrue(out.endswith(chain))

    def test_scores_are_parsed_from_the_rendered_chain(self):
        self.assertEqual(
            [1.0, 0.63],
            _Dummy._discriminabilities("x (discriminability: 1.00) y (discriminability: 0.63)"),
        )


class InjectionPathTests(unittest.TestCase):
    """
    The formatting above is only worth anything if the prompt an agent actually
    sends goes through it. An earlier version formatted correctly in a helper
    that no agent called, and every prompt silently dropped empty chains.
    These tests go through an agent's own ``get_prompt``.
    """

    def _prompt(self, mode, diagnosis):
        return MockRepairAgent(mode=mode).get_prompt(_context(diagnosis))

    def test_forecast_prompt_carries_the_chain_exactly_once(self):
        prompt = self._prompt(AgentMode.FORECAST, "1. CHAINMARK (discriminability: 0.90)")
        self.assertTrue(prompt.startswith(ISSUE))
        self.assertEqual(1, prompt.count("CHAINMARK"))

    def test_baseline_prompt_is_the_bare_issue(self):
        prompt = self._prompt(AgentMode.BASELINE, "1. CHAINMARK (discriminability: 0.90)")
        self.assertEqual(ISSUE, prompt)

    def test_empty_chain_reaches_the_prompt(self):
        self.assertIn(
            "no execution feature that separates them",
            self._prompt(AgentMode.FORECAST, ""),
        )

    def test_weak_chain_caveat_reaches_the_prompt(self):
        self.assertIn(
            "weakly", self._prompt(AgentMode.FORECAST, "1. a (discriminability: 0.01)")
        )

    def test_stored_empty_diagnosis_loads_as_empty_not_missing(self):
        """The shape pallets__flask-4992 stores: the run happened, no causes."""
        stored = {"type": "cause_chain", "discriminability": 0.0, "text": "", "causes": []}
        self.assertEqual("", diagnosis_text_from_json(stored))
        self.assertIsNone(diagnosis_text_from_json({}))
