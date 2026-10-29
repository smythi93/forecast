import json
import tempfile
import unittest
from pathlib import Path

from sflkit.analysis.analysis_type import MetaEvent
from sflkit.analysis.predicate import Comp, ReturnPredicate
from sflkit.analysis.spectra import Line
from sflkit.features.value import BinaryFeature
from sflkitlib.events.event import LineEvent

from forecast.diagnosis import (
    Atom,
    CauseChain,
    Conjunction,
    Divergence,
    Forecast,
    FunctionNames,
    Interpreter,
)
from forecast.integration import DiagnosisSerializer, diagnosis_text_from_json


def atom(line: int, discriminability: float, negated: bool = False) -> Atom:
    spectrum = Line(LineEvent("app.py", line, line))
    return Atom(
        BinaryFeature(str(spectrum), spectrum),
        negated=negated,
        discriminability=discriminability,
    )


class CauseChainDiscriminabilityTests(unittest.TestCase):
    def test_chain_reports_the_discriminability_of_its_sharpest_step(self):
        chain = CauseChain([atom(1, 0.4), atom(2, 0.9), atom(3, 0.6)])

        self.assertAlmostEqual(0.9, chain.discriminability)
        self.assertAlmostEqual(0.6333333, chain.mean_discriminability, places=5)

    def test_empty_chain_scores_zero(self):
        chain = CauseChain([])
        self.assertEqual(0.0, chain.discriminability)
        self.assertEqual(0.0, chain.mean_discriminability)

    def test_an_explicit_score_is_respected(self):
        chain = CauseChain([atom(1, 0.4)], discriminability=0.1)
        self.assertAlmostEqual(0.1, chain.discriminability)


SOURCE = """\
def load(path):
    return Loader().read(path)


class Loader:
    def read(self, path):
        def parse(text):
            return text

        return parse(path)


class Token:
    def __init__(self, text):
        self.text = text

    def size(self):
        return len(self.text)


class Keyword(Token):
    @property
    def size(self):
        return 1

    def __init__(self, text):
        super().__init__(text)
"""


def divergence(caller: str, callee: str) -> Divergence:
    return Divergence(caller=caller, callee=callee, feature=BinaryFeature(f"CALL:{callee}", None))


class FunctionNamesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "app").mkdir()
        (self.root / "app" / "tokens.py").write_text(SOURCE)

    def test_methods_are_named_with_their_class(self):
        names = FunctionNames(self.root)
        self.assertEqual("app/tokens.py:Loader.read", names.qualify("app/tokens.py:read:7"))
        self.assertEqual(
            "app/tokens.py:Loader.read.<locals>.parse", names.qualify("app/tokens.py:parse:8")
        )
        self.assertEqual("app/tokens.py:load", names.qualify("app/tokens.py:load:6"))

    def test_the_mapped_line_decides_between_definitions_of_one_name(self):
        self.assertIsNone(FunctionNames(self.root).qualify("app/tokens.py:__init__:3"))
        names = FunctionNames(self.root, {3: 14, 4: 26})
        self.assertEqual("app/tokens.py:Token.__init__", names.qualify("app/tokens.py:__init__:3"))
        self.assertEqual("app/tokens.py:Keyword.__init__", names.qualify("app/tokens.py:__init__:4"))

    def test_a_decorator_line_stands_for_its_function(self):
        names = FunctionNames(self.root, {5: 22, 6: 17})
        self.assertEqual("app/tokens.py:Keyword.size", names.qualify("app/tokens.py:size:5"))
        self.assertEqual("app/tokens.py:Token.size", names.qualify("app/tokens.py:size:6"))

    def test_unknown_names_and_files_stay_unqualified(self):
        names = FunctionNames(self.root)
        self.assertIsNone(names.qualify("app/tokens.py:missing:1"))
        self.assertIsNone(names.qualify("app/absent.py:load:1"))
        self.assertIsNone(names.qualify("FORECAST_ROOT"))

    def test_lines_are_read_from_the_function_entry_events_of_a_mapping(self):
        mapping = self.root / "mapping.json"
        mapping.write_text(json.dumps([
            {"file": "app/tokens.py", "line": 15, "id": 0, "event_type": 2,
             "function": "__init__", "function_id": 3},
            {"file": "app/tokens.py", "line": 16, "id": 1, "event_type": 3,
             "function": "__init__", "function_id": 3, "tmp_var": "t"},
            {"file": "app/tokens.py", "line": 16, "id": 2, "event_type": 0},
        ]))
        names = FunctionNames.from_mapping(self.root, mapping)
        self.assertEqual({3: 15}, names.function_lines)

    def test_the_chain_text_uses_the_qualified_names(self):
        chain = CauseChain([divergence("app/tokens.py:load:6", "app/tokens.py:read:7")])
        chain.qualify_functions(FunctionNames(self.root))
        self.assertIn(
            "called app/tokens.py:Loader.read from app/tokens.py:load,", chain.to_text()
        )


class DivergenceTextTests(unittest.TestCase):
    def test_function_ids_are_left_out_of_the_text(self):
        # Node names carry the function id, which reads like a line number.
        divergence = Divergence(
            caller="app/loader.py:load:201",
            callee="app/tokens.py:make_token:4037",
            feature=BinaryFeature("CALL:app/tokens.py:make_token:4037", None),
        )
        self.assertEqual(
            "The failing run called app/tokens.py:make_token from "
            "app/loader.py:load, which no passing run called",
            divergence.to_text(),
        )

    def test_a_call_from_the_root_names_no_caller(self):
        divergence = Divergence(
            caller="FORECAST_ROOT",
            callee="app/tokens.py:__init__:4036",
            feature=BinaryFeature("CALL:app/tokens.py:__init__:4036", None),
        )
        self.assertEqual(
            "The failing run called app/tokens.py:__init__, which no passing run called",
            divergence.to_text(),
        )


class ReturnTextTests(unittest.TestCase):
    @staticmethod
    def text(op: Comp, value, negate: bool = False) -> str:
        predicate = ReturnPredicate(MetaEvent("app.py", 7, function="load"), op, value)
        return Interpreter.interpret_return(predicate, negate=negate)

    def test_numbers_are_compared_with_zero(self):
        self.assertEqual("Function load in app.py:7 returns > 0", self.text(Comp.GT, 0))
        self.assertEqual("Function load in app.py:7 returns <= 0", self.text(Comp.GT, 0, negate=True))

    def test_none_is_named_as_none(self):
        self.assertEqual("Function load in app.py:7 returns None", self.text(Comp.EQ, None))
        self.assertEqual("Function load in app.py:7 does not return None", self.text(Comp.NE, None))
        self.assertEqual(
            "Function load in app.py:7 does not return None", self.text(Comp.EQ, None, negate=True)
        )

    def test_empty_strings_and_bytes_are_named(self):
        self.assertEqual("Function load in app.py:7 returns an empty string", self.text(Comp.EQ, ""))
        self.assertEqual("Function load in app.py:7 does not return empty bytes", self.text(Comp.NE, b""))


class SerializationTests(unittest.TestCase):
    def test_serialized_chain_carries_the_text_agents_receive(self):
        chain = CauseChain([atom(1, 1.0), atom(2, 0.8)])

        data = DiagnosisSerializer.to_json(chain)

        self.assertEqual("cause_chain", data["type"])
        self.assertIn("text", data)
        self.assertEqual(chain.to_text(), data["text"])
        self.assertAlmostEqual(1.0, data["discriminability"])
        # This is exactly what the repair pipeline reads back.
        self.assertEqual(chain.to_text(), diagnosis_text_from_json(data))

    def test_text_is_reconstructed_for_diagnoses_written_before_it_was_stored(self):
        chain = CauseChain([atom(1, 1.0), Conjunction([atom(2, 0.8), atom(3, 0.8)],
                                                      discriminability=0.8)])
        data = DiagnosisSerializer.to_json(chain)
        legacy = {k: v for k, v in data.items() if k != "text"}

        recovered = diagnosis_text_from_json(legacy)

        self.assertIsNotNone(recovered)
        self.assertIn("Line app.py:1 is executed", recovered)
        # Compound causes keep their operator.
        self.assertIn(" AND ", recovered)

    def test_missing_diagnosis_yields_no_text(self):
        self.assertIsNone(diagnosis_text_from_json({}))
        self.assertIsNone(diagnosis_text_from_json(None))

    def test_diagnosis_without_causes_yields_empty_text(self):
        # The diagnosis ran and kept nothing: a finding, not an absence.
        self.assertEqual("", diagnosis_text_from_json({"type": "cause_chain", "causes": []}))


class MccTests(unittest.TestCase):
    def test_perfect_separation_scores_one(self):
        self.assertAlmostEqual(
            1.0, Forecast.mcc(fail_with=4, pass_with=0, fail_without=0, pass_without=6)
        )

    def test_perfect_anti_correlation_scores_minus_one(self):
        self.assertAlmostEqual(
            -1.0, Forecast.mcc(fail_with=0, pass_with=6, fail_without=4, pass_without=0)
        )

    def test_no_signal_scores_zero(self):
        self.assertAlmostEqual(
            0.0, Forecast.mcc(fail_with=2, pass_with=2, fail_without=2, pass_without=2)
        )

    def test_constant_hypothesis_is_undefined_and_scores_zero(self):
        self.assertEqual(
            0.0, Forecast.mcc(fail_with=0, pass_with=0, fail_without=3, pass_without=5)
        )

    def test_score_is_symmetric_in_the_two_populations(self):
        # Swapping which population is "with" flips the sign, as MCC requires.
        forward = Forecast.mcc(
            fail_with=3, pass_with=1, fail_without=1, pass_without=5
        )
        backward = Forecast.mcc(
            fail_with=1, pass_with=5, fail_without=3, pass_without=1
        )
        self.assertAlmostEqual(forward, -backward)


if __name__ == "__main__":
    unittest.main()
