"""
Diagnosis engine for FORECAST.

This module implements the full diagnosis pipeline that converts a
:class:`~forecast.tree.TracedRelevanceTree` (built from execution traces) into
a human-readable cause-effect chain that explains why the failing tests behave
differently from the passing ones.

Key classes
-----------
* :class:`Interpreter` — translates ``sflkit`` feature objects into natural-language
  sentences.
* :class:`Diagnosis` / :class:`Atom` / :class:`Conjunction` / :class:`Disjunction` /
  :class:`CauseChain` — the diagnosis AST produced by the engine.
* :class:`Forecast` — the main diagnosis-derivation algorithm (MCC-based
  discriminability scoring, subsumption pruning, chain extraction).
"""

import abc
import ast
import enum
import json
from pathlib import Path
from typing import List, Optional

from sflkit.analysis.analysis_type import AnalysisType
from sflkit.analysis.spectra import Line, Function, Loop, DefUse, Length
from sflkit.analysis.predicate import (
    FunctionErrorPredicate,
    Branch,
    Condition,
    IsAsciiPredicate,
    ContainsDigitPredicate,
    ContainsSpecialPredicate,
    EmptyStringPredicate,
    EmptyBytesPredicate,
    ReturnPredicate,
    NonePredicate,
    VariablePredicate,
    ScalarPair,
    Comp,
)
from sflkit.features.value import BinaryFeature, FeatureValue, Feature
from sflkit.features.vector import FeatureVector
from sflkit.runners.run import TestResult
from sflkitlib.events import EventType

from forecast.logger import LOGGER
from forecast.tree import TracedRelevanceTree

EPSILON = 1e-8


class EntryPoint(enum.Enum):
    """
    Marks whether a diagnosis atom was observed at function *entry* or *exit*.

    :cvar ENTER: The feature was observed when the function was entered.
    :cvar EXIT: The feature was observed when the function returned.
    """

    ENTER = "ENTER"
    EXIT = "EXIT"


class Interpreter:
    """
    Translates ``sflkit`` feature / analysis objects into natural-language strings.

    Each ``interpret_*`` static method handles one
    :class:`~sflkit.analysis.analysis_type.AnalysisType` and returns a
    human-readable sentence such as::

        'Function \"foo\" in bar.py:42 raises an error'
        'Else-branch in baz.py:17 is taken'

    :attr:`TYPE_TO_METHOD` maps each analysis type to its method so that
    :meth:`interpret` can dispatch dynamically.
    """

    @staticmethod
    def interpret_line(feature: Line, negate: bool = False) -> str:
        """
        Describe a line-coverage feature.

        :param feature: The line spectrum feature.
        :param negate: When ``True``, describe the *absence* of execution.
        :returns: Natural-language description.
        """
        return (
            f"Line {feature.file}:{feature.line} is {'NOT ' if negate else ''}executed"
        )

    @staticmethod
    def interpret_function(feature: Function, negate: bool = False) -> str:
        """
        Describe a function-coverage feature.

        :param feature: The function spectrum feature.
        :param negate: When ``True``, describe the *absence* of the call.
        :returns: Natural-language description.
        """
        return f"Function \"{feature.function}\" in {feature.file}:{feature.line} is {'NOT ' if negate else ''}called"

    @staticmethod
    def interpret_function_error(
        feature: FunctionErrorPredicate, negate: bool = False
    ) -> str:
        """
        Describe a function-error predicate.

        :param feature: The predicate feature.
        :param negate: When ``True``, describe the case where *no* error is raised.
        :returns: Natural-language description.
        """
        action = "does NOT raise" if negate else "raises"
        return f'Function "{feature.function}" in {feature.file}:{feature.line} {action} an error'

    @staticmethod
    def interpret_loop(feature: Loop, negate: bool = False) -> str:
        """
        Describe a loop-iteration-count feature.

        :param feature: The loop spectrum feature.
        :param negate: When ``True``, negate the described iteration pattern.
        :returns: Natural-language description (never / once / multiple times).
        """
        prefix = "NOT " if negate else ""
        if feature.evaluate_hit == Loop.evaluate_hit_0:
            return f"Loop at {feature.file}:{feature.line} is {prefix}never executed"
        elif feature.evaluate_hit == Loop.evaluate_hit_1:
            return f"Loop at {feature.file}:{feature.line} is {prefix}executed once"
        elif feature.evaluate_hit == Loop.evaluate_hit_more:
            return f"Loop at {feature.file}:{feature.line} is {prefix}executed multiple times"
        return f"Loop at {feature.file}:{feature.line} is {prefix}executed based on custom condition"

    @staticmethod
    def interpret_ascii_string(feature: IsAsciiPredicate, negate: bool = False) -> str:
        """
        Describe an ASCII-string predicate.

        :param feature: The predicate feature.
        :param negate: When ``True``, negate the description.
        :returns: Natural-language description.
        """
        action = "does NOT contain" if negate else "contains"
        return f'String variable "{feature.var}" in {feature.file}:{feature.line} {action} only ASCII characters'

    @staticmethod
    def interpret_digit_string(
        feature: ContainsDigitPredicate, negate: bool = False
    ) -> str:
        """
        Describe a digit-string predicate.

        :param feature: The predicate feature.
        :param negate: When ``True``, negate the description.
        :returns: Natural-language description.
        """
        action = "does NOT contain" if negate else "contains"
        return f'String variable "{feature.var}" in {feature.file}:{feature.line} {action} at least one digit'

    @staticmethod
    def interpret_special_string(
        feature: ContainsSpecialPredicate, negate: bool = False
    ) -> str:
        """
        Describe a special-character-string predicate.

        :param feature: The predicate feature.
        :param negate: When ``True``, negate the description.
        :returns: Natural-language description.
        """
        return (
            f'String variable "{feature.var}" in {feature.file}:{feature.line} '
            f"{'does NOT contain' if negate else 'contains'} at least one special (non-ascii) character"
        )

    @staticmethod
    def interpret_empty_string(
        feature: EmptyStringPredicate, negate: bool = False
    ) -> str:
        """
        Describe an empty-string predicate.

        :param feature: The predicate feature.
        :param negate: When ``True``, negate the description.
        :returns: Natural-language description.
        """
        return f"String variable \"{feature.var}\" in {feature.file}:{feature.line} is {'NOT ' if negate else ''}empty"

    @staticmethod
    def interpret_empty_bytes(
        feature: EmptyBytesPredicate, negate: bool = False
    ) -> str:
        """
        Describe an empty-bytes predicate.

        :param feature: The predicate feature.
        :param negate: When ``True``, negate the description.
        :returns: Natural-language description.
        """
        return f"Bytes variable \"{feature.var}\" in {feature.file}:{feature.line} is {'NOT ' if negate else ''}empty"

    @staticmethod
    def interpret_branch(feature: Branch, negate: bool = False) -> str:
        """
        Describe a branch-coverage feature.

        :param feature: The branch spectrum feature.
        :param negate: When ``True``, flip the taken branch (then ↔ else).
        :returns: Natural-language description.
        """
        if feature.then:
            return f"{'Else' if negate else 'Then'}-branch in {feature.file}:{feature.line} is taken"
        else:
            return f"{'Then' if negate else 'Else'}-branch at {feature.file}:{feature.line} is taken"

    @staticmethod
    def interpret_condition(feature: Condition, negate: bool = False) -> str:
        """
        Describe a boolean-condition predicate.

        :param feature: The condition predicate feature.
        :param negate: When ``True``, flip the expected truth value.
        :returns: Natural-language description.
        """
        result = not feature.negate if not negate else feature.negate
        return f'Condition "{feature.condition}" at {feature.file}:{feature.line} evaluates to {result}'

    @staticmethod
    def interpret_def_use(feature: DefUse, negate: bool = False) -> str:
        """
        Describe a def-use (data-flow) feature.

        :param feature: The def-use spectrum feature.
        :param negate: When ``True``, negate the description.
        :returns: Natural-language description.
        """
        return (
            f'Variable "{feature.var}" defined at {feature.file}:{feature.line} '
            f"is {'NOT ' if negate else ''}used at {feature.use_file}:{feature.use_line}"
        )

    @staticmethod
    def adjust_op(op: Comp, negate: bool) -> Comp:
        """
        Return the logical complement of *op* when *negate* is ``True``,
        otherwise return *op* unchanged.

        :param op: Original comparison operator.
        :param negate: Whether to flip the operator.
        :returns: The (possibly negated) operator.
        """
        if negate:
            if op == Comp.EQ:
                return Comp.NE
            elif op == Comp.NE:
                return Comp.EQ
            elif op == Comp.LT:
                return Comp.GE
            elif op == Comp.LE:
                return Comp.GT
            elif op == Comp.GT:
                return Comp.LE
            elif op == Comp.GE:
                return Comp.LT
        return op

    @staticmethod
    def interpret_variable(feature: VariablePredicate, negate: bool = False) -> str:
        """
        Describe a variable-value predicate (comparison against 0).

        :param feature: The predicate feature.
        :param negate: Negate the comparison.
        :returns: Natural-language description.
        """
        op = Interpreter.adjust_op(feature.op, negate)
        return (
            f'Variable "{feature.var}" in {feature.file}:{feature.line} is {op.value} 0'
        )

    @staticmethod
    def interpret_none(feature: NonePredicate, negate: bool = False) -> str:
        """
        Describe a None-check predicate.

        :param feature: The predicate feature.
        :param negate: Negate the description.
        :returns: Natural-language description.
        """
        return f"Variable \"{feature.var}\" in {feature.file}:{feature.line} is {'NOT ' if negate else ''}None"

    @staticmethod
    def interpret_return(feature: ReturnPredicate, negate: bool = False) -> str:
        """
        Describe a return-value predicate.

        ``sflkit`` compares numbers with 0, strings with ``""``, bytes with
        ``b""``, and anything else with ``None``; only numbers take an order.

        :param feature: The predicate feature.
        :param negate: Negate the comparison.
        :returns: Natural-language description.
        """
        op = Interpreter.adjust_op(feature.op, negate)
        value = getattr(feature, "value", 0)
        function = f"Function {feature.function} in {feature.file}:{feature.line}"
        if value is None or isinstance(value, (str, bytes)):
            what = {
                None: "None",
                "": "an empty string",
                b"": "empty bytes",
            }.get(value, repr(value))
            if op == Comp.EQ:
                return f"{function} returns {what}"
            if op == Comp.NE:
                return f"{function} does not return {what}"
            return f"{function} returns {op.value} {what}"
        return f"{function} returns {op.value} {value}"

    @staticmethod
    def interpret_scalar_pair(feature: ScalarPair, negate: bool = False) -> str:
        """
        Describe a scalar-pair comparison predicate.

        :param feature: The predicate feature.
        :param negate: Negate the comparison.
        :returns: Natural-language description.
        """
        op = Interpreter.adjust_op(feature.op, negate)
        var2_prefix = "predefined " if feature.var1 == feature.var2 else ""
        return (
            f"In {feature.file}:{feature.line}, {feature.var1} {op.value} "
            f"{var2_prefix}{feature.var2}"
        )

    @staticmethod
    def interpret_length(feature: Length, negate: bool = False) -> str:
        """
        Describe a collection-length feature.

        :param feature: The length spectrum feature.
        :param negate: Negate the description.
        :returns: Natural-language description (0 / 1 / >1).
        """
        prefix = "NOT " if negate else ""
        if feature.evaluate_length == Length.evaluate_length_0:
            return f"Length of variable {feature.var} in {feature.file}:{feature.line} is {prefix}0"
        elif feature.evaluate_length == Length.evaluate_length_1:
            return f"Length of variable {feature.var} in {feature.file}:{feature.line} is {prefix}1"
        elif feature.evaluate_length == Length.evaluate_length_more:
            return f"Length of variable {feature.var} in {feature.file}:{feature.line} is {prefix}greater than 1"
        return f"Length of variable {feature.var} in {feature.file}:{feature.line} {prefix}meets custom condition"

    TYPE_TO_METHOD = {
        AnalysisType.LINE: interpret_line,
        AnalysisType.FUNCTION: interpret_function,
        AnalysisType.FUNCTION_ERROR: interpret_function_error,
        AnalysisType.LOOP: interpret_loop,
        AnalysisType.ASCII_STRING: interpret_ascii_string,
        AnalysisType.DIGIT_STRING: interpret_digit_string,
        AnalysisType.SPECIAL_STRING: interpret_special_string,
        AnalysisType.EMPTY_STRING: interpret_empty_string,
        AnalysisType.EMPTY_BYTES: interpret_empty_bytes,
        AnalysisType.BRANCH: interpret_branch,
        AnalysisType.CONDITION: interpret_condition,
        AnalysisType.DEF_USE: interpret_def_use,
        AnalysisType.VARIABLE: interpret_variable,
        AnalysisType.NONE: interpret_none,
        AnalysisType.RETURN: interpret_return,
        AnalysisType.SCALAR_PAIR: interpret_scalar_pair,
        AnalysisType.LENGTH: interpret_length,
    }
    """Dispatch table mapping :class:`~sflkit.analysis.analysis_type.AnalysisType`
    values to their corresponding ``interpret_*`` static method."""

    def interpret(self, feature: Feature, negate: bool = False) -> str:
        """
        Translate *feature* into a natural-language string.

        Looks up the appropriate ``interpret_*`` method in :attr:`TYPE_TO_METHOD`
        and delegates to it.  Falls back to a generic ``"Feature … of unknown type"``
        message for unrecognized analysis types.

        :param feature: The feature to describe.
        :param negate: If ``True``, produce the negated description.
        :returns: Human-readable description of the feature.
        """
        method = self.TYPE_TO_METHOD.get(feature.analysis.analysis_type())
        if method:
            # noinspection PyTypeChecker
            return method(feature.analysis, negate=negate)
        else:
            return f"Feature {feature} of unknown type"


INTERPRETER = Interpreter()
"""Module-level singleton :class:`Interpreter` used by :meth:`Atom.to_text`."""


class Diagnosis(abc.ABC):
    """
    Abstract base class for all diagnosis nodes.

    A diagnosis is a logical formula over execution features that
    distinguishes failing test runs from passing ones.  The
    :attr:`discriminability` score quantifies how well it separates the two
    populations (MCC-based, range −1 … +1; higher is better).

    :ivar discriminability: MCC-based score measuring how well this diagnosis
        separates failing from passing tests.
    :ivar level: Call-tree depth at which the diagnosis was derived (root = 0).
    :ivar point: Whether the feature was observed at function entry or exit.
    """

    def __init__(
        self, discriminability: float = 0.0, level: int = 0, point: EntryPoint = None
    ):
        """
        Initialize a diagnosis node.

        :param discriminability: Initial discriminability score.
            Defaults to 0.0.
        :param level: Call-tree depth.  Defaults to 0.
        :param point: Entry or exit observation point.  Defaults to
            ``None``.
        """
        self.discriminability: float = discriminability
        self.level = level
        self.point = point

    @abc.abstractmethod
    def get_features(self) -> set[Feature]:
        """
        Return the set of all features involved in this diagnosis.

        :returns: Feature objects referenced by this node.
        """

    def __gt__(self, other):
        """Compare by ``(discriminability, not negated)`` tuple (higher is better)."""
        if not isinstance(other, Diagnosis):
            return NotImplemented
        return (self.discriminability, not getattr(self, "negated", False)) > (
            other.discriminability,
            not getattr(other, "negated", False),
        )

    def __lt__(self, other):
        """Compare by ``(discriminability, not negated)`` tuple (lower is worse)."""
        if not isinstance(other, Diagnosis):
            return NotImplemented
        return (self.discriminability, not getattr(self, "negated", False)) < (
            other.discriminability,
            not getattr(other, "negated", False),
        )

    def __le__(self, other):
        """Less-than-or-equal comparison."""
        if not isinstance(other, Diagnosis):
            return NotImplemented
        return (self.discriminability, not getattr(self, "negated", False)) <= (
            other.discriminability,
            not getattr(other, "negated", False),
        )

    def __ge__(self, other):
        """Greater-than-or-equal comparison."""
        if not isinstance(other, Diagnosis):
            return NotImplemented
        return (self.discriminability, not getattr(self, "negated", False)) >= (
            other.discriminability,
            not getattr(other, "negated", False),
        )

    @abc.abstractmethod
    def to_text(self) -> str:
        """
        Return a human-readable description of this diagnosis.

        :returns: Natural-language representation.
        """


class CauseChain(Diagnosis):
    """
    An ordered chain of :class:`Diagnosis` nodes linked by causality.

    The chain is read as *cause₁ LEADS TO cause₂ LEADS TO … LEADS TO causeₙ*,
    representing the sequence of observations that collectively explain the test
    failure.

    :ivar causes: Ordered list of cause nodes (root cause first).
    """

    def __init__(
        self,
        causes: list[Diagnosis],
        discriminability: float = None,
        level: int = 0,
        point: EntryPoint = None,
    ):
        """
        Initialize the cause chain.

        :param causes: Ordered cause nodes.
        :param discriminability: Overall chain discriminability.  When omitted
            it is the discriminability of the sharpest step, which is how
            cleanly the chain separates passing from failing runs at its best
            point.
        :param level: Call-tree depth.  Defaults to 0.
        :param point: Observation point.  Defaults to
            ``None``.
        """
        if discriminability is None:
            discriminability = max(
                (cause.discriminability for cause in causes), default=0.0
            )
        super().__init__(discriminability, level, point)
        self.causes = causes

    @property
    def mean_discriminability(self) -> float:
        """Average discriminability across the steps of the chain."""
        if not self.causes:
            return 0.0
        return sum(cause.discriminability for cause in self.causes) / len(self.causes)

    def __repr__(self):
        return f"DiagnosisCauseChain(causes={self.causes})"

    def __str__(self):
        return " -> ".join(str(cause) for cause in self.causes)

    def get_features(self) -> set[str]:
        """
        Return the union of all features across all causes.

        :returns: Feature names from every node in the chain.
        """
        feature_names = set()
        for cause in self.causes:
            feature_names.update(cause.get_features())
        return feature_names

    def to_text(self) -> str:
        """
        Render the cause chain as a numbered list joined by ``LEADS TO``.

        :returns: Multi-line text representation.
        """
        return "\n LEADS TO \n".join(
            f"{i}. {cause.to_text()} (discriminability: {cause.discriminability:.2f})"
            for i, cause in enumerate(self.causes, start=1)
        )

    def qualify_functions(self, names: "FunctionNames") -> None:
        """
        Name the functions of the chain's divergences with their qualified
        names, such as ``Class.method``; see :class:`FunctionNames`.

        :param names: Resolver for the traced project.
        """
        for cause in self.causes:
            if isinstance(cause, Divergence):
                cause.qualify(names)


class Atom(Diagnosis):
    """
    A single feature observation — the atomic unit of a diagnosis.

    Wraps one ``sflkit`` :class:`~sflkit.features.value.Feature` with an
    optional negation flag and is rendered to text by :data:`INTERPRETER`.

    :ivar feature: The ``sflkit`` feature this atom describes.
    :ivar negated: When ``True``, the atom asserts the *absence* of the feature.
    """

    def __init__(
        self,
        feature: Feature,
        negated: bool = False,
        discriminability: float = 0.0,
        level: int = 0,
        point: EntryPoint = None,
    ):
        """
        Initialize the atom.

        :param feature: The feature being described.
        :param negated: Whether to negate the feature.  Defaults to
            ``False``.
        :param discriminability: MCC score.  Defaults to 0.0.
        :param level: Call-tree depth.  Defaults to 0.
        :param point: Observation point.  Defaults to
            ``None``.
        """
        super().__init__(discriminability, level, point)
        self.feature = feature
        self.negated = negated

    def __repr__(self):
        return f"DiagnosisAtom(feature={self.feature}, negated={self.negated})"

    def __str__(self):
        return f"{'NOT ' if self.negated else ''}{str(self.feature)}"

    def get_features(self) -> set[Feature]:
        """Return the singleton set containing :attr:`feature`."""
        return {self.feature}

    def to_text(self) -> str:
        """
        Render via :data:`INTERPRETER`.

        :returns: Natural-language description of the feature observation.
        """
        return INTERPRETER.interpret(self.feature, negate=self.negated)


class FunctionNames:
    """
    Qualified names, such as ``MatrixBase._handle_creation_inputs``, for the
    functions in call-tree node names.

    Node names give a function by file and bare name, so a method reads like a
    module-level function: ``sympy/matrices/matrices.py:_handle_creation_inputs``
    does not say that it belongs to ``MatrixBase``. Repair agents look methods
    up by class, and without the class they guess it. The class (and any
    enclosing function) is read from the source; where a file defines the name
    more than once, the line the trace mapping records for the function id
    decides.

    :ivar source_root: Root of the traced project's source.
    :ivar function_lines: Function id -> line of its definition.
    """

    def __init__(self, source_root, function_lines: Optional[dict[int, int]] = None):
        """
        :param source_root: Root of the traced project's source.
        :param function_lines: Function id -> line of its definition, as the
            trace mapping records it; without it, names defined more than once
            in a file stay unqualified.
        """
        self.source_root = Path(source_root)
        self.function_lines = function_lines or {}
        self._definitions: dict[str, list[tuple[str, set[int]]]] = {}

    @classmethod
    def from_mapping(cls, source_root, mapping_path) -> "FunctionNames":
        """
        Build the resolver with the definition lines from an ``sflkit`` mapping.

        :param source_root: Root of the traced project's source.
        :param mapping_path: The ``mapping.json`` the traces were collected with.
        """
        events = json.loads(Path(mapping_path).read_text())
        lines = {
            int(event["function_id"]): int(event["line"])
            for event in events
            if event.get("event_type") == EventType.FUNCTION_ENTER.value
        }
        return cls(source_root, lines)

    def qualify(self, node_name: str) -> Optional[str]:
        """
        Name the function of a call-tree node with its qualified name.

        :param node_name: ``<file>:<function_name>:<function_id>``.
        :returns: ``<file>:<qualified name>``, or ``None`` when the source does
            not decide it.
        """
        head, _, function_id = node_name.rpartition(":")
        file, _, function = head.rpartition(":")
        if not (file and function and function_id.isdigit()):
            return None
        candidates = [
            (qualname, lines)
            for qualname, lines in self._definitions_in(file)
            if qualname.rpartition(".")[2] == function
        ]
        if len({qualname for qualname, _ in candidates}) > 1:
            line = self.function_lines.get(int(function_id))
            candidates = [c for c in candidates if line in c[1]]
        qualnames = {qualname for qualname, _ in candidates}
        return f"{file}:{qualnames.pop()}" if len(qualnames) == 1 else None

    def _definitions_in(self, file: str) -> list[tuple[str, set[int]]]:
        """
        Every function definition in *file*: its qualified name and the lines
        that can stand for it (the ``def`` and its decorators).
        """
        if file not in self._definitions:
            definitions = []
            try:
                tree = ast.parse((self.source_root / file).read_text(errors="replace"))
            except (OSError, SyntaxError, ValueError):
                tree = None

            def visit(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualname = prefix + child.name
                        lines = {child.lineno} | {d.lineno for d in child.decorator_list}
                        definitions.append((qualname, lines))
                        visit(child, qualname + ".<locals>.")
                    elif isinstance(child, ast.ClassDef):
                        visit(child, prefix + child.name + ".")
                    else:
                        visit(child, prefix)

            if tree is not None:
                visit(tree, "")
            self._definitions[file] = definitions
        return self._definitions[file]


class Divergence(Diagnosis):
    """
    The point where a run stops behaving like the passing runs.

    A :class:`Divergence` describes a *call* that the failing runs make and the
    passing runs do not: the edge from a node both populations reach into a
    child only the failing runs enter.

    This is deliberately an edge rather than a node. Everything below such a
    child is already inside the faulty region -- effect, not cause -- so scoring
    those nodes would only report where the failure surfaced. The transition
    itself is where correct behaviour becomes incorrect, and the observation
    that discriminates there is the call, not any scalar feature: at the sole
    divergence of django-15996 not one of the 24,912 features at the parent
    separated the populations, while the call into
    ``django/db/migrations/writer.py:serialize`` separated them perfectly.

    :ivar caller: Node both populations reach.
    :ivar callee: Child only the failing runs enter.
    """

    def __init__(
        self,
        caller: str,
        callee: str,
        feature: Feature,
        discriminability: float = 0.0,
        level: int = 0,
        point: EntryPoint = None,
    ):
        """
        Initialize the divergence.

        :param caller: Name of the node both populations reach.
        :param callee: Name of the failing-only child.
        :param feature: Synthetic feature standing for "entered *callee*",
            so that this diagnosis takes part in de-duplication and
            subsumption like any other.
        :param discriminability: MCC score for the split.
        :param level: Call-tree depth of *caller*.
        :param point: Observation point.
        """
        super().__init__(discriminability, level, point)
        self.caller = caller
        self.callee = callee
        self.feature = feature
        #: Node name -> ``<file>:<qualified name>``, set by :meth:`qualify`.
        self.qualified: dict[str, str] = {}

    def __repr__(self):
        return f"Divergence(caller={self.caller}, callee={self.callee})"

    def __str__(self):
        return f"DIVERGENCE:{self.caller}->{self.callee}"

    def get_features(self) -> set[Feature]:
        """Return the singleton set containing the synthetic call feature."""
        return {self.feature}

    def to_text(self) -> str:
        """
        Describe the divergent call in natural language.

        :returns: Natural-language description of the transition.
        """
        where = (
            f" from {self._function_text(self.caller)}"
            if self.caller != "FORECAST_ROOT"
            else ""
        )
        return (
            f"The failing run called {self._function_text(self.callee)}{where}, "
            f"which no passing run called"
        )

    def qualify(self, names: FunctionNames) -> None:
        """
        Name the caller and callee with their qualified names where the
        source decides them.

        :param names: Resolver for the traced project.
        """
        for name in (self.caller, self.callee):
            qualified = names.qualify(name)
            if qualified:
                self.qualified[name] = qualified

    def _function_text(self, name: str) -> str:
        """
        Render a call-tree node name for a reader, without its function id.

        Node names are ``<file>:<function_name>:<function_id>``. The id keeps
        overloaded names apart inside the tree, but in text it reads like a
        line number, so it is left out. Where :meth:`qualify` found the
        qualified name, it replaces the bare one.

        :param name: Call-tree node name.
        :returns: ``<file>:<qualified name>`` or ``<file>:<function_name>``.
        """
        if name in self.qualified:
            return self.qualified[name]
        head, _, function_id = name.rpartition(":")
        return head if head and function_id.isdigit() else name


def _join(parts: list[Diagnosis], operator: str, render) -> str:
    """
    Join *parts* with *operator*, parenthesizing every part that is a compound
    diagnosis of the other operator.

    Without the parentheses, ``(A AND B) OR C`` would read as
    ``A AND B OR C``, a different formula than the one that was scored.

    :param parts: The diagnoses to join.
    :param operator: ``"AND"`` or ``"OR"``.
    :param render: Renders a single part, e.g. ``str`` or its ``to_text``.
    :returns: The joined text.
    """
    rendered = []
    for part in parts:
        text = render(part)
        if (
            isinstance(part, (Conjunction, Disjunction))
            and part.OPERATOR != operator
            and len(part.parts) > 1
        ):
            text = f"({text})"
        rendered.append(text)
    return f" {operator} ".join(rendered)


class Conjunction(Diagnosis):
    """
    A conjunction (AND) of :class:`Atom` nodes.

    All atoms must hold simultaneously for the conjunction to be satisfied.

    :ivar atoms: The conjuncts, flattened to atoms.
    :ivar parts: The diagnoses the conjunction joins, which keep their own
        operators; the atoms themselves unless compound diagnoses are joined.
    """

    OPERATOR = "AND"

    def __init__(
        self,
        atoms: list[Atom],
        discriminability: float = 0.0,
        level: int = 0,
        point: EntryPoint = None,
        parts: Optional[list[Diagnosis]] = None,
    ):
        """
        Initialize the conjunction.

        :param atoms: Conjunct atoms.
        :param discriminability: MCC score.  Defaults to 0.0.
        :param level: Call-tree depth.  Defaults to 0.
        :param point: Observation point.  Defaults to
            ``None``.
        :param parts: The joined diagnoses, when they are compound.
            Defaults to *atoms*.
        """
        super().__init__(discriminability, level, point)
        self.atoms = atoms
        self.parts = parts or list(atoms)

    def __repr__(self):
        return f"DiagnosisConjunction(atoms={self.atoms})"

    def __str__(self):
        return _join(self.parts, self.OPERATOR, str)

    def get_features(self) -> set[Feature]:
        """Return the set of features from all conjunct atoms."""
        return {atom.feature for atom in self.atoms}

    def to_text(self) -> str:
        """
        Render each part and join with ``AND``.

        :returns: Natural-language conjunction.
        """
        return _join(self.parts, self.OPERATOR, lambda part: part.to_text())


class Disjunction(Diagnosis):
    """
    A disjunction (OR) of :class:`Atom` nodes.

    At least one atom must hold for the disjunction to be satisfied.

    :ivar atoms: The disjuncts, flattened to atoms.
    :ivar parts: The diagnoses the disjunction joins, which keep their own
        operators; the atoms themselves unless compound diagnoses are joined.
    """

    OPERATOR = "OR"

    def __init__(
        self,
        atoms: list[Atom],
        discriminability: float = 0.0,
        level: int = 0,
        point: EntryPoint = None,
        parts: Optional[list[Diagnosis]] = None,
    ):
        """
        Initialize the disjunction.

        :param atoms: Disjunct atoms.
        :param discriminability: MCC score.  Defaults to 0.0.
        :param level: Call-tree depth.  Defaults to 0.
        :param point: Observation point.  Defaults to
            ``None``.
        :param parts: The joined diagnoses, when they are compound.
            Defaults to *atoms*.
        """
        super().__init__(discriminability, level, point)
        self.atoms = atoms
        self.parts = parts or list(atoms)

    def __repr__(self):
        return f"DiagnosisDisjunction(atoms={self.atoms})"

    def __str__(self):
        return _join(self.parts, self.OPERATOR, str)

    def get_features(self) -> set[Feature]:
        """Return the set of features from all disjunct atoms."""
        return {atom.feature for atom in self.atoms}

    def to_text(self) -> str:
        """
        Render each part and join with ``OR``.

        :returns: Natural-language disjunction.
        """
        return _join(self.parts, self.OPERATOR, lambda part: part.to_text())


class Forecast:
    """
    Main diagnosis-derivation algorithm for FORECAST.

    Takes a fully-populated :class:`~forecast.tree.TracedRelevanceTree` and
    produces a :class:`CauseChain` by:

    1. Marking discriminative nodes (:meth:`_mark_discriminative_nodes`).
    2. Extracting per-node diagnoses with MCC scoring
       (:meth:`_extract_cause_chain`, :meth:`_get_top_discriminative_diagnosis`).
    3. Filtering below :attr:`min_discriminability_threshold`.
    4. Deduplicating and removing subsumed diagnoses.
    5. Selecting the top-*k* most discriminative causes
       (:attr:`max_chain_length`).

    :ivar root: Root of the call tree to analyze.
    :ivar min_discriminability_threshold: Minimum MCC score a diagnosis must
        reach to be included in the chain.  Defaults to 0.6.
    :ivar max_chain_length: Maximum number of causes in the final chain.
        Defaults to 10.
    """

    def __init__(
        self,
        root: TracedRelevanceTree,
        min_discriminability_threshold: float = 0.6,
        max_chain_length: int = 10,
        combine: bool = True,
        divergences: bool = True,
    ):
        """
        Initialize the Forecast engine.

        :param root: Root node of the call tree.
        :param min_discriminability_threshold: MCC threshold below which
            diagnoses are discarded.  Defaults to 0.6.
        :param max_chain_length: Maximum chain length.  Defaults to 10.
        :param combine: Whether atoms are combined into conjunctions and
            disjunctions.  Defaults to ``True``; ``False`` keeps single
            features only, e.g. for ablations.
        :param divergences: Whether calls only failing runs make are reported
            as divergences.  Defaults to ``True``.
        """
        self.root = root
        self.min_discriminability_threshold = min_discriminability_threshold
        self.max_chain_length = max_chain_length
        self.combine = combine
        self.divergences = divergences

    def derive_diagnosis(self) -> CauseChain:
        """
        Derive and return the best :class:`CauseChain` from :attr:`root`.

        Runs the full pipeline:

        1. :meth:`_mark_discriminative_nodes` — recursively marks nodes that
           separate passing from failing runs.
        2. :meth:`_extract_cause_chain` — collects per-node
           :class:`Diagnosis` objects scored by MCC.
        3. Filters out causes below :attr:`min_discriminability_threshold`.
        4. Deduplicates and removes subsumed diagnoses.
        5. Selects the top-:attr:`max_chain_length` causes in temporal order.

        :returns: The final cause-effect chain (it may be empty if no
            discriminative features were found).
        """
        self._mark_discriminative_nodes(self.root)

        causes = self._extract_cause_chain(self.root)

        LOGGER.debug(
            f"Extracted {len(causes)} initial diagnoses from discriminative nodes"
        )

        if all(
            cause.discriminability < self.min_discriminability_threshold
            for cause in causes
        ):
            LOGGER.warning(
                "No discriminative causes found above threshold. Forecast may be inconclusive."
            )
            causes = causes
        else:
            causes = [
                cause
                for cause in causes
                if cause.discriminability >= self.min_discriminability_threshold
            ]
            LOGGER.debug(
                f"After discriminability threshold filter (>={self.min_discriminability_threshold}): "
                f"{len(causes)} diagnoses remain"
            )

        # Filter out duplicates and prefer simpler diagnoses with same discriminability
        # Do this BEFORE selecting top-k to ensure we select from unique diagnoses
        filtered_causes = []
        seen_diagnoses = set()

        for cause in causes:
            # Create a unique key for this diagnosis
            diagnosis_key = self._get_diagnosis_key(cause)

            # Skip if we've already seen this exact diagnosis
            if diagnosis_key in seen_diagnoses:
                LOGGER.debug(f"Skipping duplicate diagnosis: {cause}")
                continue

            # Check for subsumption relationships with existing diagnoses
            should_add = True
            indices_to_remove = []

            for idx, existing_cause in enumerate(filtered_causes):
                existing_features = existing_cause.get_features()
                new_features = cause.get_features()

                # Case 1: Same discriminability - prefer simpler diagnosis
                if (
                    abs(cause.discriminability - existing_cause.discriminability)
                    < EPSILON
                ):
                    # If new diagnosis is a superset (more complex), skip it
                    if (
                        new_features >= existing_features
                        and new_features != existing_features
                    ):
                        LOGGER.debug(
                            f"Skipping subsumed diagnosis (more complex, equal disc): {cause} "
                            f"(disc={cause.discriminability:.3f}) is subsumed by simpler "
                            f"{existing_cause} (disc={existing_cause.discriminability:.3f})"
                        )
                        should_add = False
                        break
                    # If existing diagnosis is a superset (more complex), remove it
                    elif (
                        existing_features >= new_features
                        and existing_features != new_features
                    ):
                        LOGGER.debug(
                            f"Removing subsumed diagnosis (more complex, equal disc): {existing_cause} "
                            f"(disc={existing_cause.discriminability:.3f}) subsumed by simpler "
                            f"{cause} (disc={cause.discriminability:.3f})"
                        )
                        indices_to_remove.append(idx)

                # Case 2: New diagnosis has BETTER discriminability and subsumes existing
                # Keep the more discriminative one even if more complex
                elif cause.discriminability > existing_cause.discriminability + EPSILON:
                    # If new diagnosis contains all features of existing (is superset)
                    if (
                        new_features >= existing_features
                        and existing_features  # Not empty
                    ):
                        LOGGER.debug(
                            f"Removing subsumed diagnosis (lower disc): {existing_cause} "
                            f"(disc={existing_cause.discriminability:.3f}) subsumed by better "
                            f"{cause} (disc={cause.discriminability:.3f})"
                        )
                        indices_to_remove.append(idx)
                    # Case 4: New is simpler AND better - remove complex lower-disc
                    elif new_features < existing_features and new_features:  # Not empty
                        LOGGER.debug(
                            f"Removing subsumed diagnosis (more complex, lower disc): {existing_cause} "
                            f"(disc={existing_cause.discriminability:.3f}) subsumed by simpler and better "
                            f"{cause} (disc={cause.discriminability:.3f})"
                        )
                        indices_to_remove.append(idx)

                # Case 3: Existing has BETTER discriminability and subsumes new
                # Skip the new one in favor of the better existing one
                elif existing_cause.discriminability > cause.discriminability + EPSILON:
                    # If existing contains all features of new (is superset)
                    if existing_features >= new_features and new_features:  # Not empty
                        LOGGER.debug(
                            f"Skipping subsumed diagnosis (lower disc): {cause} "
                            f"(disc={cause.discriminability:.3f}) subsumed by better "
                            f"{existing_cause} (disc={existing_cause.discriminability:.3f})"
                        )
                        should_add = False
                        break
                    # Case 5: Existing is simpler AND better - skip complex lower-disc
                    elif (
                        existing_features < new_features
                        and existing_features  # Not empty
                    ):
                        LOGGER.debug(
                            f"Skipping subsumed diagnosis (more complex, lower disc): {cause} "
                            f"(disc={cause.discriminability:.3f}) subsumed by simpler and better "
                            f"{existing_cause} (disc={existing_cause.discriminability:.3f})"
                        )
                        should_add = False
                        break

            # Remove more complex diagnoses that are subsumed
            for idx in reversed(indices_to_remove):
                removed_key = self._get_diagnosis_key(filtered_causes[idx])
                seen_diagnoses.discard(removed_key)
                filtered_causes.pop(idx)

            if should_add:
                filtered_causes.append(cause)
                seen_diagnoses.add(diagnosis_key)

        causes = filtered_causes

        LOGGER.debug(
            f"After deduplication and subsumption removal: {len(causes)} unique diagnoses"
        )

        # NOW limit to top-k most discriminative while preserving temporal order
        # This ensures we select the best k from the unique set.
        #
        # Ties break towards :class:`Divergence`. A run can produce many
        # observations that separate the populations perfectly -- scikit-learn
        # 25570 yields 219 candidates, a great many at 1.0 -- and with a plain
        # sort the transition loses to whichever tied cause happens to come
        # first, dropping out of the chain entirely. Where separating power is
        # equal, the call the failing runs alone make is the cause; the features
        # observed after it describe what followed.
        if len(causes) > self.max_chain_length:
            indices = [
                cause[0]
                for cause in sorted(
                    enumerate(causes),
                    key=lambda cause: (
                        cause[1].discriminability,
                        isinstance(cause[1], Divergence),
                    ),
                    reverse=True,
                )[: self.max_chain_length]
            ]
            causes = [causes[i] for i in sorted(indices)]

            LOGGER.debug(
                f"After top-{self.max_chain_length} selection: {len(causes)} diagnoses in final chain"
            )

        LOGGER.debug("Diagnosis inference complete")

        return CauseChain(causes)

    @staticmethod
    def _get_diagnosis_key(diagnosis: Diagnosis) -> tuple:
        """
        Return a hashable key that uniquely identifies *diagnosis* by structure.

        Used to detect duplicate diagnoses during chain construction.

        :param diagnosis: The diagnosis to key.
        :returns: A hashable key based on the diagnosis type and its features.
        """
        if isinstance(diagnosis, Divergence):
            return "Divergence", diagnosis.caller, diagnosis.callee
        elif isinstance(diagnosis, Atom):
            return "Atom", diagnosis.feature, diagnosis.negated
        elif isinstance(diagnosis, Conjunction):
            atoms = tuple(
                sorted((atom.feature, atom.negated) for atom in diagnosis.atoms)
            )
            return "Conjunction", atoms
        elif isinstance(diagnosis, Disjunction):
            atoms = tuple(
                sorted((atom.feature, atom.negated) for atom in diagnosis.atoms)
            )
            return "Disjunction", atoms
        else:
            # Fallback for unknown diagnosis types
            return type(diagnosis).__name__, str(diagnosis)

    def _mark_discriminative_nodes(self, node: TracedRelevanceTree) -> None:
        """
        Mark every node in the subtree rooted at *node* as discriminative if it
        separates passing from failing runs, or if any of its children is
        discriminative.

        A node is considered discriminative when at least one feature appears
        in both a passing and a failing feature vector at entry *or* exit.

        Iterative post-order traversal (children marked before parents): real
        call trees can be tens of thousands of frames deep, which would exceed
        Python's recursion limit.

        :param node: The root of the subtree to evaluate.
        """
        # Collect nodes root-first; a pre-order push/pop over a tree yields an
        # order whose reverse visits every child before its parent.
        order: List[TracedRelevanceTree] = []
        stack = [node]
        while stack:
            current = stack.pop()
            order.append(current)
            stack.extend(current.children.values())
        for current in reversed(order):
            self._mark_node_discriminative(current)

    @staticmethod
    def _mark_node_discriminative(node: TracedRelevanceTree) -> None:
        """
        Set ``node.is_discriminative`` assuming its children are already marked.

        :param node: The node to evaluate.
        """
        node.is_discriminative = any(
            child.is_discriminative for child in node.children.values()
        )

        if not node.is_discriminative:
            passing_features = set()
            failing_features = set()

            for feature_vector in node.enter_features:
                if feature_vector.result == TestResult.PASSING:
                    passing_features.update(feature_vector.features.keys())
                elif feature_vector.result == TestResult.FAILING:
                    failing_features.update(feature_vector.features.keys())

            if passing_features & failing_features:
                node.is_discriminative = True
            else:
                passing_features.clear()
                failing_features.clear()
                for feature_vector in node.exit_features:
                    if feature_vector.result == TestResult.PASSING:
                        passing_features.update(feature_vector.features.keys())
                    elif feature_vector.result == TestResult.FAILING:
                        failing_features.update(feature_vector.features.keys())

                if passing_features & failing_features:
                    node.is_discriminative = True

    def _extract_cause_chain(
        self,
        node: TracedRelevanceTree,
        level: int = 0,
    ) -> List[Diagnosis]:
        """
        Recursively extract discriminative :class:`Diagnosis` nodes from the
        call tree in DFS order (entry before children before exit).

        Processes only discriminative nodes.  For each such node, the best
        ENTER diagnosis is added first, then child diagnoses are collected
        recursively, then the best EXIT diagnosis is appended last — preserving
        the temporal execution order.

        :param node: Current node being processed.
        :param level: Current call-tree depth (incremented for children).
        :returns: Ordered list of cause candidates.
        """
        if not node.is_discriminative:
            return []

        causes: List[Diagnosis] = []
        # Iterative Euler tour preserving the recursive order
        # (enter node, children in order, exit node). Deep call trees would
        # overflow Python's recursion limit, so an explicit stack is used.
        # Each entry is (is_exit, node, level); children are pushed reversed so
        # they pop in their original order, and the node's own exit marker is
        # pushed before them so it pops last.
        stack: List[tuple] = [(False, node, level)]
        while stack:
            is_exit, current, lvl = stack.pop()
            if is_exit:
                exit_diagnosis = self._get_top_discriminative_diagnosis(
                    current.exit_features,
                    EntryPoint.EXIT,
                    level=lvl,
                )
                if exit_diagnosis:
                    LOGGER.debug(
                        f"Node '{current.name}' EXIT: {exit_diagnosis} (disc={exit_diagnosis.discriminability:.3f})"
                    )
                    causes.append(exit_diagnosis)
                continue

            enter_diagnosis = self._get_top_discriminative_diagnosis(
                current.enter_features,
                EntryPoint.ENTER,
                level=lvl,
            )
            if enter_diagnosis:
                LOGGER.debug(
                    f"Node '{current.name}' ENTER: {enter_diagnosis} (disc={enter_diagnosis.discriminability:.3f})"
                )
                causes.append(enter_diagnosis)

            divergence = (
                self._get_divergence(current, level=lvl) if self.divergences else None
            )
            if divergence:
                causes.extend(divergence)

            stack.append((True, current, lvl))
            discriminative_children = [
                child
                for child in current.children.values()
                if child.is_discriminative
            ]
            for child in reversed(discriminative_children):
                stack.append((False, child, lvl + 1))

        return causes

    @staticmethod
    def _run_populations(vectors: List[FeatureVector]) -> tuple[int, int]:
        """
        Count passing and failing vectors in *vectors*.

        Note that ``FeatureVector.run_id`` is a fork counter, not a run
        identifier -- every feature fork takes a fresh id -- so it cannot be
        used to count runs. At the ROOT these counts are nevertheless exactly
        the run counts, because ``post_process`` records one exit vector per
        run; deeper nodes keep up to :data:`~forecast.tree._PER_RUN_CAP`
        observations per run and must use :attr:`id_counter` instead, whose
        keys are the real run identifiers.

        :param vectors: Feature vectors recorded at one observation point.
        :returns: ``(passing, failing)`` vector counts.
        """
        passing = failing = 0
        for vector in vectors or []:
            if vector.result == TestResult.PASSING:
                passing += 1
            elif vector.result == TestResult.FAILING:
                failing += 1
        return passing, failing

    def _get_divergence(
        self, node: TracedRelevanceTree, level: int = 0
    ) -> List[Divergence]:
        """
        Report calls *node* makes in failing runs but never in passing ones.

        Such a child is where the run leaves behaviour the passing runs share,
        so the edge into it is a cause candidate even though the child itself
        can never be scored: with no passing vectors there is nothing to compare
        against, and :meth:`_get_top_discriminative_diagnosis` discards it. The
        edge is scored instead, using the same MCC as every other diagnosis so
        the numbers stay comparable — the run populations come from the child's
        entry, the totals from the root.

        :param node: The node whose children to inspect.
        :param level: Call-tree depth of *node*.
        :returns: One :class:`Divergence` per failing-only child, most
            discriminative first.
        """
        total_passing, total_failing = self._run_populations(self.root.exit_features)
        if not total_failing:
            return []
        divergences = []
        for name, child in node.children.items():
            passing, failing = self._run_populations(child.enter_features)
            if passing or not failing:
                continue
            # The child holds no passing observations, so every run recorded
            # against it is a failing one and its run count is simply the number
            # of distinct runs in id_counter. Deriving this from the vectors
            # instead would over-count: the per-run cap keeps several vectors
            # per run, which made fail_without negative and produced scores
            # above 1 (mcc(2, 0, -1, 15) = 1.4639 on django-15996, where there
            # is exactly one failing run).
            fail_with = min(
                len(getattr(child, "id_counter", None) or {}) or failing,
                total_failing,
            )
            score = self.mcc(
                fail_with,
                0,
                total_failing - fail_with,
                total_passing,
            )
            divergences.append(
                Divergence(
                    caller=str(node.name),
                    callee=str(name),
                    feature=BinaryFeature(f"CALL:{name}", name),
                    discriminability=score,
                    level=level,
                    point=EntryPoint.ENTER,
                )
            )
        divergences.sort(key=lambda d: -d.discriminability)
        return divergences

    def _get_top_discriminative_diagnosis(
        self,
        feature_vectors: List[FeatureVector],
        phase: EntryPoint,
        level: int = 0,
    ) -> Optional[Diagnosis]:
        """
        Compute the best :class:`Diagnosis` for a list of feature vectors.

        Scores each feature with MCC (both positive and negated variants),
        attempts pairwise AND / OR combinations for features below perfect
        discriminability, and returns the single best candidate after tie-breaking
        via :meth:`_select_most_useful_diagnosis`.

        :param feature_vectors: Entry or exit vectors for
                one tree node.
        :param phase: Whether these are ENTER or EXIT vectors.
        :param level: Call-tree depth (stored on returned diagnosis).
        :returns: The best :class:`Atom`, :class:`Conjunction`,
            or :class:`Disjunction`, or ``None`` if no discriminative feature
            was found.
        """
        if not feature_vectors:
            return None

        passing_features = {}
        failing_features = {}
        total_passing = 0
        total_failing = 0

        for id_, fv in enumerate(feature_vectors):
            if fv.result == TestResult.PASSING:
                total_passing += 1
                for feature, value in fv.features.items():
                    if feature not in passing_features:
                        passing_features[feature] = {
                            FeatureValue.TRUE: 0,
                            FeatureValue.FALSE: 0,
                            FeatureValue.UNDEFINED: 0,
                            "vectors": {
                                FeatureValue.TRUE: set(),
                                FeatureValue.FALSE: set(),
                                FeatureValue.UNDEFINED: set(),
                            },
                        }
                    passing_features[feature][value] += 1
                    passing_features[feature]["vectors"][value].add(id_)
            elif fv.result == TestResult.FAILING:
                total_failing += 1
                for feature, value in fv.features.items():
                    if feature not in failing_features:
                        failing_features[feature] = {
                            FeatureValue.TRUE: 0,
                            FeatureValue.FALSE: 0,
                            FeatureValue.UNDEFINED: 0,
                            "vectors": {
                                FeatureValue.TRUE: set(),
                                FeatureValue.FALSE: set(),
                                FeatureValue.UNDEFINED: set(),
                            },
                        }
                    failing_features[feature][value] += 1
                    failing_features[feature]["vectors"][value].add(id_)

        if total_failing == 0 or total_passing == 0:
            return None

        # Calculate discriminability score for each feature
        feature_scores = []
        # Sorted, so that ties between equally scored candidates break the same
        # way in every run; set order depends on string hashing.
        for feature in sorted(
            set(passing_features.keys()) | set(failing_features.keys()), key=str
        ):
            fail_with = failing_features.get(feature, {}).get(FeatureValue.TRUE, 0)
            not_fail_with = failing_features.get(feature, {}).get(FeatureValue.FALSE, 0)
            pass_with = passing_features.get(feature, {}).get(FeatureValue.TRUE, 0)
            not_pass_with = passing_features.get(feature, {}).get(FeatureValue.FALSE, 0)
            # Complements are taken against the value being tested: for the
            # positive hypothesis the remaining runs are those where the
            # feature is not TRUE, for the negated one those where it is not
            # FALSE. Features are tertiary, so these two differ whenever a run
            # never observed the feature at all.
            fail_without = total_failing - fail_with
            pass_without = total_passing - pass_with
            not_fail_without = total_failing - not_fail_with
            not_pass_without = total_passing - not_pass_with

            # Only consider if actually discriminative (not appearing equally in both)
            if fail_with / max(1, total_failing) == pass_with / max(
                1, total_passing
            ) and not_fail_with / max(1, total_failing) == not_pass_with / max(
                1, total_passing
            ):
                continue

            # Calculate discriminability for the positive case (feature=value)
            if fail_with > 0 or pass_with > 0:
                pos_discriminability = self.mcc(
                    fail_with, pass_with, fail_without, pass_without
                )
            else:
                pos_discriminability = 0.0

            # Calculate discriminability for the negated case (NOT feature=value)
            # This means the feature has a DIFFERENT value (or is absent)
            if not_fail_with > 0 or not_pass_with > 0:
                neg_discriminability = self.mcc(
                    not_fail_with,
                    not_pass_with,
                    not_fail_without,
                    not_pass_without,
                )
            else:
                neg_discriminability = 0.0

            # Choose the better discriminator (positive or negated)
            if pos_discriminability >= neg_discriminability:
                feature_scores.append(
                    Atom(
                        feature,
                        discriminability=pos_discriminability,
                        level=level,
                        point=phase,
                    )
                )
            else:
                feature_scores.append(
                    Atom(
                        feature,
                        negated=True,
                        discriminability=neg_discriminability,
                        level=level,
                        point=phase,
                    )
                )

        if not feature_scores:
            return None

        # Find all atoms with the best discriminability score
        best_atom_score = max(atom.discriminability for atom in feature_scores)
        best_atoms = [
            atom for atom in feature_scores if atom.discriminability == best_atom_score
        ]

        LOGGER.debug(
            f"Found {len(best_atoms)} best atom(s) with discriminability={best_atom_score:.3f}"
        )

        # Collect all candidate diagnoses (atoms and combinations)
        all_candidates = []

        # For each best atom, compute possible combinations
        for best_atom in best_atoms:
            # Add the atom itself as a candidate
            all_candidates.append(best_atom)

            # If not perfect discriminability, try combinations
            if self.combine and best_atom.discriminability < 1 - EPSILON:
                # Find possible combinations with other features
                for candidate in feature_scores:
                    if (
                        candidate == best_atom
                        or candidate.feature not in failing_features
                    ):
                        continue

                    # Get feature vectors for the best atom
                    best_feature = best_atom.feature
                    if best_atom.negated:
                        value = FeatureValue.FALSE
                    else:
                        value = FeatureValue.TRUE

                    failing = (
                        failing_features.get(best_feature, {})
                        .get("vectors", {})
                        .get(value, set())
                    )
                    passing = (
                        passing_features.get(best_feature, {})
                        .get("vectors", {})
                        .get(value, set())
                    )

                    # Get feature vectors for the candidate
                    candidate_feature = candidate.feature
                    if candidate.negated:
                        candidate_value = FeatureValue.FALSE
                    else:
                        candidate_value = FeatureValue.TRUE

                    candidate_failing = (
                        failing_features.get(candidate_feature, {})
                        .get("vectors", {})
                        .get(candidate_value, set())
                    )
                    candidate_passing = (
                        passing_features.get(candidate_feature, {})
                        .get("vectors", {})
                        .get(candidate_value, set())
                    )

                    # Calculate AND combination metrics
                    and_fail_with = len(failing & candidate_failing)
                    and_pass_with = len(passing & candidate_passing)
                    and_fail_without = total_failing - and_fail_with
                    and_pass_without = total_passing - and_pass_with
                    and_discriminability = self.mcc(
                        and_fail_with, and_pass_with, and_fail_without, and_pass_without
                    )

                    # Calculate OR combination metrics
                    or_fail = len(failing | candidate_failing)
                    or_pass = len(passing | candidate_passing)
                    or_fail_without = total_failing - or_fail
                    or_pass_without = total_passing - or_pass
                    or_discriminability = self.mcc(
                        or_fail, or_pass, or_fail_without, or_pass_without
                    )

                    # Add combinations if they improve discriminability
                    if and_discriminability > best_atom.discriminability:
                        LOGGER.debug(
                            f"AND combination improves discriminability: "
                            f"{best_atom.feature} (disc={best_atom.discriminability:.3f}) AND "
                            f"{candidate.feature} (disc={candidate.discriminability:.3f}) -> "
                            f"disc={and_discriminability:.3f}"
                        )
                        all_candidates.append(
                            Conjunction(
                                [best_atom, candidate],
                                discriminability=and_discriminability,
                                level=level,
                                point=phase,
                            )
                        )
                    if or_discriminability > best_atom.discriminability:
                        LOGGER.debug(
                            f"OR combination improves discriminability: "
                            f"{best_atom.feature} (disc={best_atom.discriminability:.3f}) OR "
                            f"{candidate.feature} (disc={candidate.discriminability:.3f}) -> "
                            f"disc={or_discriminability:.3f}"
                        )
                        all_candidates.append(
                            Disjunction(
                                [best_atom, candidate],
                                discriminability=or_discriminability,
                                level=level,
                                point=phase,
                            )
                        )

        # Find the best discriminability among all candidates
        if not all_candidates:
            return None

        max_discriminability = max(d.discriminability for d in all_candidates)
        best_candidates = [
            d for d in all_candidates if d.discriminability == max_discriminability
        ]

        # If multiple candidates have the same discriminability, try combining them
        if len(best_candidates) == 1:
            return best_candidates[0]
        elif len(best_candidates) > 1:
            # Try to combine the best candidates to see if we can improve discriminability
            combined = (
                self._try_combine_diagnoses(
                    best_candidates,
                    failing_features,
                    passing_features,
                    total_failing,
                    total_passing,
                    level,
                    phase,
                )
                if self.combine
                else None
            )
            # If combination improves discriminability, use it
            if combined and combined.discriminability > max_discriminability:
                LOGGER.debug(
                    f"Multi-candidate combination selected: {combined} "
                    f"(disc={combined.discriminability:.3f}) improves over "
                    f"{len(best_candidates)} candidates with disc={max_discriminability:.3f}"
                )
                return combined
            # Otherwise, select the most useful individual diagnosis
            LOGGER.debug(
                f"Multi-candidate combination did not improve, using tie-breaking "
                f"to select from {len(best_candidates)} candidates"
            )
            return self._select_most_useful_diagnosis(
                best_candidates, failing_features, total_failing
            )
        else:
            return None

    def _try_combine_diagnoses(
        self,
        diagnoses: list[Diagnosis],
        failing_features: dict,
        passing_features: dict,
        total_failing: int,
        total_passing: int,
        level: int,
        phase: EntryPoint,
    ) -> Diagnosis | None:
        """
        Attempt pairwise AND / OR combinations of *diagnoses* to find one with
        higher discriminability.

        Iterates over pairs drawn from the first five elements of *diagnoses*
        (to avoid combinatorial explosion), computes MCC scores for each `AND`
        and `OR` combination, and returns the best-improving combination.

        :param diagnoses: Candidate diagnoses that all share the same
            (maximum) discriminability score.
        :param failing_features: Feature → ``{FeatureValue: count, "vectors": …}``
            map for failing runs.
        :param passing_features: Equivalent map for passing runs.
        :param total_failing: Total number of failing test runs.
        :param total_passing: Total number of passing test runs.
        :param level: Call-tree depth to stamp on the returned diagnosis.
        :param phase: :class:`EntryPoint` to stamp on the returned diagnosis.
        :returns: The best-improving :class:`Conjunction` or
            :class:`Disjunction`, or ``None`` when no combination improves
            over the input discriminability.
        """
        if len(diagnoses) < 2:
            return None

        best_combination = None
        best_discriminability = max(d.discriminability for d in diagnoses)

        # Try pairwise combinations (limit to avoid exponential explosion)
        for i, diag1 in enumerate(diagnoses[:5]):  # Limit to first 5 for performance
            for diag2 in diagnoses[i + 1 : 6]:  # Limit to first 6 for performance
                # Extract atoms from both diagnoses
                atoms1 = self._get_atoms_from_diagnosis(diag1)
                atoms2 = self._get_atoms_from_diagnosis(diag2)

                # Skip if they share features (would be redundant)
                features1 = {atom.feature for atom in atoms1}
                features2 = {atom.feature for atom in atoms2}
                if features1 & features2:
                    continue

                # Calculate vectors for both diagnoses
                vectors1 = self._get_diagnosis_vectors(
                    diag1, failing_features, passing_features
                )
                vectors2 = self._get_diagnosis_vectors(
                    diag2, failing_features, passing_features
                )

                if not vectors1 or not vectors2:
                    continue

                fail1, pass1 = vectors1
                fail2, pass2 = vectors2

                # Try AND combination
                and_fail = fail1 & fail2
                and_pass = pass1 & pass2
                and_fail_with = len(and_fail)
                and_pass_with = len(and_pass)
                and_fail_without = total_failing - and_fail_with
                and_pass_without = total_passing - and_pass_with

                and_discriminability = self.mcc(
                    and_fail_with, and_pass_with, and_fail_without, and_pass_without
                )

                if and_discriminability > best_discriminability:
                    best_discriminability = and_discriminability
                    # Combine all atoms
                    best_combination = Conjunction(
                        atoms1 + atoms2,
                        discriminability=and_discriminability,
                        level=level,
                        point=phase,
                        parts=[diag1, diag2],
                    )
                    LOGGER.debug(
                        f"Multi-candidate AND combination improves: "
                        f"{diag1} (disc={diag1.discriminability:.3f}) AND "
                        f"{diag2} (disc={diag2.discriminability:.3f}) -> "
                        f"disc={and_discriminability:.3f}"
                    )

                # Try OR combination
                or_fail = fail1 | fail2
                or_pass = pass1 | pass2
                or_fail_with = len(or_fail)
                or_pass_with = len(or_pass)
                or_fail_without = total_failing - or_fail_with
                or_pass_without = total_passing - or_pass_with

                or_discriminability = self.mcc(
                    or_fail_with, or_pass_with, or_fail_without, or_pass_without
                )

                if or_discriminability > best_discriminability:
                    best_discriminability = or_discriminability
                    # Combine all atoms
                    best_combination = Disjunction(
                        atoms1 + atoms2,
                        discriminability=or_discriminability,
                        level=level,
                        point=phase,
                        parts=[diag1, diag2],
                    )
                    LOGGER.debug(
                        f"Multi-candidate OR combination improves: "
                        f"{diag1} (disc={diag1.discriminability:.3f}) OR "
                        f"{diag2} (disc={diag2.discriminability:.3f}) -> "
                        f"disc={or_discriminability:.3f}"
                    )

        return best_combination

    @staticmethod
    def _get_atoms_from_diagnosis(diagnosis: Diagnosis) -> list[Atom]:
        """
        Extract the flat list of :class:`Atom` objects from *diagnosis*.

        :param diagnosis: An :class:`Atom`, :class:`Conjunction`, or
            :class:`Disjunction`.
        :returns: Constituent atoms — a single-element list for an
            :class:`Atom`, ``diagnosis.atoms`` for compound types, or
            ``[]`` for unknown types.
        """
        if isinstance(diagnosis, Atom):
            return [diagnosis]
        elif isinstance(diagnosis, (Conjunction, Disjunction)):
            return diagnosis.atoms
        return []

    def _get_diagnosis_vectors(
        self, diagnosis: Diagnosis, failing_features: dict, passing_features: dict
    ) -> tuple[set, set] | None:
        """
        Compute the sets of test-vector IDs where *diagnosis* holds.

        For a :class:`Conjunction` the atom support sets are intersected; for
        a :class:`Disjunction` they are joined; for a plain :class:`Atom`
        that atom's vectors are returned directly.

        :param diagnosis: The diagnosis whose support sets to compute.
        :param failing_features: Feature → ``{FeatureValue: count, "vectors": …}``
            map for failing runs.
        :param passing_features: Equivalent map for passing runs.
        :returns: ``(failing_ids, passing_ids)`` where each set contains the
            run IDs in which *diagnosis* holds, or ``None`` if the sets cannot
            be computed (unknown diagnosis type or empty atom list).
        """
        # A combination of compound diagnoses holds where its parts hold
        # together (AND) or alone (OR), which its flattened atoms do not say.
        if isinstance(diagnosis, (Conjunction, Disjunction)) and any(
            isinstance(part, (Conjunction, Disjunction)) for part in diagnosis.parts
        ):
            part_vectors = [
                self._get_diagnosis_vectors(part, failing_features, passing_features)
                for part in diagnosis.parts
            ]
            if any(vectors is None for vectors in part_vectors):
                return None
            combine = set.intersection if isinstance(diagnosis, Conjunction) else set.union
            return (
                combine(*(failing for failing, _ in part_vectors)),
                combine(*(passing for _, passing in part_vectors)),
            )

        atoms = self._get_atoms_from_diagnosis(diagnosis)
        if not atoms:
            return None

        # Start with the first atom's vectors
        first_atom = atoms[0]
        if first_atom.negated:
            value = FeatureValue.FALSE
        else:
            value = FeatureValue.TRUE

        failing_set = (
            failing_features.get(first_atom.feature, {})
            .get("vectors", {})
            .get(value, set())
            .copy()
        )
        passing_set = (
            passing_features.get(first_atom.feature, {})
            .get("vectors", {})
            .get(value, set())
            .copy()
        )

        # For Conjunction, intersect all atoms' vectors
        # For Disjunction, union all atoms' vectors
        # For single Atom, just use the first atom's vectors
        if isinstance(diagnosis, Conjunction):
            for atom in atoms[1:]:
                if atom.negated:
                    value = FeatureValue.FALSE
                else:
                    value = FeatureValue.TRUE
                atom_failing = (
                    failing_features.get(atom.feature, {})
                    .get("vectors", {})
                    .get(value, set())
                )
                atom_passing = (
                    passing_features.get(atom.feature, {})
                    .get("vectors", {})
                    .get(value, set())
                )
                failing_set &= atom_failing
                passing_set &= atom_passing
        elif isinstance(diagnosis, Disjunction):
            for atom in atoms[1:]:
                if atom.negated:
                    value = FeatureValue.FALSE
                else:
                    value = FeatureValue.TRUE
                atom_failing = (
                    failing_features.get(atom.feature, {})
                    .get("vectors", {})
                    .get(value, set())
                )
                atom_passing = (
                    passing_features.get(atom.feature, {})
                    .get("vectors", {})
                    .get(value, set())
                )
                failing_set |= atom_failing
                passing_set |= atom_passing

        return failing_set, passing_set

    # Define actionability ranking for feature types.
    # Higher number = more actionable for a developer fixing the bug.
    ACTIONABILITY_RANK = {
        AnalysisType.CONDITION: 6,  # Most actionable: specific conditions
        AnalysisType.DEF_USE: 6,
        AnalysisType.VARIABLE: 6,
        AnalysisType.SCALAR_PAIR: 5,
        AnalysisType.RETURN: 5,
        AnalysisType.NONE: 4,
        AnalysisType.FUNCTION_ERROR: 4,
        AnalysisType.LINE: 4,  # Medium actionable: localization
        AnalysisType.FUNCTION: 4,
        AnalysisType.LOOP: 3,
        AnalysisType.LENGTH: 3,
        AnalysisType.BRANCH: 2,  # Less actionable: control flow only
        AnalysisType.ASCII_STRING: 1,  # Least actionable: symptoms
        AnalysisType.SPECIAL_STRING: 1,
        AnalysisType.DIGIT_STRING: 1,
        AnalysisType.EMPTY_STRING: 1,
        AnalysisType.EMPTY_BYTES: 1,
    }

    @staticmethod
    def _get_atoms(diagnosis: Diagnosis) -> list[Atom]:
        """
        Extract atoms from *diagnosis* (alias of :meth:`_get_atoms_from_diagnosis`,
        used internally for tie-breaking).

        :param diagnosis: Source diagnosis.
        :returns: Constituent :class:`Atom` objects.
        """
        if isinstance(diagnosis, Atom):
            return [diagnosis]
        elif isinstance(diagnosis, (Conjunction, Disjunction)):
            return diagnosis.atoms
        return []

    @staticmethod
    def _sort_atom(failing_features: dict, total_failing: int):
        """
        Return a sort-key function for tie-breaking among equal-discriminability
        diagnoses.

        The key prioritizes (in order):

        1. Simpler diagnoses (fewer atoms).
        2. Non-negated features (positive statements are more actionable).
        3. Higher support ratio in failing tests.
        4. Higher :attr:`ACTIONABILITY_RANK` of the feature type.

        :param failing_features: Feature → ``{FeatureValue: count, …}`` map
            for failing runs.
        :param total_failing: Total number of failing test runs.
        :returns: A callable suitable for ``max()`` / ``sorted()`` that maps
            a :class:`Diagnosis` to a comparison tuple.
        """

        def sort_key(diagnosis: Diagnosis):
            atoms = Forecast._get_atoms(diagnosis)
            # 1. Prefer simpler diagnoses (atoms > combinations)
            complexity = len(atoms)  # Lower is better (will be negated)
            # 2. Prefer non-negated (fewer negated atoms is better)
            negation_count = sum(1 for atom in atoms if atom.negated)
            # 3. Support in failing tests (average across all atoms)
            total_support = 0
            for atom in atoms:
                feature_data = failing_features.get(atom.feature, {})
                if atom.negated:
                    support = feature_data.get(FeatureValue.FALSE, 0)
                else:
                    support = feature_data.get(FeatureValue.TRUE, 0)
                total_support += support
            avg_support_ratio = total_support / max(1, total_failing * len(atoms))
            # 4. Actionability rank (average across all atoms)
            total_actionability = sum(
                Forecast.ACTIONABILITY_RANK.get(
                    atom.feature.analysis.analysis_type(), 0
                )
                for atom in atoms
            )
            avg_actionability = total_actionability / max(1, len(atoms))
            return (
                -complexity,  # Negative so lower complexity ranks higher
                -negation_count,  # Negative so fewer negations rank higher
                avg_support_ratio,
                avg_actionability,
            )

        return sort_key

    def _select_most_useful_diagnosis(
        self,
        diagnoses: list[Diagnosis],
        failing_features: dict,
        total_failing: int,
    ) -> Diagnosis:
        """
        Select the most useful diagnosis from a list with equal discriminability.

        Applies the tie-breaking sort key produced by :meth:`_sort_atom` to
        prefer simpler, more actionable, and better-supported diagnoses.

        :param diagnoses: Candidate diagnoses with identical discriminability.
        :param failing_features: Feature → ``{FeatureValue: count, …}`` map
            for failing runs (used to compute support ratios).
        :param total_failing: Total number of failing runs.
        :returns: The highest-ranked candidate according to the tie-breaking
            criteria.
        """
        return max(diagnoses, key=self._sort_atom(failing_features, total_failing))

    @staticmethod
    def mcc(fail_with, pass_with, fail_without, pass_without) -> float:
        """
        Compute the Matthews Correlation Coefficient (MCC) used as the
        discriminability score throughout FORECAST.

        Failing runs are the positive class, so in confusion-matrix terms
        *fail_with* are the true positives, *pass_with* the false positives,
        *fail_without* the false negatives and *pass_without* the true
        negatives.

        MCC ranges from -1 (perfectly anti-correlated) through 0 (no
        correlation) to +1 (the hypothesis holds in exactly the failing runs).
        Returns 0.0 when a denominator term vanishes, which happens when the
        hypothesis or its complement is constant across all runs and no
        correlation is defined.

        :param fail_with: Failing runs in which the hypothesis holds.
        :param pass_with: Passing runs in which the hypothesis holds.
        :param fail_without: Failing runs in which it does not hold.
        :param pass_without: Passing runs in which it does not hold.
        :returns: MCC score in ``[-1.0, 1.0]``, or ``0.0`` when undefined.
        """
        # Avoid division by zero
        if (
            (fail_with + pass_with) == 0
            or (fail_with + fail_without) == 0
            or (pass_without + pass_with) == 0
            or (pass_without + fail_without) == 0
        ):
            return 0.0
        return (fail_with * pass_without - pass_with * fail_without) / (
            (fail_with + pass_with)
            * (fail_with + fail_without)
            * (pass_without + pass_with)
            * (pass_without + fail_without)
        ) ** 0.5
