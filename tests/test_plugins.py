import types
import unittest
from unittest.mock import patch

from forecast.agents.base import RepairAgent, RepairContext
from forecast.localization.localization import default_localize
from forecast.localization.sfl import get_formula, register_formula
from forecast.plugin_loader import load_plugins
from forecast.plugins.registry import (
    create_subject,
    get_agent_class,
    get_localization_technique,
    register_agent_class,
    register_localization_technique,
    register_model_provider,
    register_subject_class,
    resolve_model_factory,
)


class _DummyAgent(RepairAgent):
    @classmethod
    def names(cls):
        return ["DummyAgent"]

    def get_prompt(self, context: RepairContext) -> str:
        return context.problem_statement

    def get_patch(self, response, context: RepairContext, log=None):
        return None


class PluginRegistryTests(unittest.TestCase):
    def test_register_agent_class(self):
        register_agent_class(_DummyAgent, replace=True)
        resolved = get_agent_class("dummyagent")
        self.assertIs(resolved, _DummyAgent)

    def test_register_agent_class_decorator(self):
        @register_agent_class(replace=True)
        class _DecorAgent(RepairAgent):
            @classmethod
            def names(cls):
                return ["DecorAgent"]

            def get_prompt(self, context: RepairContext) -> str:
                return context.problem_statement

            def get_patch(self, response, context: RepairContext, log=None):
                return None

        resolved = get_agent_class("decoragent")
        self.assertIs(resolved, _DecorAgent)

    def test_model_provider_exact_alias(self):
        def factory(**kwargs):
            return ("factory", kwargs["model"])

        register_model_provider(
            name="test-provider",
            factory=factory,
            aliases=["my-test-model"],
            replace=True,
        )
        resolved = resolve_model_factory("my-test-model")
        self.assertIsNotNone(resolved)
        self.assertEqual(("factory", "my-test-model"), resolved(model="my-test-model"))

    def test_model_provider_decorator(self):
        @register_model_provider(
            name="decor-provider",
            aliases=["decor-model"],
            replace=True,
        )
        def factory(**kwargs):
            return ("decor", kwargs["model"])

        resolved = resolve_model_factory("decor-model")
        self.assertIsNotNone(resolved)
        self.assertEqual(("decor", "decor-model"), resolved(model="decor-model"))

    def test_register_formula_without_sflkit(self):
        def my_formula(ep: int, np: int, ef: int, nf: int) -> float:
            return float(ep + np + ef + nf)

        register_formula("MyFormula", my_formula)
        resolved = get_formula("MyFormula")
        self.assertIs(resolved, my_formula)
        self.assertEqual(10.0, resolved(1, 2, 3, 4))

    def test_register_formula_decorator(self):
        @register_formula("DecorFormula", replace=True)
        def decor_formula(ep: int, np: int, ef: int, nf: int) -> float:
            return float(ep * np + ef * nf)

        resolved = get_formula("DecorFormula")
        self.assertIs(resolved, decor_formula)
        self.assertEqual(14.0, resolved(1, 2, 3, 4))

    def test_register_localization_technique_decorator(self):
        @register_localization_technique("decor-technique", replace=True)
        def handler(*, subject, location_type, formula="Tarantula", **kwargs):
            return [subject, location_type, formula, kwargs.get("extra")]

        resolved = get_localization_technique("decor-technique")
        self.assertIs(resolved, handler)
        self.assertEqual(
            ["sub", "line", "Ochiai", 3],
            resolved(subject="sub", location_type="line", formula="Ochiai", extra=3),
        )

    def test_register_subject_class_decorator(self):
        @register_subject_class("decor-subject", replace=True)
        class _DecorSubject:
            def __init__(self, value: int = 0):
                self.value = value

        created = create_subject("decor-subject", value=7)
        self.assertIsInstance(created, _DecorSubject)
        self.assertEqual(7, created.value)


class PluginLoaderTests(unittest.TestCase):
    def setUp(self):
        # Keep the default localization-technique key present.
        @register_localization_technique("sfl", replace=True)
        def _sfl_handler(**kwargs):
            return default_localize(
                subject=kwargs["subject"],
                location_type=kwargs["location_type"],
                formula=kwargs["formula"],
            )

    def test_load_plugins_calls_register_hook_with_context(self):
        plugin_module = types.ModuleType("test_plugin_module")

        def register(context):
            context.register_formula("HookFormula", lambda ep, np, ef, nf: 0.42)

        plugin_module.register = register

        class _EP:
            name = "test-plugin"

            @staticmethod
            def load():
                return plugin_module

        with patch("forecast.plugin_loader.entry_points", return_value=[_EP()]):
            loaded = load_plugins(force=True)

        self.assertEqual(["test-plugin"], loaded)
        formula = get_formula("HookFormula")
        self.assertAlmostEqual(0.42, formula(1, 1, 1, 1))

    def test_load_plugins_supports_decorator_style_registration(self):
        plugin_module = types.ModuleType("test_plugin_decorator_module")

        def register(context):
            @context.register_formula("HookDecorFormula")
            def _formula(ep, np, ef, nf):
                return 0.11

        plugin_module.register = register

        class _EP:
            name = "test-plugin-decorator"

            @staticmethod
            def load():
                return plugin_module

        with patch("forecast.plugin_loader.entry_points", return_value=[_EP()]):
            loaded = load_plugins(force=True)

        self.assertEqual(["test-plugin-decorator"], loaded)
        formula = get_formula("HookDecorFormula")
        self.assertAlmostEqual(0.11, formula(1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
