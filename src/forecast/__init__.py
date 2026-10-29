"""
FORECAST — Failure-Diagnosis-Boosted Program Repair.

Quick-start
-----------
>>> from forecast import Forecast, ForecastHandler, instrument
>>> from forecast import RepairContext, AgentMode
>>> from forecast import get_logger

Core pipeline
~~~~~~~~~~~~~
- :func:`~forecast.instrument.instrument`          — Step 1a: instrument source
- :class:`~forecast.subject.Subject`               — Step 1b: collect events
- :class:`~forecast.handler.ForecastHandler`        — Step 1c: build relevance tree
- :class:`~forecast.diagnosis.Forecast`             — Step 2 : derive diagnosis
- :class:`~forecast.agents.base.RepairAgent`        — Steps 3-5: prompt → LLM → patch
"""

# ── Core diagnosis ────────────────────────────────────────────────────────────
from forecast.diagnosis import (
    Forecast,
    Diagnosis,
    CauseChain,
    Atom,
    Conjunction,
    Disjunction,
    Interpreter,
    EntryPoint,
)

# ── Event collection & tree building ─────────────────────────────────────────
from forecast.handler import ForecastHandler
from forecast.tree import (
    TracedRelevanceTree,
    TreeVisitor,
    PrintTreeVisitor,
    ExhaustivePrintTreeVisitor,
)

# ── Integration helpers ───────────────────────────────────────────────────────
from forecast.integration import (
    DiagnosisSerializer,
    save_diagnosis,
    load_diagnosis_json,
)

# ── Agent API ─────────────────────────────────────────────────────────────────
from forecast.agents.base import (
    AgentMode,
    RepairAgent,
    RepairContext,
    RepairResult,
    normalise_patch,
)
from forecast.agents.models.utils import (
    ModelClient,
    ModelResponse,
    get_model_client,
    save_trajectory,
)
from forecast.plugins import (
    create_subject,
    list_agents,
    list_localization_techniques,
    list_model_providers,
    list_registered_formulas,
    list_subjects,
    register_agent_class,
    register_formula,
    register_localization_technique,
    register_model_provider,
    register_subject_class,
)

# ── Logging ───────────────────────────────────────────────────────────────────
from forecast.logger import get_logger, LOGGER

__version__ = "0.1.0"

__all__ = [
    # diagnosis
    "Forecast",
    "Diagnosis",
    "CauseChain",
    "Atom",
    "Conjunction",
    "Disjunction",
    "Interpreter",
    "EntryPoint",
    # collection
    "ForecastHandler",
    # tree
    "TracedRelevanceTree",
    "TreeVisitor",
    "PrintTreeVisitor",
    "ExhaustivePrintTreeVisitor",
    # integration
    "DiagnosisSerializer",
    "save_diagnosis",
    "load_diagnosis_json",
    # agents
    "AgentMode",
    "RepairAgent",
    "RepairContext",
    "RepairResult",
    "normalise_patch",
    # models
    "ModelClient",
    "ModelResponse",
    "get_model_client",
    "save_trajectory",
    # plugin registries
    "register_agent_class",
    "register_model_provider",
    "register_formula",
    "register_localization_technique",
    "register_subject_class",
    "create_subject",
    "list_agents",
    "list_model_providers",
    "list_registered_formulas",
    "list_localization_techniques",
    "list_subjects",
    # logging
    "get_logger",
    "LOGGER",
    # version
    "__version__",
]
