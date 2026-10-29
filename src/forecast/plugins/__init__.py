"""
Plugin extension points and registries for FORECAST.
"""

from forecast.plugins.registry import (
    clear_registries,
    create_subject,
    get_agent_class,
    get_formula,
    get_localization_technique,
    get_subject_class,
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
    resolve_model_factory,
)

__all__ = [
    "clear_registries",
    "create_subject",
    "get_agent_class",
    "get_formula",
    "get_localization_technique",
    "get_subject_class",
    "list_agents",
    "list_localization_techniques",
    "list_model_providers",
    "list_registered_formulas",
    "list_subjects",
    "register_agent_class",
    "register_formula",
    "register_localization_technique",
    "register_model_provider",
    "register_subject_class",
    "resolve_model_factory",
]
