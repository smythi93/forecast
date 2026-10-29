"""
Utilities for loading and rendering Jinja2-based prompt templates from YAML files.

This module provides a consistent interface for agents to:
1. Load prompt YAML configurations
2. Render Jinja2 templates with RepairContext data
3. Cache loaded prompts for efficiency
"""

import inspect
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from jinja2 import Template, TemplateError

from forecast.agents.base import RepairContext

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    Manages loading and rendering prompt templates for a repair agent.

    Features:
    - Lazy loading of YAML prompt files
    - Jinja2 template rendering with context variables
    - Template caching for performance
    - Error handling and validation

    Example:
        ```python
        manager = PromptTemplateManager(agent_dir=Path("agents/my_agent"))
        prompt = manager.render("baseline", context)
        system_prompt = manager.render_system("default")
        ```
    """

    def __init__(
        self, agent_dir: Optional[Path] = None, agent_name: Optional[str] = None
    ):
        """
        Initialize the prompt template manager.

        :param agent_dir: Directory containing the agent's ``prompts.yaml``.
            Must be provided explicitly.
        :param agent_name: Name of the agent (for logging).  ``None`` infers
            it from *agent_dir*.
        :raises FileNotFoundError: If ``prompts.yaml`` is not found in
            *agent_dir*.
        :raises ValueError: If *agent_dir* is ``None``.
        """
        if agent_dir is None:
            # Try to infer from calling module (subclass must override if needed)
            raise ValueError("agent_dir must be provided")

        self.agent_dir = Path(agent_dir)
        self.agent_name = agent_name or self.agent_dir.name
        self.yaml_path = self.agent_dir / "prompts.yaml"

        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"prompts.yaml not found in {self.agent_dir}. "
                f"Expected: {self.yaml_path}"
            )

        self._config: Optional[Dict[str, Any]] = None
        self._template_cache: Dict[str, Template] = {}

        logger.debug(f"Initialized PromptTemplateManager for {self.agent_name}")

    @property
    def config(self) -> Dict[str, Any]:
        """Load and cache the prompt configuration."""
        if self._config is None:
            self._load_config()
        return self._config

    def _load_config(self) -> None:
        """Load the prompts.yaml configuration file."""
        try:
            with open(self.yaml_path, "r") as f:
                self._config = yaml.safe_load(f)
            logger.debug(f"Loaded prompts from {self.yaml_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse {self.yaml_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.yaml_path}: {e}")

        # Validate required sections
        if not self._config:
            raise ValueError(f"{self.yaml_path} is empty")

        required_sections = ["system_prompts", "user_prompts"]
        for section in required_sections:
            if section not in self._config:
                logger.warning(f"Missing section '{section}' in {self.yaml_path}")

    def render(
        self,
        template_name: str,
        context: RepairContext,
        section: str = "user_prompts",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Render a Jinja2 user-prompt template from the YAML configuration.

        :param template_name: Name of the template (e.g. ``"baseline"``,
            ``"with_diagnosis"``).
        :param context: Repair context to use for rendering.
        :param section: Section in the YAML file.  Defaults to
            ``"user_prompts"``.
        :returns: The rendered prompt text.
        :raises KeyError: If *section* or *template_name* is not found.
        :raises TemplateError: If template rendering fails.
        """
        if section not in self.config:
            raise KeyError(f"Section '{section}' not found in prompts.yaml")

        if template_name not in self.config[section]:
            available = list(self.config[section].keys())
            raise KeyError(
                f"Template '{template_name}' not found in {section}. "
                f"Available: {available}"
            )

        # Get or compile the template
        cache_key = f"{section}:{template_name}"
        if cache_key not in self._template_cache:
            template_str = self.config[section][template_name]
            try:
                self._template_cache[cache_key] = Template(template_str)
            except TemplateError as e:
                raise TemplateError(f"Failed to parse template '{template_name}': {e}")

        template = self._template_cache[cache_key]

        # Render with context variables
        context_dict = self._context_to_dict(context)
        if extra_context:
            context_dict.update(extra_context)
        try:
            return template.render(**context_dict)
        except TemplateError as e:
            raise TemplateError(f"Failed to render template '{template_name}': {e}")

    def render_system(
        self,
        template_name: str = "default",
    ) -> Optional[str]:
        """
        Render a system-prompt template (no context variables needed).

        :param template_name: Name of the system prompt template.  Defaults
            to ``"default"``.
        :returns: The rendered system prompt, or ``None`` if the template is
            not found.
        """
        if "system_prompts" not in self.config:
            return None

        if template_name not in self.config["system_prompts"]:
            available = list(self.config["system_prompts"].keys())
            logger.warning(
                f"System prompt '{template_name}' not found. " f"Available: {available}"
            )
            return None

        cache_key = f"system_prompts:{template_name}"
        if cache_key not in self._template_cache:
            template_str = self.config["system_prompts"][template_name]
            self._template_cache[cache_key] = Template(template_str)

        template = self._template_cache[cache_key]
        return template.render()

    def render_variant(
        self,
        variant_name: str,
        context: RepairContext,
    ) -> Optional[str]:
        """
        Render a prompt variant (for multi-stage repairs).

        :param variant_name: Name of the variant (e.g. ``"analysis"``,
            ``"implementation"``).
        :param context: Repair context for rendering.
        :returns: The rendered variant prompt, or ``None`` if the
            ``"variants"`` section is absent.
        """
        if "variants" not in self.config:
            return None

        return self.render(variant_name, context, section="variants")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return agent metadata from the ``prompts.yaml`` configuration.

        :returns: Metadata dictionary (``agent_name``, ``version``,
            ``description``, etc.).
        """
        return self.config.get("metadata", {})

    def supports_diagnosis(self) -> bool:
        """
        Check if this agent's prompts support FORECAST diagnoses.

        :returns: ``True`` if ``with_diagnosis`` templates are available
            (determined by ``metadata.supports_diagnosis`` in the YAML).
        """
        metadata = self.get_metadata()
        return metadata.get("supports_diagnosis", False)

    @staticmethod
    def _context_to_dict(context: RepairContext) -> Dict[str, Any]:
        """
        Convert a :class:`~forecast.agents.base.RepairContext` to a dict for
        Jinja2 rendering.

        :param context: The repair context.
        :returns: Dictionary with all context fields as top-level keys.
        """
        return {
            "instance_id": context.instance_id,
            "problem_statement": context.problem_statement,
            "version": context.version,
            "failing_tests": context.failing_tests or [],
            "passing_tests": context.passing_tests or [],
            "diagnosis": context.diagnosis,
            "diagnosis_discriminability": context.diagnosis_discriminability or 0.0,
            "repo": context.repo,
            "base_commit": context.base_commit,
            "relevant_files": [str(f) for f in (context.relevant_files or [])],
        }


def create_manager_for_agent(agent_class) -> PromptTemplateManager:
    """
    Create a :class:`PromptTemplateManager` for an agent class.

    Infers the agent directory from the agent class's module path and
    constructs a manager automatically.

    :param agent_class: The agent class (e.g. ``ACRAgent``, ``SWEAgent``).
    :returns: A configured :class:`PromptTemplateManager` for the agent.
    """
    module = inspect.getmodule(agent_class)
    if module is None or not getattr(module, "__file__", None):
        raise ValueError(
            f"Cannot infer agent directory for {agent_class}. Missing module file."
        )

    module_path = Path(module.__file__).resolve()
    if module_path.name == "__init__.py":
        agent_dir = module_path.parent
    else:
        agent_dir = module_path.parent

    yaml_path = agent_dir / "prompts.yaml"
    if yaml_path.exists():
        return PromptTemplateManager(
            agent_dir=agent_dir,
            agent_name=agent_class.__name__,
        )

    raise ValueError(
        f"Cannot infer agent directory for {agent_class}. "
        f"Please provide agent_dir explicitly."
    )
