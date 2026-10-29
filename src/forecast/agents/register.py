from forecast.agents.base import RepairAgent
from forecast.plugins.registry import (
    get_agent_class,
    list_agents,
    register_agent_class,
)

REGISTERED_AGENTS: dict[str, type[RepairAgent]] = {}


def register_agent(cls):
    """
    Class decorator that registers a :class:`~forecast.agents.base.RepairAgent`
    subclass in the global :data:`REGISTERED_AGENTS` registry.

    Allows agents to be looked up by name when running experiments via
    :func:`~forecast.cli.add_agent_argument`.

    :param cls: The :class:`~forecast.agents.base.RepairAgent` subclass to
        register.
    :raises ValueError: If *cls* is not a :class:`~forecast.agents.base.RepairAgent`
        subclass, has no ``names()`` method, or any of its names are already
        taken.
    """
    if not issubclass(cls, RepairAgent):
        raise ValueError("Can only register subclasses of RepairAgent")
    if not hasattr(cls, "names"):
        raise ValueError("Registered agent class must have a 'names' function")
    register_agent_class(cls)
    # Keep the legacy module-level dict in sync for existing call sites.
    for name in cls.names():
        REGISTERED_AGENTS[name.lower()] = cls
    return cls


def get_registered_agent(name: str) -> type[RepairAgent] | None:
    """
    Resolve a registered agent class by name.
    """

    resolved = get_agent_class(name)
    if resolved is not None:
        return resolved
    return REGISTERED_AGENTS.get(name.lower())


def list_registered_agents() -> list[str]:
    """
    Return registered agent names.
    """

    names = set(list_agents()) | set(REGISTERED_AGENTS.keys())
    return sorted(names)
