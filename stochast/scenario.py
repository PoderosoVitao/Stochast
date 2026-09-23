from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ScenarioFunc = Callable[[Any], None]

_REGISTRY: list[Scenario] = []


@dataclass
class Scenario:
    func: ScenarioFunc
    name: str
    runs: int
    tags: list[str] = field(default_factory=list)


# Registers the decorated function as a scenario to be executed `runs` times.
def scenario(
    runs: int = 10, tags: list[str] | None = None
) -> Callable[[ScenarioFunc], ScenarioFunc]:
    def decorator(func: ScenarioFunc) -> ScenarioFunc:
        _REGISTRY.append(Scenario(func=func, name=func.__name__, runs=runs, tags=tags or []))
        return func

    return decorator


# Returns every scenario registered so far via the @scenario decorator.
def registered_scenarios() -> list[Scenario]:
    return list(_REGISTRY)


# Empties the registry; used to isolate scenario discovery between test runs.
def clear_registry() -> None:
    _REGISTRY.clear()
