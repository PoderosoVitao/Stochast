from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from stochast.adapters import AgentResult
from stochast.records import AssertionResult

_current_assertions: ContextVar[list[AssertionResult] | None] = ContextVar(
    "stochast_current_assertions", default=None
)


# Opens a window during which expect.* calls append to the yielded list,
# scoped so concurrent runs never see each other's assertions.
@contextmanager
def collecting() -> Iterator[list[AssertionResult]]:
    assertions: list[AssertionResult] = []
    token = _current_assertions.set(assertions)
    try:
        yield assertions
    finally:
        _current_assertions.reset(token)


def _record(label: str, passed: bool, detail: str = "") -> None:
    assertions = _current_assertions.get()
    if assertions is None:
        raise RuntimeError("expect.* was called outside of a scenario run")
    assertions.append(AssertionResult(label=label, passed=passed, detail=detail))


# Checks that the named tool was invoked at least once.
def tool_called(result: AgentResult, name: str) -> None:
    passed = any(call.name == name for call in result.tool_calls)
    detail = "" if passed else f"{name!r} was never called"
    _record(f"tool_called({name})", passed, detail)


# Checks that the given substring is present in the final output.
def output_contains(result: AgentResult, substring: str) -> None:
    passed = substring in result.output
    detail = "" if passed else f"output did not contain {substring!r}"
    _record(f"output_contains({substring!r})", passed, detail)
