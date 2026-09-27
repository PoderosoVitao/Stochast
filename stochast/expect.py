from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

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


# Checks that the named tool was never invoked.
def tool_not_called(result: AgentResult, name: str) -> None:
    passed = not any(call.name == name for call in result.tool_calls)
    detail = "" if passed else f"{name!r} was called but should not have been"
    _record(f"tool_not_called({name})", passed, detail)


# Checks that the named tool was invoked exactly n times.
def tool_called_times(result: AgentResult, name: str, n: int) -> None:
    count = sum(1 for call in result.tool_calls if call.name == name)
    passed = count == n
    detail = "" if passed else f"{name!r} was called {count} time(s), expected {n}"
    _record(f"tool_called_times({name}, {n})", passed, detail)


# Checks that at least one invocation of the named tool matched these
# arguments; extra arguments on the call are ignored.
def tool_args(result: AgentResult, name: str, **kwargs: Any) -> None:
    passed = any(
        call.name == name and all(call.arguments.get(k) == v for k, v in kwargs.items())
        for call in result.tool_calls
    )
    detail = "" if passed else f"no call to {name!r} matched arguments {kwargs}"
    _record(f"tool_args({name}, {kwargs})", passed, detail)


# Checks that these tool names appear, in this relative order, somewhere
# among the calls made (other calls may be interleaved between them).
def tool_order(result: AgentResult, names: list[str]) -> None:
    called = [call.name for call in result.tool_calls]
    remaining = iter(called)
    passed = all(name in remaining for name in names)
    detail = "" if passed else f"tool calls {called} did not contain {names} in order"
    _record(f"tool_order({names})", passed, detail)


# Checks that no more than n tool calls were made in total.
def max_tool_calls(result: AgentResult, n: int) -> None:
    count = len(result.tool_calls)
    passed = count <= n
    detail = "" if passed else f"{count} tool calls exceeded the max of {n}"
    _record(f"max_tool_calls({n})", passed, detail)


# Checks that the given substring is present in the final output.
def output_contains(result: AgentResult, substring: str) -> None:
    passed = substring in result.output
    detail = "" if passed else f"output did not contain {substring!r}"
    _record(f"output_contains({substring!r})", passed, detail)


# Checks that the final output matches a regular expression.
def output_matches(result: AgentResult, pattern: str) -> None:
    passed = re.search(pattern, result.output) is not None
    detail = "" if passed else f"output did not match pattern {pattern!r}"
    _record(f"output_matches({pattern!r})", passed, detail)


# Checks that no tool call made during this run raised an error.
def no_error(result: AgentResult) -> None:
    failed = [call.name for call in result.tool_calls if call.error is not None]
    passed = not failed
    detail = "" if passed else f"tool call(s) errored: {', '.join(failed)}"
    _record("no_error", passed, detail)


# Records an arbitrary predicate over the result under the given label.
def custom(result: AgentResult, fn: Callable[[AgentResult], bool], label: str) -> None:
    passed = fn(result)
    detail = "" if passed else f"custom check {label!r} failed"
    _record(label, passed, detail)
