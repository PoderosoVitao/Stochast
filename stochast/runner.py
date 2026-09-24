from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from stochast import expect
from stochast.adapters import AgentAdapter, AgentResult, TransportError
from stochast.records import AssertionResult, RunRecord, ToolCall
from stochast.scenario import Scenario


# Thin wrapper over an AgentAdapter that scenario functions receive as
# `agent`; it accumulates everything the adapter reports across however
# many `agent.run(...)` calls a scenario makes, for the runner to persist.
class AgentHandle:
    def __init__(self, adapter: AgentAdapter) -> None:
        self._adapter = adapter
        self.tool_calls: list[ToolCall] = []
        self.final_output: str = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.raw_messages: list[dict[str, Any]] = []

    def run(self, prompt: str) -> AgentResult:
        result = self._adapter.run(prompt)
        self.tool_calls.extend(result.tool_calls)
        self.final_output = result.output
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.raw_messages.extend(result.raw_messages)
        return result


class RunInterrupted(Exception):
    def __init__(self, records: list[RunRecord]) -> None:
        super().__init__(f"interrupted after {len(records)} completed run(s)")
        self.records = records


def _backoff_seconds(attempt: int) -> float:
    return float(min(2 ** (attempt - 1), 10))


# Executes a scenario's body once, translating any non-transport exception
# into the run's error rather than letting it escape.
def _run_once(scenario: Scenario, agent: AgentHandle) -> tuple[list[AssertionResult], str | None]:
    with expect.collecting() as assertions:
        try:
            scenario.func(agent)
        except TransportError:
            raise
        except Exception as exc:
            return list(assertions), f"{type(exc).__name__}: {exc}"
    return list(assertions), None


# Executes one run of a scenario, retrying the whole attempt on transport
# errors (with backoff) and discarding partial state from failed attempts,
# since a retried attempt must not mix state with the one it replaces.
def _execute_run(
    scenario: Scenario,
    run_index: int,
    adapter: AgentAdapter,
    retries: int,
    sleep: Callable[[float], None],
) -> RunRecord:
    start = time.monotonic()
    attempt = 0
    while True:
        agent = AgentHandle(adapter)
        try:
            assertions, error = _run_once(scenario, agent)
            break
        except TransportError as exc:
            attempt += 1
            if attempt > retries:
                assertions, error = [], f"transport error after {retries} retries: {exc}"
                break
            sleep(_backoff_seconds(attempt))

    return RunRecord(
        run_index=run_index,
        scenario_name=scenario.name,
        tool_calls=agent.tool_calls,
        final_output=agent.final_output,
        assertions=assertions,
        prompt_tokens=agent.prompt_tokens,
        completion_tokens=agent.completion_tokens,
        latency_ms=(time.monotonic() - start) * 1000,
        error=error,
        raw_messages=agent.raw_messages,
    )


# Runs a scenario `scenario.runs` times with bounded concurrency, returning
# one RunRecord per run sorted by run_index. Raises RunInterrupted, carrying
# whatever runs had already completed, if a KeyboardInterrupt surfaces.
def run_scenario(
    scenario: Scenario,
    adapter_factory: Callable[[], AgentAdapter],
    *,
    concurrency: int = 5,
    retries: int = 3,
    seed: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[RunRecord]:
    total = scenario.runs
    records: list[RunRecord] = []

    def build_and_run(run_index: int) -> RunRecord:
        adapter = adapter_factory()
        if seed is not None and hasattr(adapter, "seed"):
            adapter.seed = seed + run_index
        try:
            return _execute_run(scenario, run_index, adapter, retries, sleep)
        finally:
            close = getattr(adapter, "close", None)
            if close is not None:
                close()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(build_and_run, i): i for i in range(total)}
        try:
            for future in as_completed(futures):
                records.append(future.result())
                if on_progress is not None:
                    on_progress(len(records), total)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise RunInterrupted(sorted(records, key=lambda r: r.run_index)) from None

    return sorted(records, key=lambda r: r.run_index)
