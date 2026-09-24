import pytest

from stochast import expect
from stochast.adapters import AgentResult, TransportError
from stochast.runner import RunInterrupted, run_scenario
from stochast.scenario import Scenario


def make_scenario(func, runs: int = 1) -> Scenario:
    return Scenario(func=func, name=func.__name__, runs=runs, tags=[])


class OkAdapter:
    def run(self, prompt: str) -> AgentResult:
        return AgentResult(output="ok")


def test_run_scenario_executes_the_requested_number_of_runs():
    def body(agent):
        agent.run("hello")

    records = run_scenario(make_scenario(body, runs=5), OkAdapter, concurrency=3)

    assert [r.run_index for r in records] == [0, 1, 2, 3, 4]
    assert all(r.final_output == "ok" for r in records)


def test_run_scenario_records_assertion_outcomes_from_the_scenario_body():
    def body(agent):
        result = agent.run("hello")
        expect.output_contains(result, "ok")
        expect.tool_called(result, "missing_tool")

    [record] = run_scenario(make_scenario(body), OkAdapter)

    assert [a.passed for a in record.assertions] == [True, False]
    assert record.passed is False


def test_non_transport_exceptions_are_recorded_as_errors_without_retry():
    run_calls = 0

    class Adapter:
        def run(self, prompt: str) -> AgentResult:
            nonlocal run_calls
            run_calls += 1
            return AgentResult(output="ok")

    def body(agent):
        agent.run("hello")
        raise ValueError("boom")

    [record] = run_scenario(make_scenario(body), Adapter, retries=3, sleep=lambda s: None)

    assert record.error == "ValueError: boom"
    assert record.passed is False
    assert run_calls == 1


def test_transport_errors_are_retried_until_success():
    class FlakyAdapter:
        def __init__(self) -> None:
            self.attempts = 0

        def run(self, prompt: str) -> AgentResult:
            self.attempts += 1
            if self.attempts <= 2:
                raise TransportError("rate limited")
            return AgentResult(output="ok")

    def body(agent):
        agent.run("hello")

    [record] = run_scenario(make_scenario(body), FlakyAdapter, retries=3, sleep=lambda s: None)

    assert record.error is None
    assert record.final_output == "ok"


def test_transport_errors_fail_the_run_after_exhausting_retries():
    class AlwaysFlaky:
        def run(self, prompt: str) -> AgentResult:
            raise TransportError("down")

    def body(agent):
        agent.run("hello")

    [record] = run_scenario(make_scenario(body), AlwaysFlaky, retries=2, sleep=lambda s: None)

    assert record.error is not None
    assert "transport error" in record.error


def test_seed_is_derived_per_run_and_applied_when_the_adapter_supports_it():
    seen_seeds = []

    class SeedAwareAdapter:
        def __init__(self) -> None:
            self.seed: int | None = None

        def run(self, prompt: str) -> AgentResult:
            seen_seeds.append(self.seed)
            return AgentResult(output="ok")

    def body(agent):
        agent.run("hello")

    run_scenario(
        make_scenario(body, runs=3),
        SeedAwareAdapter,
        concurrency=3,
        seed=100,
        sleep=lambda s: None,
    )

    assert sorted(seen_seeds) == [100, 101, 102]


def test_adapter_close_is_called_even_when_the_run_errors():
    closed = []

    class Adapter:
        def run(self, prompt: str) -> AgentResult:
            raise ValueError("boom")

        def close(self) -> None:
            closed.append(True)

    def body(agent):
        agent.run("hello")

    run_scenario(make_scenario(body), Adapter, sleep=lambda s: None)

    assert closed == [True]


def test_keyboard_interrupt_raises_run_interrupted_with_completed_records():
    creation_order = 0

    def factory():
        nonlocal creation_order
        index = creation_order
        creation_order += 1

        class InterruptingAdapter:
            def run(self, prompt: str) -> AgentResult:
                raise KeyboardInterrupt()

        return InterruptingAdapter() if index == 1 else OkAdapter()

    def body(agent):
        agent.run("hello")

    with pytest.raises(RunInterrupted) as exc_info:
        run_scenario(make_scenario(body, runs=3), factory, concurrency=1)

    assert len(exc_info.value.records) < 3
    assert all(r.run_index != 1 for r in exc_info.value.records)
