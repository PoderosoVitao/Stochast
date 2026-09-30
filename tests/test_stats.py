import pytest

from stochast.records import AssertionResult, RunRecord, ToolCall
from stochast.stats import analyze_scenario, percentiles, tool_path_frequencies, wilson_interval


def test_wilson_interval_matches_the_spec_worked_example():
    lo, hi = wilson_interval(47, 50)
    assert lo == pytest.approx(0.8378, abs=1e-3)
    assert hi == pytest.approx(0.9794, abs=1e-3)


def test_wilson_interval_is_symmetric_around_point_five():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.403832, abs=1e-5)
    assert hi == pytest.approx(0.596168, abs=1e-5)


def test_wilson_interval_stays_within_zero_and_one_for_extreme_proportions():
    lo, hi = wilson_interval(10, 10)
    assert lo == pytest.approx(0.722467, abs=1e-5)
    assert hi == pytest.approx(1.0, abs=1e-9)

    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.277533, abs=1e-5)


def test_wilson_interval_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def make_record(
    run_index: int,
    *,
    passed_labels: dict[str, bool],
    tool_calls: list[ToolCall] | None = None,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    error: str | None = None,
) -> RunRecord:
    return RunRecord(
        run_index=run_index,
        scenario_name="refund_status_lookup",
        tool_calls=tool_calls or [],
        final_output="ok",
        assertions=[AssertionResult(label=label, passed=p) for label, p in passed_labels.items()],
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        error=error,
    )


def test_analyze_scenario_computes_pass_rate_and_its_interval():
    records = [
        make_record(0, passed_labels={"a": True}),
        make_record(1, passed_labels={"a": True}),
        make_record(2, passed_labels={"a": False}),
    ]

    stats = analyze_scenario(records)

    assert stats.scenario_name == "refund_status_lookup"
    assert stats.total_runs == 3
    assert stats.passed_runs == 2
    assert stats.pass_rate == pytest.approx(2 / 3)
    assert stats.pass_rate_interval == wilson_interval(2, 3)


def test_analyze_scenario_breaks_down_failures_per_assertion_label():
    records = [
        make_record(0, passed_labels={"a": True, "b": True}),
        make_record(1, passed_labels={"a": True, "b": False}),
        make_record(2, passed_labels={"a": False, "b": False}),
    ]

    stats = analyze_scenario(records)

    by_label = {a.label: a for a in stats.assertions}
    assert by_label["a"].failures == 1
    assert by_label["a"].total == 3
    assert by_label["b"].failures == 2
    assert by_label["b"].total == 3


def test_analyze_scenario_orders_assertions_worst_first():
    records = [
        make_record(0, passed_labels={"rarely_fails": True, "often_fails": False}),
        make_record(1, passed_labels={"rarely_fails": True, "often_fails": False}),
        make_record(2, passed_labels={"rarely_fails": False, "often_fails": False}),
    ]

    stats = analyze_scenario(records)

    assert [a.label for a in stats.assertions] == ["often_fails", "rarely_fails"]


def test_analyze_scenario_rejects_an_empty_list():
    with pytest.raises(ValueError):
        analyze_scenario([])


def test_percentiles_matches_linear_interpolation_reference_values():
    stats = percentiles([1.0, 2.0, 3.0, 4.0, 5.0])

    assert stats.min == 1.0
    assert stats.max == 5.0
    assert stats.p50 == pytest.approx(3.0)
    assert stats.p95 == pytest.approx(4.8)
    assert stats.p99 == pytest.approx(4.96)


def test_percentiles_of_a_single_value_is_that_value_everywhere():
    stats = percentiles([7.0])

    assert (stats.p50, stats.p95, stats.p99, stats.min, stats.max) == (7.0, 7.0, 7.0, 7.0, 7.0)


def test_percentiles_rejects_an_empty_list():
    with pytest.raises(ValueError):
        percentiles([])


def test_tool_path_frequencies_counts_distinct_sequences_most_common_first():
    records = [
        make_record(0, passed_labels={}, tool_calls=[ToolCall(name="a", arguments={})]),
        make_record(1, passed_labels={}, tool_calls=[ToolCall(name="a", arguments={})]),
        make_record(
            2,
            passed_labels={},
            tool_calls=[ToolCall(name="a", arguments={}), ToolCall(name="b", arguments={})],
        ),
        make_record(3, passed_labels={}, tool_calls=[]),
    ]

    frequencies = tool_path_frequencies(records)

    assert frequencies[0].path == ("a",)
    assert frequencies[0].count == 2
    assert frequencies[0].frequency == pytest.approx(0.5)
    assert ("a", "b") in [f.path for f in frequencies]
    assert () in [f.path for f in frequencies]


def test_analyze_scenario_includes_tool_paths_percentiles_and_failing_samples():
    records = [
        make_record(
            0,
            passed_labels={"a": True},
            tool_calls=[ToolCall(name="lookup_order", arguments={})],
            latency_ms=100.0,
            cost_usd=0.01,
        ),
        make_record(
            1,
            passed_labels={"a": False},
            tool_calls=[ToolCall(name="lookup_order", arguments={})],
            latency_ms=200.0,
            cost_usd=0.02,
            error="boom",
        ),
    ]

    stats = analyze_scenario(records)

    assert stats.tool_paths[0].path == ("lookup_order",)
    assert stats.tool_paths[0].count == 2
    assert stats.latency_ms.min == 100.0
    assert stats.latency_ms.max == 200.0
    assert stats.cost_usd.max == pytest.approx(0.02)
    assert [r.run_index for r in stats.failing_samples] == [1]


def test_analyze_scenario_caps_failing_samples():
    records = [make_record(i, passed_labels={"a": False}) for i in range(10)]

    stats = analyze_scenario(records)

    assert len(stats.failing_samples) == 5
