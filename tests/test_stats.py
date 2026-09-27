import pytest

from stochast.records import AssertionResult, RunRecord
from stochast.stats import analyze_scenario, wilson_interval


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


def make_record(run_index: int, *, passed_labels: dict[str, bool]) -> RunRecord:
    return RunRecord(
        run_index=run_index,
        scenario_name="refund_status_lookup",
        tool_calls=[],
        final_output="ok",
        assertions=[AssertionResult(label=label, passed=p) for label, p in passed_labels.items()],
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
