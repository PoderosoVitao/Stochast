from stochast.report import render_markdown
from stochast.stats import AssertionStats, ScenarioStats


def make_stats(name: str, passed: int, total: int, assertions=()) -> ScenarioStats:
    return ScenarioStats(
        scenario_name=name,
        total_runs=total,
        passed_runs=passed,
        pass_rate=passed / total,
        pass_rate_interval=(passed / total - 0.1, passed / total + 0.1),
        assertions=list(assertions),
    )


def test_render_markdown_includes_scenario_name_and_pass_rate():
    stats = make_stats("refund_status_lookup", 47, 50)

    output = render_markdown([stats])

    assert "## refund_status_lookup" in output
    assert "47/50" in output
    assert "94%" in output


def test_render_markdown_includes_an_assertion_breakdown_table():
    assertion = AssertionStats(
        label="tool_called(lookup_order)",
        total=50,
        failures=3,
        failure_rate=0.06,
        failure_rate_interval=(0.02, 0.16),
    )
    stats = make_stats("refund_status_lookup", 47, 50, assertions=[assertion])

    output = render_markdown([stats])

    assert "tool_called(lookup_order)" in output
    assert "3/50" in output
    assert "6%" in output


def test_render_markdown_omits_the_table_when_there_are_no_assertions():
    stats = make_stats("refund_status_lookup", 50, 50)

    output = render_markdown([stats])

    assert "| Assertion |" not in output


def test_render_markdown_renders_a_section_per_scenario():
    stats_a = make_stats("first_scenario", 10, 10)
    stats_b = make_stats("second_scenario", 5, 10)

    output = render_markdown([stats_a, stats_b])

    assert "## first_scenario" in output
    assert "## second_scenario" in output
    assert output.index("## first_scenario") < output.index("## second_scenario")
