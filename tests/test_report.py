from stochast.records import AssertionResult, RunRecord, ToolCall
from stochast.report import render_markdown
from stochast.stats import AssertionStats, Percentiles, ScenarioStats, ToolPathFrequency

ZERO_PERCENTILES = Percentiles(p50=0.0, p95=0.0, p99=0.0, min=0.0, max=0.0)


def make_stats(
    name: str,
    passed: int,
    total: int,
    assertions=(),
    tool_paths=(),
    latency_ms=ZERO_PERCENTILES,
    cost_usd=ZERO_PERCENTILES,
    failing_samples=(),
) -> ScenarioStats:
    return ScenarioStats(
        scenario_name=name,
        total_runs=total,
        passed_runs=passed,
        pass_rate=passed / total,
        pass_rate_interval=(passed / total - 0.1, passed / total + 0.1),
        assertions=list(assertions),
        tool_paths=list(tool_paths),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        failing_samples=list(failing_samples),
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


def test_render_markdown_includes_the_tool_path_frequency_table():
    tool_paths = [
        ToolPathFrequency(path=("lookup_order", "format_response"), count=38, frequency=0.76),
        ToolPathFrequency(
            path=("lookup_order", "issue_refund", "format_response"), count=1, frequency=0.02
        ),
    ]
    stats = make_stats("refund_status_lookup", 49, 50, tool_paths=tool_paths)

    output = render_markdown([stats])

    assert "38/50  lookup_order -> format_response" in output
    assert " 1/50  lookup_order -> issue_refund -> format_response" in output


def test_render_markdown_labels_a_tool_path_with_no_calls():
    tool_paths = [ToolPathFrequency(path=(), count=4, frequency=0.08)]
    stats = make_stats("refund_status_lookup", 50, 50, tool_paths=tool_paths)

    output = render_markdown([stats])

    assert "(no tool calls)" in output


def test_render_markdown_includes_latency_and_cost_percentiles():
    latency = Percentiles(p50=120.0, p95=450.0, p99=900.0, min=80.0, max=1200.0)
    cost = Percentiles(p50=0.001, p95=0.004, p99=0.01, min=0.0005, max=0.02)
    stats = make_stats("refund_status_lookup", 50, 50, latency_ms=latency, cost_usd=cost)

    output = render_markdown([stats])

    assert "p50 120ms" in output
    assert "p99 900ms" in output
    assert "$0.0010" in output


def test_render_markdown_omits_failing_samples_section_when_none_failed():
    stats = make_stats("refund_status_lookup", 50, 50)

    output = render_markdown([stats])

    assert "Sample failing runs" not in output


def test_render_markdown_includes_a_failing_run_trace():
    record = RunRecord(
        run_index=3,
        scenario_name="refund_status_lookup",
        tool_calls=[ToolCall(name="lookup_order", arguments={"order_id": 4471})],
        final_output="I don't know.",
        assertions=[
            AssertionResult(label="output_contains('4471')", passed=False, detail="missing 4471")
        ],
    )
    stats = make_stats("refund_status_lookup", 49, 50, failing_samples=[record])

    output = render_markdown([stats])

    assert "Run 3" in output
    assert "output_contains('4471')" in output
    assert "missing 4471" in output
    assert "lookup_order" in output
    assert "I don't know." in output
