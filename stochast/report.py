from __future__ import annotations

from stochast.stats import Percentiles, ScenarioStats


# Renders one or more scenarios' statistics as a single self-contained
# markdown report.
def render_markdown(stats: list[ScenarioStats]) -> str:
    return "# Stochast Report\n\n" + "\n".join(_render_scenario(s) for s in stats)


# Renders one scenario's pass rate, per-assertion breakdown, tool-call
# paths, latency/cost percentiles, and a sample of its failing runs.
def _render_scenario(stats: ScenarioStats) -> str:
    lo, hi = stats.pass_rate_interval
    lines = [
        f"## {stats.scenario_name}",
        "",
        f"Pass rate: **{stats.passed_runs}/{stats.total_runs}** "
        f"({stats.pass_rate:.0%}, 95% CI [{lo:.0%}, {hi:.0%}])",
        "",
    ]
    lines += _render_assertions(stats)
    lines += _render_tool_paths(stats)
    lines += _render_percentiles(stats)
    lines += _render_failing_samples(stats)
    return "\n".join(lines)


# Renders the per-assertion failure-rate table.
def _render_assertions(stats: ScenarioStats) -> list[str]:
    if not stats.assertions:
        return []

    lines = ["| Assertion | Failures | Failure rate | 95% CI |", "|---|---|---|---|"]
    for assertion in stats.assertions:
        a_lo, a_hi = assertion.failure_rate_interval
        lines.append(
            f"| `{assertion.label}` | {assertion.failures}/{assertion.total} "
            f"| {assertion.failure_rate:.0%} | [{a_lo:.0%}, {a_hi:.0%}] |"
        )
    lines.append("")
    return lines


# Renders the tool-call path frequency table, in the "count/total  path"
# format from the spec so rare bad paths are easy to spot at a glance.
def _render_tool_paths(stats: ScenarioStats) -> list[str]:
    if not stats.tool_paths:
        return []

    width = len(str(stats.total_runs))
    lines = ["**Tool-call paths**", "", "```"]
    for tp in stats.tool_paths:
        path = " -> ".join(tp.path) if tp.path else "(no tool calls)"
        lines.append(f"{tp.count:>{width}}/{stats.total_runs}  {path}")
    lines += ["```", ""]
    return lines


# Renders latency and cost percentiles, never a bare mean.
def _render_percentiles(stats: ScenarioStats) -> list[str]:
    return [
        f"**Latency**: {_format_percentiles(stats.latency_ms, 'ms', 0)}",
        "",
        f"**Cost**: {_format_percentiles(stats.cost_usd, '$', 4, prefix=True)}",
        "",
    ]


def _format_percentiles(p: Percentiles, unit: str, decimals: int, prefix: bool = False) -> str:
    def fmt(value: float) -> str:
        text = f"{value:.{decimals}f}"
        return f"{unit}{text}" if prefix else f"{text}{unit}"

    return (
        f"p50 {fmt(p.p50)}, p95 {fmt(p.p95)}, p99 {fmt(p.p99)}, min {fmt(p.min)}, max {fmt(p.max)}"
    )


# Renders a sample of failing runs with enough of their trace to debug:
# which assertions failed and why, the tool-call path, and the final output.
def _render_failing_samples(stats: ScenarioStats) -> list[str]:
    if not stats.failing_samples:
        return []

    lines = ["**Sample failing runs**", ""]
    for record in stats.failing_samples:
        lines.append(f"- Run {record.run_index}:")
        if record.error:
            lines.append(f"  - error: {record.error}")
        for assertion in record.assertions:
            if not assertion.passed:
                detail = f" — {assertion.detail}" if assertion.detail else ""
                lines.append(f"  - failed `{assertion.label}`{detail}")
        path = " -> ".join(call.name for call in record.tool_calls) or "(no tool calls)"
        lines.append(f"  - tool calls: {path}")
        lines.append(f"  - output: {record.final_output!r}")
    lines.append("")
    return lines
