from __future__ import annotations

from stochast.stats import ScenarioStats


# Renders one or more scenarios' statistics as a single self-contained
# markdown report.
def render_markdown(stats: list[ScenarioStats]) -> str:
    return "# Stochast Report\n\n" + "\n".join(_render_scenario(s) for s in stats)


# Renders one scenario's pass rate and per-assertion failure breakdown.
def _render_scenario(stats: ScenarioStats) -> str:
    lo, hi = stats.pass_rate_interval
    lines = [
        f"## {stats.scenario_name}",
        "",
        f"Pass rate: **{stats.passed_runs}/{stats.total_runs}** "
        f"({stats.pass_rate:.0%}, 95% CI [{lo:.0%}, {hi:.0%}])",
        "",
    ]
    if stats.assertions:
        lines += ["| Assertion | Failures | Failure rate | 95% CI |", "|---|---|---|---|"]
        for assertion in stats.assertions:
            a_lo, a_hi = assertion.failure_rate_interval
            lines.append(
                f"| `{assertion.label}` | {assertion.failures}/{assertion.total} "
                f"| {assertion.failure_rate:.0%} | [{a_lo:.0%}, {a_hi:.0%}] |"
            )
        lines.append("")
    return "\n".join(lines)
