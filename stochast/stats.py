from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from stochast.records import RunRecord

Interval = tuple[float, float]

MAX_FAILING_SAMPLES = 5


# Computes a Wilson score confidence interval for a binomial proportion.
# Unlike the normal approximation, it stays within [0, 1] and stays
# reasonable for small samples and proportions near 0 or 1.
def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    phat = successes / n
    denominator = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denominator
    margin = z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


# Computes the p-th percentile of a list of values by linear interpolation
# between the two closest ranks (the same convention numpy's default uses).
def _percentile(sorted_values: list[float], p: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (p / 100)
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]

    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


@dataclass
class Percentiles:
    p50: float
    p95: float
    p99: float
    min: float
    max: float


# Summarizes a list of values as percentiles rather than a mean, since
# latency and cost are right-skewed by retries and variable-length loops.
def percentiles(values: list[float]) -> Percentiles:
    if not values:
        raise ValueError("cannot compute percentiles of an empty list")

    ordered = sorted(values)
    return Percentiles(
        p50=_percentile(ordered, 50),
        p95=_percentile(ordered, 95),
        p99=_percentile(ordered, 99),
        min=ordered[0],
        max=ordered[-1],
    )


@dataclass
class ToolPathFrequency:
    path: tuple[str, ...]
    count: int
    frequency: float


# Collapses each run's tool calls into an ordered sequence of names and
# counts how often each distinct sequence occurred, most common first.
# This surfaces rare bad paths that a pass rate alone averages away.
def tool_path_frequencies(records: list[RunRecord]) -> list[ToolPathFrequency]:
    counts: dict[tuple[str, ...], int] = {}
    for record in records:
        path = tuple(call.name for call in record.tool_calls)
        counts[path] = counts.get(path, 0) + 1

    total = len(records)
    frequencies = [
        ToolPathFrequency(path=path, count=count, frequency=count / total)
        for path, count in counts.items()
    ]
    frequencies.sort(key=lambda f: (-f.count, f.path))
    return frequencies


@dataclass
class AssertionStats:
    label: str
    total: int
    failures: int
    failure_rate: float
    failure_rate_interval: Interval


@dataclass
class ScenarioStats:
    scenario_name: str
    total_runs: int
    passed_runs: int
    pass_rate: float
    pass_rate_interval: Interval
    assertions: list[AssertionStats]
    tool_paths: list[ToolPathFrequency]
    latency_ms: Percentiles
    cost_usd: Percentiles
    failing_samples: list[RunRecord]


# Aggregates one scenario's RunRecords into a pass rate and a per-assertion
# failure-rate breakdown, each with a 95% Wilson confidence interval.
# Assertions are ordered worst-first, since that's what a reader wants to
# see immediately.
def analyze_scenario(records: list[RunRecord]) -> ScenarioStats:
    if not records:
        raise ValueError("cannot analyze an empty list of run records")

    total = len(records)
    passed = sum(1 for r in records if r.passed)

    counts: dict[str, list[int]] = {}
    for record in records:
        for assertion in record.assertions:
            entry = counts.setdefault(assertion.label, [0, 0])
            entry[1] += 1
            if not assertion.passed:
                entry[0] += 1

    assertions = [
        AssertionStats(
            label=label,
            total=occurrences,
            failures=failures,
            failure_rate=failures / occurrences,
            failure_rate_interval=wilson_interval(failures, occurrences),
        )
        for label, (failures, occurrences) in counts.items()
    ]
    assertions.sort(key=lambda a: (-a.failure_rate, a.label))

    failing_samples = [r for r in records if not r.passed][:MAX_FAILING_SAMPLES]

    return ScenarioStats(
        scenario_name=records[0].scenario_name,
        total_runs=total,
        passed_runs=passed,
        pass_rate=passed / total,
        pass_rate_interval=wilson_interval(passed, total),
        assertions=assertions,
        tool_paths=tool_path_frequencies(records),
        latency_ms=percentiles([r.latency_ms for r in records]),
        cost_usd=percentiles([r.cost_usd for r in records]),
        failing_samples=failing_samples,
    )


# Computes the two-sided exact p-value for a 2x2 contingency table under
# Fisher's exact test: the probability, under fixed row/column totals, of a
# table at least as extreme as the one observed. Used instead of a normal
# approximation because eval batches are typically too small for one to be
# reliable (the same reasoning behind using Wilson over the normal interval).
def fisher_exact_p_value(successes_a: int, total_a: int, successes_b: int, total_b: int) -> float:
    if total_a <= 0 or total_b <= 0:
        raise ValueError("total_a and total_b must be positive")
    if not 0 <= successes_a <= total_a or not 0 <= successes_b <= total_b:
        raise ValueError("successes must be between 0 and the corresponding total")

    successes = successes_a + successes_b
    n = total_a + total_b
    low = max(0, successes - total_b)
    high = min(total_a, successes)

    def table_probability(a: int) -> float:
        return math.comb(total_a, a) * math.comb(total_b, successes - a) / math.comb(n, successes)

    probabilities = {a: table_probability(a) for a in range(low, high + 1)}
    observed = probabilities[successes_a]
    as_extreme = (p for p in probabilities.values() if p <= observed * (1 + 1e-7))
    return min(1.0, sum(as_extreme))


# Projects, via a two-proportion z-test, how many total runs (scaling both
# arms up proportionally, holding the observed rates fixed) would be needed
# for the observed difference to reach significance. This is an estimate
# based on a normal approximation, not a guarantee: it answers "if this
# trend holds up, roughly how much more data would it take."
def _runs_needed_for_significance(
    successes_a: int, total_a: int, successes_b: int, total_b: int, alpha: float
) -> int | None:
    rate_a = successes_a / total_a
    rate_b = successes_b / total_b
    if rate_a == rate_b:
        return None

    pooled = (successes_a + successes_b) / (total_a + total_b)
    se = (pooled * (1 - pooled) * (1 / total_a + 1 / total_b)) ** 0.5
    z = (rate_b - rate_a) / se
    z_alpha: float = NormalDist().inv_cdf(1 - alpha / 2)

    multiplier: float = (z_alpha / abs(z)) ** 2
    if multiplier <= 1:
        return total_a + total_b
    return math.ceil(multiplier * (total_a + total_b))


@dataclass
class ComparisonResult:
    baseline_total: int
    baseline_passed: int
    baseline_pass_rate: float
    variant_total: int
    variant_passed: int
    variant_pass_rate: float
    difference: float
    p_value: float
    significant: bool
    runs_needed_for_significance: int | None
    verdict: str


# Compares two batches of RunRecords (typically the same scenario run under
# two configurations) and reports whether their pass rates differ
# significantly, using Fisher's exact test. When they don't, says so plainly
# instead of printing a misleading winner, and estimates how much more data
# would be needed to tell them apart.
def compare_pass_rates(
    baseline: list[RunRecord], variant: list[RunRecord], alpha: float = 0.05
) -> ComparisonResult:
    if not baseline or not variant:
        raise ValueError("cannot compare an empty list of run records")

    baseline_total = len(baseline)
    baseline_passed = sum(1 for r in baseline if r.passed)
    variant_total = len(variant)
    variant_passed = sum(1 for r in variant if r.passed)

    baseline_rate = baseline_passed / baseline_total
    variant_rate = variant_passed / variant_total
    p_value = fisher_exact_p_value(baseline_passed, baseline_total, variant_passed, variant_total)
    significant = p_value < alpha

    runs_needed = None
    if not significant:
        runs_needed = _runs_needed_for_significance(
            baseline_passed, baseline_total, variant_passed, variant_total, alpha
        )

    return ComparisonResult(
        baseline_total=baseline_total,
        baseline_passed=baseline_passed,
        baseline_pass_rate=baseline_rate,
        variant_total=variant_total,
        variant_passed=variant_passed,
        variant_pass_rate=variant_rate,
        difference=variant_rate - baseline_rate,
        p_value=p_value,
        significant=significant,
        runs_needed_for_significance=runs_needed,
        verdict=_verdict(baseline_rate, variant_rate, p_value, significant, runs_needed, alpha),
    )


def _verdict(
    baseline_rate: float,
    variant_rate: float,
    p_value: float,
    significant: bool,
    runs_needed: int | None,
    alpha: float,
) -> str:
    change = f"{baseline_rate:.0%} -> {variant_rate:.0%}"
    if significant:
        direction = "better" if variant_rate > baseline_rate else "worse"
        return (
            f"The variant is significantly {direction} than the baseline "
            f"({change}, p={p_value:.3f})."
        )

    estimate = (
        f" Roughly {runs_needed} total runs would be needed to detect this "
        f"difference at p<{alpha:g}."
        if runs_needed is not None
        else ""
    )
    return (
        f"No significant difference detected ({change}, p={p_value:.3f}). The sample is too "
        f"small to distinguish the two configurations.{estimate}"
    )
