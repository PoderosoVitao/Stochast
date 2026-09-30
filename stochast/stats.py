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
