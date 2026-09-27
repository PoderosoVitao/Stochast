from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

from stochast.records import RunRecord

Interval = tuple[float, float]


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

    return ScenarioStats(
        scenario_name=records[0].scenario_name,
        total_runs=total,
        passed_runs=passed,
        pass_rate=passed / total,
        pass_rate_interval=wilson_interval(passed, total),
        assertions=assertions,
    )
