# Stochast

A testing tool for LLM agents that reports pass rates instead of pass/fail verdicts. Agents are
non-deterministic: the same prompt can take a different tool-call path on every run, so a single
pass or fail tells you almost nothing. Stochast runs a scenario N times and reports how often it
passed.

```python
# scenarios.py
from stochast import scenario, expect
from stochast.adapters.openai import OpenAIAdapter, ToolSpec


def lookup_order(order_id: int) -> dict:
    return {"order_id": order_id, "status": "shipped"}


def build_adapter():
    return OpenAIAdapter(
        model="gpt-4o-mini",
        api_key="sk-...",
        tools=[
            ToolSpec(
                name="lookup_order",
                description="Look up an order by id",
                parameters={
                    "type": "object",
                    "properties": {"order_id": {"type": "integer"}},
                    "required": ["order_id"],
                },
                handler=lookup_order,
            )
        ],
    )


@scenario(runs=20)
def refund_status_lookup(agent):
    result = agent.run("What's the status of order 4471?")
    expect.tool_called(result, "lookup_order")
    expect.output_contains(result, "4471")
```

```
stochast run scenarios.py --adapter scenarios:build_adapter
```

```
refund_status_lookup (20 runs)
  pass rate: 18/20 (90%)
```

Every run is persisted as JSON under `stochast-results/`, alongside a `report.md` summarizing
pass rate and per-assertion failure rates (each with a 95% confidence interval), the distinct
tool-call paths taken and how often each occurred, latency and cost percentiles, and a sample of
failing runs with their traces. See [`examples/refund_agent`](examples/refund_agent) for a
complete, runnable version of the example above, with three scenarios covering the full assertion
vocabulary.

## Did my change actually help?

```
stochast compare stochast-results/baseline.json stochast-results/variant.json
```

```
baseline: 10/10 (100%)
variant:  5/10 (50%)
difference: -50%  (p=0.0325)
The variant is significantly worse than the baseline (100% -> 50%, p=0.033).
```

When the sample is too small to tell, it says so instead of printing a misleading winner, and
estimates how many more runs it would take to find out:

```
No significant difference detected (50% -> 60%, p=1.000). The sample is too small to
distinguish the two configurations. Roughly 381 total runs would be needed to detect this
difference at p<0.05.
```

## Retry policy

Stochast retries transport errors (timeouts, connection failures, 429s, 5xxs) with backoff,
because those are infrastructure problems. It never retries anything else: a model calling the
wrong tool or giving a bad answer is data, and retrying it would silently destroy the measurement
you're trying to take.

## Status

Implemented: the `@scenario` decorator, an OpenAI-compatible tool-calling adapter, a concurrent
runner with the retry policy above and Ctrl-C-safe partial results, the full assertion vocabulary
(`tool_called`, `tool_not_called`, `tool_called_times`, `tool_args`, `tool_order`,
`max_tool_calls`, `output_contains`, `output_matches`, `no_error`, `custom`), Wilson confidence
intervals, per-assertion failure-rate breakdowns, tool-call path frequency tables, cost/latency
percentiles (cost tracking is opt-in — pass per-token pricing to `OpenAIAdapter` if you want it,
since stochast ships no built-in price list to go stale), failing-run traces, a markdown report,
and A/B comparison via Fisher's exact test with an explicit insufficient-sample verdict.

By design, this stays a CLI tool: no web UI or dashboard, no database (JSON on disk), no
LLM-as-judge grading, and no framework-specific adapters — write your own `AgentAdapter` for
anything beyond the OpenAI-compatible wire format, which takes about twenty lines.

## Install

```
pip install stochast
```

Requires Python 3.11+.
