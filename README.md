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
pass rate and per-assertion failure rates, each with a 95% confidence interval. See
[`examples/refund_agent`](examples/refund_agent) for a complete, runnable version of the example
above, with three scenarios covering the full assertion vocabulary.

## Retry policy

Stochast retries transport errors (timeouts, connection failures, 429s, 5xxs) with backoff,
because those are infrastructure problems. It never retries anything else: a model calling the
wrong tool or giving a bad answer is data, and retrying it would silently destroy the measurement
you're trying to take.

## Status

Early and incomplete. Currently implemented: the `@scenario` decorator, an OpenAI-compatible
tool-calling adapter, a concurrent runner with the retry policy above and Ctrl-C-safe partial
results, the full assertion vocabulary (`tool_called`, `tool_not_called`, `tool_called_times`,
`tool_args`, `tool_order`, `max_tool_calls`, `output_contains`, `output_matches`, `no_error`,
`custom`), Wilson confidence intervals, per-assertion failure-rate breakdowns, and a markdown
report. Tool-path frequency tables, cost/latency percentiles, and A/B comparison are planned but
not yet built.

## Install

```
pip install stochast
```

Requires Python 3.11+.
