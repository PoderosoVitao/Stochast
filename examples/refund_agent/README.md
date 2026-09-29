# refund_agent

A small customer-support agent with two tools (`lookup_order`, `issue_refund`) and three
scenarios exercising the full assertion vocabulary: tool-call ordering and arguments,
`tool_not_called` for the refund path that shouldn't be taken, and graceful recovery when a
tool call errors.

Runs against Claude through Anthropic's OpenAI-compatible endpoint by default:

```
export ANTHROPIC_API_KEY=sk-ant-...
stochast run examples/refund_agent --adapter examples/refund_agent/agent.py:build_adapter
```

Costs a few cents against Haiku. Set `STOCHAST_EXAMPLE_MODEL` and `STOCHAST_EXAMPLE_BASE_URL`
to point it at OpenAI or another OpenAI-compatible provider instead.
