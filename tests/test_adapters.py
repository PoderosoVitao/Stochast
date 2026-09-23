from stochast.adapters import AgentAdapter, AgentResult
from stochast.records import ToolCall


class FakeAdapter:
    def run(self, prompt: str) -> AgentResult:
        return AgentResult(
            output=f"handled: {prompt}",
            tool_calls=[ToolCall(name="lookup_order", arguments={"order_id": 4471})],
            prompt_tokens=12,
            completion_tokens=5,
        )


def test_fake_adapter_satisfies_the_agent_adapter_protocol():
    assert isinstance(FakeAdapter(), AgentAdapter)


def test_agent_result_defaults_to_empty_tool_calls_and_messages():
    result = AgentResult(output="done")
    assert result.tool_calls == []
    assert result.raw_messages == []
    assert result.prompt_tokens == 0
