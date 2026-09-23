import json

import httpx
import pytest

from stochast.adapters import TransportError
from stochast.adapters.openai import OpenAIAdapter, ToolSpec


def completion(content: str | None = None, tool_calls: list[dict] | None = None) -> httpx.Response:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }
    return httpx.Response(200, json=body)


def adapter_with(handler, **kwargs) -> OpenAIAdapter:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIAdapter(model="gpt-4o-mini", api_key="test-key", client=client, **kwargs)


def test_run_returns_output_and_token_usage_when_model_answers_directly():
    def handler(request: httpx.Request) -> httpx.Response:
        return completion(content="Order 4471 is shipped.")

    result = adapter_with(handler).run("Where is order 4471?")

    assert result.output == "Order 4471 is shipped."
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 4
    assert result.tool_calls == []


def test_run_executes_a_requested_tool_and_returns_final_answer():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return completion(
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "lookup_order",
                            "arguments": json.dumps({"order_id": 4471}),
                        },
                    }
                ]
            )
        return completion(content="Order 4471 is shipped.")

    def lookup_order(order_id: int) -> dict:
        return {"order_id": order_id, "status": "shipped"}

    tool = ToolSpec(
        name="lookup_order", description="Look up an order", parameters={}, handler=lookup_order
    )
    result = adapter_with(handler, tools=[tool]).run("Where is order 4471?")

    assert result.output == "Order 4471 is shipped."
    [call] = result.tool_calls
    assert call.name == "lookup_order"
    assert call.arguments == {"order_id": 4471}
    assert call.result == {"order_id": 4471, "status": "shipped"}
    assert call.error is None


def test_run_records_tool_handler_errors_and_feeds_them_back_to_the_model():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return completion(
                tool_calls=[
                    {"id": "call_1", "function": {"name": "lookup_order", "arguments": "{}"}}
                ]
            )
        return completion(content="I couldn't find that order.")

    def lookup_order() -> dict:
        raise ValueError("order not found")

    tool = ToolSpec(
        name="lookup_order", description="Look up an order", parameters={}, handler=lookup_order
    )
    result = adapter_with(handler, tools=[tool]).run("Where is order 4471?")

    [call] = result.tool_calls
    assert call.error == "order not found"
    assert result.output == "I couldn't find that order."


def test_run_stops_after_max_tool_iterations_without_a_final_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        return completion(
            tool_calls=[{"id": "call_1", "function": {"name": "noop", "arguments": "{}"}}]
        )

    tool = ToolSpec(name="noop", description="", parameters={}, handler=lambda: None)
    result = adapter_with(handler, tools=[tool], max_tool_iterations=2).run("loop forever")

    assert len(result.tool_calls) == 2


def test_run_raises_transport_error_on_connection_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(TransportError):
        adapter_with(handler).run("hello")


@pytest.mark.parametrize("status_code", [429, 503])
def test_run_raises_transport_error_on_retryable_status_codes(status_code):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="rate limited")

    with pytest.raises(TransportError):
        adapter_with(handler).run("hello")


def test_run_raises_http_status_error_on_non_retryable_status_codes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(httpx.HTTPStatusError):
        adapter_with(handler).run("hello")
