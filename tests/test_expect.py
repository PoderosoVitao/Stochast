import pytest

from stochast import expect
from stochast.adapters import AgentResult
from stochast.records import ToolCall


def result_with_tools(*names: str, output: str = "") -> AgentResult:
    return AgentResult(output=output, tool_calls=[ToolCall(name=n, arguments={}) for n in names])


def test_assertions_raise_outside_a_collecting_block():
    with pytest.raises(RuntimeError):
        expect.tool_called(result_with_tools("lookup_order"), "lookup_order")


def test_tool_called_passes_when_the_tool_was_invoked():
    with expect.collecting() as assertions:
        expect.tool_called(result_with_tools("lookup_order", "format_response"), "lookup_order")

    [assertion] = assertions
    assert assertion.passed is True


def test_tool_called_fails_when_the_tool_was_never_invoked():
    with expect.collecting() as assertions:
        expect.tool_called(result_with_tools("format_response"), "lookup_order")

    [assertion] = assertions
    assert assertion.passed is False
    assert "lookup_order" in assertion.detail


def test_output_contains_passes_when_substring_is_present():
    with expect.collecting() as assertions:
        expect.output_contains(result_with_tools(output="Order 4471 is shipped."), "4471")

    [assertion] = assertions
    assert assertion.passed is True


def test_output_contains_fails_when_substring_is_absent():
    with expect.collecting() as assertions:
        expect.output_contains(result_with_tools(output="Order 4471 is shipped."), "9999")

    [assertion] = assertions
    assert assertion.passed is False


def test_collecting_blocks_do_not_leak_assertions_into_each_other():
    with expect.collecting() as first:
        expect.tool_called(result_with_tools("a"), "a")
        with expect.collecting() as second:
            expect.tool_called(result_with_tools("b"), "b")
        assert len(second) == 1

    assert len(first) == 1
