import pytest

from stochast import expect
from stochast.adapters import AgentResult
from stochast.records import ToolCall


def result_with_tools(*names: str, output: str = "") -> AgentResult:
    return AgentResult(output=output, tool_calls=[ToolCall(name=n, arguments={}) for n in names])


def result_with_calls(*calls: ToolCall, output: str = "") -> AgentResult:
    return AgentResult(output=output, tool_calls=list(calls))


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


def test_tool_not_called_passes_when_the_tool_was_never_invoked():
    with expect.collecting() as assertions:
        expect.tool_not_called(result_with_tools("format_response"), "issue_refund")

    [assertion] = assertions
    assert assertion.passed is True


def test_tool_not_called_fails_when_the_tool_was_invoked():
    with expect.collecting() as assertions:
        expect.tool_not_called(result_with_tools("issue_refund"), "issue_refund")

    [assertion] = assertions
    assert assertion.passed is False


def test_tool_called_times_passes_on_an_exact_match():
    with expect.collecting() as assertions:
        expect.tool_called_times(result_with_tools("a", "a", "b"), "a", 2)

    [assertion] = assertions
    assert assertion.passed is True


def test_tool_called_times_fails_on_a_mismatch():
    with expect.collecting() as assertions:
        expect.tool_called_times(result_with_tools("a"), "a", 2)

    [assertion] = assertions
    assert assertion.passed is False
    assert "expected 2" in assertion.detail


def test_tool_args_passes_when_a_call_matches_the_given_arguments():
    call = ToolCall(name="lookup_order", arguments={"order_id": 4471, "verbose": True})
    with expect.collecting() as assertions:
        expect.tool_args(result_with_calls(call), "lookup_order", order_id=4471)

    [assertion] = assertions
    assert assertion.passed is True


def test_tool_args_fails_when_no_call_matches():
    call = ToolCall(name="lookup_order", arguments={"order_id": 1})
    with expect.collecting() as assertions:
        expect.tool_args(result_with_calls(call), "lookup_order", order_id=4471)

    [assertion] = assertions
    assert assertion.passed is False


def test_tool_order_passes_when_names_appear_in_relative_order():
    with expect.collecting() as assertions:
        expect.tool_order(result_with_tools("a", "x", "b", "y", "c"), ["a", "b", "c"])

    [assertion] = assertions
    assert assertion.passed is True


def test_tool_order_fails_when_names_are_out_of_order():
    with expect.collecting() as assertions:
        expect.tool_order(result_with_tools("b", "a", "c"), ["a", "b", "c"])

    [assertion] = assertions
    assert assertion.passed is False


def test_max_tool_calls_passes_when_under_the_limit():
    with expect.collecting() as assertions:
        expect.max_tool_calls(result_with_tools("a", "b"), 3)

    [assertion] = assertions
    assert assertion.passed is True


def test_max_tool_calls_fails_when_over_the_limit():
    with expect.collecting() as assertions:
        expect.max_tool_calls(result_with_tools("a", "b", "c"), 2)

    [assertion] = assertions
    assert assertion.passed is False


def test_output_matches_passes_on_a_regex_match():
    with expect.collecting() as assertions:
        expect.output_matches(result_with_tools(output="Order 4471 is shipped."), r"\b4471\b")

    [assertion] = assertions
    assert assertion.passed is True


def test_output_matches_fails_when_the_pattern_is_absent():
    with expect.collecting() as assertions:
        expect.output_matches(result_with_tools(output="Order 4471 is shipped."), r"\b9999\b")

    [assertion] = assertions
    assert assertion.passed is False


def test_no_error_passes_when_no_tool_call_errored():
    call = ToolCall(name="lookup_order", arguments={})
    with expect.collecting() as assertions:
        expect.no_error(result_with_calls(call))

    [assertion] = assertions
    assert assertion.passed is True


def test_no_error_fails_when_a_tool_call_errored():
    call = ToolCall(name="lookup_order", arguments={}, error="not found")
    with expect.collecting() as assertions:
        expect.no_error(result_with_calls(call))

    [assertion] = assertions
    assert assertion.passed is False
    assert "lookup_order" in assertion.detail


def test_custom_records_the_predicate_result_under_the_given_label():
    with expect.collecting() as assertions:
        expect.custom(result_with_tools(output="ok"), lambda r: r.output == "ok", "output is ok")

    [assertion] = assertions
    assert assertion.label == "output is ok"
    assert assertion.passed is True


def test_custom_fails_when_the_predicate_returns_false():
    with expect.collecting() as assertions:
        expect.custom(result_with_tools(output="bad"), lambda r: r.output == "ok", "output is ok")

    [assertion] = assertions
    assert assertion.passed is False
