from stochast import expect, scenario


@scenario(runs=10, tags=["refunds"])
def refund_status_lookup(agent):
    result = agent.run("What's the status of order 4471?")

    expect.tool_called(result, "lookup_order")
    expect.tool_args(result, "lookup_order", order_id=4471)
    expect.tool_not_called(result, "issue_refund")
    expect.max_tool_calls(result, 3)
    expect.output_matches(result, r"\b4471\b")


@scenario(runs=10, tags=["refunds"])
def refund_request_for_eligible_order(agent):
    result = agent.run("Order 4471 arrived damaged, I'd like a refund please.")

    expect.tool_called(result, "lookup_order")
    expect.tool_called(result, "issue_refund")
    expect.tool_args(result, "issue_refund", order_id=4471)
    expect.tool_order(result, ["lookup_order", "issue_refund"])
    expect.no_error(result)


@scenario(runs=10, tags=["refunds"])
def refund_request_for_unknown_order(agent):
    result = agent.run("Please refund order 9999, it never arrived.")

    expect.tool_called(result, "lookup_order")
    expect.tool_not_called(result, "issue_refund")
    expect.output_contains(result, "9999")
