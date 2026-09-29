import os
from typing import Any

from stochast.adapters.openai import OpenAIAdapter, ToolSpec

SYSTEM_PROMPT = (
    "You are a customer support agent for an online store. Always look up an "
    "order before answering questions about it. Only issue a refund when the "
    "customer explicitly asks for one and the order is eligible."
)

LOOKUP_ORDER_PARAMS = {
    "type": "object",
    "properties": {"order_id": {"type": "integer"}},
    "required": ["order_id"],
}

ISSUE_REFUND_PARAMS = {
    "type": "object",
    "properties": {
        "order_id": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["order_id", "reason"],
}


# Builds a fresh, isolated order database and the tools bound to it, so that
# concurrent runs of a scenario never see each other's mutations.
def make_tools() -> list[ToolSpec]:
    orders: dict[int, dict[str, Any]] = {
        4471: {"status": "shipped", "eligible_for_refund": True},
    }

    def lookup_order(order_id: int) -> dict[str, Any]:
        order = orders.get(order_id)
        if order is None:
            raise ValueError(f"no such order: {order_id}")
        return {"order_id": order_id, **order}

    def issue_refund(order_id: int, reason: str) -> dict[str, Any]:
        order = orders.get(order_id)
        if order is None:
            raise ValueError(f"no such order: {order_id}")
        if not order["eligible_for_refund"]:
            raise ValueError(f"order {order_id} is not eligible for a refund")
        order["status"] = "refunded"
        return {"order_id": order_id, "status": "refunded"}

    return [
        ToolSpec(
            name="lookup_order",
            description="Look up an order's status by id",
            parameters=LOOKUP_ORDER_PARAMS,
            handler=lookup_order,
        ),
        ToolSpec(
            name="issue_refund",
            description="Issue a refund for an eligible order",
            parameters=ISSUE_REFUND_PARAMS,
            handler=issue_refund,
        ),
    ]


# Builds the adapter that scenarios.py's scenarios run against. Talks to
# Anthropic's OpenAI-compatible endpoint by default; set STOCHAST_EXAMPLE_BASE_URL
# and STOCHAST_EXAMPLE_MODEL to point it at OpenAI or another provider instead.
def build_adapter() -> OpenAIAdapter:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("set ANTHROPIC_API_KEY to run this example")

    return OpenAIAdapter(
        model=os.environ.get("STOCHAST_EXAMPLE_MODEL", "claude-haiku-4-5-20251001"),
        api_key=api_key,
        base_url=os.environ.get("STOCHAST_EXAMPLE_BASE_URL", "https://api.anthropic.com/v1"),
        system_prompt=SYSTEM_PROMPT,
        tools=make_tools(),
    )
