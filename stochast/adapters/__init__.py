from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from stochast.records import ToolCall


@dataclass
class AgentResult:
    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_messages: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class AgentAdapter(Protocol):
    def run(self, prompt: str) -> AgentResult: ...


# Raised by an adapter for infrastructure failures (timeouts, connection
# errors, 429/5xx responses). The runner retries these; every other
# exception is treated as model behaviour and is never retried.
class TransportError(Exception):
    pass
