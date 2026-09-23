from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx

from stochast.adapters import AgentResult, TransportError
from stochast.records import ToolCall


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    # Renders this tool in the OpenAI chat-completions `tools` wire format.
    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class OpenAIAdapter:
    def __init__(
        self,
        model: str,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        max_tool_iterations: int = 10,
        seed: int | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_tool_iterations = max_tool_iterations
        self.seed = seed
        self._handlers = {tool.name: tool.handler for tool in self.tools}
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # Drives the chat-completions tool-calling loop for a single prompt,
    # executing any requested tools locally, until the model answers
    # without requesting further tool calls or max_tool_iterations is hit.
    def run(self, prompt: str) -> AgentResult:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        tool_calls: list[ToolCall] = []
        prompt_tokens = 0
        completion_tokens = 0

        for _ in range(self.max_tool_iterations):
            response = self._complete(messages)
            usage = response.get("usage") or {}
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)

            message = response["choices"][0]["message"]
            messages.append(message)

            requested_calls = message.get("tool_calls") or []
            if not requested_calls:
                return AgentResult(
                    output=message.get("content") or "",
                    tool_calls=tool_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    raw_messages=messages,
                )

            for call in requested_calls:
                tool_calls.append(self._execute_tool_call(call, messages))

        return AgentResult(
            output=messages[-1].get("content") or "",
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_messages=messages,
        )

    # Sends one chat-completions request, translating network-level and
    # 429/5xx failures into TransportError so the runner knows to retry.
    def _complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.tools:
            payload["tools"] = [tool.to_openai_format() for tool in self.tools]
        if self.seed is not None:
            payload["seed"] = self.seed

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except httpx.TransportError as exc:
            raise TransportError(str(exc)) from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise TransportError(f"{response.status_code}: {response.text}")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    # Runs one model-requested tool call locally and records its outcome,
    # appending the tool's response (or its error) back into the transcript.
    def _execute_tool_call(self, call: dict[str, Any], messages: list[dict[str, Any]]) -> ToolCall:
        name = call["function"]["name"]
        arguments = json.loads(call["function"]["arguments"] or "{}")
        handler = self._handlers.get(name)

        start = time.monotonic()
        error: str | None = None
        result: Any = None
        try:
            if handler is None:
                raise LookupError(f"no handler registered for tool {name!r}")
            result = handler(**arguments)
        except Exception as exc:
            error = str(exc)
        duration_ms = (time.monotonic() - start) * 1000

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": f"Error: {error}" if error else json.dumps(result),
            }
        )
        return ToolCall(
            name=name, arguments=arguments, result=result, error=error, duration_ms=duration_ms
        )
