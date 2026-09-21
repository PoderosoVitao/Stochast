from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class AssertionResult:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class RunRecord:
    run_index: int
    scenario_name: str
    tool_calls: list[ToolCall]
    final_output: str
    assertions: list[AssertionResult]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    raw_messages: list[dict[str, Any]] = field(default_factory=list)

    # True when every assertion in this run passed and no error was raised.
    @property
    def passed(self) -> bool:
        return self.error is None and all(a.passed for a in self.assertions)


# Writes a batch of run records for one scenario to a single JSON file.
def save_run_records(records: list[RunRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    path.write_text(json.dumps(payload, indent=2, default=str))


# Reads back a batch of run records previously written by save_run_records.
def load_run_records(path: Path) -> list[RunRecord]:
    payload = json.loads(path.read_text())
    records = []
    for entry in payload:
        tool_calls = [ToolCall(**tc) for tc in entry.pop("tool_calls")]
        assertions = [AssertionResult(**a) for a in entry.pop("assertions")]
        records.append(RunRecord(tool_calls=tool_calls, assertions=assertions, **entry))
    return records
