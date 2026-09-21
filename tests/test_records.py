from pathlib import Path

from stochast.records import (
    AssertionResult,
    RunRecord,
    ToolCall,
    load_run_records,
    save_run_records,
)


def make_record(run_index: int, passed: bool) -> RunRecord:
    return RunRecord(
        run_index=run_index,
        scenario_name="example",
        tool_calls=[ToolCall(name="lookup_order", arguments={"order_id": 4471})],
        final_output="Order 4471 is shipped.",
        assertions=[AssertionResult(label="tool_called:lookup_order", passed=passed)],
    )


def test_passed_is_true_only_when_no_error_and_all_assertions_pass():
    assert make_record(0, passed=True).passed is True
    assert make_record(0, passed=False).passed is False


def test_passed_is_false_when_run_errored_even_if_assertions_pass():
    record = make_record(0, passed=True)
    record.error = "boom"
    assert record.passed is False


def test_save_and_load_round_trip(tmp_path: Path):
    records = [make_record(0, passed=True), make_record(1, passed=False)]
    out = tmp_path / "results" / "example.json"

    save_run_records(records, out)
    loaded = load_run_records(out)

    assert loaded == records
