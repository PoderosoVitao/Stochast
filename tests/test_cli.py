import json
from pathlib import Path

from typer.testing import CliRunner

from stochast.cli import app
from stochast.records import AssertionResult, RunRecord, save_run_records

runner = CliRunner()

ADAPTER_SOURCE = """
from stochast.adapters import AgentResult

class FakeAdapter:
    def run(self, prompt):
        return AgentResult(output="Order 4471 is shipped.")

def build_adapter():
    return FakeAdapter()
"""

FAILING_ADAPTER_SOURCE = """
from stochast.adapters import AgentResult

class FakeAdapter:
    def run(self, prompt):
        return AgentResult(output="no idea")

def build_adapter():
    return FakeAdapter()
"""

SCENARIO_SOURCE = """
from stochast import scenario, expect

@scenario(runs=3)
def refund_status_lookup(agent):
    result = agent.run("Where's my order?")
    expect.output_contains(result, "4471")
"""

TWO_SCENARIOS_SOURCE = """
from stochast import scenario, expect

@scenario(runs=2)
def refund_status_lookup(agent):
    result = agent.run("Where's my order?")
    expect.output_contains(result, "4471")

@scenario(runs=2)
def unrelated_scenario(agent):
    result = agent.run("hi")
    expect.output_contains(result, "4471")
"""


def write(path: Path, source: str) -> Path:
    path.write_text(source)
    return path


def test_run_reports_full_pass_rate_and_writes_json(tmp_path: Path):
    scenario_file = write(tmp_path / "scenarios.py", SCENARIO_SOURCE)
    adapter_file = write(tmp_path / "adapter.py", ADAPTER_SOURCE)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            str(scenario_file),
            "--adapter",
            f"{adapter_file}:build_adapter",
            "--concurrency",
            "1",
            "-o",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "3/3" in result.output

    records = json.loads((out_dir / "refund_status_lookup.json").read_text())
    assert len(records) == 3
    assert all(a["passed"] for r in records for a in r["assertions"])


def test_run_exits_nonzero_when_assertions_fail(tmp_path: Path):
    scenario_file = write(tmp_path / "scenarios.py", SCENARIO_SOURCE)
    adapter_file = write(tmp_path / "adapter.py", FAILING_ADAPTER_SOURCE)

    result = runner.invoke(
        app,
        [
            "run",
            str(scenario_file),
            "--adapter",
            f"{adapter_file}:build_adapter",
            "--concurrency",
            "1",
            "-o",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "0/3" in result.output


def test_run_filters_scenarios_by_keyword(tmp_path: Path):
    scenario_file = write(tmp_path / "scenarios.py", TWO_SCENARIOS_SOURCE)
    adapter_file = write(tmp_path / "adapter.py", ADAPTER_SOURCE)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            str(scenario_file),
            "--adapter",
            f"{adapter_file}:build_adapter",
            "-k",
            "refund",
            "--concurrency",
            "1",
            "-o",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "refund_status_lookup.json").exists()
    assert not (out_dir / "unrelated_scenario.json").exists()


def test_run_overrides_run_count(tmp_path: Path):
    scenario_file = write(tmp_path / "scenarios.py", SCENARIO_SOURCE)
    adapter_file = write(tmp_path / "adapter.py", ADAPTER_SOURCE)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            str(scenario_file),
            "--adapter",
            f"{adapter_file}:build_adapter",
            "--runs",
            "5",
            "--concurrency",
            "1",
            "-o",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    records = json.loads((out_dir / "refund_status_lookup.json").read_text())
    assert len(records) == 5


def test_run_reports_no_scenarios_matched(tmp_path: Path):
    scenario_file = write(tmp_path / "scenarios.py", SCENARIO_SOURCE)
    adapter_file = write(tmp_path / "adapter.py", ADAPTER_SOURCE)

    result = runner.invoke(
        app,
        [
            "run",
            str(scenario_file),
            "--adapter",
            f"{adapter_file}:build_adapter",
            "-k",
            "nope",
        ],
    )

    assert result.exit_code == 1
    assert "No scenarios matched" in result.output


def test_run_writes_a_combined_markdown_report(tmp_path: Path):
    scenario_file = write(tmp_path / "scenarios.py", SCENARIO_SOURCE)
    adapter_file = write(tmp_path / "adapter.py", ADAPTER_SOURCE)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            str(scenario_file),
            "--adapter",
            f"{adapter_file}:build_adapter",
            "--concurrency",
            "1",
            "-o",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    report_text = (out_dir / "report.md").read_text()
    assert "## refund_status_lookup" in report_text
    assert "3/3" in report_text


def test_report_command_renders_markdown_from_saved_json(tmp_path: Path):
    scenario_file = write(tmp_path / "scenarios.py", SCENARIO_SOURCE)
    adapter_file = write(tmp_path / "adapter.py", ADAPTER_SOURCE)
    out_dir = tmp_path / "out"

    runner.invoke(
        app,
        [
            "run",
            str(scenario_file),
            "--adapter",
            f"{adapter_file}:build_adapter",
            "--concurrency",
            "1",
            "-o",
            str(out_dir),
        ],
    )

    result = runner.invoke(app, ["report", str(out_dir / "refund_status_lookup.json")])

    assert result.exit_code == 0, result.output
    assert "## refund_status_lookup" in result.output


def test_report_command_writes_to_a_file_when_out_is_given(tmp_path: Path):
    scenario_file = write(tmp_path / "scenarios.py", SCENARIO_SOURCE)
    adapter_file = write(tmp_path / "adapter.py", ADAPTER_SOURCE)
    out_dir = tmp_path / "out"

    runner.invoke(
        app,
        [
            "run",
            str(scenario_file),
            "--adapter",
            f"{adapter_file}:build_adapter",
            "--concurrency",
            "1",
            "-o",
            str(out_dir),
        ],
    )

    report_file = tmp_path / "custom-report.md"
    result = runner.invoke(
        app,
        [
            "report",
            str(out_dir / "refund_status_lookup.json"),
            "-o",
            str(report_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "## refund_status_lookup" in report_file.read_text()


def make_records(passed: int, total: int) -> list[RunRecord]:
    return [
        RunRecord(
            run_index=i,
            scenario_name="refund_status_lookup",
            tool_calls=[],
            final_output="ok",
            assertions=[AssertionResult(label="a", passed=i < passed)],
        )
        for i in range(total)
    ]


def test_compare_reports_a_significant_regression_and_exits_nonzero(tmp_path: Path):
    baseline_file = tmp_path / "baseline.json"
    variant_file = tmp_path / "variant.json"
    save_run_records(make_records(10, 10), baseline_file)
    save_run_records(make_records(5, 10), variant_file)

    result = runner.invoke(app, ["compare", str(baseline_file), str(variant_file)])

    assert result.exit_code == 1, result.output
    assert "significantly worse" in result.output


def test_compare_reports_insufficient_sample_and_exits_zero(tmp_path: Path):
    baseline_file = tmp_path / "baseline.json"
    variant_file = tmp_path / "variant.json"
    save_run_records(make_records(5, 10), baseline_file)
    save_run_records(make_records(6, 10), variant_file)

    result = runner.invoke(app, ["compare", str(baseline_file), str(variant_file)])

    assert result.exit_code == 0, result.output
    assert "too small" in result.output


def test_compare_exits_zero_for_a_significant_improvement(tmp_path: Path):
    baseline_file = tmp_path / "baseline.json"
    variant_file = tmp_path / "variant.json"
    save_run_records(make_records(5, 10), baseline_file)
    save_run_records(make_records(10, 10), variant_file)

    result = runner.invoke(app, ["compare", str(baseline_file), str(variant_file)])

    assert result.exit_code == 0, result.output
    assert "significantly better" in result.output
