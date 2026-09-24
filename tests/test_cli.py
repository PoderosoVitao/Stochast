import json
from pathlib import Path

from typer.testing import CliRunner

from stochast.cli import app

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
