from pathlib import Path

import pytest

from stochast.discovery import discover_scenarios, import_file, resolve_adapter_factory
from stochast.scenario import registered_scenarios

SCENARIO_SOURCE = """
from stochast import scenario

@scenario()
def my_scenario(agent):
    pass
"""

ADAPTER_SOURCE = """
def build_adapter():
    return "an adapter"
"""


def test_import_file_executes_the_module(tmp_path: Path):
    path = tmp_path / "scenarios.py"
    path.write_text(SCENARIO_SOURCE)

    import_file(path)

    assert [s.name for s in registered_scenarios()] == ["my_scenario"]


def test_discover_scenarios_imports_every_py_file_under_a_directory(tmp_path: Path):
    (tmp_path / "a.py").write_text(SCENARIO_SOURCE.replace("my_scenario", "a_scenario"))
    (tmp_path / "b.py").write_text(SCENARIO_SOURCE.replace("my_scenario", "b_scenario"))

    discover_scenarios(tmp_path)

    assert {s.name for s in registered_scenarios()} == {"a_scenario", "b_scenario"}


def test_discover_scenarios_accepts_a_single_file(tmp_path: Path):
    path = tmp_path / "scenarios.py"
    path.write_text(SCENARIO_SOURCE)

    discover_scenarios(path)

    assert [s.name for s in registered_scenarios()] == ["my_scenario"]


def test_resolve_adapter_factory_from_a_file_path(tmp_path: Path):
    path = tmp_path / "adapter.py"
    path.write_text(ADAPTER_SOURCE)

    factory = resolve_adapter_factory(f"{path}:build_adapter")

    assert factory() == "an adapter"


def test_resolve_adapter_factory_rejects_a_spec_without_a_colon():
    with pytest.raises(ValueError, match="module:factory"):
        resolve_adapter_factory("no_colon_here")
