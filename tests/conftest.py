import pytest

from stochast.scenario import clear_registry


@pytest.fixture(autouse=True)
def _isolated_scenario_registry():
    clear_registry()
    yield
    clear_registry()
