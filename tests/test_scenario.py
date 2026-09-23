from stochast.scenario import registered_scenarios, scenario


def test_scenario_decorator_registers_function_with_defaults():
    @scenario()
    def my_scenario(agent):
        pass

    [registered] = registered_scenarios()
    assert registered.func is my_scenario
    assert registered.name == "my_scenario"
    assert registered.runs == 10
    assert registered.tags == []


def test_scenario_decorator_captures_runs_and_tags():
    @scenario(runs=50, tags=["refunds"])
    def refund_status_lookup(agent):
        pass

    [registered] = registered_scenarios()
    assert registered.runs == 50
    assert registered.tags == ["refunds"]


def test_scenario_decorator_returns_the_original_function_unchanged():
    @scenario()
    def my_scenario(agent):
        return "called"

    assert my_scenario(None) == "called"


def test_multiple_scenarios_are_registered_in_declaration_order():
    @scenario()
    def first(agent):
        pass

    @scenario()
    def second(agent):
        pass

    names = [s.name for s in registered_scenarios()]
    assert names == ["first", "second"]
