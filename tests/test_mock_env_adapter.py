from adapters.mock_env import MockEnvAdapter


def test_mock_env_sequence_advances_and_finishes():
    env = MockEnvAdapter("mock_scenarios/correct_process.json")
    obs1 = env.get_observation()
    assert obs1.procedure_hint == "S03"
    obs2 = env.get_observation()
    assert obs2.payload["throttle_right"] == "IDLE"
    obs3 = env.get_observation()
    obs4 = env.get_observation()
    assert env.get_observation() is None
    assert env.remaining() == 0


def test_reset_replays_sequence():
    env = MockEnvAdapter("mock_scenarios/premature_acceleration.json")
    env.get_observation()
    env.reset()
    obs1 = env.get_observation()
    assert obs1.payload["throttle_right"] == "IDLE_EARLY"


def test_missing_steps_scenario_loaded():
    env = MockEnvAdapter("mock_scenarios/missing_steps.json")
    obs = env.get_observation()
    assert obs.payload["battery"] == "ON"
    assert env.remaining() == 2

