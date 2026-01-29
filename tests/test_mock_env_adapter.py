from adapters.mock_env import MockEnvAdapter


def test_mock_env_sequence_advances_and_finishes():
    env = MockEnvAdapter("mock_scenarios/correct_process.json")
    obs1 = env.get_observation()
    assert obs1.procedure_hint == "S01"
    obs2 = env.get_observation()
    obs3 = env.get_observation()
    obs4 = env.get_observation()
    # consume rest; ensure throttle event present
    throttle_seen = False
    while True:
        ob = env.get_observation()
        if ob is None:
            break
        if ob.payload.get("throttle_right") == "IDLE":
            throttle_seen = True
    assert throttle_seen
    assert env.remaining() == 0


def test_reset_replays_sequence():
    env = MockEnvAdapter("mock_scenarios/premature_acceleration.json")
    env.get_observation()
    env.reset()
    obs1 = env.get_observation()
    # first obs is now S01; advance to premature throttle
    env.get_observation()  # S02
    env.get_observation()  # S03
    env.get_observation()  # S04
    obs5 = env.get_observation()
    assert obs5.payload["throttle_right"] == "IDLE_EARLY"


def test_missing_steps_scenario_loaded():
    env = MockEnvAdapter("mock_scenarios/missing_steps.json")
    obs = env.get_observation()
    assert obs.payload["battery"] == "ON"
    assert env.remaining() == 2
