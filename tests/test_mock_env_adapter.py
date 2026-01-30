import json
from pathlib import Path
from uuid import uuid4

import pytest

from adapters.mock_env import MockEnvAdapter


SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "mock_scenarios"
TEMP_DIR = Path("tests/.tmp_mock_env")


def write_temp_scenario(data) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMP_DIR / f"scenario_{uuid4().hex}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_mock_env_sequence_advances_and_finishes():
    env = MockEnvAdapter(str(SCENARIOS_DIR / "correct_process.json"))
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
    env = MockEnvAdapter(str(SCENARIOS_DIR / "premature_acceleration.json"))
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
    env = MockEnvAdapter(str(SCENARIOS_DIR / "missing_steps.json"))
    obs = env.get_observation()
    assert obs.payload["battery"] == "ON"
    assert env.remaining() == 2


def test_missing_scenario_path_raises():
    missing = SCENARIOS_DIR / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        MockEnvAdapter(str(missing))


def test_scenario_not_a_list_raises():
    path = write_temp_scenario({"not": "a list"})
    with pytest.raises(ValueError, match="Scenario must be a list"):
        MockEnvAdapter(str(path))


def test_scenario_entry_not_dict_raises():
    path = write_temp_scenario([{"payload": {}}, "bad"])
    with pytest.raises(ValueError, match="entry at index 1"):
        MockEnvAdapter(str(path))


def test_scenario_path_is_directory_raises():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="not a file"):
        MockEnvAdapter(str(TEMP_DIR))
