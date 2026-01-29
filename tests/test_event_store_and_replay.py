from pathlib import Path

from simtutor.runner import replay_log, run_simulation


def test_run_writes_log_and_replay_ok():
    log_path = Path("logs/test_run.jsonl")
    produced = run_simulation(
        pack_path="packs/fa18c_startup/pack.yaml",
        scenario_path="mock_scenarios/correct_process.json",
        log_path=str(log_path),
    )
    assert produced.exists()
    ok, msg = replay_log(str(produced), "packs/fa18c_startup/pack.yaml")
    assert ok, msg
