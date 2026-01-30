from pathlib import Path
from time import sleep
from uuid import uuid4

from core.event_store import JsonlEventStore
from simtutor.runner import replay_log, run_simulation


BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = BASE_DIR / "mock_scenarios"
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
TEMP_DIR = Path("tests/.tmp_logs")


def write_temp_log(events) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMP_DIR / f"log_{uuid4().hex}.jsonl"
    with JsonlEventStore(path, mode="w") as store:
        for ev in events:
            store.append(ev)
    return path


def cleanup_path(path: Path) -> None:
    for _ in range(3):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            sleep(0.01)


def test_run_writes_log_and_replay_ok():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TEMP_DIR / "test_run.jsonl"
    produced = run_simulation(
        pack_path=str(PACK_PATH),
        scenario_path=str(SCENARIOS_DIR / "correct_process.json"),
        log_path=str(log_path),
    )
    try:
        assert produced.exists()
        ok, msg = replay_log(str(produced), str(PACK_PATH))
        assert ok, msg
    finally:
        cleanup_path(produced)


def test_replay_log_missing_step_id():
    log_path = write_temp_log([{"kind": "step_activated", "payload": {}}])
    try:
        ok, msg = replay_log(str(log_path), str(PACK_PATH))
        assert not ok
        assert "missing step_id" in msg
    finally:
        cleanup_path(log_path)


def test_replay_log_unknown_step_id():
    log_path = write_temp_log([{"kind": "step_activated", "payload": {"step_id": "S99"}}])
    try:
        ok, msg = replay_log(str(log_path), str(PACK_PATH))
        assert not ok
        assert "unknown step_id" in msg
    finally:
        cleanup_path(log_path)


def test_replay_log_out_of_order_activation():
    log_path = write_temp_log([{"kind": "step_activated", "payload": {"step_id": "S02"}}])
    try:
        ok, msg = replay_log(str(log_path), str(PACK_PATH))
        assert not ok
        assert "out of order" in msg
    finally:
        cleanup_path(log_path)


def test_replay_log_ends_with_active_step():
    log_path = write_temp_log([{"kind": "step_activated", "payload": {"step_id": "S01"}}])
    try:
        ok, msg = replay_log(str(log_path), str(PACK_PATH))
        assert not ok
        assert "ended with active" in msg
    finally:
        cleanup_path(log_path)


def test_replay_log_malformed_file():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TEMP_DIR / f"bad_{uuid4().hex}.jsonl"
    log_path.write_text("{not-json}\n", encoding="utf-8")
    try:
        ok, msg = replay_log(str(log_path), str(PACK_PATH))
        assert not ok
        assert "failed to load log" in msg
    finally:
        cleanup_path(log_path)
