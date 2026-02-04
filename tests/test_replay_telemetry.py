from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from adapters.event_store.telemetry_writer import TelemetryWriter
from simtutor.runner import replay_telemetry


def _tmp_dir() -> Path:
    base = Path("tests/.tmp_telemetry")
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_replay_telemetry_monotonic() -> None:
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry.jsonl"
    with TelemetryWriter(path) as writer:
        writer.append({"schema_version": "v2", "seq": 1, "t_wall": 1.0, "source": "derived"})
        writer.append({"schema_version": "v2", "seq": 2, "t_wall": 2.0, "source": "derived"})
    ok, msg = replay_telemetry([str(path)])
    assert ok is True
    assert msg == "ok"


def test_replay_telemetry_detects_bad_seq() -> None:
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_bad.jsonl"
    with TelemetryWriter(path) as writer:
        writer.append({"schema_version": "v2", "seq": 2, "t_wall": 1.0, "source": "derived"})
        writer.append({"schema_version": "v2", "seq": 1, "t_wall": 2.0, "source": "derived"})
    ok, msg = replay_telemetry([str(path)])
    assert ok is False
    assert "non-monotonic seq" in msg
