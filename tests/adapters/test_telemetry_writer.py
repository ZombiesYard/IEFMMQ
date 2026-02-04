from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from adapters.event_store.telemetry_writer import TelemetryWriter
from core.types_v2 import TelemetryFrame


def _frame(seq: int) -> TelemetryFrame:
    return TelemetryFrame(seq=seq, t_wall=float(seq), source="derived", vars={"k": seq})


def _tmp_dir() -> Path:
    base = Path("tests/.tmp_telemetry")
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_writer_round_trip_jsonl() -> None:
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry.jsonl"
    with TelemetryWriter(path) as writer:
        writer.append(_frame(1))
        writer.append(_frame(2))

    frames = TelemetryWriter.load(path)
    assert len(frames) == 2
    assert frames[0]["seq"] == 1
    assert frames[1]["seq"] == 2


def test_writer_round_trip_gzip() -> None:
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry.jsonl.gz"
    with TelemetryWriter(path, compression="gzip") as writer:
        writer.append(_frame(1))

    frames = TelemetryWriter.load(path)
    assert len(frames) == 1
    assert frames[0]["vars"]["k"] == 1
