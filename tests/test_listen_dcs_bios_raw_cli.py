from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import tools.listen_dcs_bios_raw as cli


def test_continuous_mode_stops_at_duration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeObservation:
        payload = {"schema_version": "v2", "seq": 1, "bios": {"TEST": 1}, "delta": {"TEST": 1}}

    class FakeReceiver:
        def __init__(self, *args, **kwargs) -> None:
            self._returned = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def get_observation(self):
            if self._returned:
                return None
            self._returned = True
            return FakeObservation()

    ticks = [0.0, 0.1, 1.1]

    def fake_time() -> float:
        return ticks.pop(0) if ticks else 1.1

    out = tmp_path / "bios_1s.jsonl"
    monkeypatch.setattr(cli, "DcsBiosRawReceiver", FakeReceiver)
    monkeypatch.setattr(cli.time, "time", fake_time)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "listen_dcs_bios_raw.py",
            "--aircraft",
            "FA-18C_hornet",
            "--duration",
            "1",
            "--output",
            str(out),
        ],
    )

    rc = cli.main()
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["bios"]["TEST"] == 1


def test_rejects_once_with_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "listen_dcs_bios_raw.py",
            "--aircraft",
            "FA-18C_hornet",
            "--once",
            "--duration",
            "1",
        ],
    )
    with pytest.raises(SystemExit, match="cannot be combined"):
        cli.main()
