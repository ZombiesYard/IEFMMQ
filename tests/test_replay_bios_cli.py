from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from core.event_store import JsonlEventStore
from core.types import Observation, TutorResponse
from simtutor.__main__ import main


def _bios_frame(seq: int, t_wall: float, *, apu_switch: int) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "seq": seq,
        "t_wall": t_wall,
        "aircraft": "FA-18C_hornet",
        "bios": {
            "BATTERY_SW": 2,
            "L_GEN_SW": 1,
            "R_GEN_SW": 1,
            "APU_CONTROL_SW": apu_switch,
            "APU_READY_LT": 0,
            "ENGINE_CRANK_SW": 0,
        },
        "delta": {"APU_CONTROL_SW": apu_switch},
    }


def _write_replay(path: Path, frames: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(frame, ensure_ascii=False) + "\n" for frame in frames)
    path.write_text(text, encoding="utf-8")


def _reserve_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


class OverlayModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id if request else None,
            message="Turn on APU.",
            actions=[],
            explanations=["Turn on APU."],
            metadata={
                "provider": "mock_qwen",
                "help_response": {
                    "diagnosis": {"step_id": "S02", "error_category": "OM"},
                    "next": {"step_id": "S03"},
                    "overlay": {"targets": ["apu_switch"]},
                    "explanations": ["Turn on APU."],
                    "confidence": 0.9,
                },
            },
        )


def test_cli_replay_bios_udp_help_generates_help_cycle_and_dry_run_overlay(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_replay.jsonl"
    _write_replay(
        replay_path,
        [
            _bios_frame(1, 10.0, apu_switch=0),
            _bios_frame(2, 10.4, apu_switch=0),
        ],
    )
    output_path = tmp_path / "replay_events.jsonl"
    udp_port = _reserve_udp_port()

    monkeypatch.setattr("simtutor.__main__._build_replay_model_from_args", lambda _args: OverlayModel())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
            "--output",
            str(output_path),
            "--speed",
            "1.0",
            "--max-frames",
            "2",
            "--help-udp-port",
            str(udp_port),
            "--lang",
            "en",
        ],
    )

    def _send_help() -> None:
        time.sleep(0.05)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(b"help", ("127.0.0.1", udp_port))
        finally:
            sock.close()

    sender = threading.Thread(target=_send_help, daemon=True)
    sender.start()
    code = main()
    sender.join(timeout=1.0)

    assert code == 0
    events = JsonlEventStore.load(output_path)
    kinds = [event.get("kind") or event.get("type") for event in events]
    assert "observation" in kinds
    assert "tutor_request" in kinds
    assert "tutor_response" in kinds
    assert "overlay_dry_run" in kinds
