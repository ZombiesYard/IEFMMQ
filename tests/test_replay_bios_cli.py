from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from adapters.model_stub import ModelStub
from adapters.vision_frames import build_frame_channel_dir, build_frame_filename, build_frame_id, build_frame_manifest_path
from adapters.vision_prompting import load_vision_layout, solve_layout_geometry
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
            "ENGINE_CRANK_SW": 1,
        },
        "delta": {"APU_CONTROL_SW": apu_switch},
    }


def _write_replay(path: Path, frames: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(frame, ensure_ascii=False) + "\n" for frame in frames)
    path.write_text(text, encoding="utf-8")


def _default_policy_path() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge_source_policy.yaml"


def _make_source_frame(path: Path, *, width: int = 1920, height: int = 1080) -> None:
    layout = load_vision_layout()
    solved = solve_layout_geometry(layout, output_width=width, output_height=height)
    strip = solved["strip_rect"]

    image = Image.new("RGB", (width, height), (200, 32, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (
            strip["x"],
            strip["y"],
            strip["x"] + strip["width"],
            strip["y"] + strip["height"],
        ),
        fill=(16, 20, 22),
    )
    for region in solved["regions"]:
        draw.rectangle(
            (
                region["x"],
                region["y"],
                region["x"] + region["width"],
                region["y"] + region["height"],
            ),
            fill=(28, 41, 48),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _append_manifest_entry(channel_dir: Path, entry: dict[str, object]) -> None:
    manifest_path = build_frame_manifest_path(channel_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_sidecar_frame(
    *,
    saved_games_dir: Path,
    session_id: str,
    capture_wall_ms: int,
    frame_seq: int,
) -> None:
    channel_dir = build_frame_channel_dir(
        saved_games_dir=saved_games_dir,
        session_id=session_id,
        channel="composite_panel",
    )
    frame_path = channel_dir / build_frame_filename(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq)
    _make_source_frame(frame_path)
    _append_manifest_entry(
        channel_dir,
        {
            "schema_version": "v2",
            "frame_id": build_frame_id(capture_wall_ms=capture_wall_ms, frame_seq=frame_seq),
            "capture_wall_ms": capture_wall_ms,
            "frame_seq": frame_seq,
            "channel": "composite_panel",
            "layout_id": "fa18c_composite_panel_v2",
            "image_path": str(frame_path.resolve()),
            "width": 1920,
            "height": 1080,
            "source_session_id": session_id,
        },
    )


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
                    "overlay": {
                        "targets": ["apu_switch"],
                        "evidence": [
                            {
                                "target": "apu_switch",
                                "type": "delta",
                                "ref": "RECENT_UI_TARGETS.apu_switch",
                                "quote": "Recent delta shows APU switch activity.",
                            }
                        ],
                    },
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
    trigger_instances: list["FakeUdpHelpTrigger"] = []

    class FakeUdpHelpTrigger:
        def __init__(self, host: str = "127.0.0.1", port: int = 7794, timeout: float = 0.2) -> None:
            self.host = host
            self.port = port
            self.timeout = timeout
            self._pending = False
            trigger_instances.append(self)

        @property
        def bound_port(self) -> int:
            return 1

        def start(self) -> None:
            return

        def poll(self) -> bool:
            if self._pending:
                self._pending = False
                return True
            return False

        def close(self) -> None:
            return

    monkeypatch.setattr("simtutor.__main__._build_replay_model_from_args", lambda _args: OverlayModel())
    monkeypatch.setattr("live_dcs.UdpHelpTrigger", FakeUdpHelpTrigger)
    monkeypatch.setattr(
        "adapters.action_executor.DcsOverlaySender",
        lambda **_kwargs: type("DummySender", (), {"close": lambda self: None})(),
    )
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
            "1",
            "--lang",
            "en",
        ],
    )

    def _send_help() -> None:
        deadline = time.time() + 2.0
        while not trigger_instances and time.time() < deadline:
            time.sleep(0.01)
        assert trigger_instances
        trigger_instances[0]._pending = True

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


def test_cli_replay_bios_closes_source_when_store_enter_fails(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_store_fail.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    output_path = tmp_path / "replay_fail.jsonl"

    close_state = {"closed": False}

    class CloseTrackingReceiver:
        def __init__(self, *_args, **_kwargs) -> None:
            return

        def close(self) -> None:
            close_state["closed"] = True

    class FailingStore:
        def __init__(self, *_args, **_kwargs) -> None:
            return

        def __enter__(self):
            raise RuntimeError("store enter failed")

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("live_dcs.ReplayBiosReceiver", CloseTrackingReceiver)
    monkeypatch.setattr("core.event_store.JsonlEventStore", FailingStore)
    monkeypatch.setattr("simtutor.__main__._build_replay_model_from_args", lambda _args: ModelStub(mode="A"))
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
        ],
    )

    code = main()
    assert code == 1
    assert close_state["closed"] is True


def test_cli_replay_bios_log_raw_llm_text_can_disable_env_default(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_log_raw_env_on.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, bool] = {}

    def _fake_run_replay(args):
        captured["log_raw_llm_text"] = bool(args.log_raw_llm_text)
        return 0

    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "1")
    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
            "--no-log-raw-llm-text",
        ],
    )

    code = main()
    assert code == 0
    assert captured["log_raw_llm_text"] is False


def test_cli_replay_bios_log_raw_llm_text_can_enable_when_env_default_off(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_log_raw_env_off.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, bool] = {}

    def _fake_run_replay(args):
        captured["log_raw_llm_text"] = bool(args.log_raw_llm_text)
        return 0

    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "0")
    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
            "--log-raw-llm-text",
        ],
    )

    code = main()
    assert code == 0
    assert captured["log_raw_llm_text"] is True


def test_cli_replay_bios_model_max_tokens_reads_env_default(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_model_max_tokens_env.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, int] = {}

    def _fake_run_replay(args):
        captured["model_max_tokens"] = int(args.model_max_tokens)
        return 0

    monkeypatch.setenv("SIMTUTOR_MODEL_MAX_TOKENS", "256")
    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
        ],
    )

    code = main()
    assert code == 0
    assert captured["model_max_tokens"] == 256


def test_cli_replay_bios_model_max_tokens_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_model_max_tokens_arg.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, int] = {}

    def _fake_run_replay(args):
        captured["model_max_tokens"] = int(args.model_max_tokens)
        return 0

    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
            "--model-max-tokens",
            "128",
        ],
    )

    code = main()
    assert code == 0
    assert captured["model_max_tokens"] == 128


def test_cli_replay_bios_model_max_tokens_invalid_env_falls_back_to_zero(monkeypatch, tmp_path: Path, caplog) -> None:
    replay_path = tmp_path / "bios_cli_model_max_tokens_invalid_env.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, int] = {}

    def _fake_run_replay(args):
        captured["model_max_tokens"] = int(args.model_max_tokens)
        return 0

    monkeypatch.setenv("SIMTUTOR_MODEL_MAX_TOKENS", "")
    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
        ],
    )

    code = main()
    assert code == 0
    assert captured["model_max_tokens"] == 0
    assert "SIMTUTOR_MODEL_MAX_TOKENS" in caplog.text


def test_cli_replay_bios_model_max_tokens_rejects_negative_value(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_model_max_tokens_negative.jsonl"
    replay_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
            "--model-max-tokens",
            "-1",
        ],
    )

    with pytest.raises(SystemExit):
        main()


def test_cli_replay_bios_log_raw_llm_text_reads_true_false_env(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_log_raw_true_false.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, bool] = {}

    def _fake_run_replay(args):
        captured["log_raw_llm_text"] = bool(args.log_raw_llm_text)
        return 0

    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)

    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "true")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
        ],
    )
    code = main()
    assert code == 0
    assert captured["log_raw_llm_text"] is True

    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "false")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
        ],
    )
    code = main()
    assert code == 0
    assert captured["log_raw_llm_text"] is False


def test_cli_replay_bios_log_raw_llm_text_invalid_env_falls_back_false_with_warning(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    replay_path = tmp_path / "bios_cli_log_raw_invalid.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, bool] = {}

    def _fake_run_replay(args):
        captured["log_raw_llm_text"] = bool(args.log_raw_llm_text)
        return 0

    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "abc")
    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
        ],
    )

    with caplog.at_level("WARNING"):
        code = main()
    assert code == 0
    assert captured["log_raw_llm_text"] is False
    assert any(
        "SIMTUTOR_LOG_RAW_LLM_TEXT" in record.message and "Invalid boolean environment value" in record.message
        for record in caplog.records
    )


def test_cli_replay_bios_rejects_missing_policy_in_cold_start_production(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    replay_path = tmp_path / "bios_cli_cold_start_missing_policy.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    output_path = tmp_path / "replay_missing_policy.jsonl"
    missing_policy_path = tmp_path / "dir with spaces" / "missing_knowledge_source_policy.yaml"

    monkeypatch.setattr(
        "adapters.action_executor.DcsOverlaySender",
        lambda **_kwargs: type("DummySender", (), {"close": lambda self: None})(),
    )
    monkeypatch.setattr("simtutor.__main__._build_replay_model_from_args", lambda _args: ModelStub(mode="A"))
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
            "--cold-start-production",
            "--knowledge-source-policy",
            str(missing_policy_path),
            "--max-frames",
            "1",
        ],
    )

    code = main()
    out = capsys.readouterr().out
    assert code == 1
    assert "cold-start production requires valid knowledge source policy" in out
    assert "missing_knowledge_source_policy.yaml" in out
    assert str(missing_policy_path) not in out


def test_cli_replay_bios_cold_start_production_prints_policy_summary(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    replay_path = tmp_path / "bios_cli_cold_start_ok.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    output_path = tmp_path / "replay_cold_start_ok.jsonl"

    monkeypatch.setattr(
        "adapters.action_executor.DcsOverlaySender",
        lambda **_kwargs: type("DummySender", (), {"close": lambda self: None})(),
    )
    monkeypatch.setattr("simtutor.__main__._build_replay_model_from_args", lambda _args: ModelStub(mode="A"))
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
            "--cold-start-production",
            "--knowledge-source-policy",
            str(_default_policy_path()),
            "--max-frames",
            "1",
        ],
    )

    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "当前仅使用 cold-start 白名单块" in out


def test_cli_replay_bios_cold_start_production_reads_env_default(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_cold_start_env_default.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, bool] = {}

    def _fake_run_replay(args):
        captured["cold_start_production"] = bool(args.cold_start_production)
        return 0

    monkeypatch.setenv("SIMTUTOR_COLD_START_PRODUCTION", "true")
    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
        ],
    )

    code = main()
    assert code == 0
    assert captured["cold_start_production"] is True


def test_cli_replay_bios_scenario_profile_defaults_airfield(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_scenario_profile_default.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, str] = {}

    def _fake_run_replay(args):
        captured["scenario_profile"] = str(args.scenario_profile)
        return 0

    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
        ],
    )

    code = main()
    assert code == 0
    assert captured["scenario_profile"] == "airfield"


def test_cli_replay_bios_scenario_profile_accepts_carrier(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_scenario_profile_carrier.jsonl"
    replay_path.write_text("", encoding="utf-8")
    captured: dict[str, str] = {}

    def _fake_run_replay(args):
        captured["scenario_profile"] = str(args.scenario_profile)
        return 0

    monkeypatch.setattr("simtutor.__main__._run_replay_bios", _fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-bios",
            "--input",
            str(replay_path),
            "--scenario-profile",
            "carrier",
        ],
    )

    code = main()
    assert code == 0
    assert captured["scenario_profile"] == "carrier"


def test_cli_replay_bios_consumes_sidecar_frames_and_selects_stable_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_cli_with_sidecar.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 1772872445.0, apu_switch=0)])
    output_path = tmp_path / "replay_with_sidecar.jsonl"
    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    _write_sidecar_frame(
        saved_games_dir=saved_games_dir,
        session_id="sess-replay",
        capture_wall_ms=1772872444950,
        frame_seq=122,
    )
    _write_sidecar_frame(
        saved_games_dir=saved_games_dir,
        session_id="sess-replay",
        capture_wall_ms=1772872445010,
        frame_seq=123,
    )
    captured: dict[str, Any] = {}

    class RecordingVisionModel:
        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            captured["vision"] = dict(request.context["vision"]) if request else {}
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Turn on APU.",
                actions=[],
                explanations=["Turn on APU."],
                metadata={"provider": "mock_qwen"},
            )

    class DummySender:
        def close(self) -> None:
            return

    monkeypatch.setattr("adapters.action_executor.DcsOverlaySender", lambda **_kwargs: DummySender())
    monkeypatch.setattr("simtutor.__main__._build_replay_model_from_args", lambda _args: RecordingVisionModel())
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
            "0",
            "--max-frames",
            "1",
            "--auto-help-once",
            "--vision-saved-games-dir",
            str(saved_games_dir),
            "--vision-session-id",
            "sess-replay",
        ],
    )

    code = main()

    assert code == 0
    assert captured["vision"]["status"] == "available"
    assert captured["vision"]["frame_ids"] == ["1772872444950_000122", "1772872445010_000123"]


def test_cli_replay_bios_without_sidecar_marks_vision_unavailable(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_cli_without_sidecar.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    output_path = tmp_path / "replay_without_sidecar.jsonl"
    captured: dict[str, Any] = {}

    class RecordingVisionModel:
        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            captured["vision"] = dict(request.context["vision"]) if request else {}
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Turn on APU.",
                actions=[],
                explanations=["Turn on APU."],
                metadata={"provider": "mock_qwen"},
            )

    class DummySender:
        def close(self) -> None:
            return

    monkeypatch.setattr("adapters.action_executor.DcsOverlaySender", lambda **_kwargs: DummySender())
    monkeypatch.setattr("simtutor.__main__._build_replay_model_from_args", lambda _args: RecordingVisionModel())
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
            "0",
            "--max-frames",
            "1",
            "--auto-help-once",
        ],
    )

    code = main()

    assert code == 0
    assert captured["vision"]["status"] == "vision_unavailable"
    assert captured["vision"]["sync_miss_reason"] == "vision_port_unconfigured"
