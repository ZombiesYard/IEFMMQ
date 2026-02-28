from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import yaml

from core.types import Observation, TutorResponse
from live_dcs import LiveDcsTutorLoop, ReplayBiosReceiver, StdinHelpTrigger, _load_overlay_allowlist


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


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        self.calls.append({"observation": observation, "request": request})
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


class FailingModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        raise RuntimeError("model unavailable")


class RecordingExecutor:
    def __init__(self, *, include_dry_run: bool = False, dry_run: bool = False) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.include_dry_run = include_dry_run
        self.dry_run = dry_run

    def execute_actions(self, actions):
        actions_list = [dict(item) for item in actions if isinstance(item, dict)]
        self.calls.append(actions_list)
        report = {"executed": actions_list, "rejected": [], "dropped": [], "dry_run": []}
        if self.include_dry_run:
            report["dry_run"] = [{"target": item.get("target")} for item in actions_list]
        return report

    def close(self) -> None:  # pragma: no cover
        return


def _write_replay(path: Path, frames: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(frame, ensure_ascii=False) + "\n" for frame in frames)
    path.write_text(text, encoding="utf-8")


def _apu_element_id_from_ui_map() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    ui_map_path = repo_root / "packs" / "fa18c_startup" / "ui_map.yaml"
    ui_map = yaml.safe_load(ui_map_path.read_text(encoding="utf-8"))
    return str(ui_map["cockpit_elements"]["apu_switch"]["dcs_id"])


def test_live_loop_offline_single_sample_runs_help_response_and_actions(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_one.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        stats = loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert stats["frames"] == 1
    assert stats["help_cycles"] == 1
    assert stats["model_calls"] == 1
    assert len(model.calls) == 1
    request = model.calls[0]["request"]
    assert request is not None
    assert request.intent == "help"
    assert "candidate_steps" in request.context
    assert "recent_deltas" in request.context
    assert "recent_actions" in request.context
    assert "deterministic_step_hint" in request.context
    assert request.metadata["prompt_hash"]

    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    assert executor.calls[0][0]["type"] == "overlay"
    assert executor.calls[0][0]["target"] == "apu_switch"
    assert executor.calls[0][0]["element_id"] == _apu_element_id_from_ui_map()

    kinds = [event.kind for event in events]
    assert "observation" in kinds
    assert "tutor_request" in kinds
    assert "tutor_response" in kinds


def test_live_loop_reuses_cached_result_for_same_state_within_cooldown(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_two.jsonl"
    _write_replay(
        replay_path,
        [
            _bios_frame(1, 10.0, apu_switch=0),
            _bios_frame(2, 10.1, apu_switch=0),
        ],
    )

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=30.0,
        lang="en",
        event_sink=events.append,
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=1)
    finally:
        loop.close()

    assert stats["frames"] == 2
    assert stats["help_cycles"] == 2
    assert stats["model_calls"] == 1
    assert stats["cache_hits"] == 1
    assert len(model.calls) == 1
    assert len(executor.calls) == 2
    tutor_response_payloads = [event.payload for event in events if event.kind == "tutor_response"]
    assert len(tutor_response_payloads) == 2
    assert tutor_response_payloads[0]["response_id"] != tutor_response_payloads[1]["response_id"]


def test_live_loop_dry_run_overlay_prints_planned_actions(tmp_path: Path, capsys) -> None:
    replay_path = tmp_path / "bios_dry_run.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 11.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor(include_dry_run=True)
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        dry_run_overlay=True,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(executor.calls) == 0
    out = capsys.readouterr().out
    assert "dry_run_actions" in out
    assert "apu_switch" in out


def test_live_loop_dry_run_overlay_uses_executor_when_executor_is_dry_run(tmp_path: Path, capsys) -> None:
    replay_path = tmp_path / "bios_dry_run_exec.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 11.5, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor(include_dry_run=True, dry_run=True)
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        dry_run_overlay=True,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    out = capsys.readouterr().out
    assert "dry_run_actions" in out
    assert "apu_switch" in out


def test_replay_receiver_streams_and_only_parses_on_demand(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_streaming.jsonl"
    replay_path.write_text(
        json.dumps(_bios_frame(1, 12.0, apu_switch=0), ensure_ascii=False)
        + "\n"
        + "{bad-json}\n",
        encoding="utf-8",
    )

    source = ReplayBiosReceiver(replay_path)
    try:
        first = source.get_observation()
        assert first is not None
        assert first.payload["seq"] == 1

        try:
            source.get_observation()
            assert False, "expected ValueError for invalid second line"
        except ValueError as exc:
            assert "invalid JSON" in str(exc)
            assert source.is_exhausted is True
            assert source._fh.closed is True
    finally:
        source.close()


def test_replay_receiver_skips_non_mapping_json_values(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_non_mapping_values.jsonl"
    replay_path.write_text(
        "[]\n"
        + json.dumps(_bios_frame(7, 20.0, apu_switch=1), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    source = ReplayBiosReceiver(replay_path)
    try:
        obs = source.get_observation()
        assert obs is not None
        assert obs.payload["seq"] == 7
    finally:
        source.close()


def test_stdin_help_trigger_reader_does_not_enqueue_after_stop_set_during_input(
    monkeypatch,
) -> None:
    trigger = StdinHelpTrigger()

    def _fake_input() -> str:
        trigger._stop.set()
        return "help"

    monkeypatch.setattr(builtins, "input", _fake_input)
    trigger._reader()
    assert trigger.poll() is False


def test_load_overlay_allowlist_raises_when_pack_ui_targets_narrow_to_zero(tmp_path: Path) -> None:
    ui_map = tmp_path / "ui_map.yaml"
    pack = tmp_path / "pack.yaml"
    ui_map.write_text(
        "version: v1\n"
        "cockpit_elements:\n"
        "  apu_switch:\n"
        "    dcs_id: pnt_375\n",
        encoding="utf-8",
    )
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "ui_targets:\n"
        "  - not_in_ui_map\n",
        encoding="utf-8",
    )

    try:
        _load_overlay_allowlist(pack, ui_map)
        assert False, "expected ValueError when narrowed allowlist is empty"
    except ValueError as exc:
        assert "narrows overlay allowlist to zero valid targets" in str(exc)


def test_load_overlay_allowlist_uses_step_ui_targets_union_when_top_level_missing(tmp_path: Path) -> None:
    ui_map = tmp_path / "ui_map.yaml"
    pack = tmp_path / "pack.yaml"
    ui_map.write_text(
        "version: v1\n"
        "cockpit_elements:\n"
        "  apu_switch:\n"
        "    dcs_id: pnt_375\n"
        "  battery_switch:\n"
        "    dcs_id: pnt_404\n"
        "  fire_test_switch:\n"
        "    dcs_id: pnt_331\n",
        encoding="utf-8",
    )
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    ui_targets:\n"
        "      - battery_switch\n"
        "  - id: S02\n"
        "    ui_targets:\n"
        "      - apu_switch\n"
        "  - id: S03\n"
        "    ui_targets: []\n",
        encoding="utf-8",
    )

    allowlist = _load_overlay_allowlist(pack, ui_map)
    assert allowlist == ["apu_switch", "battery_switch"]


def test_live_loop_counts_model_attempt_when_model_raises(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_model_error.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 13.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = FailingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
    )
    try:
        stats = loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert stats["help_cycles"] == 1
    assert stats["model_calls"] == 1
    assert len(executor.calls) == 1
    assert executor.calls[0] == []


def test_live_loop_cache_key_ignores_numeric_churn_when_discrete_state_unchanged(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_numeric_churn.jsonl"
    frame1 = _bios_frame(1, 15.0, apu_switch=0)
    frame2 = _bios_frame(2, 15.1, apu_switch=0)
    frame1["bios"]["IFEI_RPM_R"] = "10"
    frame1["delta"]["IFEI_RPM_R"] = "10"
    frame2["bios"]["IFEI_RPM_R"] = "11"
    frame2["delta"]["IFEI_RPM_R"] = "11"
    _write_replay(replay_path, [frame1, frame2])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=30.0,
        lang="en",
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=1)
    finally:
        loop.close()

    assert stats["help_cycles"] == 2
    assert stats["model_calls"] == 1
    assert stats["cache_hits"] == 1


def test_live_loop_does_not_cache_error_response_and_retries_model_within_cooldown(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_retry_on_error.jsonl"
    _write_replay(
        replay_path,
        [
            _bios_frame(1, 16.0, apu_switch=0),
            _bios_frame(2, 16.1, apu_switch=0),
        ],
    )

    source = ReplayBiosReceiver(replay_path)
    model = FailingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=30.0,
        lang="en",
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=1)
    finally:
        loop.close()

    assert stats["help_cycles"] == 2
    assert stats["model_calls"] == 2
    assert stats["cache_hits"] == 0
