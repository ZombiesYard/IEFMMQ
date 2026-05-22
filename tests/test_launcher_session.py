from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from simtutor.launcher_processes import ProcessSnapshot, ProcessState
from simtutor.launcher_session import (
    LauncherExperimentSession,
    LauncherSessionError,
    build_launcher_session_plan,
    format_dry_run_plan,
    resolve_unique_live_log_path,
)
from simtutor.launcher_settings import LauncherSettings
from simtutor.launcher_tunnel import LauncherTunnelError


@dataclass(frozen=True)
class FakeEntry:
    status: str
    code: str
    message: str


class FakeReport:
    def __init__(self, ok: bool = True, text: str = "ready") -> None:
        self.ok = ok
        self.text = text
        self.entries = (FakeEntry("pass" if ok else "error", "fake", text),)

    def to_text(self) -> str:
        return self.text


class FakeProcess:
    def __init__(
        self,
        *,
        name: str,
        command: list[str],
        events: list[str],
        stop_returncode: int = 0,
        initial_returncode: int | None = None,
        **_kwargs: object,
    ) -> None:
        self.name = name
        self.command = tuple(command)
        self.events = events
        self.stop_returncode = stop_returncode
        self.initial_returncode = initial_returncode
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.events.append(f"start:{self.name}")
        self.started = True
        if self.name == "live-dcs":
            _write_fake_live_log(self.command)

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self.events.append(f"stop:{self.name}")
        self.stopped = True
        if self.initial_returncode is not None:
            self.stop_returncode = self.initial_returncode

    def wait_for_output(self, *, timeout_s: float = 1.0) -> None:
        self.events.append(f"wait:{self.name}")

    def snapshot(self) -> ProcessSnapshot:
        state = ProcessState.STOPPED if self.stopped or self.initial_returncode is not None else ProcessState.RUNNING
        returncode = self.stop_returncode if self.stopped else self.initial_returncode
        return ProcessSnapshot(
            name=self.name,
            command=self.command,
            state=state,
            returncode=returncode,
            log_lines=(f"{self.name} log",),
        )


class FakeTunnel:
    def __init__(self, settings: LauncherSettings, events: list[str], *, fail_start: bool = False) -> None:
        self.settings = settings
        self.events = events
        self.fail_start = fail_start

    def start(self) -> None:
        self.events.append("start:tunnel")
        if self.fail_start:
            raise RuntimeError("tunnel failed")

    def stop(self) -> None:
        self.events.append("stop:tunnel")

    def snapshot(self):
        return type("TunnelSnapshot", (), {"running": True, "returncode": None, "command": ("ssh",)})()


class OccupiedPortTunnel(FakeTunnel):
    def start(self) -> None:
        self.events.append("start:tunnel")
        raise LauncherTunnelError("local TCP port 16324 is already occupied")


def _settings(tmp_path: Path, *, model_profile_mode: str = "remote_direct") -> LauncherSettings:
    saved_games = tmp_path / "Saved Games" / "DCS"
    logs = tmp_path / "logs"
    saved_games.mkdir(parents=True)
    logs.mkdir()
    return LauncherSettings(
        saved_games_path=str(saved_games),
        dcs_variant="DCS",
        monitor_mode="fa18c_composite_panel_v2",
        language="zh",
        scenario_profile="airfield",
        output_log_directory=str(logs),
        participant_id="P01",
        condition="with_tutor",
        trial_id="T01",
        study_id="study-alpha",
        participant_group="novice",
        experimenter_id="E01",
        questionnaire_ref="questionnaires/P01.yaml",
        recording_ref="recordings/P01_T01.mp4",
        model_provider="openai_compat",
        model_profile_mode=model_profile_mode,
        model_base_url="http://127.0.0.1:16324",
        text_model_name="simtutor-base",
        vision_model_name="simtutor-vision",
        model_timeout_s=60,
        model_enable_multimodal=True,
        max_overlay_targets=4,
        dcs_aircraft="FA-18C_hornet",
        dcs_mission="cold_start.miz",
        vr_setup="none",
        monitor_setup="fa18c_composite_panel_v2",
        prompt_version="launcher-v0.4",
    )


def _write_fake_live_log(command: tuple[str, ...]) -> None:
    if "--output" not in command:
        return
    output = Path(command[command.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}\n", encoding="utf-8")


def test_build_launcher_session_plan_uses_production_live_dcs_flags(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    log_path = tmp_path / "logs" / "live.jsonl"

    plan = build_launcher_session_plan(settings, log_path=log_path, session_id="sess-P01-T01", python_executable="python")

    assert plan.live_log_path == log_path
    assert plan.sidecar_command[:3] == ("python", "-u", "tools/capture_vision_sidecar.py")
    assert plan.live_dcs_command[:4] == ("python", "-m", "simtutor", "live-dcs")
    assert "--bios-source" in plan.live_dcs_command
    assert "raw" in plan.live_dcs_command
    assert "--global-help-hotkey" in plan.live_dcs_command
    assert "X1" in plan.live_dcs_command
    assert "--max-overlay-targets" in plan.live_dcs_command
    assert "4" in plan.live_dcs_command
    assert "--model-enable-multimodal" in plan.live_dcs_command
    assert "--log-raw-llm-text" in plan.live_dcs_command
    assert "--print-model-io" in plan.live_dcs_command
    assert "--help-udp-port" in plan.live_dcs_command
    assert "7792" in plan.live_dcs_command
    assert "--no-cold-start-production" in plan.live_dcs_command
    assert plan.export_command[:4] == ("python", "-m", "simtutor", "experiment-export")
    assert "--strict" in plan.export_command
    assert "--study-id" in plan.export_command
    assert "--recording-ref" in plan.export_command
    assert plan.analyze_command[:4] == ("python", "-m", "simtutor", "experiment-analyze")


def test_dry_run_formats_exact_commands_and_unique_log_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = tmp_path / "logs" / "live_dcs_P01_T01_20260521_153000.jsonl"
    first.write_text("", encoding="utf-8")

    log_path = resolve_unique_live_log_path(settings, timestamp="20260521_153000")
    plan = build_launcher_session_plan(settings, log_path=log_path, session_id="sess-P01-T01", python_executable="python")
    text = format_dry_run_plan(plan)

    assert log_path.name == "live_dcs_P01_T01_20260521_153000_2.jsonl"
    assert "python -m simtutor live-dcs" in text
    assert str(log_path) in text
    assert "python -m simtutor experiment-export" in text
    assert "python -m simtutor experiment-analyze" in text


def test_session_starts_preflight_tunnel_model_sidecar_then_live_and_stops_in_reverse(tmp_path: Path) -> None:
    settings = _settings(tmp_path, model_profile_mode="remote_tunnel")
    settings.ssh_user = "yz"
    settings.ssh_host = "example.invalid"
    events: list[str] = []

    def process_factory(*, name: str, command: list[str], **kwargs: object) -> FakeProcess:
        return FakeProcess(name=name, command=command, events=events, **kwargs)

    session = LauncherExperimentSession(
        settings,
        process_factory=process_factory,
        tunnel_factory=lambda received: FakeTunnel(received, events),
        preflight_runner=lambda received: events.append("preflight") or FakeReport(ok=True),
        model_validator=lambda received: events.append("model") or FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    result = session.start()
    session.stop(run_post_run=False)

    assert result.live_log_path.name == "live_dcs_P01_T01_20260521_153000.jsonl"
    assert events == [
        "preflight",
        "start:tunnel",
        "model",
        "start:vision-sidecar",
        "start:live-dcs",
        "stop:live-dcs",
        "stop:vision-sidecar",
        "stop:tunnel",
    ]


def test_session_reuses_existing_tunnel_when_local_forward_port_is_occupied(tmp_path: Path) -> None:
    settings = _settings(tmp_path, model_profile_mode="remote_tunnel")
    settings.ssh_user = "yz"
    settings.ssh_host = "example.invalid"
    events: list[str] = []

    def process_factory(*, name: str, command: list[str], **kwargs: object) -> FakeProcess:
        return FakeProcess(name=name, command=command, events=events, **kwargs)

    session = LauncherExperimentSession(
        settings,
        process_factory=process_factory,
        tunnel_factory=lambda received: OccupiedPortTunnel(received, events),
        preflight_runner=lambda received: events.append("preflight") or FakeReport(ok=True),
        model_validator=lambda received: events.append("model") or FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    session.start()
    session.stop(run_post_run=False)

    assert events == [
        "preflight",
        "start:tunnel",
        "model",
        "start:vision-sidecar",
        "start:live-dcs",
        "stop:live-dcs",
        "stop:vision-sidecar",
    ]


def test_session_stop_runs_export_then_analysis_after_live_processes_stop(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    events: list[str] = []

    def process_factory(*, name: str, command: list[str], **kwargs: object) -> FakeProcess:
        return FakeProcess(name=name, command=command, events=events, **kwargs)

    session = LauncherExperimentSession(
        settings,
        process_factory=process_factory,
        preflight_runner=lambda _settings: FakeReport(ok=True),
        model_validator=lambda _settings: FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    session.start()
    session.stop(run_post_run=True)

    assert events == [
        "start:vision-sidecar",
        "start:live-dcs",
        "stop:live-dcs",
        "stop:vision-sidecar",
        "start:experiment-export",
        "wait:experiment-export",
        "start:experiment-analyze",
        "wait:experiment-analyze",
    ]


def test_session_stop_allows_user_terminated_live_process_when_log_exists(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    events: list[str] = []

    def process_factory(*, name: str, command: list[str], **kwargs: object) -> FakeProcess:
        return FakeProcess(
            name=name,
            command=command,
            events=events,
            stop_returncode=(-15 if name == "live-dcs" else 0),
            **kwargs,
        )

    session = LauncherExperimentSession(
        settings,
        process_factory=process_factory,
        preflight_runner=lambda _settings: FakeReport(ok=True),
        model_validator=lambda _settings: FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    session.start()
    session.stop(run_post_run=True)

    assert "start:experiment-export" in events


def test_session_rejects_repeated_start_while_live_process_is_running(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    events: list[str] = []
    session = LauncherExperimentSession(
        settings,
        process_factory=lambda **kwargs: FakeProcess(events=events, **kwargs),
        preflight_runner=lambda _settings: FakeReport(ok=True),
        model_validator=lambda _settings: FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    session.start()

    with pytest.raises(LauncherSessionError, match="already running"):
        session.start()

    assert events.count("start:live-dcs") == 1


def test_post_run_rejects_live_process_that_is_still_running(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    events: list[str] = []
    session = LauncherExperimentSession(
        settings,
        process_factory=lambda **kwargs: FakeProcess(events=events, **kwargs),
        preflight_runner=lambda _settings: FakeReport(ok=True),
        model_validator=lambda _settings: FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    session.start()

    with pytest.raises(LauncherSessionError, match="still running"):
        session.run_post_run()

    assert "start:experiment-export" not in events


def test_preflight_failure_short_circuits_before_live_dcs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    events: list[str] = []

    def process_factory(*, name: str, command: list[str], **kwargs: object) -> FakeProcess:
        return FakeProcess(name=name, command=command, events=events, **kwargs)

    session = LauncherExperimentSession(
        settings,
        process_factory=process_factory,
        preflight_runner=lambda _settings: FakeReport(ok=False, text="missing hook"),
        model_validator=lambda _settings: events.append("model") or FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    with pytest.raises(LauncherSessionError, match="preflight failed"):
        session.start()

    assert events == []


def test_missing_strict_export_metadata_short_circuits_before_preflight(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.study_id = ""
    events: list[str] = []

    session = LauncherExperimentSession(
        settings,
        process_factory=lambda **kwargs: FakeProcess(events=events, **kwargs),
        preflight_runner=lambda _settings: events.append("preflight") or FakeReport(ok=True),
        model_validator=lambda _settings: events.append("model") or FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    with pytest.raises(LauncherSessionError, match="study_id"):
        session.start()

    assert events == []


def test_model_failure_stops_launcher_owned_tunnel_before_live_dcs(tmp_path: Path) -> None:
    settings = _settings(tmp_path, model_profile_mode="remote_tunnel")
    settings.ssh_user = "yz"
    settings.ssh_host = "example.invalid"
    events: list[str] = []

    session = LauncherExperimentSession(
        settings,
        process_factory=lambda **kwargs: FakeProcess(events=events, **kwargs),
        tunnel_factory=lambda received: FakeTunnel(received, events),
        preflight_runner=lambda _settings: FakeReport(ok=True),
        model_validator=lambda _settings: FakeReport(ok=False, text="missing model"),
        timestamp=lambda: "20260521_153000",
    )

    with pytest.raises(LauncherSessionError, match="model validation failed"):
        session.start()

    assert events == ["start:tunnel", "stop:tunnel"]


def test_stop_blocks_post_run_when_live_log_is_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    events: list[str] = []

    class NoLogProcess(FakeProcess):
        def start(self) -> None:
            self.events.append(f"start:{self.name}")
            self.started = True

    session = LauncherExperimentSession(
        settings,
        process_factory=lambda **kwargs: NoLogProcess(events=events, **kwargs),
        preflight_runner=lambda _settings: FakeReport(ok=True),
        model_validator=lambda _settings: FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    session.start()

    with pytest.raises(LauncherSessionError, match="live-dcs log was not written"):
        session.stop(run_post_run=True)

    assert "start:experiment-export" not in events


def test_stop_blocks_post_run_when_live_process_already_failed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    events: list[str] = []

    def process_factory(*, name: str, command: list[str], **kwargs: object) -> FakeProcess:
        return FakeProcess(
            name=name,
            command=command,
            events=events,
            initial_returncode=(1 if name == "live-dcs" else None),
            **kwargs,
        )

    session = LauncherExperimentSession(
        settings,
        process_factory=process_factory,
        preflight_runner=lambda _settings: FakeReport(ok=True),
        model_validator=lambda _settings: FakeReport(ok=True),
        timestamp=lambda: "20260521_153000",
    )

    session.start()

    with pytest.raises(LauncherSessionError, match="live-dcs exited with code 1"):
        session.stop(run_post_run=True)

    assert "start:experiment-export" not in events
