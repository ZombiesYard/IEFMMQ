"""Experiment session orchestration for the SimTutor launcher."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shlex
import re
import sys
from typing import Any, Callable

from simtutor.launcher_model_check import validate_launcher_model_profile
from simtutor.launcher_preflight import run_preflight
from simtutor.launcher_processes import LauncherProcess, ProcessSnapshot, ProcessState
from simtutor.launcher_settings import LauncherSettings, validate_settings
from simtutor.launcher_tunnel import LauncherTunnel


class LauncherSessionError(RuntimeError):
    """Raised when an experiment session cannot advance safely."""


@dataclass(frozen=True)
class LauncherSessionPlan:
    session_id: str
    live_log_path: Path
    export_output_dir: Path
    analysis_output_dir: Path
    sidecar_command: tuple[str, ...]
    live_dcs_command: tuple[str, ...]
    export_command: tuple[str, ...]
    analyze_command: tuple[str, ...]


@dataclass(frozen=True)
class LauncherSessionStartResult:
    session_id: str
    live_log_path: Path
    plan: LauncherSessionPlan


ProcessFactory = Callable[..., Any]
TunnelFactory = Callable[[LauncherSettings], Any]
ReportRunner = Callable[[LauncherSettings], Any]
TimestampFactory = Callable[[], str]


def resolve_unique_live_log_path(settings: LauncherSettings, *, timestamp: str | None = None) -> Path:
    out_dir = Path(settings.output_log_directory).expanduser()
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    participant = _slug(settings.participant_id or "participant")
    trial = _slug(settings.trial_id or "trial")
    stem = f"live_dcs_{participant}_{trial}_{stamp}"
    for suffix in ("", *[f"_{index}" for index in range(2, 1000)]):
        candidate = out_dir / f"{stem}{suffix}.jsonl"
        if not candidate.exists():
            return candidate
    raise LauncherSessionError(f"could not resolve unique live log path in {out_dir}")


def build_launcher_session_plan(
    settings: LauncherSettings,
    *,
    log_path: str | Path,
    session_id: str,
    python_executable: str | None = None,
) -> LauncherSessionPlan:
    executable = python_executable or sys.executable
    saved_games_dir = Path(settings.saved_games_path).expanduser()
    live_log_path = Path(log_path).expanduser()
    export_output_dir = Path(settings.export_output_directory).expanduser()
    analysis_output_dir = Path(settings.analysis_output_directory).expanduser()

    sidecar_command = [
        executable,
        "-u",
        "tools/capture_vision_sidecar.py",
        "--saved-games-dir",
        str(saved_games_dir),
        "--session-id",
        session_id,
        "--trigger-host",
        settings.vision_capture_trigger_host,
        "--trigger-port",
        str(settings.vision_capture_trigger_port),
        "--capture-fps",
        "0",
    ]

    live_dcs_command = [
        executable,
        "-m",
        "simtutor",
        "live-dcs",
        "--bios-source",
        settings.dcs_bios_source,
        "--raw-bios-host",
        settings.raw_bios_host,
        "--raw-bios-port",
        str(settings.raw_bios_port),
        "--raw-bios-aircraft",
        settings.dcs_aircraft,
        "--raw-bios-control-dir",
        settings.raw_bios_control_dir,
        "--timeout",
        "0.5",
        "--pack",
        settings.pack_path,
        "--ui-map",
        settings.ui_map_path,
        "--telemetry-map",
        settings.telemetry_map_path,
        "--bios-to-ui",
        settings.bios_to_ui_path,
        "--knowledge-index",
        settings.knowledge_index_path,
        "--rag-top-k",
        str(settings.rag_top_k),
        "--max-overlay-targets",
        str(settings.max_overlay_targets),
        "--output",
        str(live_log_path),
        "--session-id",
        session_id,
        "--vision-saved-games-dir",
        str(saved_games_dir),
        "--vision-session-id",
        session_id,
        "--vision-trigger-wait-ms",
        str(settings.vision_trigger_wait_ms),
        "--vision-capture-trigger-host",
        settings.vision_capture_trigger_host,
        "--vision-capture-trigger-port",
        str(settings.vision_capture_trigger_port),
        "--global-help-hotkey",
        settings.global_help_hotkey,
        "--global-help-cooldown-ms",
        str(settings.global_help_cooldown_ms),
        "--model-provider",
        settings.model_provider,
        "--model-base-url",
        settings.model_base_url,
        "--model-name",
        settings.text_model_name,
        "--vision-model-name",
        settings.vision_model_name,
        "--model-timeout-s",
        str(settings.model_timeout_s),
        "--lang",
        settings.language,
        "--scenario-profile",
        settings.scenario_profile,
        "--no-cold-start-production",
    ]
    if settings.global_help_modifiers.strip():
        live_dcs_command.extend(["--global-help-modifiers", settings.global_help_modifiers.strip()])
    if settings.model_enable_multimodal:
        live_dcs_command.append("--model-enable-multimodal")
    else:
        live_dcs_command.append("--no-model-enable-multimodal")

    export_command = [
        executable,
        "-m",
        "simtutor",
        "experiment-export",
        str(live_log_path),
        "--output-dir",
        str(export_output_dir),
        "--trial-id",
        settings.trial_id,
        "--study-id",
        settings.study_id,
        "--participant-id",
        settings.participant_id,
        "--condition",
        settings.condition,
        "--recording-ref",
        settings.recording_ref or str(live_log_path),
        "--pack",
        settings.pack_path,
        "--taxonomy",
        settings.taxonomy_path,
        "--ui-map",
        settings.ui_map_path,
        "--bios-to-ui",
        settings.bios_to_ui_path,
        "--model-provider",
        settings.model_provider,
        "--model-name",
        settings.text_model_name,
        "--vision-model-name",
        settings.vision_model_name,
        "--scenario-profile",
        settings.scenario_profile,
        "--dcs-aircraft",
        settings.dcs_aircraft,
        "--monitor-setup",
        settings.monitor_setup or settings.monitor_mode,
        "--strict",
    ]
    _append_optional(export_command, "--group", settings.participant_group)
    _append_optional(export_command, "--experimenter-id", settings.experimenter_id)
    _append_optional(export_command, "--questionnaire", settings.questionnaire_ref)
    _append_optional(export_command, "--prompt-version", settings.prompt_version)
    _append_optional(export_command, "--prompt-hash", settings.prompt_hash)
    _append_optional(export_command, "--dcs-mission", settings.dcs_mission)
    _append_optional(export_command, "--vr-setup", settings.vr_setup)

    analyze_command = [
        executable,
        "-m",
        "simtutor",
        "experiment-analyze",
        str(export_output_dir),
        "--output-dir",
        str(analysis_output_dir),
    ]

    return LauncherSessionPlan(
        session_id=session_id,
        live_log_path=live_log_path,
        export_output_dir=export_output_dir,
        analysis_output_dir=analysis_output_dir,
        sidecar_command=tuple(sidecar_command),
        live_dcs_command=tuple(live_dcs_command),
        export_command=tuple(export_command),
        analyze_command=tuple(analyze_command),
    )


def format_dry_run_plan(plan: LauncherSessionPlan) -> str:
    lines = [
        f"session_id: {plan.session_id}",
        f"live_log_path: {plan.live_log_path}",
        "vision sidecar:",
        shlex.join(plan.sidecar_command),
        "live-dcs:",
        shlex.join(plan.live_dcs_command),
        "experiment-export:",
        shlex.join(plan.export_command),
        "experiment-analyze:",
        shlex.join(plan.analyze_command),
    ]
    return "\n".join(lines)


class LauncherExperimentSession:
    def __init__(
        self,
        settings: LauncherSettings,
        *,
        process_factory: ProcessFactory = LauncherProcess,
        tunnel_factory: TunnelFactory = LauncherTunnel,
        preflight_runner: ReportRunner = run_preflight,
        model_validator: ReportRunner = validate_launcher_model_profile,
        timestamp: TimestampFactory | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.settings = settings
        self.process_factory = process_factory
        self.tunnel_factory = tunnel_factory
        self.preflight_runner = preflight_runner
        self.model_validator = model_validator
        self.timestamp = timestamp or (lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.python_executable = python_executable
        self.plan: LauncherSessionPlan | None = None
        self.tunnel: Any | None = None
        self.sidecar_process: Any | None = None
        self.live_process: Any | None = None
        self.export_process: Any | None = None
        self.analyze_process: Any | None = None
        self.post_run_complete = False
        self._stop_requested = False

    def dry_run_text(self) -> str:
        validate_session_settings(self.settings)
        return format_dry_run_plan(self._build_plan())

    def start(self, *, dry_run: bool = False) -> LauncherSessionStartResult:
        if self._has_running_live_process():
            raise LauncherSessionError("experiment session is already running")
        validate_session_settings(self.settings)
        plan = self._build_plan()
        self.plan = plan
        if dry_run:
            return LauncherSessionStartResult(session_id=plan.session_id, live_log_path=plan.live_log_path, plan=plan)

        preflight_report = self.preflight_runner(self.settings)
        if not getattr(preflight_report, "ok", False):
            raise LauncherSessionError(f"preflight failed:\n{preflight_report.to_text()}")

        try:
            if self.settings.model_profile_mode == "remote_tunnel":
                self.tunnel = self.tunnel_factory(self.settings)
                self.tunnel.start()

            model_report = self.model_validator(self.settings)
            if not getattr(model_report, "ok", False):
                raise LauncherSessionError(f"model validation failed:\n{model_report.to_text()}")

            self.sidecar_process = self.process_factory(
                name="vision-sidecar",
                command=list(plan.sidecar_command),
            )
            self.sidecar_process.start()
            self.live_process = self.process_factory(
                name="live-dcs",
                command=list(plan.live_dcs_command),
            )
            self.live_process.start()
        except Exception:
            self._stop_live_processes()
            self._stop_tunnel()
            raise

        return LauncherSessionStartResult(session_id=plan.session_id, live_log_path=plan.live_log_path, plan=plan)

    def stop(self, *, run_post_run: bool = True) -> None:
        self._stop_live_processes()
        self._stop_tunnel()
        if run_post_run:
            self.run_post_run()

    def run_post_run(self) -> None:
        if self.plan is None:
            raise LauncherSessionError("session has not been started")
        if self.post_run_complete:
            return
        self._ensure_live_log_ready_for_export()
        self.export_process = self.process_factory(
            name="experiment-export",
            command=list(self.plan.export_command),
        )
        self.export_process.start()
        self._wait_for_process_exit(self.export_process, timeout_s=600.0)
        export_snapshot = self.export_process.snapshot()
        if export_snapshot.returncode not in (0, None):
            raise LauncherSessionError(f"experiment-export failed with code {export_snapshot.returncode}")
        self.analyze_process = self.process_factory(
            name="experiment-analyze",
            command=list(self.plan.analyze_command),
        )
        self.analyze_process.start()
        self._wait_for_process_exit(self.analyze_process, timeout_s=600.0)
        analyze_snapshot = self.analyze_process.snapshot()
        if analyze_snapshot.returncode not in (0, None):
            raise LauncherSessionError(f"experiment-analyze failed with code {analyze_snapshot.returncode}")
        self.post_run_complete = True

    def snapshots(self) -> tuple[ProcessSnapshot, ...]:
        snapshots: list[ProcessSnapshot] = []
        for process in (
            self.sidecar_process,
            self.live_process,
            self.export_process,
            self.analyze_process,
        ):
            if process is not None:
                snapshots.append(process.snapshot())
        return tuple(snapshots)

    def _build_plan(self) -> LauncherSessionPlan:
        timestamp = self.timestamp()
        log_path = resolve_unique_live_log_path(self.settings, timestamp=timestamp)
        session_id = f"sess-{_slug(self.settings.participant_id)}-{_slug(self.settings.trial_id)}-{timestamp}"
        return build_launcher_session_plan(
            self.settings,
            log_path=log_path,
            session_id=session_id,
            python_executable=self.python_executable,
        )

    def _stop_live_processes(self) -> None:
        if self.live_process is not None:
            was_running = self.live_process.snapshot().state == ProcessState.RUNNING
            self.live_process.stop()
            if was_running:
                self._stop_requested = True
        if self.sidecar_process is not None:
            self.sidecar_process.stop()

    def _stop_tunnel(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()

    def _ensure_live_log_ready_for_export(self) -> None:
        if self.plan is None:
            raise LauncherSessionError("session has not been started")
        if self.live_process is not None:
            snapshot = self.live_process.snapshot()
            if snapshot.state == ProcessState.RUNNING:
                raise LauncherSessionError("live-dcs is still running; stop the session before export")
            if snapshot.returncode not in (0, None) and not self._stop_requested:
                raise LauncherSessionError(f"live-dcs exited with code {snapshot.returncode}; export skipped")
        if not self.plan.live_log_path.is_file():
            raise LauncherSessionError(f"live-dcs log was not written: {self.plan.live_log_path}")
        if self.plan.live_log_path.stat().st_size <= 0:
            raise LauncherSessionError(f"live-dcs log is empty: {self.plan.live_log_path}")

    @staticmethod
    def _wait_for_process_exit(process: Any, *, timeout_s: float) -> None:
        child = getattr(process, "process", None)
        if child is not None and hasattr(child, "wait"):
            child.wait(timeout=timeout_s)
        process.wait_for_output(timeout_s=1.0)

    def _has_running_live_process(self) -> bool:
        return self.live_process is not None and self.live_process.snapshot().state == ProcessState.RUNNING


def _append_optional(command: list[str], flag: str, value: str | None) -> None:
    if isinstance(value, str) and value.strip():
        command.extend([flag, value.strip()])


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return slug or "unknown"


def validate_session_settings(settings: LauncherSettings) -> None:
    validate_settings(settings)
    missing = [
        name
        for name in (
            "study_id",
            "participant_id",
            "condition",
            "trial_id",
            "participant_group",
            "experimenter_id",
            "questionnaire_ref",
            "recording_ref",
            "model_provider",
            "text_model_name",
            "vision_model_name",
            "scenario_profile",
            "dcs_mission",
            "dcs_aircraft",
        )
        if not _setting_text_present(getattr(settings, name))
    ]
    if not (_setting_text_present(settings.prompt_version) or _setting_text_present(settings.prompt_hash)):
        missing.append("prompt_version or prompt_hash")
    if not (_setting_text_present(settings.vr_setup) or _setting_text_present(settings.monitor_setup)):
        missing.append("vr_setup or monitor_setup")
    if missing:
        raise LauncherSessionError("missing experiment session metadata: " + ", ".join(missing))


def _setting_text_present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "LauncherExperimentSession",
    "LauncherSessionError",
    "LauncherSessionPlan",
    "LauncherSessionStartResult",
    "build_launcher_session_plan",
    "format_dry_run_plan",
    "resolve_unique_live_log_path",
    "validate_session_settings",
]
