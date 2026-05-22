"""Tkinter MVP launcher for local SimTutor experiment sessions."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError as exc:
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _TK_IMPORT_ERROR = None

from simtutor.launcher_processes import LauncherProcess, ProcessState
from simtutor.launcher_session import LauncherExperimentSession, format_dry_run_plan
from simtutor.launcher_model_check import validate_launcher_model_profile
from simtutor.launcher_preflight import run_launcher_install, run_preflight
from simtutor.launcher_settings import (
    LauncherSettings,
    LauncherSettingsError,
    default_saved_games_path,
    load_profile,
    load_settings,
    save_profile,
    save_settings,
    validate_settings,
)
from simtutor.launcher_tunnel import LauncherTunnel


NORMAL_FIELDS = (
    ("monitor_mode", "Monitor mode"),
    ("language", "Language"),
    ("scenario_profile", "Scenario profile"),
    ("output_log_directory", "Runtime log output directory"),
    ("export_output_directory", "Experiment output directory"),
    ("analysis_output_directory", "Analysis output directory"),
    ("__metadata", "Experiment metadata"),
    ("participant_id", "Participant ID"),
    ("condition", "Condition"),
    ("trial_id", "Trial ID"),
    ("study_id", "Study ID"),
    ("participant_group", "Participant group"),
    ("experimenter_id", "Experimenter ID"),
    ("questionnaire_ref", "Questionnaire reference"),
    ("recording_ref", "Recording reference"),
)

ADVANCED_FIELDS = (
    ("model_provider", "Model provider"),
    ("model_profile_mode", "Model profile mode"),
    ("model_base_url", "Model base URL"),
    ("text_model_name", "Text model name"),
    ("vision_model_name", "Vision model name"),
    ("model_timeout_s", "Model timeout"),
    ("model_enable_multimodal", "VLM facts enabled"),
    ("log_raw_llm_text", "Log raw LLM text"),
    ("print_model_io", "Print model IO"),
    ("max_overlay_targets", "Max overlay targets"),
    ("pack_path", "Pack path"),
    ("taxonomy_path", "Taxonomy path"),
    ("ui_map_path", "UI map path"),
    ("telemetry_map_path", "Telemetry map path"),
    ("bios_to_ui_path", "BIOS-to-UI path"),
    ("knowledge_index_path", "Knowledge index path"),
    ("rag_top_k", "RAG top-k"),
    ("dcs_bios_source", "DCS-BIOS source"),
    ("raw_bios_host", "Raw BIOS host"),
    ("raw_bios_port", "Raw BIOS port"),
    ("raw_bios_control_dir", "Raw BIOS control dir"),
    ("dcs_aircraft", "DCS aircraft"),
    ("dcs_mission", "DCS mission"),
    ("vr_setup", "VR setup"),
    ("monitor_setup", "Monitor setup"),
    ("prompt_version", "Prompt version"),
    ("prompt_hash", "Prompt hash"),
    ("global_help_hotkey", "Global help hotkey"),
    ("global_help_modifiers", "Global help modifiers"),
    ("global_help_cooldown_ms", "Global help cooldown ms"),
    ("vision_trigger_wait_ms", "Vision trigger wait ms"),
    ("vision_capture_trigger_host", "Vision trigger host"),
    ("vision_capture_trigger_port", "Vision trigger port"),
    ("ssh_tunnel_profile", "SSH tunnel profile"),
    ("ssh_executable_path", "SSH executable path"),
    ("ssh_user", "SSH user"),
    ("ssh_host", "SSH host"),
    ("ssh_local_port", "SSH local port"),
    ("ssh_remote_host", "SSH remote host"),
    ("ssh_remote_port", "SSH remote port"),
    ("ssh_identity_file_path", "SSH identity file path"),
    ("preflight_profile", "Preflight profile"),
    ("export_profile", "Export profile"),
)

DIRECTORY_FIELDS = {
    "output_log_directory",
    "export_output_directory",
    "analysis_output_directory",
    "raw_bios_control_dir",
}

FILE_FIELDS = {
    "pack_path",
    "taxonomy_path",
    "ui_map_path",
    "telemetry_map_path",
    "bios_to_ui_path",
    "knowledge_index_path",
    "ssh_executable_path",
    "ssh_identity_file_path",
}

FIELD_CHOICES = {
    "monitor_mode": ("single-monitor", "extended-right", "ultrawide-left-stack", "fa18c_composite_panel_v2"),
    "language": ("zh", "en"),
    "scenario_profile": ("airfield", "carrier"),
    "condition": ("with_tutor", "without_tutor", "baseline", "treatment"),
    "participant_group": ("", "novice", "experienced", "pilot", "student"),
    "model_provider": ("openai_compat", "stub", "ollama"),
    "model_profile_mode": ("remote_tunnel", "remote_direct", "local_stub"),
    "model_timeout_s": ("60.0", "20.0", "30.0", "120.0"),
    "model_enable_multimodal": ("true", "false"),
    "log_raw_llm_text": ("true", "false"),
    "print_model_io": ("true", "false"),
    "max_overlay_targets": ("2", "4", "3", "1", "0"),
    "rag_top_k": ("5", "8", "10", "3", "1"),
    "dcs_bios_source": ("raw", "decoded"),
    "raw_bios_host": ("239.255.50.10", "127.0.0.1", "0.0.0.0"),
    "raw_bios_port": ("5010",),
    "dcs_aircraft": ("FA-18C_hornet",),
    "dcs_mission": ("", "cold_start", "freeflight", "training"),
    "vr_setup": ("", "none", "Quest 3", "VR headset"),
    "monitor_setup": ("", "single-monitor", "extended-right", "ultrawide-left-stack", "fa18c_composite_panel_v2"),
    "prompt_version": ("launcher-v0.4",),
    "prompt_hash": ("",),
    "global_help_hotkey": ("X1", "X2", "F10", "F12", ""),
    "global_help_modifiers": ("", "ctrl", "alt", "shift", "ctrl+shift"),
    "global_help_cooldown_ms": ("800", "1000", "1500", "2000", "0"),
    "vision_trigger_wait_ms": ("4000", "3000", "5000", "8000", "0"),
    "vision_capture_trigger_host": ("127.0.0.1", "localhost", "0.0.0.0"),
    "vision_capture_trigger_port": ("7795",),
    "ssh_tunnel_profile": ("cloud-247-vllm", "", "default", "tu-clausthal", "local"),
    "ssh_user": ("yz50", ""),
    "ssh_host": ("cloud-247.rz.tu-clausthal.de", ""),
    "ssh_local_port": ("16324",),
    "ssh_remote_host": ("127.0.0.1", "localhost"),
    "ssh_remote_port": ("6324",),
    "preflight_profile": ("", "default", "quick", "strict"),
    "export_profile": ("", "default", "experiment", "analysis"),
}

READONLY_CHOICE_FIELDS = {
    "monitor_mode",
    "language",
    "scenario_profile",
    "model_provider",
    "model_profile_mode",
    "model_enable_multimodal",
    "log_raw_llm_text",
    "print_model_io",
    "dcs_bios_source",
}


ProcessFactory = Callable[..., LauncherProcess]
CommandBuilder = Callable[[LauncherSettings], list[str]]


class LauncherApp(ttk.Frame if ttk is not None else object):
    def __init__(
        self,
        master: tk.Tk,
        *,
        settings: LauncherSettings | None = None,
        process_factory: ProcessFactory = LauncherProcess,
        command_builder: CommandBuilder | None = None,
    ) -> None:
        _require_tk()
        super().__init__(master, padding=12)
        self.master = master
        self.settings = settings or load_settings()
        self.process_factory = process_factory
        self.command_builder = command_builder or _fake_process_command
        self.variables: dict[str, tk.StringVar] = {}
        self.status_variables = {
            name: tk.StringVar(value="stopped")
            for name in ("DCS", "model", "tunnel", "tutor", "vision", "export", "analysis")
        }
        self.process: LauncherProcess | None = None
        self.tunnel: LauncherTunnel | None = None
        self.session: LauncherExperimentSession | None = None
        self.session_log_header: list[str] = []

        master.title("SimTutor Experiment Launcher")
        master.minsize(900, 680)
        self.grid(row=0, column=0, sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_settings_tabs()
        self._build_status_area()
        self._build_buttons()
        self._build_log_area()
        self._poll_process()

    def _build_settings_tabs(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")
        notebook.columnconfigure(0, weight=1)
        notebook.rowconfigure(0, weight=1)
        self._add_settings_tab(notebook, "Normal", NORMAL_FIELDS)
        self._add_settings_tab(notebook, "Advanced", ADVANCED_FIELDS)

    def _add_settings_tab(self, notebook: ttk.Notebook, title: str, field_defs: tuple[tuple[str, str], ...]) -> None:
        outer = ttk.Frame(notebook)
        notebook.add(outer, text=title)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        frame = ttk.Frame(canvas, padding=10)
        window_id = canvas.create_window((0, 0), window=frame, anchor="nw")

        def update_scroll_region(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_inner_width(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def on_linux_scroll_up(_event: tk.Event) -> None:
            canvas.yview_scroll(-3, "units")

        def on_linux_scroll_down(_event: tk.Event) -> None:
            canvas.yview_scroll(3, "units")

        frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_inner_width)
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", on_mousewheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<Button-4>", on_linux_scroll_up), add="+")
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<Button-4>"), add="+")
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<Button-5>", on_linux_scroll_down), add="+")
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<Button-5>"), add="+")

        frame.columnconfigure(1, weight=1)
        row = 0
        for name, label in field_defs:
            if name.startswith("__"):
                ttk.Separator(frame).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 6))
                row += 1
                ttk.Label(frame, text=label).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 3))
                row += 1
                continue
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            var = tk.StringVar(value=_format_setting_value(getattr(self.settings, name)))
            self.variables[name] = var
            choices = FIELD_CHOICES.get(name)
            if choices is None:
                field = ttk.Entry(frame, textvariable=var)
            else:
                state = "readonly" if name in READONLY_CHOICE_FIELDS else "normal"
                field = ttk.Combobox(frame, textvariable=var, values=choices, state=state)
            field.grid(row=row, column=1, sticky="ew", pady=3)
            if name in DIRECTORY_FIELDS or name in FILE_FIELDS:
                button = ttk.Button(frame, text="Browse", command=lambda key=name: self._browse_path(key))
                button.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=3)
            row += 1

    def _build_status_area(self) -> None:
        frame = ttk.LabelFrame(self, text="Status", padding=10)
        frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for column, (name, var) in enumerate(self.status_variables.items()):
            ttk.Label(frame, text=name).grid(row=0, column=column, sticky="w", padx=(0, 18))
            ttk.Label(frame, textvariable=var).grid(row=1, column=column, sticky="w", padx=(0, 18))

    def _build_buttons(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        frame.columnconfigure(10, weight=1)
        ttk.Button(frame, text="Load", command=self._load).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(frame, text="Save", command=self._save).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(frame, text="Save Profile", command=self._save_profile).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(frame, text="Load Profile", command=self._load_profile).grid(row=0, column=3, padx=(0, 18))
        ttk.Button(frame, text="Dry Run", command=self._dry_run).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(frame, text="Start", command=self._start).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(frame, text="Stop", command=self._stop).grid(row=0, column=6, padx=(0, 18))
        ttk.Button(frame, text="Install", command=self._install).grid(row=0, column=7, padx=(0, 6))
        ttk.Button(frame, text="Preflight", command=self._preflight).grid(row=0, column=8, padx=(0, 6))
        ttk.Button(frame, text="Validate Model", command=self._validate_model).grid(row=0, column=9, padx=(0, 6))
        ttk.Button(frame, text="Export", command=self._export).grid(row=0, column=10, padx=(0, 6))

    def _build_log_area(self) -> None:
        frame = ttk.LabelFrame(self, text="Process log", padding=10)
        frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        self.log_text = tk.Text(frame, height=12, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _browse_path(self, key: str) -> None:
        assert filedialog is not None
        selected = filedialog.askopenfilename() if key in FILE_FIELDS else filedialog.askdirectory()
        if selected:
            self.variables[key].set(selected)

    def _collect_settings(self) -> LauncherSettings:
        values = {name: var.get() for name, var in self.variables.items()}
        values["saved_games_path"] = default_saved_games_path()
        values["dcs_variant"] = "DCS"
        return LauncherSettings.from_dict(values)

    def _apply_settings(self, settings: LauncherSettings) -> None:
        self.settings = settings
        for name, var in self.variables.items():
            var.set(_format_setting_value(getattr(settings, name)))

    def _load(self) -> None:
        try:
            self._apply_settings(load_settings())
        except LauncherSettingsError as exc:
            assert messagebox is not None
            messagebox.showerror("Load failed", str(exc))

    def _save(self) -> None:
        try:
            settings = self._collect_settings()
            save_settings(settings)
            self.settings = settings
        except LauncherSettingsError as exc:
            assert messagebox is not None
            messagebox.showerror("Save failed", str(exc))

    def _save_profile(self) -> None:
        try:
            settings = self._collect_settings()
            validate_settings(settings)
            name = f"{settings.participant_id}_{settings.trial_id}"
            path = save_profile(name, settings)
            self._append_log(f"saved profile: {path}")
        except LauncherSettingsError as exc:
            assert messagebox is not None
            messagebox.showerror("Profile save failed", str(exc))

    def _load_profile(self) -> None:
        try:
            settings = self._collect_settings()
            name = f"{settings.participant_id}_{settings.trial_id}"
            self._apply_settings(load_profile(name))
        except LauncherSettingsError as exc:
            assert messagebox is not None
            messagebox.showerror("Profile load failed", str(exc))

    def _start(self) -> None:
        if self.session is not None and any(
            snapshot.state == ProcessState.RUNNING
            for snapshot in self.session.snapshots()
        ):
            assert messagebox is not None
            messagebox.showerror("Start failed", "An experiment session is already running.")
            self._append_log("start failed: an experiment session is already running")
            return
        try:
            settings = self._collect_settings()
            save_settings(settings)
        except LauncherSettingsError as exc:
            assert messagebox is not None
            messagebox.showerror("Start failed", str(exc))
            return
        self.settings = settings
        try:
            session = LauncherExperimentSession(
                settings,
                process_factory=self.process_factory,
                tunnel_factory=LauncherTunnel,
                python_executable=sys.executable,
            )
            result = session.start()
        except Exception as exc:
            assert messagebox is not None
            messagebox.showerror("Start failed", str(exc))
            self._append_log(f"start failed: {exc}")
            self.session = None
            return
        self.session = session
        self.status_variables["DCS"].set("external")
        self.status_variables["model"].set(settings.model_profile_mode)
        self.status_variables["tunnel"].set("running" if settings.model_profile_mode == "remote_tunnel" else "not required")
        self.status_variables["tutor"].set("running")
        self.status_variables["vision"].set("running")
        self.status_variables["export"].set("pending")
        self.status_variables["analysis"].set("pending")
        self.session_log_header = [
            f"session started: {result.session_id}",
            f"live log: {result.live_log_path}",
        ]
        self._append_log("\n".join(self.session_log_header))

    def _stop(self) -> None:
        if self.session is not None:
            try:
                self.session.stop(run_post_run=True)
                self.status_variables["export"].set("complete")
                self.status_variables["analysis"].set("complete")
                if self.session.plan is not None:
                    self.session_log_header.extend(
                        [
                            f"export output: {self.session.plan.export_output_dir}",
                            f"analysis output: {self.session.plan.analysis_output_dir}",
                        ]
                    )
                    self._append_log("\n".join(self.session_log_header[-2:]))
            except Exception as exc:
                assert messagebox is not None
                messagebox.showerror("Stop/export failed", str(exc))
                self._append_log(f"stop/export failed: {exc}")
        elif self.process is not None:
            self.process.stop()
        if self.tunnel is not None:
            self.tunnel.stop()
        self.status_variables["tunnel"].set("stopped")
        self.status_variables["tutor"].set("stopped")
        self.status_variables["vision"].set("stopped")

    def _poll_process(self) -> None:
        if self.session is not None:
            lines: list[str] = list(self.session_log_header)
            for snapshot in self.session.snapshots():
                lines.extend(f"[{snapshot.name}] {line}" for line in snapshot.log_lines)
                if snapshot.name == "live-dcs" and snapshot.state == ProcessState.EXITED:
                    self.status_variables["tutor"].set(f"exited ({snapshot.returncode})")
                if snapshot.name == "vision-sidecar" and snapshot.state == ProcessState.EXITED:
                    self.status_variables["vision"].set(f"exited ({snapshot.returncode})")
            if lines:
                self.log_text.delete("1.0", "end")
                self.log_text.insert("end", "\n".join(lines))
                self.log_text.see("end")
        elif self.process is not None:
            snapshot = self.process.snapshot()
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", "\n".join(snapshot.log_lines))
            self.log_text.see("end")
            if snapshot.state == ProcessState.EXITED:
                self.status_variables["tutor"].set(f"exited ({snapshot.returncode})")
        self.after(250, self._poll_process)

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def _install(self) -> None:
        try:
            settings = self._collect_settings()
            summary = _run_install_action(settings)
        except Exception as exc:
            assert messagebox is not None
            messagebox.showerror("Install failed", str(exc))
            self._append_log(f"install failed: {exc}")
            return
        self.settings = settings
        self._append_log(summary)

    def _preflight(self) -> None:
        try:
            settings = self._collect_settings()
        except LauncherSettingsError as exc:
            assert messagebox is not None
            messagebox.showerror("Preflight failed", str(exc))
            self._append_log(f"preflight failed: {exc}")
            return
        self.settings = settings
        report, text = _run_preflight_action(settings)
        self._append_log(text)
        if not report.ok:
            assert messagebox is not None
            messagebox.showwarning("Preflight found issues", "See the process log for details.")

    def _validate_model(self) -> None:
        try:
            settings = self._collect_settings()
            validate_settings(settings)
            if settings.model_profile_mode == "remote_tunnel":
                if self.tunnel is None or not self.tunnel.snapshot().running:
                    self.tunnel = LauncherTunnel(settings)
                    self.tunnel.start()
                self.status_variables["tunnel"].set("running")
            report, text = _run_model_check_action(settings)
        except Exception as exc:
            assert messagebox is not None
            messagebox.showerror("Model validation failed", str(exc))
            self._append_log(f"model validation failed: {exc}")
            return
        self.settings = settings
        self.status_variables["model"].set(settings.model_profile_mode)
        self._append_log(text)
        if not report.ok:
            assert messagebox is not None
            messagebox.showwarning("Model validation found issues", "See the process log for details.")

    def _dry_run(self) -> None:
        try:
            settings = self._collect_settings()
            session = LauncherExperimentSession(
                settings,
                process_factory=self.process_factory,
                python_executable=sys.executable,
            )
            plan = session.start(dry_run=True).plan
            text = (
                "workflow:\n"
                "- validate settings and preflight\n"
                "- start SSH tunnel when model_profile_mode=remote_tunnel\n"
                "- validate model endpoint\n"
                "- launch the commands below\n"
                + format_dry_run_plan(plan)
            )
        except Exception as exc:
            assert messagebox is not None
            messagebox.showerror("Dry run failed", str(exc))
            self._append_log(f"dry run failed: {exc}")
            return
        self._append_log("dry run:\n" + text)

    def _export(self) -> None:
        if self.session is None:
            self._append_log("export skipped: no completed session is available")
            return
        try:
            self.session.run_post_run()
        except Exception as exc:
            assert messagebox is not None
            messagebox.showerror("Export failed", str(exc))
            self._append_log(f"export failed: {exc}")
            return
        self.status_variables["export"].set("complete")
        self.status_variables["analysis"].set("complete")


def _format_setting_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _fake_process_command(settings: LauncherSettings) -> list[str]:
    participant = json.dumps(settings.participant_id)
    trial = json.dumps(settings.trial_id)
    script = (
        "import time; "
        "print('SimTutor fake session starting', flush=True); "
        f"participant = {participant}; "
        f"trial = {trial}; "
        "print(f'participant={participant} trial={trial}', flush=True); "
        "time.sleep(0.2); "
        "print('DCS/model/tutor process shell ready', flush=True); "
        "time.sleep(0.2)"
    )
    return [sys.executable, "-u", "-c", script]


def _run_install_action(
    settings: LauncherSettings,
    installer: Callable[[LauncherSettings], Any] = run_launcher_install,
) -> str:
    result = installer(settings)
    return (
        "install complete: "
        f"files_copied={result.files_copied}, "
        f"export_patched={result.export_patched}, "
        f"config_written={result.config_written}, "
        f"monitor_setup_written={result.monitor_setup_written}, "
        f"vlm_frame_enabled={result.vlm_frame_enabled}"
    )


def _run_preflight_action(
    settings: LauncherSettings,
    runner: Callable[[LauncherSettings], Any] = run_preflight,
) -> tuple[Any, str]:
    report = runner(settings)
    return report, "preflight report:\n" + report.to_text()


def _run_model_check_action(
    settings: LauncherSettings,
    runner: Callable[[LauncherSettings], Any] = validate_launcher_model_profile,
) -> tuple[Any, str]:
    report = runner(settings)
    return report, "model validation report:\n" + report.to_text()


def main() -> int:
    if _TK_IMPORT_ERROR is not None:
        print(f"[SIMTUTOR_LAUNCHER] tkinter is required for the Windows GUI launcher: {_TK_IMPORT_ERROR}", file=sys.stderr)
        return 1
    try:
        assert tk is not None
        root = tk.Tk()
        LauncherApp(root)
        root.mainloop()
    except Exception as exc:
        print(f"[SIMTUTOR_LAUNCHER] failed to start GUI: {exc}", file=sys.stderr)
        return 1
    return 0


def _require_tk() -> None:
    if _TK_IMPORT_ERROR is not None or tk is None or ttk is None:
        raise RuntimeError(f"tkinter is required for the Windows GUI launcher: {_TK_IMPORT_ERROR}")


if __name__ == "__main__":
    raise SystemExit(main())
