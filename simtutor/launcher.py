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
from simtutor.launcher_preflight import run_launcher_install, run_preflight
from simtutor.launcher_settings import (
    LauncherSettings,
    LauncherSettingsError,
    load_profile,
    load_settings,
    save_profile,
    save_settings,
    validate_settings,
)


NORMAL_FIELDS = (
    ("saved_games_path", "Saved Games path"),
    ("dcs_variant", "DCS variant"),
    ("monitor_mode", "Monitor mode"),
    ("language", "Language"),
    ("scenario_profile", "Scenario profile"),
    ("output_log_directory", "Output log directory"),
    ("participant_id", "Participant id"),
    ("condition", "Condition"),
    ("trial_id", "Trial id"),
)

ADVANCED_FIELDS = (
    ("model_provider", "Model provider"),
    ("model_base_url", "Model base URL"),
    ("text_model_name", "Text model name"),
    ("vision_model_name", "Vision model name"),
    ("model_timeout_s", "Model timeout"),
    ("max_overlay_targets", "Max overlay targets"),
    ("ssh_tunnel_profile", "SSH tunnel profile"),
    ("preflight_profile", "Preflight profile"),
    ("export_profile", "Export profile"),
)


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
            for name in ("DCS", "model", "tunnel", "tutor", "vision")
        }
        self.process: LauncherProcess | None = None

        master.title("SimTutor Experiment Launcher")
        master.minsize(860, 620)
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
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text=title)
        frame.columnconfigure(1, weight=1)
        for row, (name, label) in enumerate(field_defs):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            var = tk.StringVar(value=str(getattr(self.settings, name)))
            self.variables[name] = var
            entry = ttk.Entry(frame, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            if name in {"saved_games_path", "output_log_directory"}:
                button = ttk.Button(frame, text="Browse", command=lambda key=name: self._browse_directory(key))
                button.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=3)

    def _build_status_area(self) -> None:
        frame = ttk.LabelFrame(self, text="Status", padding=10)
        frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for column, (name, var) in enumerate(self.status_variables.items()):
            ttk.Label(frame, text=name).grid(row=0, column=column, sticky="w", padx=(0, 18))
            ttk.Label(frame, textvariable=var).grid(row=1, column=column, sticky="w", padx=(0, 18))

    def _build_buttons(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        frame.columnconfigure(9, weight=1)
        ttk.Button(frame, text="Load", command=self._load).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(frame, text="Save", command=self._save).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(frame, text="Save Profile", command=self._save_profile).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(frame, text="Load Profile", command=self._load_profile).grid(row=0, column=3, padx=(0, 18))
        ttk.Button(frame, text="Start", command=self._start).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(frame, text="Stop", command=self._stop).grid(row=0, column=5, padx=(0, 18))
        ttk.Button(frame, text="Install", command=self._install).grid(row=0, column=6, padx=(0, 6))
        ttk.Button(frame, text="Preflight", command=self._preflight).grid(row=0, column=7, padx=(0, 6))
        ttk.Button(frame, text="Export", command=self._placeholder_export).grid(row=0, column=8, padx=(0, 6))

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

    def _browse_directory(self, key: str) -> None:
        assert filedialog is not None
        selected = filedialog.askdirectory()
        if selected:
            self.variables[key].set(selected)

    def _collect_settings(self) -> LauncherSettings:
        values = {name: var.get() for name, var in self.variables.items()}
        return LauncherSettings.from_dict(values)

    def _apply_settings(self, settings: LauncherSettings) -> None:
        self.settings = settings
        for name, var in self.variables.items():
            var.set(str(getattr(settings, name)))

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
        try:
            settings = self._collect_settings()
            save_settings(settings)
        except LauncherSettingsError as exc:
            assert messagebox is not None
            messagebox.showerror("Start failed", str(exc))
            return
        self.settings = settings
        if self.process is None or self.process.snapshot().state != ProcessState.RUNNING:
            self.process = self.process_factory(name="simtutor-fake-session", command=self.command_builder(settings))
        try:
            self.process.start()
        except Exception as exc:
            assert messagebox is not None
            messagebox.showerror("Start failed", str(exc))
            return
        self.status_variables["DCS"].set("external")
        self.status_variables["model"].set(settings.model_provider)
        self.status_variables["tunnel"].set("placeholder")
        self.status_variables["tutor"].set("running")
        self.status_variables["vision"].set("placeholder")

    def _stop(self) -> None:
        if self.process is not None:
            self.process.stop()
        self.status_variables["tutor"].set("stopped")

    def _poll_process(self) -> None:
        if self.process is not None:
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

    def _placeholder_export(self) -> None:
        self._append_log("export placeholder: experiment export implementation is out of scope for this MVP")


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
