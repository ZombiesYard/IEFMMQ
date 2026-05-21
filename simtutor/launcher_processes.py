"""Child process shell for the SimTutor experiment launcher."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess
import threading
from typing import Any, Callable, Mapping, Sequence


class ProcessState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    EXITED = "exited"


class LauncherProcessError(RuntimeError):
    """Raised when a managed launcher process cannot be started or stopped."""


@dataclass(frozen=True)
class ProcessSnapshot:
    name: str
    command: tuple[str, ...]
    state: ProcessState
    returncode: int | None
    log_lines: tuple[str, ...]


PopenFactory = Callable[..., Any]


class LauncherProcess:
    def __init__(
        self,
        *,
        name: str,
        command: Sequence[str],
        popen_factory: PopenFactory = subprocess.Popen,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        max_log_lines: int = 200,
    ) -> None:
        if not name.strip():
            raise LauncherProcessError("process name is required")
        if not command:
            raise LauncherProcessError("process command is required")
        self.name = name
        self.command = tuple(str(part) for part in command)
        self.popen_factory = popen_factory
        self.cwd = Path(cwd) if cwd is not None else None
        self.env = dict(env) if env is not None else None
        self.max_log_lines = max_log_lines
        self.process: Any | None = None
        self._state = ProcessState.STOPPED
        self._log_lines: deque[str] = deque(maxlen=max_log_lines)
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            raise LauncherProcessError(f"process already running: {self.name}")
        self.process = self.popen_factory(
            list(self.command),
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._state = ProcessState.RUNNING
        self._reader = threading.Thread(target=self._read_stdout, name=f"{self.name}-stdout", daemon=True)
        self._reader.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self.process is None:
            self._state = ProcessState.STOPPED
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout_s)
        self.wait_for_output(timeout_s=0.2)
        self._state = ProcessState.STOPPED

    def drain_output(self) -> None:
        self.wait_for_output(timeout_s=0.0)

    def wait_for_output(self, *, timeout_s: float = 1.0) -> None:
        if self._reader is not None:
            self._reader.join(timeout=max(0.0, timeout_s))

    def snapshot(self) -> ProcessSnapshot:
        returncode = self.process.poll() if self.process is not None else None
        state = self._state
        if state == ProcessState.RUNNING and returncode is not None:
            state = ProcessState.EXITED
            self._state = state
        with self._lock:
            log_lines = tuple(self._log_lines)
        return ProcessSnapshot(
            name=self.name,
            command=self.command,
            state=state,
            returncode=returncode,
            log_lines=log_lines,
        )

    def _read_stdout(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            self._append_line(line)

    def _append_line(self, line: str) -> None:
        clean = line.rstrip("\r\n")
        with self._lock:
            self._log_lines.append(clean)


__all__ = [
    "LauncherProcess",
    "LauncherProcessError",
    "ProcessSnapshot",
    "ProcessState",
]
