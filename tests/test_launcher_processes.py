from __future__ import annotations

import subprocess
from collections import deque

import pytest

from simtutor.launcher_processes import LauncherProcess, LauncherProcessError, ProcessState


class FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = deque(lines)

    def readline(self) -> str:
        if self._lines:
            return self._lines.popleft()
        return ""


class FakePopen:
    def __init__(self, command: list[str], **kwargs: object) -> None:
        self.command = command
        self.kwargs = kwargs
        self.stdout = FakeStdout(["booting\n", "ready\n"])
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_launcher_process_starts_and_captures_log_tail() -> None:
    proc = LauncherProcess(
        name="fake",
        command=["fake.exe", "--demo"],
        popen_factory=FakePopen,
        max_log_lines=10,
    )

    proc.start()
    proc.wait_for_output(timeout_s=1.0)
    snapshot = proc.snapshot()

    assert snapshot.name == "fake"
    assert snapshot.command == ("fake.exe", "--demo")
    assert snapshot.state == ProcessState.RUNNING
    assert snapshot.returncode is None
    assert snapshot.log_lines == ("booting", "ready")


def test_launcher_process_stop_terminates_running_child() -> None:
    proc = LauncherProcess(name="fake", command=["fake.exe"], popen_factory=FakePopen)

    proc.start()
    proc.stop()
    snapshot = proc.snapshot()

    assert snapshot.state == ProcessState.STOPPED
    assert snapshot.returncode == 0
    assert proc.process is not None
    assert proc.process.terminated is True


def test_launcher_process_rejects_double_start() -> None:
    proc = LauncherProcess(name="fake", command=["fake.exe"], popen_factory=FakePopen)

    proc.start()

    with pytest.raises(LauncherProcessError, match="already running"):
        proc.start()


def test_launcher_process_uses_safe_subprocess_defaults() -> None:
    captured: dict[str, object] = {}

    def factory(command: list[str], **kwargs: object) -> FakePopen:
        captured.update(kwargs)
        return FakePopen(command, **kwargs)

    proc = LauncherProcess(name="fake", command=["fake.exe"], popen_factory=factory)

    proc.start()

    assert captured["stdout"] == subprocess.PIPE
    assert captured["stderr"] == subprocess.STDOUT
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["text"] is True
