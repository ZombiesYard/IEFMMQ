from __future__ import annotations

from simtutor.launcher_settings import LauncherSettings
from simtutor.launcher_tunnel import (
    LauncherTunnel,
    LauncherTunnelError,
    build_ssh_tunnel_command,
)

import pytest


class FakeProcess:
    def __init__(self) -> None:
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


class FakeExitedProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.returncode = 255
        self.stdout = FakeStdout("Permission denied (publickey).\n")


class FakeStdout:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> str:
        return self.text


def _remote_tunnel_settings() -> LauncherSettings:
    return LauncherSettings(
        saved_games_path="C:/Users/test/Saved Games/DCS",
        output_log_directory="C:/SimTutor/logs",
        participant_id="P01",
        condition="with_tutor",
        trial_id="T01",
        model_provider="openai_compat",
        model_profile_mode="remote_tunnel",
        text_model_name="qwen36_27b",
        vision_model_name="simtutor-vision",
        ssh_user="pilot",
        ssh_host="cloud.example.test",
        ssh_local_port=16324,
        ssh_remote_host="127.0.0.1",
        ssh_remote_port=6324,
        ssh_identity_file_path="C:/Users/test/.ssh/id_ed25519",
    )


def test_build_ssh_tunnel_command_uses_exit_on_forward_failure_and_identity() -> None:
    settings = _remote_tunnel_settings()

    command = build_ssh_tunnel_command(settings, ssh_executable="C:/Windows/System32/OpenSSH/ssh.exe")

    assert command == [
        "C:/Windows/System32/OpenSSH/ssh.exe",
        "-N",
        "-L",
        "16324:127.0.0.1:6324",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        "-i",
        "C:/Users/test/.ssh/id_ed25519",
        "pilot@cloud.example.test",
    ]


def test_launcher_tunnel_rejects_occupied_local_port_before_start() -> None:
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        calls.append(command)
        return FakeProcess()

    tunnel = LauncherTunnel(
        _remote_tunnel_settings(),
        popen_factory=fake_popen,
        port_checker=lambda _host, _port: False,
        ssh_resolver=lambda _settings: "ssh.exe",
    )

    with pytest.raises(LauncherTunnelError, match="local TCP port 16324 is already occupied"):
        tunnel.start()

    assert calls == []


def test_launcher_tunnel_stops_only_owned_process() -> None:
    created: list[FakeProcess] = []

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        process = FakeProcess()
        created.append(process)
        return process

    tunnel = LauncherTunnel(
        _remote_tunnel_settings(),
        popen_factory=fake_popen,
        port_checker=lambda _host, _port: True,
        ssh_resolver=lambda _settings: "ssh.exe",
    )

    tunnel.start()
    tunnel.stop()

    assert len(created) == 1
    assert created[0].terminated is True


def test_launcher_tunnel_reports_immediate_key_auth_failure() -> None:
    tunnel = LauncherTunnel(
        _remote_tunnel_settings(),
        popen_factory=lambda _command, **_kwargs: FakeExitedProcess(),
        port_checker=lambda _host, _port: True,
        ssh_resolver=lambda _settings: "ssh.exe",
    )

    with pytest.raises(LauncherTunnelError, match="password prompts are disabled"):
        tunnel.start()
