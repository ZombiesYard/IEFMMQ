"""SSH tunnel management for the SimTutor experiment launcher."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import socket
import subprocess
import sys
from typing import Any, Callable

from simtutor.launcher_settings import LauncherSettings


class LauncherTunnelError(RuntimeError):
    """Raised when the launcher cannot start or stop its SSH tunnel."""


PopenFactory = Callable[..., Any]
PortChecker = Callable[[str, int], bool]
SshResolver = Callable[[LauncherSettings], str]


@dataclass(frozen=True)
class LauncherTunnelSnapshot:
    command: tuple[str, ...]
    running: bool
    returncode: int | None


def tcp_port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def resolve_ssh_executable(settings: LauncherSettings) -> str:
    configured = settings.ssh_executable_path.strip() if isinstance(settings.ssh_executable_path, str) else ""
    if configured:
        return configured
    detected = shutil.which("ssh.exe") or shutil.which("ssh")
    if detected:
        return detected
    return "ssh.exe" if sys.platform == "win32" else "ssh"


def build_ssh_tunnel_command(settings: LauncherSettings, *, ssh_executable: str | None = None) -> list[str]:
    executable = ssh_executable or resolve_ssh_executable(settings)
    user = _required_text(settings.ssh_user, "ssh_user")
    host = _required_text(settings.ssh_host, "ssh_host")
    remote_host = _required_text(settings.ssh_remote_host, "ssh_remote_host")
    local_port = _required_port(settings.ssh_local_port, "ssh_local_port")
    remote_port = _required_port(settings.ssh_remote_port, "ssh_remote_port")

    command = [
        executable,
        "-N",
        "-L",
        f"{local_port}:{remote_host}:{remote_port}",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
    ]
    identity = settings.ssh_identity_file_path.strip() if isinstance(settings.ssh_identity_file_path, str) else ""
    if identity:
        command.extend(["-i", identity])
    command.append(f"{user}@{host}")
    return command


class LauncherTunnel:
    def __init__(
        self,
        settings: LauncherSettings,
        *,
        popen_factory: PopenFactory = subprocess.Popen,
        port_checker: PortChecker = tcp_port_is_available,
        ssh_resolver: SshResolver = resolve_ssh_executable,
    ) -> None:
        self.settings = settings
        self.popen_factory = popen_factory
        self.port_checker = port_checker
        self.ssh_resolver = ssh_resolver
        self.process: Any | None = None
        self.command: tuple[str, ...] = ()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            raise LauncherTunnelError("launcher-owned SSH tunnel is already running")
        local_port = _required_port(self.settings.ssh_local_port, "ssh_local_port")
        if not self.port_checker("127.0.0.1", local_port):
            raise LauncherTunnelError(f"local TCP port {local_port} is already occupied")
        command = build_ssh_tunnel_command(self.settings, ssh_executable=self.ssh_resolver(self.settings))
        self.command = tuple(command)
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self.process = self.popen_factory(command, **kwargs)
        except OSError as exc:
            raise LauncherTunnelError(f"could not start SSH tunnel: {exc}") from exc
        self._raise_if_exited_immediately()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout_s)

    def snapshot(self) -> LauncherTunnelSnapshot:
        returncode = self.process.poll() if self.process is not None else None
        return LauncherTunnelSnapshot(
            command=self.command,
            running=self.process is not None and returncode is None,
            returncode=returncode,
        )

    def _raise_if_exited_immediately(self) -> None:
        if self.process is None:
            return
        returncode = self.process.poll()
        if returncode is None:
            return
        output = _read_process_output(self.process)
        detail = f": {output}" if output else ""
        hint = ""
        lowered = output.lower()
        if "permission denied" in lowered or "publickey" in lowered or "password" in lowered:
            hint = " Configure SSH key authentication; launcher password prompts are disabled."
        raise LauncherTunnelError(f"SSH tunnel exited during startup with code {returncode}{detail}.{hint}")


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LauncherTunnelError(f"{name} is required")
    return value.strip()


def _required_port(value: int, name: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise LauncherTunnelError(f"{name} must be a TCP port") from exc
    if port <= 0 or port > 65535:
        raise LauncherTunnelError(f"{name} must be between 1 and 65535")
    return port


def _read_process_output(process: Any) -> str:
    stdout = getattr(process, "stdout", None)
    if stdout is None or not hasattr(stdout, "read"):
        return ""
    try:
        text = stdout.read()
    except Exception:
        return ""
    return str(text).strip()


__all__ = [
    "LauncherTunnel",
    "LauncherTunnelError",
    "LauncherTunnelSnapshot",
    "build_ssh_tunnel_command",
    "resolve_ssh_executable",
    "tcp_port_is_available",
]
