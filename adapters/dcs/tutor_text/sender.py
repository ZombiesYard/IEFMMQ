from __future__ import annotations

import socket
from typing import Any

from adapters.dcs.tutor_text.codec import command_from_message, decode_ack, encode_command


class DcsTutorTextSender:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7783,
        *,
        timeout: float = 0.5,
        enabled: bool = True,
    ) -> None:
        self.server = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.enabled = enabled

    def close(self) -> None:
        self.sock.close()

    def _failed_result(
        self,
        *,
        cmd: dict[str, Any] | None,
        text: str,
        display_time_s: float,
        clear_view: bool,
        failure_class: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "cmd_id": cmd.get("cmd_id") if isinstance(cmd, dict) else None,
            "status": "failed",
            "failure_class": failure_class,
            "reason": reason,
            "text": text,
            "display_time_s": display_time_s,
            "clear_view": clear_view,
        }

    def send_text(
        self,
        text: str,
        *,
        display_time_s: float = 12.0,
        clear_view: bool = False,
        expect_ack: bool = True,
        cmd_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_text = text.strip() if isinstance(text, str) else ""
        if not self.enabled:
            return self._failed_result(
                cmd=None,
                text=normalized_text,
                display_time_s=display_time_s,
                clear_view=clear_view,
                failure_class="sender_disabled",
                reason="tutor text sender disabled",
            )

        cmd = command_from_message(
            normalized_text,
            display_time_s=display_time_s,
            clear_view=clear_view,
            cmd_id=cmd_id,
        )
        try:
            self.sock.sendto(encode_command(cmd), self.server)
        except OSError as exc:
            return self._failed_result(
                cmd=cmd,
                text=normalized_text,
                display_time_s=display_time_s,
                clear_view=clear_view,
                failure_class="transport_error",
                reason=str(exc),
            )

        if not expect_ack:
            return {
                "cmd_id": cmd["cmd_id"],
                "status": "ok",
                "text": normalized_text,
                "display_time_s": display_time_s,
                "clear_view": clear_view,
                "ack_skipped": True,
            }

        try:
            payload, _addr = self.sock.recvfrom(4096)
        except socket.timeout:
            return self._failed_result(
                cmd=cmd,
                text=normalized_text,
                display_time_s=display_time_s,
                clear_view=clear_view,
                failure_class="ack_timeout",
                reason="timed out waiting for DCS tutor text ack",
            )
        except OSError as exc:
            return self._failed_result(
                cmd=cmd,
                text=normalized_text,
                display_time_s=display_time_s,
                clear_view=clear_view,
                failure_class="transport_error",
                reason=str(exc),
            )

        try:
            ack = decode_ack(payload)
        except ValueError as exc:
            return self._failed_result(
                cmd=cmd,
                text=normalized_text,
                display_time_s=display_time_s,
                clear_view=clear_view,
                failure_class="invalid_ack",
                reason=str(exc),
            )
        result = {
            "cmd_id": ack.get("cmd_id"),
            "status": ack.get("status"),
            "reason": ack.get("reason"),
            "text": normalized_text,
            "display_time_s": display_time_s,
            "clear_view": clear_view,
        }
        if ack.get("status") == "failed":
            result["failure_class"] = "remote_failure"
        return result
