"""
Register a global Windows low-level keyboard/mouse hook and send UDP help
triggers to live_dcs.py.

Examples:
  python tools/send_help_hotkey.py --hotkey F8 --modifiers Ctrl+Shift
  python tools/send_help_hotkey.py --hotkey X1
  python tools/send_help_hotkey.py --hotkey X2 --cooldown-ms 600
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from typing import Any

from adapters.windows_global_help_trigger import DEFAULT_GLOBAL_HELP_COOLDOWN_MS, WindowsGlobalHelpTrigger


class HelpTriggerHook:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        hotkey: str,
        modifiers: str,
        cooldown_ms: int,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.hotkey = hotkey
        self.modifiers = modifiers
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._payload = b"help"
        self._stop_requested = False
        self._trigger = WindowsGlobalHelpTrigger(
            hotkey=hotkey,
            modifiers=modifiers,
            cooldown_ms=cooldown_ms,
            emit_hook=self._send_help,
        )

    def run(self) -> int:
        self._install_signal_handlers()
        try:
            self._trigger.start()
            print(f"[HOTKEY] listening on global trigger {self._trigger.hotkey_label} -> udp {self.host}:{self.port}")
            print("[HOTKEY] press Ctrl+C to exit")
            while not self._stop_requested:
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.request_stop()
        finally:
            self.close()
        return 0

    def close(self) -> None:
        self._trigger.close()
        self._sock.close()

    def request_stop(self) -> None:
        self._stop_requested = True
        self._trigger.request_stop()

    def _install_signal_handlers(self) -> None:
        def _sigint_handler(_signum: int, _frame: Any) -> None:
            self.request_stop()

        signal.signal(signal.SIGINT, _sigint_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _sigint_handler)

    def _send_help(self) -> None:
        try:
            self._sock.sendto(self._payload, (self.host, self.port))
        except OSError as exc:
            print(f"[HOTKEY] send failed to {self.host}:{self.port}: {type(exc).__name__}: {exc}")
            return
        print(f"[HOTKEY] sent help to {self.host}:{self.port}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register a global Windows trigger and send UDP help packets.")
    parser.add_argument("--host", default="127.0.0.1", help="live_dcs.py help UDP host")
    parser.add_argument("--port", type=int, default=7792, help="live_dcs.py help UDP port")
    parser.add_argument("--hotkey", default="F8", help="Trigger key: ESC, F1-F24, X1/MOUSE4, X2/MOUSE5")
    parser.add_argument("--modifiers", default="", help="Optional modifiers joined by +, e.g. Ctrl+Shift")
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
        help="Debounce window in milliseconds",
    )
    return parser


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("send_help_hotkey.py only runs on Windows")
    args = build_arg_parser().parse_args()
    runner = HelpTriggerHook(
        host=args.host,
        port=args.port,
        hotkey=args.hotkey,
        modifiers=args.modifiers,
        cooldown_ms=args.cooldown_ms,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
