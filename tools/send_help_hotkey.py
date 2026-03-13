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
import ctypes
import signal
import socket
import sys
import time
from ctypes import wintypes

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
HC_ACTION = 0
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_XBUTTONDOWN = 0x020B
CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_F1 = 0x70
VK_ESCAPE = 0x1B
XBUTTON1 = 0x0001
XBUTTON2 = 0x0002
LLKHF_INJECTED = 0x00000010
WM_QUIT = 0x0012

_USER32 = ctypes.WinDLL("user32", use_last_error=True)
_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


LowLevelProc = ctypes.WINFUNCTYPE(wintypes.LPARAM, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

_USER32.SetWindowsHookExW.argtypes = (ctypes.c_int, LowLevelProc, wintypes.HINSTANCE, wintypes.DWORD)
_USER32.SetWindowsHookExW.restype = wintypes.HHOOK
_USER32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
_USER32.CallNextHookEx.restype = wintypes.LPARAM
_USER32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
_USER32.UnhookWindowsHookEx.restype = wintypes.BOOL
_USER32.GetMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
_USER32.GetMessageW.restype = wintypes.BOOL
_USER32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
_USER32.PostThreadMessageW.restype = wintypes.BOOL
_USER32.TranslateMessage.argtypes = (ctypes.POINTER(wintypes.MSG),)
_USER32.TranslateMessage.restype = wintypes.BOOL
_USER32.DispatchMessageW.argtypes = (ctypes.POINTER(wintypes.MSG),)
_USER32.DispatchMessageW.restype = wintypes.LPARAM
_USER32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
_USER32.GetAsyncKeyState.restype = ctypes.c_short
_KERNEL32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
_KERNEL32.GetModuleHandleW.restype = wintypes.HMODULE
_KERNEL32.GetCurrentThreadId.argtypes = ()
_KERNEL32.GetCurrentThreadId.restype = wintypes.DWORD
_KERNEL32.SetConsoleCtrlHandler.argtypes = (ctypes.c_void_p, wintypes.BOOL)
_KERNEL32.SetConsoleCtrlHandler.restype = wintypes.BOOL


def _modifier_mask(text: str) -> int:
    mask = 0
    raw = str(text).strip()
    if not raw:
        return mask
    for item in raw.split("+"):
        token = item.strip().lower()
        if token in {"ctrl", "control"}:
            mask |= 0x01
        elif token == "alt":
            mask |= 0x02
        elif token == "shift":
            mask |= 0x04
        elif token in {"win", "windows"}:
            mask |= 0x08
        elif token:
            raise ValueError(f"unsupported modifier: {item.strip()}")
    return mask


def _parse_hotkey(name: str) -> tuple[str, int]:
    normalized = str(name).strip().upper()
    if normalized == "ESC":
        return ("keyboard", VK_ESCAPE)
    if normalized.startswith("F") and normalized[1:].isdigit():
        index = int(normalized[1:])
        if 1 <= index <= 24:
            return ("keyboard", VK_F1 + index - 1)
    if normalized in {"X1", "MOUSE4", "SIDE1"}:
        return ("mouse", XBUTTON1)
    if normalized in {"X2", "MOUSE5", "SIDE2"}:
        return ("mouse", XBUTTON2)
    raise ValueError("hotkey must be one of ESC, F1-F24, X1/MOUSE4, X2/MOUSE5")


def _modifier_state_mask() -> int:
    mask = 0
    if _USER32.GetAsyncKeyState(VK_CONTROL) & 0x8000:
        mask |= 0x01
    if _USER32.GetAsyncKeyState(VK_MENU) & 0x8000:
        mask |= 0x02
    if _USER32.GetAsyncKeyState(VK_SHIFT) & 0x8000:
        mask |= 0x04
    if (_USER32.GetAsyncKeyState(VK_LWIN) | _USER32.GetAsyncKeyState(VK_RWIN)) & 0x8000:
        mask |= 0x08
    return mask


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
        self.trigger_kind, self.trigger_code = _parse_hotkey(hotkey)
        self.required_modifiers = _modifier_mask(modifiers)
        self.hotkey_label = f"{modifiers}+{hotkey}" if modifiers else hotkey
        self.cooldown_s = max(0.0, float(cooldown_ms) / 1000.0)
        self._last_sent = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._payload = b"help"
        self._keyboard_hook = None
        self._mouse_hook = None
        self._keyboard_proc = None
        self._mouse_proc = None
        self._ctrl_handler = None
        self._loop_thread_id = 0
        self._stop_requested = False

    def run(self) -> int:
        self._install_hooks()
        self._install_console_handlers()
        self._loop_thread_id = int(_KERNEL32.GetCurrentThreadId())
        msg = wintypes.MSG()
        print(f"[HOTKEY] listening on global trigger {self.hotkey_label} -> udp {self.host}:{self.port}")
        print("[HOTKEY] press Ctrl+C to exit")
        try:
            while _USER32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                if self._stop_requested or int(msg.message) == WM_QUIT:
                    break
                _USER32.TranslateMessage(ctypes.byref(msg))
                _USER32.DispatchMessageW(ctypes.byref(msg))
        except KeyboardInterrupt:
            self.request_stop()
        finally:
            self.close()
        return 0

    def close(self) -> None:
        if self._keyboard_hook:
            _USER32.UnhookWindowsHookEx(self._keyboard_hook)
            self._keyboard_hook = None
        if self._mouse_hook:
            _USER32.UnhookWindowsHookEx(self._mouse_hook)
            self._mouse_hook = None
        self._sock.close()

    def request_stop(self) -> None:
        self._stop_requested = True
        if self._loop_thread_id:
            _USER32.PostThreadMessageW(self._loop_thread_id, WM_QUIT, 0, 0)

    def _install_hooks(self) -> None:
        module = _KERNEL32.GetModuleHandleW(None)
        if self.trigger_kind == "keyboard":
            self._keyboard_proc = LowLevelProc(self._keyboard_callback)
            self._keyboard_hook = _USER32.SetWindowsHookExW(WH_KEYBOARD_LL, self._keyboard_proc, module, 0)
            if not self._keyboard_hook:
                error = ctypes.get_last_error()
                raise OSError(f"failed to install global keyboard hook (GetLastError={error})")
        if self.trigger_kind == "mouse":
            self._mouse_proc = LowLevelProc(self._mouse_callback)
            self._mouse_hook = _USER32.SetWindowsHookExW(WH_MOUSE_LL, self._mouse_proc, module, 0)
            if not self._mouse_hook:
                error = ctypes.get_last_error()
                raise OSError(f"failed to install global mouse hook (GetLastError={error})")

    def _install_console_handlers(self) -> None:
        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        def _console_handler(ctrl_type: int) -> int:
            if ctrl_type in {
                CTRL_C_EVENT,
                CTRL_BREAK_EVENT,
                CTRL_CLOSE_EVENT,
                CTRL_LOGOFF_EVENT,
                CTRL_SHUTDOWN_EVENT,
            }:
                self.request_stop()
                return 1
            return 0

        self._ctrl_handler = _console_handler
        _KERNEL32.SetConsoleCtrlHandler(self._ctrl_handler, True)

        def _sigint_handler(_signum, _frame) -> None:
            self.request_stop()

        signal.signal(signal.SIGINT, _sigint_handler)

    def _send_help(self) -> None:
        now = time.monotonic()
        if self.cooldown_s > 0 and (now - self._last_sent) < self.cooldown_s:
            return
        self._last_sent = now
        self._sock.sendto(self._payload, (self.host, self.port))
        print(f"[HOTKEY] sent help to {self.host}:{self.port}")

    def _keyboard_callback(self, code: int, wparam: int, lparam: int) -> int:
        try:
            if code == HC_ACTION and self.trigger_kind == "keyboard" and wparam in {WM_KEYDOWN, WM_SYSKEYDOWN}:
                event = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if not (event.flags & LLKHF_INJECTED) and int(event.vkCode) == self.trigger_code:
                    if _modifier_state_mask() == self.required_modifiers:
                        self._send_help()
        except KeyboardInterrupt:
            self.request_stop()
        except Exception:
            pass
        return _USER32.CallNextHookEx(None, code, wparam, lparam)

    def _mouse_callback(self, code: int, wparam: int, lparam: int) -> int:
        try:
            if code == HC_ACTION and self.trigger_kind == "mouse" and wparam == WM_XBUTTONDOWN:
                event = ctypes.cast(lparam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                button = (int(event.mouseData) >> 16) & 0xFFFF
                if button == self.trigger_code and _modifier_state_mask() == self.required_modifiers:
                    self._send_help()
        except KeyboardInterrupt:
            self.request_stop()
        except Exception:
            pass
        return _USER32.CallNextHookEx(None, code, wparam, lparam)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register a global Windows trigger and send UDP help packets.")
    parser.add_argument("--host", default="127.0.0.1", help="live_dcs.py help UDP host")
    parser.add_argument("--port", type=int, default=7792, help="live_dcs.py help UDP port")
    parser.add_argument("--hotkey", default="F8", help="Trigger key: ESC, F1-F24, X1/MOUSE4, X2/MOUSE5")
    parser.add_argument("--modifiers", default="", help="Optional modifiers joined by +, e.g. Ctrl+Shift")
    parser.add_argument("--cooldown-ms", type=int, default=400, help="Debounce window in milliseconds")
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
