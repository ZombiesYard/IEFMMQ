from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
from ctypes import wintypes
from typing import Callable

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
HC_ACTION = 0
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_XBUTTONDOWN = 0x020B
WM_QUIT = 0x0012
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
DEFAULT_GLOBAL_HELP_COOLDOWN_MS = 400

if sys.platform == "win32":
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
    _USER32.TranslateMessage.argtypes = (ctypes.POINTER(wintypes.MSG),)
    _USER32.TranslateMessage.restype = wintypes.BOOL
    _USER32.DispatchMessageW.argtypes = (ctypes.POINTER(wintypes.MSG),)
    _USER32.DispatchMessageW.restype = wintypes.LPARAM
    _USER32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    _USER32.PostThreadMessageW.restype = wintypes.BOOL
    _USER32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
    _USER32.GetAsyncKeyState.restype = ctypes.c_short
    _KERNEL32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
    _KERNEL32.GetModuleHandleW.restype = wintypes.HMODULE
    _KERNEL32.GetCurrentThreadId.argtypes = ()
    _KERNEL32.GetCurrentThreadId.restype = wintypes.DWORD
else:
    _USER32 = None
    _KERNEL32 = None
    LowLevelProc = None


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
    if sys.platform != "win32" or _USER32 is None:
        return 0
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


class WindowsGlobalHelpTrigger:
    def __init__(
        self,
        *,
        hotkey: str,
        modifiers: str = "",
        cooldown_ms: int = DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
        emit_hook: Callable[[], None] | None = None,
    ) -> None:
        self.trigger_kind, self.trigger_code = _parse_hotkey(hotkey)
        self.required_modifiers = _modifier_mask(modifiers)
        self.hotkey_label = f"{modifiers}+{hotkey}" if modifiers else hotkey
        self.cooldown_s = max(0.0, float(cooldown_ms) / 1000.0)
        self.emit_hook = emit_hook
        self._last_sent = 0.0
        self._queue: queue.SimpleQueue[None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._started = False
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._stop_requested = False
        self._startup_error: Exception | None = None
        self._loop_thread_id = 0
        self._keyboard_hook = None
        self._mouse_hook = None
        self._keyboard_proc = None
        self._mouse_proc = None
        self._last_callback_error_at = 0.0

    def start(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("global help hotkey is only supported on Windows")
        if self._started:
            return
        self._started = True
        self._thread.start()
        self._ready.wait(timeout=2.0)
        if self._startup_error is not None:
            raise self._startup_error
        if not self._ready.is_set():
            raise RuntimeError("timed out while starting global help hotkey hook")

    def poll(self) -> bool:
        try:
            self._queue.get_nowait()
            return True
        except queue.Empty:
            return False

    def close(self) -> None:
        self.request_stop()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def request_stop(self) -> None:
        self._stop_requested = True
        if sys.platform == "win32" and self._loop_thread_id and _USER32 is not None:
            _USER32.PostThreadMessageW(self._loop_thread_id, WM_QUIT, 0, 0)

    def _run_loop(self) -> None:
        try:
            self._loop_thread_id = int(_KERNEL32.GetCurrentThreadId())
            self._install_hooks()
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            self._closed.set()
            return

        self._ready.set()
        msg = wintypes.MSG()
        try:
            while not self._stop_requested:
                result = int(_USER32.GetMessageW(ctypes.byref(msg), None, 0, 0))
                if result == -1:
                    error = ctypes.get_last_error()
                    raise OSError(f"GetMessageW failed (GetLastError={error})")
                if result == 0:
                    break
                if int(msg.message) == WM_QUIT:
                    break
                _USER32.TranslateMessage(ctypes.byref(msg))
                _USER32.DispatchMessageW(ctypes.byref(msg))
        finally:
            self._uninstall_hooks()
            self._closed.set()

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

    def _uninstall_hooks(self) -> None:
        if self._keyboard_hook:
            _USER32.UnhookWindowsHookEx(self._keyboard_hook)
            self._keyboard_hook = None
        if self._mouse_hook:
            _USER32.UnhookWindowsHookEx(self._mouse_hook)
            self._mouse_hook = None

    def _emit_help(self) -> None:
        now = time.monotonic()
        if self.cooldown_s > 0 and (now - self._last_sent) < self.cooldown_s:
            return
        self._last_sent = now
        if self.emit_hook is not None:
            self.emit_hook()
            return
        self._queue.put(None)

    def _report_callback_error(self, hook_name: str, exc: Exception) -> None:
        now = time.monotonic()
        if (now - self._last_callback_error_at) < 5.0:
            return
        self._last_callback_error_at = now
        print(
            f"[GLOBAL_HELP_TRIGGER] {hook_name} hook callback failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    def _keyboard_callback(self, code: int, wparam: int, lparam: int) -> int:
        try:
            if code == HC_ACTION and self.trigger_kind == "keyboard" and wparam in {WM_KEYDOWN, WM_SYSKEYDOWN}:
                event = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if not (event.flags & LLKHF_INJECTED) and int(event.vkCode) == self.trigger_code:
                    if _modifier_state_mask() == self.required_modifiers:
                        self._emit_help()
        except Exception as exc:
            self._report_callback_error("keyboard", exc)
        return _USER32.CallNextHookEx(None, code, wparam, lparam)

    def _mouse_callback(self, code: int, wparam: int, lparam: int) -> int:
        try:
            if code == HC_ACTION and self.trigger_kind == "mouse" and wparam == WM_XBUTTONDOWN:
                event = ctypes.cast(lparam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                button = (int(event.mouseData) >> 16) & 0xFFFF
                if button == self.trigger_code and _modifier_state_mask() == self.required_modifiers:
                    self._emit_help()
        except Exception as exc:
            self._report_callback_error("mouse", exc)
        return _USER32.CallNextHookEx(None, code, wparam, lparam)
