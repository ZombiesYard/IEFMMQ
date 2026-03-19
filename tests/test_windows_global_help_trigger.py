from __future__ import annotations

import ctypes

import pytest

from adapters import windows_global_help_trigger
from adapters.windows_global_help_trigger import WM_XBUTTONDOWN, WindowsGlobalHelpTrigger


def test_report_callback_error_is_rate_limited(monkeypatch) -> None:
    trigger = WindowsGlobalHelpTrigger(hotkey="F8")
    calls: list[tuple[str, str]] = []
    moments = iter([10.0, 12.0, 16.0])

    monkeypatch.setattr("adapters.windows_global_help_trigger.time.monotonic", lambda: next(moments))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: calls.append((str(args[0]), str(kwargs.get("file")))),
    )

    trigger._report_callback_error("keyboard", RuntimeError("boom-1"))
    trigger._report_callback_error("keyboard", RuntimeError("boom-2"))
    trigger._report_callback_error("mouse", RuntimeError("boom-3"))

    assert len(calls) == 2
    assert "keyboard hook callback failed" in calls[0][0]
    assert "mouse hook callback failed" in calls[1][0]


def test_run_loop_raises_when_getmessagew_returns_error(monkeypatch) -> None:
    trigger = WindowsGlobalHelpTrigger(hotkey="F8")

    class DummyUser32:
        def PeekMessageW(self, *args, **kwargs):
            return 0

        def GetMessageW(self, *args, **kwargs):
            return -1

    class DummyKernel32:
        def GetCurrentThreadId(self):
            return 77

    monkeypatch.setattr("adapters.windows_global_help_trigger._USER32", DummyUser32())
    monkeypatch.setattr("adapters.windows_global_help_trigger._KERNEL32", DummyKernel32())
    monkeypatch.setattr("adapters.windows_global_help_trigger.sys.platform", "win32")
    monkeypatch.setattr(trigger, "_install_hooks", lambda: None)
    monkeypatch.setattr(trigger, "_uninstall_hooks", lambda: None)
    monkeypatch.setattr(
        "adapters.windows_global_help_trigger.ctypes.get_last_error",
        lambda: 1234,
        raising=False,
    )

    with pytest.raises(OSError, match="GetMessageW failed"):
        trigger._run_loop()


def test_close_warns_when_hook_thread_does_not_stop(monkeypatch) -> None:
    trigger = WindowsGlobalHelpTrigger(hotkey="F8")
    trigger._started = True

    class DummyThread:
        def __init__(self) -> None:
            self.join_calls: list[float] = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout if timeout is not None else -1.0)

    class DummyEvent:
        def __init__(self) -> None:
            self.wait_calls: list[float] = []

        def wait(self, timeout: float | None = None) -> bool:
            self.wait_calls.append(timeout if timeout is not None else -1.0)
            return False

    thread = DummyThread()
    event = DummyEvent()
    trigger._thread = thread
    trigger._closed = event
    monkeypatch.setattr(trigger, "request_stop", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: calls.append(str(args[0])))

    trigger.close()

    assert event.wait_calls == [1.0]
    assert thread.join_calls == [1.0]
    assert any("shutdown timed out" in call for call in calls)


def test_run_loop_primes_message_queue_before_setting_ready(monkeypatch) -> None:
    trigger = WindowsGlobalHelpTrigger(hotkey="F8")
    calls: list[str] = []

    class DummyUser32:
        def PeekMessageW(self, *args, **kwargs):
            calls.append("peek")
            return 0

        def GetMessageW(self, *args, **kwargs):
            calls.append("get")
            return 0

    class DummyKernel32:
        def GetCurrentThreadId(self):
            return 77

    monkeypatch.setattr("adapters.windows_global_help_trigger._USER32", DummyUser32())
    monkeypatch.setattr("adapters.windows_global_help_trigger._KERNEL32", DummyKernel32())
    monkeypatch.setattr(trigger, "_install_hooks", lambda: calls.append("install"))
    monkeypatch.setattr(trigger, "_uninstall_hooks", lambda: calls.append("uninstall"))

    trigger._run_loop()

    assert calls[:3] == ["install", "peek", "get"]


def test_request_stop_retries_post_thread_message_when_queue_not_ready(monkeypatch) -> None:
    trigger = WindowsGlobalHelpTrigger(hotkey="F8")
    trigger._loop_thread_id = 99
    calls: list[int] = []

    class DummyUser32:
        def PostThreadMessageW(self, thread_id, message, wparam, lparam):
            calls.append(thread_id)
            return 1 if len(calls) == 3 else 0

    monkeypatch.setattr("adapters.windows_global_help_trigger._USER32", DummyUser32())
    monkeypatch.setattr("adapters.windows_global_help_trigger.sys.platform", "win32")
    sleeps: list[float] = []
    monkeypatch.setattr("adapters.windows_global_help_trigger.time.sleep", lambda value: sleeps.append(float(value)))

    trigger.request_stop()

    assert calls == [99, 99, 99]
    assert sleeps == [0.01, 0.01]


def test_mouse_callback_ignores_injected_xbutton_events(monkeypatch) -> None:
    trigger = WindowsGlobalHelpTrigger(hotkey="X1")
    trigger.trigger_kind = "mouse"
    trigger.trigger_code = 0x0001
    trigger.required_modifiers = 0
    emitted: list[str] = []
    monkeypatch.setattr(trigger, "_emit_help", lambda: emitted.append("help"))
    monkeypatch.setattr("adapters.windows_global_help_trigger._modifier_state_mask", lambda: 0)

    class DummyUser32:
        def CallNextHookEx(self, *args, **kwargs):
            return 0

    monkeypatch.setattr("adapters.windows_global_help_trigger._USER32", DummyUser32())
    event = type("DummyMouseEvent", (), {"mouseData": 0x0001 << 16, "flags": 0x00000001})()

    class DummyPointer:
        def __init__(self, contents):
            self.contents = contents

    monkeypatch.setattr(
        windows_global_help_trigger.ctypes,
        "cast",
        lambda value, pointer_type: DummyPointer(event),
    )

    trigger._mouse_callback(0, WM_XBUTTONDOWN, ctypes.c_void_p(0))

    assert emitted == []
