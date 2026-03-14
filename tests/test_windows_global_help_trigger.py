from __future__ import annotations

import pytest

from adapters.windows_global_help_trigger import WindowsGlobalHelpTrigger


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
        def GetMessageW(self, *args, **kwargs):
            return -1

    class DummyKernel32:
        def GetCurrentThreadId(self):
            return 77

    monkeypatch.setattr("adapters.windows_global_help_trigger._USER32", DummyUser32())
    monkeypatch.setattr("adapters.windows_global_help_trigger._KERNEL32", DummyKernel32())
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
