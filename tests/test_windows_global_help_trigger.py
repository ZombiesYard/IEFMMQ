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
