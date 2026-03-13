from __future__ import annotations

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
