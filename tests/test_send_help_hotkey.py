from __future__ import annotations

from tools import send_help_hotkey


def test_help_trigger_hook_reuses_shared_windows_trigger(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSocket:
        def __init__(self) -> None:
            self.sent: list[tuple[bytes, tuple[str, int]]] = []

        def sendto(self, payload: bytes, addr: tuple[str, int]) -> None:
            self.sent.append((payload, addr))

        def close(self) -> None:
            captured["socket_closed"] = True

    sock = FakeSocket()

    class FakeTrigger:
        def __init__(self, *, hotkey, modifiers="", cooldown_ms=0, emit_hook=None) -> None:
            captured["trigger_init"] = {
                "hotkey": hotkey,
                "modifiers": modifiers,
                "cooldown_ms": cooldown_ms,
                "emit_hook_is_callable": callable(emit_hook),
            }
            self.hotkey_label = f"{modifiers}+{hotkey}" if modifiers else hotkey
            self._emit_hook = emit_hook

        def start(self) -> None:
            if self._emit_hook is not None:
                self._emit_hook()

        def request_stop(self) -> None:
            captured["trigger_stop_called"] = True

        def close(self) -> None:
            captured["trigger_closed"] = True

    monkeypatch.setattr(send_help_hotkey.socket, "socket", lambda *args, **kwargs: sock)
    monkeypatch.setattr(send_help_hotkey, "WindowsGlobalHelpTrigger", FakeTrigger)

    runner = send_help_hotkey.HelpTriggerHook(
        host="127.0.0.1",
        port=7792,
        hotkey="X1",
        modifiers="Ctrl",
        cooldown_ms=650,
    )
    runner._trigger.start()
    runner.request_stop()
    runner.close()

    assert captured["trigger_init"] == {
        "hotkey": "X1",
        "modifiers": "Ctrl",
        "cooldown_ms": 650,
        "emit_hook_is_callable": True,
    }
    assert sock.sent == [(b"help", ("127.0.0.1", 7792))]
    assert captured["trigger_stop_called"] is True
    assert captured["trigger_closed"] is True
    assert captured["socket_closed"] is True


def test_send_help_logs_and_continues_on_socket_oserror(monkeypatch, capsys) -> None:
    class FailingSocket:
        def sendto(self, payload: bytes, addr: tuple[str, int]) -> None:
            raise OSError("network down")

        def close(self) -> None:
            return

    monkeypatch.setattr(send_help_hotkey.socket, "socket", lambda *args, **kwargs: FailingSocket())
    monkeypatch.setattr(
        send_help_hotkey,
        "WindowsGlobalHelpTrigger",
        lambda **kwargs: type(
            "FakeTrigger",
            (),
            {
                "hotkey_label": "Ctrl+X1",
                "request_stop": lambda self: None,
                "close": lambda self: None,
                "start": lambda self: None,
            },
        )(),
    )
    runner = send_help_hotkey.HelpTriggerHook(
        host="127.0.0.1",
        port=7792,
        hotkey="X1",
        modifiers="Ctrl",
        cooldown_ms=650,
    )

    runner._send_help()

    captured = capsys.readouterr()
    assert "send failed to 127.0.0.1:7792" in captured.out
