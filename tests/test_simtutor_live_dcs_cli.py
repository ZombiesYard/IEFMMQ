from __future__ import annotations

import builtins
import sys


def test_simtutor_live_dcs_subcommand_forwards_args(monkeypatch) -> None:
    from simtutor import __main__ as simtutor_main

    captured: dict[str, object] = {}

    def _fake_live_dcs_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr("live_dcs.main", _fake_live_dcs_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "live-dcs",
            "--global-help-hotkey",
            "X1",
            "--vision-session-id",
            "sess-live",
        ],
    )

    result = simtutor_main.main()

    assert result == 0
    assert captured["argv"] == ["--global-help-hotkey", "X1", "--vision-session-id", "sess-live"]


def test_simtutor_live_dcs_subcommand_reports_exception_type_to_stderr(
    monkeypatch,
    capsys,
) -> None:
    from simtutor import __main__ as simtutor_main

    def _fake_live_dcs_main(argv):
        raise RuntimeError("boom")

    monkeypatch.setattr("live_dcs.main", _fake_live_dcs_main)
    monkeypatch.setattr(sys, "argv", ["simtutor", "live-dcs"])

    result = simtutor_main.main()
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "[LIVE_DCS] RuntimeError: boom" in captured.err


def test_simtutor_cli_startup_does_not_import_live_dcs_for_non_live_subcommands(
    monkeypatch,
    capsys,
) -> None:
    from simtutor import __main__ as simtutor_main

    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "live_dcs":
            raise AssertionError("live_dcs should not be imported during generic CLI startup")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    monkeypatch.setattr(sys, "argv", ["simtutor"])

    result = simtutor_main.main()
    captured = capsys.readouterr()

    assert result == 0
    assert "live-dcs" in captured.out
