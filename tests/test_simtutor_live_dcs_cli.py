from __future__ import annotations

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
