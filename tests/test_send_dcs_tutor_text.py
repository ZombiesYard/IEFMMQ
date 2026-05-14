from __future__ import annotations

import json
from pathlib import Path

from tools.send_dcs_tutor_text import (
    DEFAULT_TUTOR_TEXT_HOST,
    DEFAULT_TUTOR_TEXT_PORT,
    TutorTextConfig,
    load_tutor_text_config,
    main,
    resolve_saved_games_dir,
)


def test_load_tutor_text_config_reads_explicit_block(tmp_path: Path) -> None:
    config_path = tmp_path / "SimTutorConfig.lua"
    config_path.write_text(
        'return {\n'
        '    tutor_text = {\n'
        '        host = "10.0.0.7",\n'
        "        port = 8899,\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )

    config = load_tutor_text_config(config_path)

    assert config == TutorTextConfig(
        host="10.0.0.7",
        port=8899,
        config_path=config_path.resolve(),
        source="config_tutor_text_block",
    )


def test_load_tutor_text_config_falls_back_when_block_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "SimTutorConfig.lua"
    config_path.write_text("return {\n    vision = {},\n}\n", encoding="utf-8")

    config = load_tutor_text_config(config_path)

    assert config.host == DEFAULT_TUTOR_TEXT_HOST
    assert config.port == DEFAULT_TUTOR_TEXT_PORT
    assert config.source == "default_missing_block"


def test_resolve_saved_games_dir_prefers_explicit_argument(tmp_path: Path) -> None:
    resolved = resolve_saved_games_dir(str(tmp_path / "Saved Games" / "DCS"))

    assert resolved == (tmp_path / "Saved Games" / "DCS").resolve()


def test_main_uses_config_defaults_and_prints_result(monkeypatch, capsys, tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    config_dir = saved_games_dir / "Scripts" / "SimTutor"
    config_dir.mkdir(parents=True)
    (config_dir / "SimTutorConfig.lua").write_text(
        'return {\n'
        '    tutor_text = {\n'
        '        host = "127.0.0.9",\n'
        "        port = 7788,\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeSender:
        def __init__(self, **kwargs) -> None:
            captured["sender_kwargs"] = dict(kwargs)

        def send_text(self, message: str, **kwargs) -> dict[str, object]:
            captured["message"] = message
            captured["send_kwargs"] = dict(kwargs)
            return {"status": "ok", "cmd_id": "abc"}

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr("tools.send_dcs_tutor_text.DcsTutorTextSender", FakeSender)

    code = main(
        [
            "--saved-games-dir",
            str(saved_games_dir),
            "--display-time-s",
            "7",
            "--clear-view",
            "Tutor test",
        ]
    )

    payload = json.loads(capsys.readouterr().out.strip())
    assert code == 0
    assert captured["sender_kwargs"] == {
        "host": "127.0.0.9",
        "port": 7788,
        "timeout": 1.0,
        "enabled": True,
    }
    assert captured["message"] == "Tutor test"
    assert captured["send_kwargs"] == {
        "display_time_s": 7.0,
        "clear_view": True,
        "expect_ack": True,
        "cmd_id": None,
    }
    assert captured["closed"] is True
    assert payload["config_source"] == "config_tutor_text_block"
    assert payload["result"]["status"] == "ok"


def test_main_falls_back_to_default_port_when_current_config_has_no_tutor_text_block(monkeypatch, capsys, tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    config_dir = saved_games_dir / "Scripts" / "SimTutor"
    config_dir.mkdir(parents=True)
    (config_dir / "SimTutorConfig.lua").write_text("return {\n    vision = {},\n}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeSender:
        def __init__(self, **kwargs) -> None:
            captured["sender_kwargs"] = dict(kwargs)

        def send_text(self, message: str, **kwargs) -> dict[str, object]:
            captured["message"] = message
            captured["send_kwargs"] = dict(kwargs)
            return {"status": "failed", "failure_class": "ack_timeout"}

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr("tools.send_dcs_tutor_text.DcsTutorTextSender", FakeSender)

    code = main(["--saved-games-dir", str(saved_games_dir), "Fallback test"])

    payload = json.loads(capsys.readouterr().out.strip())
    assert code == 1
    assert captured["sender_kwargs"]["host"] == DEFAULT_TUTOR_TEXT_HOST
    assert captured["sender_kwargs"]["port"] == DEFAULT_TUTOR_TEXT_PORT
    assert payload["config_source"] == "default_missing_block"
    assert payload["result"]["failure_class"] == "ack_timeout"
