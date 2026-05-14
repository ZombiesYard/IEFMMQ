from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from adapters.dcs.tutor_text import DcsTutorTextSender
from tools.install_dcs_hook import DEFAULT_TUTOR_TEXT_HOST, DEFAULT_TUTOR_TEXT_PORT


@dataclass(frozen=True)
class TutorTextConfig:
    host: str
    port: int
    config_path: Path | None
    source: str


def _match_optional(content: str, pattern: str) -> str | None:
    match = re.search(pattern, content, flags=re.DOTALL)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _find_wsl_saved_games_dir() -> Path | None:
    base = Path("/mnt/c/Users")
    if not base.exists():
        return None
    matches = sorted(path for path in base.glob("*/Saved Games/DCS") if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    env_user = os.getenv("WIN_USERNAME", "").strip()
    if env_user:
        candidate = base / env_user / "Saved Games" / "DCS"
        if candidate.is_dir():
            return candidate
    return None


def resolve_saved_games_dir(raw_saved_games_dir: str | None) -> Path:
    if isinstance(raw_saved_games_dir, str) and raw_saved_games_dir.strip():
        return Path(raw_saved_games_dir).expanduser().resolve()
    env_path = os.getenv("SIMTUTOR_SAVED_GAMES_DIR", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    wsl_dir = _find_wsl_saved_games_dir()
    if wsl_dir is not None:
        return wsl_dir.resolve()
    return (Path.home() / "Saved Games" / "DCS").resolve()


def _config_path(saved_games_dir: Path) -> Path:
    return saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua"


def load_tutor_text_config(config_path: Path) -> TutorTextConfig:
    resolved = config_path.expanduser().resolve()
    if not resolved.exists():
        return TutorTextConfig(
            host=DEFAULT_TUTOR_TEXT_HOST,
            port=DEFAULT_TUTOR_TEXT_PORT,
            config_path=resolved,
            source="default_missing_config",
        )
    content = resolved.read_text(encoding="utf-8")
    tutor_text_match = re.search(r"tutor_text\s*=\s*\{(.*?)\}", content, flags=re.DOTALL)
    if tutor_text_match is None:
        return TutorTextConfig(
            host=DEFAULT_TUTOR_TEXT_HOST,
            port=DEFAULT_TUTOR_TEXT_PORT,
            config_path=resolved,
            source="default_missing_block",
        )
    tutor_text_content = tutor_text_match.group(1)
    host = _match_optional(tutor_text_content, r'host\s*=\s*"([^"]+)"') or DEFAULT_TUTOR_TEXT_HOST
    raw_port = _match_optional(tutor_text_content, r"port\s*=\s*(\d+)")
    port = int(raw_port) if raw_port is not None else DEFAULT_TUTOR_TEXT_PORT
    return TutorTextConfig(
        host=host,
        port=port,
        config_path=resolved,
        source="config_tutor_text_block",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a direct DCS tutor-text test message through the Saved Games Lua hook.")
    parser.add_argument("message", nargs="?", default="SimTutor 直连测试消息", help="Text to display inside DCS.")
    parser.add_argument("--saved-games-dir", default=None, help="Saved Games/DCS root. Defaults to current machine detection.")
    parser.add_argument("--config-path", default=None, help="Optional explicit SimTutorConfig.lua path.")
    parser.add_argument("--host", default=None, help="Optional override for tutor text UDP host.")
    parser.add_argument("--port", type=int, default=None, help="Optional override for tutor text UDP port.")
    parser.add_argument("--display-time-s", type=float, default=12.0, help="How long DCS should display the text.")
    parser.add_argument("--clear-view", action="store_true", help="Ask DCS to clear previous top-right text before showing this message.")
    parser.add_argument("--timeout", type=float, default=1.0, help="UDP ack wait timeout in seconds.")
    parser.add_argument("--no-ack", action="store_true", help="Do not wait for the Lua hook ack.")
    parser.add_argument("--cmd-id", default=None, help="Optional explicit UUID command id for debugging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.port is not None and int(args.port) <= 0:
        raise ValueError("--port must be positive")
    if float(args.display_time_s) <= 0:
        raise ValueError("--display-time-s must be > 0")
    if float(args.timeout) <= 0:
        raise ValueError("--timeout must be > 0")

    saved_games_dir = resolve_saved_games_dir(args.saved_games_dir)
    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else _config_path(saved_games_dir)
    config = load_tutor_text_config(config_path)
    host = str(args.host).strip() if isinstance(args.host, str) and args.host.strip() else config.host
    port = int(args.port) if args.port is not None else config.port

    sender = DcsTutorTextSender(host=host, port=port, timeout=float(args.timeout), enabled=True)
    try:
        result = sender.send_text(
            args.message,
            display_time_s=float(args.display_time_s),
            clear_view=bool(args.clear_view),
            expect_ack=not bool(args.no_ack),
            cmd_id=args.cmd_id,
        )
    finally:
        sender.close()

    print(
        json.dumps(
            {
                "saved_games_dir": str(saved_games_dir),
                "config_path": str(config.config_path) if config.config_path is not None else None,
                "config_source": config.source,
                "host": host,
                "port": port,
                "result": result,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
