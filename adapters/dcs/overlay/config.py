from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DcsOverlayConfigInspection:
    config_path: Path
    exists: bool
    hilite_id: int | None = None
    hilite_ids: tuple[int, ...] = ()
    hilite_ids_declared: bool = False

    @property
    def declared_slot_count(self) -> int:
        if self.hilite_ids_declared:
            return len(self.hilite_ids)
        return 1 if self.hilite_id is not None else 0


def default_simtutor_config_path(*, dcs_variant: str = "DCS") -> Path:
    return Path.home() / "Saved Games" / dcs_variant / "Scripts" / "SimTutor" / "SimTutorConfig.lua"


def simtutor_config_path_from_saved_games_dir(saved_games_dir: str | Path | None) -> Path | None:
    if saved_games_dir is None:
        return None
    return Path(saved_games_dir).expanduser() / "Scripts" / "SimTutor" / "SimTutorConfig.lua"


def _strip_line_comments(text: str) -> str:
    return re.sub(r"--[^\r\n]*", "", text)


def _find_named_table(text: str, name: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*{{", text)
    if not match:
        return None
    start = text.find("{", match.start())
    depth = 0
    quote: str | None = None
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx]
    return None


def _parse_nonnegative_int(value: str) -> int | None:
    parsed = int(value)
    return parsed if parsed >= 0 else None


def _parse_hilite_id(overlay_block: str) -> int | None:
    match = re.search(r"\bhilite_id\s*=\s*(-?\d+)", overlay_block)
    if not match:
        return None
    return _parse_nonnegative_int(match.group(1))


def _parse_hilite_ids(overlay_block: str) -> tuple[int, ...]:
    raw_table = _find_named_table(overlay_block, "hilite_ids")
    if raw_table is None:
        return ()
    seen: set[int] = set()
    ids: list[int] = []
    for raw_entry in raw_table.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        keyed_match = re.fullmatch(r"\[\s*-?\d+\s*\]\s*=\s*(-?\d+)", entry)
        if keyed_match:
            raw_value = keyed_match.group(1)
        elif "=" in entry:
            continue
        else:
            raw_value = entry
        if not re.fullmatch(r"-?\d+", raw_value.strip()):
            continue
        value = _parse_nonnegative_int(raw_value.strip())
        if value is not None and value not in seen:
            seen.add(value)
            ids.append(value)
    return tuple(ids)


def inspect_overlay_config(config_path: Path) -> DcsOverlayConfigInspection:
    config_path = Path(config_path)
    if not config_path.exists():
        return DcsOverlayConfigInspection(config_path=config_path, exists=False)

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return DcsOverlayConfigInspection(config_path=config_path, exists=True)

    text = _strip_line_comments(text)
    overlay_block = _find_named_table(text, "overlay")
    if overlay_block is None:
        return DcsOverlayConfigInspection(config_path=config_path, exists=True)

    hilite_ids_declared = re.search(r"\bhilite_ids\s*=", overlay_block) is not None
    return DcsOverlayConfigInspection(
        config_path=config_path,
        exists=True,
        hilite_id=_parse_hilite_id(overlay_block),
        hilite_ids=_parse_hilite_ids(overlay_block),
        hilite_ids_declared=hilite_ids_declared,
    )


def build_multi_target_overlay_config_warning(
    *,
    max_overlay_targets: int,
    config_path: Path | None = None,
) -> str | None:
    requested_slots = max(0, int(max_overlay_targets))
    if requested_slots <= 1:
        return None

    effective_config_path = config_path or default_simtutor_config_path()
    inspection = inspect_overlay_config(effective_config_path)
    if not inspection.exists:
        return None

    declared_slots = inspection.declared_slot_count
    if declared_slots >= requested_slots:
        return None

    slot_word = "slot" if declared_slots == 1 else "slots"
    return (
        f"--max-overlay-targets={requested_slots} requested, but {inspection.config_path} "
        f"declares only {declared_slots} DCS highlight {slot_word}. "
        "Update the installed DCS hook/config so overlay.hilite_ids contains enough IDs, "
        "for example by rerunning python -m tools.install_dcs_hook --install-composite-panel."
    )
