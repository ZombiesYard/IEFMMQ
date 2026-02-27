from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class BiosUiMapError(ValueError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_bios_to_ui_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "bios_to_ui.yaml"


def _default_ui_map_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "ui_map.yaml"


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise BiosUiMapError(f"{label} must be a YAML mapping: {path}")
    return data


def _load_allowed_targets(ui_map_path: Path) -> set[str]:
    ui_map = _load_yaml_mapping(ui_map_path, "ui_map.yaml")
    cockpit_elements = ui_map.get("cockpit_elements")
    if not isinstance(cockpit_elements, dict):
        raise BiosUiMapError(f"ui_map.yaml missing cockpit_elements mapping: {ui_map_path}")
    return {key for key in cockpit_elements if isinstance(key, str) and key}


def _normalize_targets(raw: Any, bios_key: str) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        targets = raw
    elif isinstance(raw, Mapping):
        targets = raw.get("targets")
    else:
        raise BiosUiMapError(
            f"bios_to_ui mapping for key {bios_key!r} must be string/list/mapping with targets"
        )

    if not isinstance(targets, list):
        raise BiosUiMapError(f"bios_to_ui mapping for key {bios_key!r} has non-list targets")

    ordered: list[str] = []
    seen: set[str] = set()
    for idx, target in enumerate(targets):
        if not isinstance(target, str) or not target:
            raise BiosUiMapError(
                f"bios_to_ui mapping for key {bios_key!r} targets[{idx}] must be non-empty string"
            )
        if target in seen:
            continue
        seen.add(target)
        ordered.append(target)
    return ordered


@dataclass(frozen=True)
class BiosUiMapper:
    rules: dict[str, tuple[str, ...]]
    allowed_targets: frozenset[str]

    @classmethod
    def from_yaml(
        cls,
        bios_to_ui_path: str | Path | None = None,
        ui_map_path: str | Path | None = None,
    ) -> "BiosUiMapper":
        bios_path = Path(bios_to_ui_path) if bios_to_ui_path else _default_bios_to_ui_path()
        ui_path = Path(ui_map_path) if ui_map_path else _default_ui_map_path()

        bios_map = _load_yaml_mapping(bios_path, "bios_to_ui.yaml")
        mappings = bios_map.get("mappings")
        if not isinstance(mappings, Mapping):
            raise BiosUiMapError(f"bios_to_ui.yaml missing mappings: {bios_path}")

        allowed_targets = _load_allowed_targets(ui_path)
        rules: dict[str, tuple[str, ...]] = {}

        for raw_key, raw_value in mappings.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise BiosUiMapError("bios_to_ui mappings key must be non-empty string")
            targets = _normalize_targets(raw_value, raw_key)
            for target in targets:
                if target not in allowed_targets:
                    raise BiosUiMapError(
                        f"bios_to_ui key {raw_key!r} references unknown ui target {target!r}"
                    )
            rules[raw_key] = tuple(targets)

        return cls(rules=rules, allowed_targets=frozenset(allowed_targets))

    def targets_for_key(self, bios_key: str) -> list[str]:
        if not isinstance(bios_key, str):
            raise TypeError("bios_key must be a string")
        return list(self.rules.get(bios_key, ()))

    def map_delta(self, delta: Mapping[str, Any]) -> list[str]:
        if not isinstance(delta, Mapping):
            raise TypeError("delta must be a mapping")

        ordered: list[str] = []
        seen: set[str] = set()

        for key in delta.keys():
            if not isinstance(key, str):
                continue
            targets = self.rules.get(key, ())
            for target in targets:
                if target in seen:
                    continue
                seen.add(target)
                ordered.append(target)
        return ordered


__all__ = ["BiosUiMapError", "BiosUiMapper"]
