from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator, FormatChecker

_BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PACK_PATH = _BASE_DIR / "packs/fa18c_startup/pack.yaml"
DEFAULT_UI_MAP_PATH = _BASE_DIR / "packs/fa18c_startup/ui_map.yaml"
DEFAULT_TAXONOMY_PATH = _BASE_DIR / "packs/fa18c_startup/taxonomy.yaml"

_DEFAULT_ERROR_CATEGORIES = ["OM", "CO", "OR", "PA", "SV"]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return data


def _step_ids(pack_path: Path) -> list[str]:
    data = _load_yaml(pack_path)
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError("pack.yaml 'steps' must be a list")
    ids = [step.get("id") for step in steps if isinstance(step, dict) and isinstance(step.get("id"), str)]
    if not ids:
        raise ValueError("pack.yaml missing step ids")
    return ids


def _ui_targets(ui_map_path: Path) -> list[str]:
    data = _load_yaml(ui_map_path)
    elements = data.get("cockpit_elements")
    if not isinstance(elements, dict):
        raise ValueError("ui_map.yaml 'cockpit_elements' must be a mapping")
    targets = [name for name in elements if isinstance(name, str)]
    if not targets:
        raise ValueError("ui_map.yaml missing cockpit element keys")
    return targets


def _error_categories(taxonomy_path: Path | None) -> list[str]:
    if taxonomy_path and taxonomy_path.exists():
        data = _load_yaml(taxonomy_path)
        taxonomy = data.get("taxonomy")
        if isinstance(taxonomy, dict):
            categories = taxonomy.get("categories")
            if isinstance(categories, list):
                codes = [c.get("code") for c in categories if isinstance(c, dict) and isinstance(c.get("code"), str)]
                if codes:
                    return codes
    return _DEFAULT_ERROR_CATEGORIES


def build_help_response_schema(
    pack_path: str | Path = DEFAULT_PACK_PATH,
    ui_map_path: str | Path = DEFAULT_UI_MAP_PATH,
    taxonomy_path: str | Path | None = DEFAULT_TAXONOMY_PATH,
) -> dict[str, Any]:
    pack = Path(pack_path)
    ui_map = Path(ui_map_path)
    taxonomy = Path(taxonomy_path) if taxonomy_path is not None else None

    step_ids = _step_ids(pack)
    targets = _ui_targets(ui_map)
    error_categories = _error_categories(taxonomy)

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "HelpResponse",
        "type": "object",
        "additionalProperties": False,
        "required": ["diagnosis", "next", "overlay", "explanations", "confidence"],
        "properties": {
            "diagnosis": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step_id", "error_category"],
                "properties": {
                    "step_id": {"type": "string", "enum": step_ids},
                    "error_category": {"type": "string", "enum": error_categories},
                },
            },
            "next": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step_id"],
                "properties": {"step_id": {"type": "string", "enum": step_ids}},
            },
            "overlay": {
                "type": "object",
                "additionalProperties": False,
                "required": ["targets"],
                "properties": {
                    "targets": {
                        "type": "array",
                        "items": {"type": "string", "enum": targets},
                        "minItems": 1,
                    }
                },
            },
            "explanations": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


def validate_help_response(
    payload: Mapping[str, Any],
    *,
    pack_path: str | Path = DEFAULT_PACK_PATH,
    ui_map_path: str | Path = DEFAULT_UI_MAP_PATH,
    taxonomy_path: str | Path | None = DEFAULT_TAXONOMY_PATH,
) -> None:
    schema = build_help_response_schema(pack_path=pack_path, ui_map_path=ui_map_path, taxonomy_path=taxonomy_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        err = errors[0]
        location = ".".join(str(p) for p in err.path) or "<root>"
        raise ValueError(f"HelpResponse invalid at {location}: {err.message}")

