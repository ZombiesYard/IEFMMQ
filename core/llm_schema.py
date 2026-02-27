"""
HelpResponse schema builder and validator for LLM outputs.

Unlike v1/v2 transport schemas stored under `simtutor/schemas/`, this schema is
generated dynamically from pack/ui map/taxonomy artifacts so enum constraints stay
simulator-agnostic and pack-driven at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
import copy
import re
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_REQUIRED_PROPERTY_RE = re.compile(r"'([^']+)' is a required property")
_HELP_RESPONSE_SCHEMA_ID = "https://simtutor.dev/schemas/v2/help_response.generated.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_pack_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "pack.yaml"


def _default_ui_map_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "ui_map.yaml"


def _default_taxonomy_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "taxonomy.yaml"


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a YAML mapping: {path}")
    return data


def _load_step_ids(pack_path: Path) -> list[str]:
    data = _load_yaml_mapping(pack_path, "pack.yaml")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"pack.yaml missing non-empty 'steps': {pack_path}")
    step_ids: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"pack.yaml step entry must be a mapping: {pack_path}")
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"pack.yaml step missing string id: {pack_path}")
        step_ids.append(step_id)
    return sorted(set(step_ids))


def _load_overlay_targets(pack_path: Path, ui_map_path: Path) -> list[str]:
    ui_data = _load_yaml_mapping(ui_map_path, "ui_map.yaml")
    cockpit_elements = ui_data.get("cockpit_elements")
    if not isinstance(cockpit_elements, dict) or not cockpit_elements:
        raise ValueError(f"ui_map.yaml missing non-empty 'cockpit_elements': {ui_map_path}")
    targets = [target for target in cockpit_elements.keys() if isinstance(target, str) and target]

    pack_data = _load_yaml_mapping(pack_path, "pack.yaml")
    pack_targets = pack_data.get("ui_targets")
    if pack_targets is not None:
        if not isinstance(pack_targets, list):
            raise ValueError(f"pack.yaml field 'ui_targets' must be a list: {pack_path}")
        for target in pack_targets:
            if not isinstance(target, str) or not target:
                raise ValueError(f"pack.yaml field 'ui_targets' must contain non-empty strings: {pack_path}")
            targets.append(target)
    return sorted(set(targets))


def _load_error_categories(taxonomy_path: Path | None) -> list[str]:
    if taxonomy_path is None:
        return ["OM", "CO", "OR", "PA", "SV", "DE", "HR"]

    data = _load_yaml_mapping(taxonomy_path, "taxonomy.yaml")
    taxonomy = data.get("taxonomy")
    if not isinstance(taxonomy, dict):
        raise ValueError(f"taxonomy.yaml missing 'taxonomy' mapping: {taxonomy_path}")

    categories: list[str] = []
    for entry in taxonomy.get("categories", []):
        if isinstance(entry, dict) and isinstance(entry.get("code"), str) and entry["code"]:
            categories.append(entry["code"])
    for entry in taxonomy.get("trial_flags", []):
        if isinstance(entry, dict) and isinstance(entry.get("code"), str) and entry["code"]:
            categories.append(entry["code"])

    if not categories:
        raise ValueError(f"taxonomy.yaml missing category/trial flag codes: {taxonomy_path}")
    return sorted(set(categories))


def build_help_response_schema(
    step_ids: Sequence[str],
    overlay_targets: Sequence[str],
    error_categories: Sequence[str],
) -> dict[str, Any]:
    if not step_ids:
        raise ValueError("step_ids must be non-empty")
    if not overlay_targets:
        raise ValueError("overlay_targets must be non-empty")
    if not error_categories:
        raise ValueError("error_categories must be non-empty")

    evidence_item_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["target", "type", "ref", "quote", "grounding_confidence"],
        "properties": {
            "target": {"type": "string", "enum": list(overlay_targets)},
            "type": {"type": "string", "enum": ["var", "gate", "rag", "delta"]},
            "ref": {"type": "string", "minLength": 1},
            "quote": {"type": "string", "minLength": 1, "maxLength": 120},
            "grounding_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }

    overlay_target_evidence_rules: list[dict[str, Any]] = []
    for target in overlay_targets:
        overlay_target_evidence_rules.append(
            {
                "if": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "contains": {"const": target},
                        }
                    },
                    "required": ["targets"],
                },
                "then": {
                    "type": "object",
                    "properties": {
                        "evidence": {
                            "type": "array",
                            "contains": {
                                "type": "object",
                                "properties": {"target": {"const": target}},
                                "required": ["target"],
                            },
                        }
                    },
                    "required": ["evidence"],
                },
            }
        )

    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": _HELP_RESPONSE_SCHEMA_ID,
        "$comment": (
            "Runtime-generated schema: enums come from pack.yaml/ui_map.yaml/taxonomy.yaml "
            "and are therefore not persisted as a static schema artifact."
        ),
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
                    "step_id": {"type": "string", "enum": list(step_ids)},
                    "error_category": {"type": "string", "enum": list(error_categories)},
                },
            },
            "next": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step_id"],
                "properties": {
                    "step_id": {"type": "string", "enum": list(step_ids)},
                },
            },
            "overlay": {
                "type": "object",
                "additionalProperties": False,
                "required": ["targets", "evidence"],
                "properties": {
                    "targets": {
                        "type": "array",
                        "minItems": 0,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": list(overlay_targets)},
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 0,
                        "items": evidence_item_schema,
                    }
                },
                "allOf": overlay_target_evidence_rules,
            },
            "explanations": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }
    return schema


@lru_cache(maxsize=16)
def _cached_help_schema(
    pack_path: str,
    ui_map_path: str,
    taxonomy_path: str | None,
) -> dict[str, Any]:
    pack = Path(pack_path)
    ui_map = Path(ui_map_path)
    taxonomy = Path(taxonomy_path) if taxonomy_path else None
    step_ids = _load_step_ids(pack)
    overlay_targets = _load_overlay_targets(pack, ui_map)
    error_categories = _load_error_categories(taxonomy)
    return build_help_response_schema(step_ids, overlay_targets, error_categories)


def get_help_response_schema(
    pack_path: str | Path | None = None,
    ui_map_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
) -> dict[str, Any]:
    pack = str((Path(pack_path) if pack_path else _default_pack_path()).resolve())
    ui_map = str((Path(ui_map_path) if ui_map_path else _default_ui_map_path()).resolve())
    taxonomy = str((Path(taxonomy_path) if taxonomy_path else _default_taxonomy_path()).resolve())
    return copy.deepcopy(_cached_help_schema(pack, ui_map, taxonomy))


def _format_path_segment(segment: Any) -> str:
    if isinstance(segment, int):
        return f"[{segment}]"
    if isinstance(segment, str) and segment.isidentifier():
        return f".{segment}"
    return f"['{segment}']"


def _error_json_path(error: ValidationError) -> str:
    parts = list(error.path)
    if error.validator == "required":
        match = _REQUIRED_PROPERTY_RE.search(error.message)
        if match:
            parts.append(match.group(1))
    return "$" + "".join(_format_path_segment(p) for p in parts)


def validate_help_response(obj: Mapping[str, Any]) -> None:
    schema = get_help_response_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(dict(obj)), key=lambda err: (list(err.path), err.message))
    if not errors:
        return

    first_error = errors[0]
    path = _error_json_path(first_error)
    raise ValidationError(f"HelpResponse validation failed at {path}: {first_error.message}")


__all__ = [
    "build_help_response_schema",
    "get_help_response_schema",
    "validate_help_response",
]

