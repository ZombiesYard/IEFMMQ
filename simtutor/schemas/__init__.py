from __future__ import annotations

import json
from importlib import resources
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_INDEX: dict[str, tuple[str, str]] = {
    "event": ("simtutor.schemas.v1", "event.schema.json"),
    "observation": ("simtutor.schemas.v1", "observation.schema.json"),
    "tutor_request": ("simtutor.schemas.v1", "tutor_request.schema.json"),
    "tutor_response": ("simtutor.schemas.v1", "tutor_response.schema.json"),
    "dcs_observation": ("simtutor.schemas.v2", "dcs_observation.json"),
    "dcs_bios_frame": ("simtutor.schemas.v2", "dcs_bios_frame.json"),
    "telemetry_frame": ("simtutor.schemas.v2", "telemetry_frame.json"),
    "dcs_overlay_command": ("simtutor.schemas.v2", "dcs_overlay_command.json"),
    "dcs_overlay_ack": ("simtutor.schemas.v2", "dcs_overlay_ack.json"),
    "dcs_hello": ("simtutor.schemas.v2", "dcs_hello.json"),
    "dcs_caps": ("simtutor.schemas.v2", "dcs_caps.json"),
    "vision_observation": ("simtutor.schemas.v2", "vision_observation.json"),
    "vision_frame_manifest_entry": ("simtutor.schemas.v2", "vision_frame_manifest_entry.json"),
    "vision_fact_observation": ("simtutor.schemas.v2", "vision_fact_observation.json"),
}


def load_schema(name: str) -> Mapping[str, Any]:
    if name not in SCHEMA_INDEX:
        raise FileNotFoundError(f"Unknown schema: {name}")
    schema_pkg, schema_file = SCHEMA_INDEX[name]
    try:
        schema_path = resources.files(schema_pkg) / schema_file
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(f"Schema package not found: {schema_pkg}") from exc
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(name), format_checker=FormatChecker())


def validate_instance(instance: Mapping[str, Any], schema_name: str) -> None:
    get_validator(schema_name).validate(instance)


__all__ = ["SCHEMA_INDEX", "get_validator", "load_schema", "validate_instance"]
