from __future__ import annotations

import json
from importlib import resources
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from core.types_v2 import DcsObservation

_SCHEMA_PACKAGE = "simtutor.schemas.v2"
_SCHEMA_NAME = "dcs_observation.json"


def _load_schema() -> Mapping[str, Any]:
    try:
        schema_path = resources.files(_SCHEMA_PACKAGE) / _SCHEMA_NAME
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(f"Schema package not found: {_SCHEMA_PACKAGE}") from exc
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


_SCHEMA = _load_schema()
_VALIDATOR = Draft202012Validator(_SCHEMA, format_checker=FormatChecker())


def _error_path_key(error: Exception) -> tuple[str, ...]:
    path = getattr(error, "path", ())
    return tuple(str(p) for p in path)


def validate_dcs_observation(payload: Mapping[str, Any]) -> None:
    errors = sorted(_VALIDATOR.iter_errors(payload), key=_error_path_key)
    if errors:
        err = errors[0]
        location = ".".join([str(p) for p in err.path]) or "<root>"
        raise ValueError(f"dcs_observation invalid at {location}: {err.message}")


def decode_dcs_observation(data: bytes) -> DcsObservation:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("DCS telemetry payload must be a JSON object")
    validate_dcs_observation(payload)
    return DcsObservation(
        schema_version=payload["schema_version"],
        seq=int(payload["seq"]),
        sim_time=float(payload["sim_time"]),
        aircraft=str(payload["aircraft"]),
        cockpit=dict(payload.get("cockpit", {})),
        raw=payload.get("raw"),
    )


def encode_dcs_observation(obs: DcsObservation) -> bytes:
    payload = obs.to_dict()
    validate_dcs_observation(payload)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")

