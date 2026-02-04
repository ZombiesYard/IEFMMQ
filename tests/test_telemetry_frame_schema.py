import json
from importlib import resources

from jsonschema import Draft202012Validator, FormatChecker

from core.types_v2 import TelemetryFrame


def load_schema() -> dict:
    schema_path = resources.files("simtutor.schemas.v2") / "telemetry_frame.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_telemetry_frame_schema_accepts_defaults():
    frame = TelemetryFrame(seq=1, t_wall=1.0, source="derived").to_dict()
    schema = load_schema()
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(frame)

