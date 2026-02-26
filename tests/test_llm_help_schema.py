from jsonschema.exceptions import ValidationError
import pytest

from core.llm_schema import get_help_response_schema, validate_help_response


def _valid_help_response() -> dict:
    return {
        "diagnosis": {
            "step_id": "S02",
            "error_category": "OM",
        },
        "next": {
            "step_id": "S03",
        },
        "overlay": {
            "targets": ["apu_switch"],
        },
        "explanations": [
            "Power is available but APU is still off.",
            "Turn on APU before engine crank.",
        ],
        "confidence": 0.87,
    }


def test_help_response_schema_injects_step_and_target_enums() -> None:
    schema = get_help_response_schema()
    diagnosis_step_enum = schema["properties"]["diagnosis"]["properties"]["step_id"]["enum"]
    next_step_enum = schema["properties"]["next"]["properties"]["step_id"]["enum"]
    overlay_target_enum = schema["properties"]["overlay"]["properties"]["targets"]["items"]["enum"]

    assert "S01" in diagnosis_step_enum
    assert "S10" in diagnosis_step_enum
    assert diagnosis_step_enum == next_step_enum
    assert "battery_switch" in overlay_target_enum
    assert "apu_switch" in overlay_target_enum


def test_validate_help_response_accepts_valid_payload() -> None:
    validate_help_response(_valid_help_response())


def test_validate_help_response_reports_missing_field_path() -> None:
    payload = _valid_help_response()
    del payload["diagnosis"]["error_category"]

    with pytest.raises(ValidationError, match=r"\$\.diagnosis\.error_category"):
        validate_help_response(payload)


def test_validate_help_response_reports_invalid_step_id_path() -> None:
    payload = _valid_help_response()
    payload["diagnosis"]["step_id"] = "S99"

    with pytest.raises(ValidationError, match=r"\$\.diagnosis\.step_id"):
        validate_help_response(payload)


def test_validate_help_response_reports_invalid_target_path() -> None:
    payload = _valid_help_response()
    payload["overlay"]["targets"] = ["hacked_target"]

    with pytest.raises(ValidationError, match=r"\$\.overlay\.targets\[0\]"):
        validate_help_response(payload)


def test_validate_help_response_reports_type_error_path() -> None:
    payload = _valid_help_response()
    payload["confidence"] = "high"

    with pytest.raises(ValidationError, match=r"\$\.confidence"):
        validate_help_response(payload)

