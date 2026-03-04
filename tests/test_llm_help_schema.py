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
            "evidence": [
                {
                    "target": "apu_switch",
                    "type": "delta",
                    "ref": "RECENT_UI_TARGETS.apu_switch",
                    "quote": "Recent UI delta shows APU switch movement.",
                    "grounding_confidence": 0.88,
                }
            ],
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
    assert "S25" in diagnosis_step_enum
    assert diagnosis_step_enum == next_step_enum
    assert "battery_switch" in overlay_target_enum
    assert "apu_switch" in overlay_target_enum


def test_validate_help_response_accepts_valid_payload() -> None:
    validate_help_response(_valid_help_response())


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_validate_help_response_accepts_confidence_boundaries(confidence: float) -> None:
    payload = _valid_help_response()
    payload["confidence"] = confidence

    validate_help_response(payload)


def test_validate_help_response_accepts_multiple_overlay_targets_with_evidence() -> None:
    payload = _valid_help_response()
    payload["overlay"]["targets"] = ["apu_switch", "battery_switch"]
    payload["overlay"]["evidence"] = [
        {
            "target": "apu_switch",
            "type": "delta",
            "ref": "RECENT_UI_TARGETS.apu_switch",
            "quote": "APU switch changed recently.",
            "grounding_confidence": 0.9,
        },
        {
            "target": "battery_switch",
            "type": "var",
            "ref": "VARS.battery_on",
            "quote": "Battery var indicates power path check.",
            "grounding_confidence": 0.7,
        },
    ]

    validate_help_response(payload)


def test_validate_help_response_reports_missing_field_path() -> None:
    payload = _valid_help_response()
    del payload["diagnosis"]["error_category"]

    with pytest.raises(ValidationError, match=r"\$\.diagnosis\.error_category"):
        validate_help_response(payload)


def test_validate_help_response_reports_missing_top_level_required_field_path() -> None:
    payload = _valid_help_response()
    del payload["diagnosis"]

    with pytest.raises(ValidationError, match=r"\$\.diagnosis"):
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


def test_validate_help_response_accepts_empty_overlay_targets_with_empty_evidence() -> None:
    payload = _valid_help_response()
    payload["overlay"]["targets"] = []
    payload["overlay"]["evidence"] = []

    validate_help_response(payload)


def test_validate_help_response_rejects_missing_overlay_evidence_field_path() -> None:
    payload = _valid_help_response()
    del payload["overlay"]["evidence"]

    with pytest.raises(ValidationError, match=r"\$\.overlay\.evidence"):
        validate_help_response(payload)


def test_validate_help_response_rejects_missing_target_specific_evidence() -> None:
    payload = _valid_help_response()
    payload["overlay"]["targets"] = ["apu_switch", "battery_switch"]
    payload["overlay"]["evidence"] = [
        {
            "target": "apu_switch",
            "type": "delta",
            "ref": "RECENT_UI_TARGETS.apu_switch",
            "quote": "APU switch changed recently.",
            "grounding_confidence": 0.9,
        }
    ]

    with pytest.raises(ValidationError, match=r"\$\.overlay"):
        validate_help_response(payload)


def test_validate_help_response_rejects_invalid_evidence_type_path() -> None:
    payload = _valid_help_response()
    payload["overlay"]["evidence"][0]["type"] = "model_guess"

    with pytest.raises(ValidationError, match=r"\$\.overlay\.evidence\[0\]\.type"):
        validate_help_response(payload)


def test_validate_help_response_rejects_empty_evidence_ref_path() -> None:
    payload = _valid_help_response()
    payload["overlay"]["evidence"][0]["ref"] = ""

    with pytest.raises(ValidationError, match=r"\$\.overlay\.evidence\[0\]\.ref"):
        validate_help_response(payload)


def test_validate_help_response_rejects_quote_too_long_path() -> None:
    payload = _valid_help_response()
    payload["overlay"]["evidence"][0]["quote"] = "x" * 121

    with pytest.raises(ValidationError, match=r"\$\.overlay\.evidence\[0\]\.quote"):
        validate_help_response(payload)


@pytest.mark.parametrize("grounding_confidence", [-0.01, 1.01])
def test_validate_help_response_rejects_grounding_confidence_out_of_range(grounding_confidence: float) -> None:
    payload = _valid_help_response()
    payload["overlay"]["evidence"][0]["grounding_confidence"] = grounding_confidence

    with pytest.raises(ValidationError, match=r"\$\.overlay\.evidence\[0\]\.grounding_confidence"):
        validate_help_response(payload)


def test_validate_help_response_rejects_duplicate_overlay_targets() -> None:
    payload = _valid_help_response()
    payload["overlay"]["targets"] = ["apu_switch", "apu_switch"]

    with pytest.raises(ValidationError, match=r"\$\.overlay\.targets"):
        validate_help_response(payload)


def test_validate_help_response_rejects_empty_explanation_string() -> None:
    payload = _valid_help_response()
    payload["explanations"] = [""]

    with pytest.raises(ValidationError, match=r"\$\.explanations\[0\]"):
        validate_help_response(payload)


def test_validate_help_response_rejects_empty_explanations_array() -> None:
    payload = _valid_help_response()
    payload["explanations"] = []

    with pytest.raises(ValidationError, match=r"\$\.explanations"):
        validate_help_response(payload)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_validate_help_response_rejects_confidence_out_of_range(confidence: float) -> None:
    payload = _valid_help_response()
    payload["confidence"] = confidence

    with pytest.raises(ValidationError, match=r"\$\.confidence"):
        validate_help_response(payload)


@pytest.mark.parametrize(
    ("field_name", "extra_key", "expected_path"),
    [
        ("diagnosis", "unexpected", r"\$\.diagnosis"),
        ("next", "unexpected", r"\$\.next"),
        ("overlay", "unexpected", r"\$\.overlay"),
    ],
)
def test_validate_help_response_rejects_additional_properties(
    field_name: str,
    extra_key: str,
    expected_path: str,
) -> None:
    payload = _valid_help_response()
    payload[field_name][extra_key] = "x"

    with pytest.raises(ValidationError, match=expected_path):
        validate_help_response(payload)
