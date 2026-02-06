import pytest

from simtutor.llm_schema import validate_help_response


def _base_payload():
    return {
        "diagnosis": {"step_id": "S01", "error_category": "OM"},
        "next": {"step_id": "S02"},
        "overlay": {"targets": ["battery_switch"]},
        "explanations": ["Turn battery on to power the jet."],
        "confidence": 0.9,
    }


def test_validate_help_response_accepts_valid_payload():
    validate_help_response(_base_payload())


def test_missing_required_field_reports_root_path():
    payload = _base_payload()
    payload.pop("diagnosis")
    with pytest.raises(ValueError) as excinfo:
        validate_help_response(payload)
    msg = str(excinfo.value)
    assert "<root>" in msg
    assert "diagnosis" in msg


def test_invalid_step_id_rejected():
    payload = _base_payload()
    payload["next"]["step_id"] = "S99"
    with pytest.raises(ValueError) as excinfo:
        validate_help_response(payload)
    msg = str(excinfo.value)
    assert "next.step_id" in msg
    assert "S99" in msg


def test_invalid_overlay_target_rejected():
    payload = _base_payload()
    payload["overlay"]["targets"] = ["unknown_target"]
    with pytest.raises(ValueError) as excinfo:
        validate_help_response(payload)
    msg = str(excinfo.value)
    assert "overlay.targets.0" in msg
    assert "unknown_target" in msg


def test_wrong_type_rejected():
    payload = _base_payload()
    payload["explanations"] = "not a list"
    with pytest.raises(ValueError) as excinfo:
        validate_help_response(payload)
    msg = str(excinfo.value)
    assert "explanations" in msg
    assert "array" in msg
