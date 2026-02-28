import json
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

import adapters.response_mapping as response_mapping
from adapters.response_mapping import map_help_response_to_tutor_response
from core.types import TutorRequest


def _load_tutor_response_schema() -> dict:
    schema_path = resources.files("simtutor.schemas.v1") / "tutor_response.schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_tutor_response(payload: dict) -> None:
    schema = _load_tutor_response_schema()
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)


def test_mapping_generates_overlay_action_aligned_with_overlay_intent() -> None:
    help_obj = {
        "overlay": {"targets": ["apu_switch"]},
        "explanations": ["Turn on APU first."],
    }
    req = TutorRequest(intent="help")

    res = map_help_response_to_tutor_response(help_obj, request=req)
    payload = res.to_dict()

    assert res.status == "ok"
    assert payload["in_reply_to"] == req.request_id
    assert payload["actions"] == [
        {
            "type": "overlay",
            "intent": "highlight",
            "target": "apu_switch",
            "element_id": "pnt_375",
            "style": {"style": "highlight", "color": "#00ffcc", "thickness": 2},
        }
    ]
    _validate_tutor_response(payload)


def test_mapping_dedupes_targets_and_limits_max_overlay_count() -> None:
    help_obj = {
        "overlay": {"targets": ["apu_switch", "apu_switch", "battery_switch"]},
        "explanations": ["Do step S03."],
    }

    res = map_help_response_to_tutor_response(help_obj, max_overlay_targets=1)
    payload = res.to_dict()

    assert len(payload["actions"]) == 1
    assert payload["actions"][0]["target"] == "apu_switch"
    assert payload["metadata"]["dropped_targets"] == ["battery_switch"]
    _validate_tutor_response(payload)


def test_mapping_rejects_unknown_target_into_metadata() -> None:
    help_obj = {
        "overlay": {"targets": ["unknown_target", "apu_switch"]},
        "explanations": ["Check highlighted controls."],
    }

    res = map_help_response_to_tutor_response(help_obj, max_overlay_targets=2)
    payload = res.to_dict()

    assert [a["target"] for a in payload["actions"]] == ["apu_switch"]
    assert payload["metadata"]["rejected_targets"] == ["unknown_target"]
    assert payload["metadata"]["overlay_mapping_failures"][0]["target"] == "unknown_target"
    assert payload["metadata"]["overlay_mapping_failures"][0]["error_type"] == "KeyError"
    assert payload["metadata"]["overlay_mapping_failures"][0]["error_code"] == "target_not_mapped"
    assert "error" not in payload["metadata"]["overlay_mapping_failures"][0]
    _validate_tutor_response(payload)


def test_mapping_with_missing_or_empty_fields_still_returns_usable_tutor_response() -> None:
    res_missing = map_help_response_to_tutor_response({}, status="ok")
    payload_missing = res_missing.to_dict()
    assert res_missing.status == "ok"
    assert payload_missing["actions"] == []
    _validate_tutor_response(payload_missing)

    res_invalid_status = map_help_response_to_tutor_response(None, status="bad_status")
    payload_invalid_status = res_invalid_status.to_dict()
    assert res_invalid_status.status == "error"
    assert payload_invalid_status["metadata"]["mapping_error"] == "invalid_status:bad_status"
    assert payload_invalid_status["metadata"]["mapping_errors"] == ["invalid_status:bad_status"]
    _validate_tutor_response(payload_invalid_status)


def test_mapping_invalid_overlay_intent_coerces_to_highlight_and_records_error() -> None:
    help_obj = {"overlay": {"targets": ["apu_switch"]}, "explanations": ["x"]}

    res = map_help_response_to_tutor_response(help_obj, overlay_intent="execute")
    payload = res.to_dict()

    assert payload["actions"][0]["intent"] == "highlight"
    assert payload["metadata"]["mapping_error"] == "invalid_overlay_intent:execute"
    assert payload["metadata"]["mapping_errors"] == ["invalid_overlay_intent:execute"]
    _validate_tutor_response(payload)


def test_mapping_accumulates_multiple_mapping_errors_without_overwriting_first() -> None:
    help_obj = {"overlay": {"targets": ["apu_switch"]}, "explanations": ["x"]}

    res = map_help_response_to_tutor_response(help_obj, overlay_intent="execute", status="bad_status")
    payload = res.to_dict()

    assert res.status == "error"
    assert payload["metadata"]["mapping_error"] == "invalid_overlay_intent:execute"
    assert payload["metadata"]["mapping_errors"] == [
        "invalid_overlay_intent:execute",
        "invalid_status:bad_status",
    ]
    _validate_tutor_response(payload)


def test_mapping_reuses_overlay_planner_cache(monkeypatch, tmp_path: Path) -> None:
    ui_map_path = tmp_path / "ui_map.yaml"
    ui_map_path.write_text(
        "version: v1\n"
        "cockpit_elements:\n"
        "  apu_switch:\n"
        "    dcs_id: pnt_375\n",
        encoding="utf-8",
    )

    init_calls: list[str] = []
    original_cls = response_mapping.OverlayPlanner
    response_mapping._get_overlay_planner.cache_clear()

    class CountingOverlayPlanner(original_cls):
        def __init__(self, path: str):
            init_calls.append(path)
            super().__init__(path)

    monkeypatch.setattr(response_mapping, "OverlayPlanner", CountingOverlayPlanner)
    try:
        help_obj = {"overlay": {"targets": ["apu_switch"]}, "explanations": ["x"]}
        map_help_response_to_tutor_response(help_obj, ui_map_path=ui_map_path)
        map_help_response_to_tutor_response(help_obj, ui_map_path=ui_map_path)
    finally:
        response_mapping._get_overlay_planner.cache_clear()
        monkeypatch.setattr(response_mapping, "OverlayPlanner", original_cls)

    assert len(init_calls) == 1


def test_mapping_skips_planner_loading_when_no_selected_targets(monkeypatch) -> None:
    calls = {"count": 0}
    original_get = response_mapping._get_overlay_planner

    def _counting_get(_ui_map_path: str):
        calls["count"] += 1
        return original_get(_ui_map_path)

    monkeypatch.setattr(response_mapping, "_get_overlay_planner", _counting_get)
    try:
        res = map_help_response_to_tutor_response({"overlay": {"targets": []}, "explanations": ["x"]})
    finally:
        monkeypatch.setattr(response_mapping, "_get_overlay_planner", original_get)

    payload = res.to_dict()
    assert payload["actions"] == []
    assert calls["count"] == 0
    assert "overlay_mapping_error" not in payload["metadata"]


def test_mapping_sanitizes_planner_initialization_error_metadata() -> None:
    bad_path = Path("this/path/does/not/exist/ui_map.yaml")
    res = map_help_response_to_tutor_response(
        {"overlay": {"targets": ["apu_switch"]}, "explanations": ["x"]},
        ui_map_path=bad_path,
    )
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_mapping_error"]["error_code"] == "ui_map_not_found"
    assert payload["metadata"]["overlay_mapping_error"]["error_type"] == "FileNotFoundError"
    assert isinstance(payload["metadata"]["overlay_mapping_error"], dict)
