import json
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
import pytest

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


def _request_with_evidence_context() -> TutorRequest:
    return TutorRequest(
        intent="help",
        context={
            "vars": {"battery_on": True, "apu_on": False},
            "gates": [{"gate_id": "power_available", "status": "pass"}],
            "vision_facts": [
                {
                    "fact_id": "fcs_page_visible",
                    "state": "seen",
                    "source_frame_id": "1772872445010_000123",
                    "confidence": 0.96,
                    "evidence_note": "FCS page title and channel boxes are visible.",
                }
            ],
            "recent_deltas": [
                {"ui_target": "apu_switch", "bios_key": "APU_CONTROL_SW"},
                {"ui_target": "battery_switch", "bios_key": "BATTERY_SW"},
            ],
            "delta_summary": {
                "changed_keys_sample": ["APU_CONTROL_SW", "BATTERY_SW"],
                "recent_key_changes_topk": [{"key": "APU_CONTROL_SW"}, {"key": "BATTERY_SW"}],
            },
            "rag_topk": [{"snippet_id": "manual_s03_1", "snippet": "APU ON"}],
        },
    )


def _evidence(target: str, *, kind: str, ref: str, quote: str = "evidence") -> dict[str, str]:
    return {"target": target, "type": kind, "ref": ref, "quote": quote}


def _request_with_recent_delta_k_only() -> TutorRequest:
    return TutorRequest(
        intent="help",
        context={
            "vars": {"battery_on": True},
            "recent_deltas": [
                {"k": "APU_CONTROL_SW", "mapped_ui_target": "apu_switch"},
            ],
        },
    )


def test_mapping_generates_overlay_action_aligned_with_overlay_intent() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["Turn on APU first."],
    }
    req = _request_with_evidence_context()

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
            "evidence_required": True,
            "evidence_refs": ["RECENT_UI_TARGETS.apu_switch"],
        }
    ]
    _validate_tutor_response(payload)


def test_mapping_dedupes_targets_and_limits_max_overlay_count() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch", "apu_switch", "battery_switch"],
            "evidence": [
                _evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch"),
                _evidence("battery_switch", kind="delta", ref="RECENT_UI_TARGETS.battery_switch"),
            ],
        },
        "explanations": ["Do step S03."],
    }

    res = map_help_response_to_tutor_response(
        help_obj,
        max_overlay_targets=1,
        request=_request_with_evidence_context(),
    )
    payload = res.to_dict()

    assert len(payload["actions"]) == 1
    assert payload["actions"][0]["target"] == "apu_switch"
    assert payload["metadata"]["dropped_targets"] == ["battery_switch"]
    _validate_tutor_response(payload)


def test_mapping_max_overlay_zero_drops_targets_without_evidence_metadata() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(
        help_obj,
        max_overlay_targets=0,
        request=_request_with_evidence_context(),
    )
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["dropped_targets"] == ["apu_switch"]
    assert "allowed_evidence_ref_count" not in payload["metadata"]
    assert "overlay_rejected" not in payload["metadata"]
    assert "overlay_rejected_reasons" not in payload["metadata"]
    _validate_tutor_response(payload)


def test_mapping_rejects_unknown_target_into_metadata() -> None:
    help_obj = {
        "overlay": {
            "targets": ["unknown_target", "apu_switch"],
            "evidence": [
                _evidence("unknown_target", kind="var", ref="VARS.battery_on"),
                _evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch"),
            ],
        },
        "explanations": ["Check highlighted controls."],
    }

    res = map_help_response_to_tutor_response(
        help_obj,
        max_overlay_targets=2,
        request=_request_with_evidence_context(),
    )
    payload = res.to_dict()

    assert [a["target"] for a in payload["actions"]] == ["apu_switch"]
    assert payload["metadata"]["rejected_targets"] == ["unknown_target"]
    assert payload["metadata"]["overlay_mapping_failures"][0]["target"] == "unknown_target"
    assert payload["metadata"]["overlay_mapping_failures"][0]["error_type"] == "KeyError"
    assert payload["metadata"]["overlay_mapping_failures"][0]["error_code"] == "target_not_mapped"
    assert "error" not in payload["metadata"]["overlay_mapping_failures"][0]
    _validate_tutor_response(payload)


def test_mapping_unknown_target_does_not_consume_limit_slot() -> None:
    help_obj = {
        "overlay": {
            "targets": ["unknown_target", "apu_switch", "battery_switch"],
            "evidence": [
                _evidence("unknown_target", kind="var", ref="VARS.battery_on"),
                _evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch"),
                _evidence("battery_switch", kind="delta", ref="RECENT_UI_TARGETS.battery_switch"),
            ],
        },
        "explanations": ["Check highlighted controls."],
    }

    res = map_help_response_to_tutor_response(
        help_obj,
        max_overlay_targets=1,
        request=_request_with_evidence_context(),
    )
    payload = res.to_dict()

    assert [a["target"] for a in payload["actions"]] == ["apu_switch"]
    assert payload["metadata"]["rejected_targets"] == ["unknown_target"]
    assert payload["metadata"]["dropped_targets"] == ["battery_switch"]
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
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["x"],
    }

    res = map_help_response_to_tutor_response(
        help_obj,
        overlay_intent="execute",
        request=_request_with_evidence_context(),
    )
    payload = res.to_dict()

    assert payload["actions"][0]["intent"] == "highlight"
    assert payload["metadata"]["mapping_error"] == "invalid_overlay_intent:execute"
    assert payload["metadata"]["mapping_errors"] == ["invalid_overlay_intent:execute"]
    _validate_tutor_response(payload)


def test_mapping_accumulates_multiple_mapping_errors_without_overwriting_first() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["x"],
    }

    res = map_help_response_to_tutor_response(
        help_obj,
        overlay_intent="execute",
        status="bad_status",
        request=_request_with_evidence_context(),
    )
    payload = res.to_dict()

    assert res.status == "error"
    assert payload["metadata"]["mapping_error"] == "invalid_overlay_intent:execute"
    assert payload["metadata"]["mapping_errors"] == [
        "invalid_overlay_intent:execute",
        "invalid_status:bad_status",
    ]
    _validate_tutor_response(payload)


def test_mapping_marks_partial_visual_confirmation_and_downgrades_confidence() -> None:
    help_obj = {
        "diagnosis": {"step_id": "S14", "error_category": "OM"},
        "next": {"step_id": "S15"},
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["Check the highlighted control and continue the sequence."],
        "confidence": 0.91,
    }
    req = _request_with_evidence_context()
    req.context["deterministic_step_hint"] = {
        "inferred_step_id": "S15",
        "observability": "partially",
        "step_evidence_requirements": ["visual", "gate"],
    }

    res = map_help_response_to_tutor_response(help_obj, request=req)
    payload = res.to_dict()

    assert payload["metadata"]["observability_status"] == "partial"
    assert payload["metadata"]["requires_visual_confirmation"] is True
    assert payload["metadata"]["model_confidence"] == 0.91
    assert payload["metadata"]["effective_confidence"] < payload["metadata"]["model_confidence"]
    assert payload["metadata"]["confidence_adjustment_reason"] == "observability:partial"
    assert payload["metadata"]["evidence_strength"] == "limited"
    assert any("待视觉确认" in item for item in payload["explanations"])
    _validate_tutor_response(payload)


def test_mapping_normalizes_unobservable_status_from_request_hint() -> None:
    help_obj = {
        "diagnosis": {"step_id": "S20", "error_category": "OM"},
        "next": {"step_id": "S21"},
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["Use the highlighted control as the next single action."],
        "confidence": 0.88,
    }
    req = _request_with_evidence_context()
    req.context["deterministic_step_hint"] = {
        "inferred_step_id": "S20",
        "observability_status": "unobservable",
        "requires_visual_confirmation": True,
    }

    res = map_help_response_to_tutor_response(help_obj, request=req)
    payload = res.to_dict()

    assert payload["metadata"]["observability_status"] == "unobservable"
    assert payload["metadata"]["requires_visual_confirmation"] is True
    assert payload["metadata"]["effective_confidence"] < payload["metadata"]["model_confidence"]
    assert any("待视觉确认" in item for item in payload["explanations"])
    _validate_tutor_response(payload)


def test_mapping_emits_english_visual_confirmation_note_when_lang_is_en() -> None:
    help_obj = {
        "diagnosis": {"step_id": "S20", "error_category": "OM"},
        "next": {"step_id": "S21"},
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["Use the highlighted control as the next single action."],
        "confidence": 0.88,
    }
    req = _request_with_evidence_context()
    req.context["deterministic_step_hint"] = {
        "inferred_step_id": "S20",
        "observability_status": "unobservable",
        "requires_visual_confirmation": True,
    }

    res = map_help_response_to_tutor_response(help_obj, request=req, lang="en")

    assert any("Visual confirmation required" in item for item in res.explanations)
    assert not any("待视觉确认" in item for item in res.explanations)


def test_mapping_uses_unspecified_adjustment_reason_when_visual_confirmation_lacks_observability() -> None:
    help_obj = {
        "diagnosis": {"step_id": "S14", "error_category": "OM"},
        "next": {"step_id": "S15"},
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["Check the highlighted control and continue the sequence."],
        "confidence": 0.91,
    }
    req = _request_with_evidence_context()
    req.context["deterministic_step_hint"] = {
        "inferred_step_id": "S15",
        "step_evidence_requirements": ["visual", "gate"],
    }

    res = map_help_response_to_tutor_response(help_obj, request=req)

    assert res.metadata["requires_visual_confirmation"] is True
    assert "observability_status" not in res.metadata
    assert res.metadata["confidence_adjustment_reason"] == "observability:unspecified"


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
        help_obj = {
            "overlay": {
                "targets": ["apu_switch"],
                "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
            },
            "explanations": ["x"],
        }
        req = _request_with_evidence_context()
        map_help_response_to_tutor_response(help_obj, ui_map_path=ui_map_path, request=req)
        map_help_response_to_tutor_response(help_obj, ui_map_path=ui_map_path, request=req)
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
        {
            "overlay": {
                "targets": ["apu_switch"],
                "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
            },
            "explanations": ["x"],
        },
        ui_map_path=bad_path,
        request=_request_with_evidence_context(),
    )
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_mapping_error"]["error_code"] == "ui_map_not_found"
    assert payload["metadata"]["overlay_mapping_error"]["error_type"] == "FileNotFoundError"
    assert isinstance(payload["metadata"]["overlay_mapping_error"], dict)


def test_mapping_sanitizes_yaml_parse_error_as_ui_map_invalid(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "ui_map_bad.yaml"
    bad_yaml.write_text("version: v1\ncockpit_elements: [unclosed\n", encoding="utf-8")

    res = map_help_response_to_tutor_response(
        {
            "overlay": {
                "targets": ["apu_switch"],
                "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
            },
            "explanations": ["x"],
        },
        ui_map_path=bad_yaml,
        request=_request_with_evidence_context(),
    )
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_mapping_error"]["error_code"] == "ui_map_invalid"
    assert payload["metadata"]["overlay_mapping_error"]["error_type"] == "ParserError"


def test_mapping_reraises_unexpected_planner_init_exception(monkeypatch) -> None:
    original_get = response_mapping._get_overlay_planner

    def _raise_unexpected(_ui_map_path: str):
        raise RuntimeError("unexpected init failure")

    monkeypatch.setattr(response_mapping, "_get_overlay_planner", _raise_unexpected)
    try:
        with pytest.raises(RuntimeError, match="unexpected init failure"):
            map_help_response_to_tutor_response(
                {
                    "overlay": {
                        "targets": ["apu_switch"],
                        "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
                    },
                    "explanations": ["x"],
                },
                request=_request_with_evidence_context(),
            )
    finally:
        monkeypatch.setattr(response_mapping, "_get_overlay_planner", original_get)


def test_mapping_reraises_unexpected_plan_exception(monkeypatch) -> None:
    class _BadPlanner:
        def plan(self, target: str, intent: str = "highlight"):
            raise RuntimeError(f"unexpected plan failure:{target}:{intent}")

    original_get = response_mapping._get_overlay_planner
    monkeypatch.setattr(response_mapping, "_get_overlay_planner", lambda _ui_map_path: _BadPlanner())
    try:
        with pytest.raises(RuntimeError, match="unexpected plan failure"):
            map_help_response_to_tutor_response(
                {
                    "overlay": {
                        "targets": ["apu_switch"],
                        "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
                    },
                    "explanations": ["x"],
                },
                request=_request_with_evidence_context(),
            )
    finally:
        monkeypatch.setattr(response_mapping, "_get_overlay_planner", original_get)


def test_mapping_rejects_overlay_when_targets_without_evidence() -> None:
    help_obj = {
        "overlay": {"targets": ["apu_switch"]},
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_evidence_context())
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_rejected"] is True
    assert "missing_overlay_evidence" in payload["metadata"]["overlay_rejected_reasons"]
    assert payload["metadata"]["allowed_evidence_ref_count"] == 9


def test_mapping_rejects_overlay_when_evidence_ref_unknown() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.unknown_target")],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_evidence_context())
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_rejected"] is True
    assert any(
        reason.startswith("unknown_evidence_ref:")
        for reason in payload["metadata"]["overlay_rejected_reasons"]
    )


def test_mapping_accepts_delta_key_evidence_from_recent_deltas_k_field() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="DELTA_KEYS.APU_CONTROL_SW")],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_recent_delta_k_only())
    payload = res.to_dict()

    assert payload["actions"] != []
    assert payload["actions"][0]["target"] == "apu_switch"
    assert payload["actions"][0]["evidence_refs"] == ["DELTA_KEYS.APU_CONTROL_SW"]
    assert payload["metadata"].get("overlay_rejected") is not True


def test_mapping_accepts_visual_evidence_from_request_context() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [
                _evidence(
                    "apu_switch",
                    kind="visual",
                    ref="VISION_FACTS.fcs_page_visible@1772872445010_000123",
                )
            ],
        },
        "explanations": ["Visual cue confirms the page state."],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_evidence_context())
    payload = res.to_dict()

    assert payload["actions"] != []
    assert payload["actions"][0]["target"] == "apu_switch"
    assert payload["actions"][0]["evidence_refs"] == ["VISION_FACTS.fcs_page_visible@1772872445010_000123"]
    assert payload["metadata"].get("overlay_rejected") is not True


def test_mapping_rejects_overlay_when_no_verifiable_refs_from_request_context() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=None)
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_rejected"] is True
    assert payload["metadata"]["overlay_rejected_reasons"] == ["no_verifiable_evidence_refs"]


@pytest.mark.parametrize(
    "evidence_item",
    [
        "not-a-mapping",
        {"type": "delta", "ref": "RECENT_UI_TARGETS.apu_switch"},
        {"target": "apu_switch", "type": "unknown", "ref": "RECENT_UI_TARGETS.apu_switch"},
        {"target": "apu_switch", "type": "delta"},
    ],
)
def test_mapping_rejects_overlay_when_evidence_item_invalid(evidence_item: object) -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [evidence_item],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_evidence_context())
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_rejected"] is True
    assert "invalid_overlay_evidence_item" in payload["metadata"]["overlay_rejected_reasons"]


def test_mapping_rejects_overlay_when_evidence_type_ref_prefix_mismatch() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("apu_switch", kind="var", ref="RECENT_UI_TARGETS.apu_switch")],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_evidence_context())
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_rejected"] is True
    assert any(
        reason.startswith("evidence_type_ref_mismatch:")
        for reason in payload["metadata"]["overlay_rejected_reasons"]
    )


def test_mapping_rejects_overlay_when_evidence_target_not_declared_in_overlay_targets() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [_evidence("battery_switch", kind="delta", ref="RECENT_UI_TARGETS.battery_switch")],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_evidence_context())
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_rejected"] is True
    assert "evidence_target_not_in_overlay_targets" in payload["metadata"]["overlay_rejected_reasons"]


def test_mapping_rejects_overlay_when_target_has_no_valid_evidence_even_if_other_target_valid() -> None:
    help_obj = {
        "overlay": {
            "targets": ["apu_switch", "battery_switch"],
            "evidence": [
                _evidence("apu_switch", kind="delta", ref="RECENT_UI_TARGETS.apu_switch"),
            ],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_evidence_context())
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_rejected"] is True
    assert "missing_target_evidence:battery_switch" in payload["metadata"]["overlay_rejected_reasons"]


def test_mapping_missing_target_evidence_reason_is_sorted_deterministically() -> None:
    help_obj = {
        "overlay": {
            "targets": ["battery_switch", "apu_switch"],
            "evidence": [],
        },
        "explanations": ["x"],
    }
    res = map_help_response_to_tutor_response(help_obj, request=_request_with_evidence_context())
    payload = res.to_dict()

    assert payload["actions"] == []
    assert payload["metadata"]["overlay_rejected"] is True
    assert payload["metadata"]["overlay_rejected_reasons"] == [
        "missing_target_evidence:apu_switch,battery_switch"
    ]
