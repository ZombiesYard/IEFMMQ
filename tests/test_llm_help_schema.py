from jsonschema.exceptions import ValidationError
import os
from pathlib import Path
import time

import pytest
import yaml

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
    payload["explanations"] = "high"

    with pytest.raises(ValidationError, match=r"\$\.explanations"):
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


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _bump_mtime(path: Path) -> None:
    now = time.time()
    current = path.stat().st_mtime
    bumped = max(current + 2.0, now + 2.0)
    os.utime(path, (bumped, bumped))


def _registry_payload(first_short_explanation: str) -> dict:
    steps = []
    for i in range(1, 26):
        sid = f"S{i:02d}"
        short = first_short_explanation if i == 1 else f"step-{sid}"
        steps.append(
            {
                "step_id": sid,
                "phase": "P1",
                "official_description": f"Official {sid}",
                "short_explanation": short,
                "source_chunk_refs": [f"doc/chunk:{i}-{i}"],
            }
        )
    return {"schema_version": "v1", "steps": steps}


def test_help_schema_cache_invalidates_when_registry_file_removed(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    ui_map_path = tmp_path / "ui_map.yaml"
    taxonomy_path = tmp_path / "taxonomy.yaml"
    registry_path = tmp_path / "step_registry.yaml"

    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "steps": [{"id": "S01"}, {"id": "S02"}],
        },
    )
    _write_yaml(
        ui_map_path,
        {"cockpit_elements": {"target_a": {"description": "A"}}},
    )
    _write_yaml(
        taxonomy_path,
        {"taxonomy": {"categories": [{"code": "OM"}], "trial_flags": []}},
    )
    _write_yaml(registry_path, _registry_payload("first"))

    schema_with_registry = get_help_response_schema(
        pack_path=pack_path,
        ui_map_path=ui_map_path,
        taxonomy_path=taxonomy_path,
        step_registry_path=registry_path,
    )
    step_ids_with_registry = schema_with_registry["properties"]["next"]["properties"]["step_id"]["enum"]
    assert len(step_ids_with_registry) == 25
    assert step_ids_with_registry[0] == "S01"
    assert step_ids_with_registry[-1] == "S25"

    registry_path.unlink()

    schema_without_registry = get_help_response_schema(
        pack_path=pack_path,
        ui_map_path=ui_map_path,
        taxonomy_path=taxonomy_path,
        step_registry_path=registry_path,
    )
    step_ids_without_registry = schema_without_registry["properties"]["next"]["properties"]["step_id"]["enum"]
    assert step_ids_without_registry == ["S01", "S02"]


def test_help_schema_cache_invalidates_when_ui_map_and_taxonomy_change(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    ui_map_path = tmp_path / "ui_map.yaml"
    taxonomy_path = tmp_path / "taxonomy.yaml"
    registry_path = tmp_path / "missing_registry.yaml"

    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "steps": [{"id": "S01"}],
        },
    )
    _write_yaml(
        ui_map_path,
        {"cockpit_elements": {"target_a": {"description": "A"}}},
    )
    _write_yaml(
        taxonomy_path,
        {"taxonomy": {"categories": [{"code": "OM"}], "trial_flags": []}},
    )

    schema_before = get_help_response_schema(
        pack_path=pack_path,
        ui_map_path=ui_map_path,
        taxonomy_path=taxonomy_path,
        step_registry_path=registry_path,
    )
    overlay_before = schema_before["properties"]["overlay"]["properties"]["targets"]["items"]["enum"]
    categories_before = schema_before["properties"]["diagnosis"]["properties"]["error_category"]["enum"]
    assert overlay_before == ["target_a"]
    assert categories_before == ["OM"]

    _write_yaml(
        ui_map_path,
        {"cockpit_elements": {"target_a": {"description": "A"}, "target_b": {"description": "B"}}},
    )
    _bump_mtime(ui_map_path)
    _write_yaml(
        taxonomy_path,
        {"taxonomy": {"categories": [{"code": "OM"}, {"code": "CO"}], "trial_flags": []}},
    )
    _bump_mtime(taxonomy_path)

    schema_after = get_help_response_schema(
        pack_path=pack_path,
        ui_map_path=ui_map_path,
        taxonomy_path=taxonomy_path,
        step_registry_path=registry_path,
    )
    overlay_after = schema_after["properties"]["overlay"]["properties"]["targets"]["items"]["enum"]
    categories_after = schema_after["properties"]["diagnosis"]["properties"]["error_category"]["enum"]
    assert overlay_after == ["target_a", "target_b"]
    assert set(categories_after) == {"OM", "CO"}


def test_help_schema_falls_back_to_pack_steps_when_pack_registry_metadata_is_invalid(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    ui_map_path = tmp_path / "ui_map.yaml"
    taxonomy_path = tmp_path / "taxonomy.yaml"

    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "metadata": {"step_registry_path": 123},
            "steps": [{"id": "S01"}, {"id": "S02"}],
        },
    )
    _write_yaml(
        ui_map_path,
        {"cockpit_elements": {"target_a": {"description": "A"}}},
    )
    _write_yaml(
        taxonomy_path,
        {"taxonomy": {"categories": [{"code": "OM"}], "trial_flags": []}},
    )

    schema = get_help_response_schema(
        pack_path=pack_path,
        ui_map_path=ui_map_path,
        taxonomy_path=taxonomy_path,
    )
    step_ids = schema["properties"]["next"]["properties"]["step_id"]["enum"]
    assert step_ids == ["S01", "S02"]
