import os
from pathlib import Path
import time

import yaml
from adapters.step_inference import StepInferenceResult, extract_recent_ui_targets, infer_step_id, load_pack_steps


def _pack_steps() -> list[dict]:
    return [
        {"id": "S01"},
        {"id": "S02"},
        {"id": "S03"},
        {"id": "S04"},
        {"id": "S05"},
        {"id": "S06"},
        {"id": "S07"},
    ]


def test_infer_step_s01_when_power_blocked() -> None:
    result = infer_step_id(
        _pack_steps(),
        {"power_available": False, "battery_on": False, "l_gen_on": True, "r_gen_on": True},
        [],
    )
    assert result.inferred_step_id == "S01"
    assert "vars.battery_on==true" in result.missing_conditions


def test_infer_step_s02_when_fire_test_not_seen_and_apu_not_started() -> None:
    result = infer_step_id(
        _pack_steps(),
        {"power_available": True, "apu_on": False, "apu_ready": False, "rpm_r": 0},
        [],
    )
    assert result.inferred_step_id == "S02"
    assert "recent_ui_targets has fire_test_switch" in result.missing_conditions


def test_infer_step_s02_when_fire_test_is_still_active() -> None:
    result = infer_step_id(
        _pack_steps(),
        {"power_available": True, "apu_on": False, "apu_ready": False, "fire_test_active": True, "rpm_r": 0},
        [],
    )
    assert result.inferred_step_id == "S02"
    assert result.missing_conditions == ("vars.fire_test_active==false (complete FIRE TEST A/B)",)


def test_infer_step_s03_when_apu_not_ready() -> None:
    result = infer_step_id(
        _pack_steps(),
        {"power_available": True, "apu_on": True, "apu_ready": False},
        ["apu_switch", "fire_test_switch"],
    )
    assert result.inferred_step_id == "S03"
    assert "vars.apu_ready==true" in result.missing_conditions


def test_infer_step_s04_when_apu_ready_but_no_engine_crank() -> None:
    result = infer_step_id(
        _pack_steps(),
        {"power_available": True, "apu_ready": True, "engine_crank_right": False, "rpm_r": 0},
        ["apu_switch"],
    )
    assert result.inferred_step_id == "S04"
    assert "vars.engine_crank_right==true" in result.missing_conditions


def test_infer_step_s05_when_rpm_below_25() -> None:
    result = infer_step_id(
        _pack_steps(),
        {"power_available": True, "apu_ready": True, "engine_crank_right": True, "rpm_r": 22},
        ["eng_crank_switch"],
    )
    assert result.inferred_step_id == "S05"
    assert result.missing_conditions == ("vars.rpm_r>=25",)


def test_infer_step_s06_when_rpm_over_60_but_bleed_action_missing() -> None:
    result = infer_step_id(
        _pack_steps(),
        {"power_available": True, "apu_ready": True, "engine_crank_right": True, "rpm_r": 63},
        ["eng_crank_switch"],
    )
    assert result.inferred_step_id == "S06"
    assert "recent_ui_targets has bleed_air_knob" in result.missing_conditions


def test_infer_step_s06_when_rpm_between_25_and_60() -> None:
    result = infer_step_id(
        _pack_steps(),
        {"power_available": True, "apu_ready": True, "engine_crank_right": True, "rpm_r": 45},
        ["eng_crank_switch"],
    )
    assert result.inferred_step_id == "S06"
    assert result.missing_conditions == ("vars.rpm_r>=60",)


def test_infer_step_s05_when_throttle_not_moved_from_off_after_rpm_25() -> None:
    result = infer_step_id(
        _pack_steps(),
        {
            "power_available": True,
            "apu_ready": True,
            "engine_crank_right": True,
            "rpm_r": 30,
            "throttle_r_not_off": False,
        },
        ["eng_crank_switch"],
    )
    assert result.inferred_step_id == "S05"
    assert result.missing_conditions == ("vars.throttle_r_not_off==true",)


def test_infer_step_s05_when_throttle_state_is_unknown_after_rpm_25() -> None:
    result = infer_step_id(
        _pack_steps(),
        {
            "power_available": True,
            "apu_ready": True,
            "engine_crank_right": True,
            "rpm_r": 30,
            "throttle_r_not_off": None,
        },
        ["eng_crank_switch"],
    )
    assert result.inferred_step_id == "S05"
    assert result.missing_conditions == ("vars.throttle_r_not_off==true",)


def test_infer_step_s05_when_throttle_state_is_unparseable_after_rpm_25() -> None:
    result = infer_step_id(
        _pack_steps(),
        {
            "power_available": True,
            "apu_ready": True,
            "engine_crank_right": True,
            "rpm_r": 30,
            "throttle_r_not_off": "unknown",
        },
        ["eng_crank_switch"],
    )
    assert result.inferred_step_id == "S05"
    assert result.missing_conditions == ("vars.throttle_r_not_off==true",)


def test_infer_step_advances_to_s06_when_throttle_key_is_missing_after_rpm_25() -> None:
    result = infer_step_id(
        _pack_steps(),
        {
            "power_available": True,
            "apu_ready": True,
            "engine_crank_right": True,
            "rpm_r": 45,
            # throttle_r_not_off intentionally omitted
        },
        ["eng_crank_switch"],
    )
    assert result.inferred_step_id == "S06"
    assert result.missing_conditions == ("vars.rpm_r>=60",)


def test_infer_step_is_robust_on_invalid_inputs() -> None:
    result = infer_step_id(
        _pack_steps(),
        vars_map={"power_available": "unknown"},
        recent_ui_targets=["apu_switch", "", 1],  # type: ignore[list-item]
    )
    assert isinstance(result, StepInferenceResult)
    assert result.inferred_step_id in {"S02", "S03", "S04", "S05", "S06", "S07"}


def test_infer_step_distinguishes_false_vs_source_missing_unknown_for_power() -> None:
    result_false = infer_step_id(
        _pack_steps(),
        {
            "power_available": False,
            "battery_on": False,
            "l_gen_on": True,
            "r_gen_on": True,
        },
        [],
    )
    assert result_false.inferred_step_id == "S01"

    result_unknown = infer_step_id(
        _pack_steps(),
        {
            "power_available": False,
            "battery_on": False,
            "l_gen_on": True,
            "r_gen_on": True,
            "vars_source_missing": ["power_available", "battery_on"],
        },
        [],
    )
    assert result_unknown.inferred_step_id == "S02"


def test_extract_recent_ui_targets_prefers_direct_recent_ui_targets() -> None:
    context = {
        "recent_ui_targets": ["eng_crank_switch", "eng_crank_switch", "apu_switch"],
        "recent_actions": {"recent_buttons": ["fire_test_switch"]},
    }
    assert extract_recent_ui_targets(context) == ["eng_crank_switch", "apu_switch"]


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


def test_load_pack_steps_cache_invalidates_when_registry_changes(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    registry_path = tmp_path / "step_registry.yaml"

    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "metadata": {"step_registry_path": "step_registry.yaml"},
            "steps": [{"id": "S01", "marker": "fallback"}],
        },
    )
    _write_yaml(registry_path, _registry_payload("first"))

    first = load_pack_steps(pack_path)
    assert first[0]["short_explanation"] == "first"

    _write_yaml(registry_path, _registry_payload("updated"))
    _bump_mtime(registry_path)

    second = load_pack_steps(pack_path)
    assert second[0]["short_explanation"] == "updated"


def test_load_pack_steps_cache_invalidates_when_pack_fallback_changes(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "steps": [{"id": "S01", "marker": "first"}],
        },
    )

    first = load_pack_steps(pack_path)
    assert first[0]["marker"] == "first"

    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "steps": [{"id": "S01", "marker": "updated"}],
        },
    )
    _bump_mtime(pack_path)

    second = load_pack_steps(pack_path)
    assert second[0]["marker"] == "updated"


def test_load_pack_steps_falls_back_when_pack_metadata_registry_path_is_invalid(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "metadata": {"step_registry_path": 123},
            "steps": [{"id": "S01", "marker": "from_pack"}],
        },
    )

    loaded = load_pack_steps(pack_path)
    assert loaded[0]["id"] == "S01"
    assert loaded[0]["marker"] == "from_pack"
