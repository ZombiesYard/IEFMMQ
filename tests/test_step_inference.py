from adapters.step_inference import StepInferenceResult, extract_recent_ui_targets, infer_step_id


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
    assert result.missing_conditions == ["vars.fire_test_active==false (complete FIRE TEST A/B)"]


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
    assert result.missing_conditions == ["vars.rpm_r>=25"]


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
    assert result.missing_conditions == ["vars.rpm_r>=60"]


def test_infer_step_is_robust_on_invalid_inputs() -> None:
    result = infer_step_id(
        _pack_steps(),
        vars_map={"power_available": "unknown"},
        recent_ui_targets=["apu_switch", "", 1],  # type: ignore[list-item]
    )
    assert isinstance(result, StepInferenceResult)
    assert result.inferred_step_id in {"S02", "S03", "S04", "S05", "S06", "S07"}


def test_extract_recent_ui_targets_prefers_direct_recent_ui_targets() -> None:
    context = {
        "recent_ui_targets": ["eng_crank_switch", "eng_crank_switch", "apu_switch"],
        "recent_actions": {"recent_buttons": ["fire_test_switch"]},
    }
    assert extract_recent_ui_targets(context) == ["eng_crank_switch", "apu_switch"]
