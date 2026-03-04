from pathlib import Path

import yaml
from core.step_signal_metadata import STEP_EVIDENCE_REQUIREMENT_VALUES, STEP_OBSERVABILITY_VALUES


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
UI_MAP_PATH = BASE_DIR / "packs" / "fa18c_startup" / "ui_map.yaml"
REQUIRED_STEP_IDS = tuple(f"S{i:02d}" for i in range(1, 26))
REQUIRED_NON_EMPTY_UI_TARGET_STEPS = (
    "S01",
    "S02",
    "S03",
    "S04",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S15",
    "S18",
)


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must load to a YAML mapping"
    return data


def test_step_ui_targets_must_exist_and_reference_ui_map_keys() -> None:
    pack = _load_yaml(PACK_PATH)
    ui_map = _load_yaml(UI_MAP_PATH)

    steps = pack.get("steps")
    assert isinstance(steps, list) and steps, "pack.yaml must have non-empty steps"

    cockpit_elements = ui_map.get("cockpit_elements")
    assert isinstance(cockpit_elements, dict) and cockpit_elements, "ui_map.yaml must define cockpit_elements"
    allowed_targets = set(cockpit_elements.keys())

    for step in steps:
        assert isinstance(step, dict), "each step must be a mapping"
        step_id = step.get("id")
        assert isinstance(step_id, str) and step_id, "each step must contain non-empty id"
        assert "ui_targets" in step, f"step {step_id} missing ui_targets"

        ui_targets = step["ui_targets"]
        assert isinstance(ui_targets, list), f"step {step_id} ui_targets must be a list"

        for idx, target in enumerate(ui_targets):
            assert isinstance(target, str) and target, f"step {step_id} ui_targets[{idx}] must be non-empty string"
            assert target in allowed_targets, f"step {step_id} ui_targets[{idx}]={target!r} not found in ui_map.yaml"


def test_required_steps_exist_for_full_cold_start() -> None:
    pack = _load_yaml(PACK_PATH)
    steps = pack.get("steps")
    assert isinstance(steps, list)

    by_id = {step.get("id"): step for step in steps if isinstance(step, dict)}
    for step_id in REQUIRED_STEP_IDS:
        assert step_id in by_id, f"required step {step_id} missing in pack"


def test_clickable_subset_steps_have_non_empty_ui_targets() -> None:
    pack = _load_yaml(PACK_PATH)
    steps = pack.get("steps")
    assert isinstance(steps, list)

    by_id = {step.get("id"): step for step in steps if isinstance(step, dict)}
    for step_id in REQUIRED_NON_EMPTY_UI_TARGET_STEPS:
        assert step_id in by_id, f"required step {step_id} missing in pack"
        ui_targets = by_id[step_id].get("ui_targets")
        assert isinstance(ui_targets, list), f"step {step_id} ui_targets must be list"
        assert ui_targets, f"step {step_id} ui_targets must not be empty for clickable coverage"


def test_step_signal_metadata_values_are_valid() -> None:
    pack = _load_yaml(PACK_PATH)
    steps = pack.get("steps")
    assert isinstance(steps, list)

    for idx, step in enumerate(steps):
        assert isinstance(step, dict), f"pack.steps[{idx}] must be mapping"
        step_id = step.get("id")
        assert isinstance(step_id, str) and step_id, f"pack.steps[{idx}].id must be non-empty string"

        observability = step.get("observability")
        assert isinstance(observability, str), f"step {step_id} observability must be string"
        assert observability in STEP_OBSERVABILITY_VALUES, f"step {step_id} observability invalid: {observability!r}"

        evidence_requirements = step.get("evidence_requirements")
        assert isinstance(evidence_requirements, list), f"step {step_id} evidence_requirements must be list"
        for req_idx, req in enumerate(evidence_requirements):
            assert isinstance(req, str) and req, (
                f"step {step_id} evidence_requirements[{req_idx}] must be non-empty string"
            )
            assert req in STEP_EVIDENCE_REQUIREMENT_VALUES, (
                f"step {step_id} evidence_requirements[{req_idx}] invalid: {req!r}"
            )


def test_pack_steps_do_not_duplicate_registry_text_fields() -> None:
    pack = _load_yaml(PACK_PATH)
    steps = pack.get("steps")
    assert isinstance(steps, list)

    for idx, step in enumerate(steps):
        assert isinstance(step, dict), f"pack.steps[{idx}] must be mapping"
        assert "official_step" not in step, f"pack.steps[{idx}] must not define official_step"
        assert "short_explanation" not in step, f"pack.steps[{idx}] must not define short_explanation"
        assert "cockpit_area" not in step, f"pack.steps[{idx}] must not define cockpit_area"
