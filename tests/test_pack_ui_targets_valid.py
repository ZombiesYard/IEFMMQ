import os
import re
from pathlib import Path

import pytest
import yaml

from core.step_signal_metadata import STEP_EVIDENCE_REQUIREMENT_VALUES, STEP_OBSERVABILITY_VALUES


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
UI_MAP_PATH = BASE_DIR / "packs" / "fa18c_startup" / "ui_map.yaml"
BIOS_TO_UI_PATH = BASE_DIR / "packs" / "fa18c_startup" / "bios_to_ui.yaml"
CONTROL_ALIGNMENT_INVENTORY_PATH = BASE_DIR / "packs" / "fa18c_startup" / "control_alignment_inventory.yaml"
DEFAULT_CLICKABLEDATA_PATH = BASE_DIR / "CockpitScripts" / "clickabledata.lua"
CLICKABLE_IDS_FIXTURE_PATH = BASE_DIR / "tests" / "fixtures" / "fa18c_clickable_ids.txt"
CLICKABLEDATA_ENV_VAR = "SIMTUTOR_FA18C_CLICKABLEDATA_PATH"
REQUIRED_STEP_IDS = tuple(f"S{i:02d}" for i in range(1, 34))
_PNT_ID_PATTERN = re.compile(r"^pnt_[0-9]+(?:_[0-9]+)?$")
_CLICKABLE_ID_PATTERN = re.compile(r'^\s*elements\["(?P<id>pnt_[0-9_]+)"\]\s*=', re.MULTILINE)
_ALLOWED_MULTI_ACTION_DCS_IDS = {"pnt_124", "pnt_126"}
_INTERACTION_POLICY_KEYS = {"two_position", "multi_position", "buttons", "wheel", "hotkeys"}
_INTERACTION_CLICK_TYPES = {"left", "right", "wheel_up", "wheel_down", "keyboard"}
_INTERACTION_DETENT_DIRECTIONS = {"clockwise", "counter_clockwise"}


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must load to a YAML mapping"
    return data


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_clickable_ids(text: str) -> set[str]:
    ids = {match.group("id") for match in _CLICKABLE_ID_PATTERN.finditer(text)}
    if ids:
        return ids
    return {line.strip() for line in text.splitlines() if _PNT_ID_PATTERN.fullmatch(line.strip())}


def _load_clickable_ids() -> tuple[set[str], Path]:
    candidate_paths: list[Path] = []

    env_path = os.getenv(CLICKABLEDATA_ENV_VAR)
    if env_path:
        candidate_paths.append(Path(env_path))

    candidate_paths.append(DEFAULT_CLICKABLEDATA_PATH)
    candidate_paths.append(CLICKABLE_IDS_FIXTURE_PATH)

    for path in candidate_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        ids = _extract_clickable_ids(text)
        if ids:
            return ids, path

    pytest.skip(
        "clickabledata source not found; set "
        f"{CLICKABLEDATA_ENV_VAR} or provide {DEFAULT_CLICKABLEDATA_PATH} / {CLICKABLE_IDS_FIXTURE_PATH}"
    )


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


def test_all_operable_steps_have_non_empty_ui_targets() -> None:
    pack = _load_yaml(PACK_PATH)
    steps = pack.get("steps")
    assert isinstance(steps, list)

    by_id = {step.get("id"): step for step in steps if isinstance(step, dict)}
    operable_step_ids = tuple(
        step_id
        for step_id, step in by_id.items()
        if isinstance(step_id, str) and step.get("overlay_enabled") is not False
    )
    for step_id in operable_step_ids:
        assert step_id in by_id, f"required step {step_id} missing in pack"
        ui_targets = by_id[step_id].get("ui_targets")
        assert isinstance(ui_targets, list), f"step {step_id} ui_targets must be list"
        assert ui_targets, f"step {step_id} ui_targets must not be empty for operable-step coverage"


def test_ui_map_entries_have_unique_dcs_ids_and_required_fields() -> None:
    ui_map = _load_yaml(UI_MAP_PATH)
    cockpit_elements = ui_map.get("cockpit_elements")
    assert isinstance(cockpit_elements, dict) and cockpit_elements, "ui_map.yaml must define cockpit_elements"

    dcs_to_targets: dict[str, list[str]] = {}
    for target, entry in cockpit_elements.items():
        assert isinstance(target, str) and target, "ui_map target key must be non-empty string"
        assert isinstance(entry, dict), f"ui_map target {target!r} must map to a mapping"

        dcs_id = entry.get("dcs_id")
        assert isinstance(dcs_id, str) and dcs_id, f"ui_map target {target!r} must have non-empty dcs_id"
        assert _PNT_ID_PATTERN.fullmatch(dcs_id), f"ui_map target {target!r} has invalid dcs_id format: {dcs_id!r}"
        dcs_to_targets.setdefault(dcs_id, []).append(target)

        aliases = entry.get("aliases")
        assert isinstance(aliases, list) and aliases, f"ui_map target {target!r} aliases must be non-empty list"
        normalized_aliases = []
        for idx, alias in enumerate(aliases):
            assert isinstance(alias, str) and alias.strip(), (
                f"ui_map target {target!r} aliases[{idx}] must be non-empty string"
            )
            normalized_aliases.append(alias.strip().lower())
        assert len(normalized_aliases) == len(set(normalized_aliases)), (
            f"ui_map target {target!r} aliases must be unique (case-insensitive)"
        )

        panel_area = entry.get("panel_area")
        assert isinstance(panel_area, str) and panel_area.strip(), (
            f"ui_map target {target!r} panel_area must be non-empty string"
        )

        interaction_hint = entry.get("interaction_hint")
        if interaction_hint is not None:
            assert isinstance(interaction_hint, dict), (
                f"ui_map target {target!r} interaction_hint must be a mapping when provided"
            )
            assert isinstance(interaction_hint.get("zh"), str) and interaction_hint["zh"].strip(), (
                f"ui_map target {target!r} interaction_hint.zh must be a non-empty string"
            )
            assert isinstance(interaction_hint.get("en"), str) and interaction_hint["en"].strip(), (
                f"ui_map target {target!r} interaction_hint.en must be a non-empty string"
            )
        interaction = entry.get("interaction")
        if interaction is not None:
            assert isinstance(interaction, dict), (
                f"ui_map target {target!r} interaction must be a mapping when provided"
            )
            click_type = interaction.get("click_type")
            if click_type is not None:
                assert click_type in _INTERACTION_CLICK_TYPES, (
                    f"ui_map target {target!r} interaction.click_type invalid: {click_type!r}"
                )
            detent_direction = interaction.get("detent_direction")
            if detent_direction is not None:
                assert detent_direction in _INTERACTION_DETENT_DIRECTIONS, (
                    f"ui_map target {target!r} interaction.detent_direction invalid: {detent_direction!r}"
                )
            hotkey = interaction.get("hotkey")
            if hotkey is not None:
                assert isinstance(hotkey, str) and hotkey.strip(), (
                    f"ui_map target {target!r} interaction.hotkey must be non-empty string"
                )
            click_type_by_value = interaction.get("click_type_by_value")
            if click_type_by_value is not None:
                assert isinstance(click_type_by_value, dict) and click_type_by_value, (
                    f"ui_map target {target!r} interaction.click_type_by_value must be non-empty mapping"
                )
                for value_name, value_click_type in click_type_by_value.items():
                    assert isinstance(value_name, str) and value_name.strip(), (
                        f"ui_map target {target!r} interaction.click_type_by_value keys must be non-empty strings"
                    )
                    assert value_click_type in _INTERACTION_CLICK_TYPES, (
                        f"ui_map target {target!r} interaction.click_type_by_value[{value_name!r}] invalid: {value_click_type!r}"
                    )
            hotkey_by_action = interaction.get("hotkey_by_action")
            if hotkey_by_action is not None:
                assert isinstance(hotkey_by_action, dict) and hotkey_by_action, (
                    f"ui_map target {target!r} interaction.hotkey_by_action must be non-empty mapping"
                )
                for action_name, action_hotkey in hotkey_by_action.items():
                    assert isinstance(action_name, str) and action_name.strip(), (
                        f"ui_map target {target!r} interaction.hotkey_by_action keys must be non-empty strings"
                    )
                    assert isinstance(action_hotkey, str) and action_hotkey.strip(), (
                        f"ui_map target {target!r} interaction.hotkey_by_action[{action_name!r}] must be non-empty string"
                    )

    duplicated_dcs_ids = {dcs_id: targets for dcs_id, targets in dcs_to_targets.items() if len(targets) > 1}
    assert set(duplicated_dcs_ids.keys()).issubset(_ALLOWED_MULTI_ACTION_DCS_IDS), (
        "only UFC COMM channel selector controls may share dcs_id in ui_map"
    )
    for dcs_id, targets in duplicated_dcs_ids.items():
        assert len(targets) == 2, f"{dcs_id} should map to exactly two semantic targets, got {targets!r}"


def test_ui_map_dcs_ids_align_with_cockpit_clickabledata() -> None:
    ui_map = _load_yaml(UI_MAP_PATH)
    cockpit_elements = ui_map.get("cockpit_elements")
    assert isinstance(cockpit_elements, dict) and cockpit_elements, "ui_map.yaml must define cockpit_elements"

    clickable_ids, source_path = _load_clickable_ids()
    assert clickable_ids, f"{source_path} must expose at least one pnt_* id"

    for target, entry in cockpit_elements.items():
        assert isinstance(entry, dict), f"ui_map target {target!r} must map to a mapping"
        dcs_id = entry.get("dcs_id")
        assert isinstance(dcs_id, str) and dcs_id
        assert dcs_id in clickable_ids, (
            f"ui_map target {target!r} uses dcs_id {dcs_id!r} not found in clickable reference source: {source_path}"
        )


def test_control_alignment_inventory_covers_current_ui_map_targets() -> None:
    pack = _load_yaml(PACK_PATH)
    ui_map = _load_yaml(UI_MAP_PATH)
    bios_to_ui = _load_yaml(BIOS_TO_UI_PATH)
    inventory = _load_yaml(CONTROL_ALIGNMENT_INVENTORY_PATH)

    cockpit_elements = ui_map.get("cockpit_elements")
    controls = inventory.get("controls")
    steps = pack.get("steps")
    assert isinstance(cockpit_elements, dict) and cockpit_elements
    assert isinstance(controls, dict) and controls
    assert isinstance(steps, list) and steps
    assert set(controls) == set(cockpit_elements)

    expected_steps_by_target: dict[str, list[str]] = {}
    for step in steps:
        assert isinstance(step, dict)
        step_id = step.get("id")
        assert isinstance(step_id, str) and step_id
        ui_targets = step.get("ui_targets")
        assert isinstance(ui_targets, list)
        for target in ui_targets:
            if isinstance(target, str) and target:
                expected_steps_by_target.setdefault(target, []).append(step_id)

    mapped_keys_by_target: dict[str, set[str]] = {}
    mappings = bios_to_ui.get("mappings")
    assert isinstance(mappings, dict) and mappings
    for bios_key, raw_targets in mappings.items():
        assert isinstance(bios_key, str) and bios_key
        if isinstance(raw_targets, list):
            targets = raw_targets
        elif isinstance(raw_targets, dict):
            targets = raw_targets.get("targets")
        else:
            targets = [raw_targets]
        assert isinstance(targets, list)
        for target in targets:
            if isinstance(target, str) and target:
                mapped_keys_by_target.setdefault(target, set()).add(bios_key)

    for target, entry in controls.items():
        assert isinstance(entry, dict), f"inventory target {target!r} must map to a mapping"
        assert entry.get("dcs_id") == cockpit_elements[target].get("dcs_id")
        clickable = entry.get("clickabledata")
        assert isinstance(clickable, dict), f"inventory target {target!r} missing clickabledata"
        assert isinstance(clickable.get("label"), str) and clickable["label"].strip()
        assert isinstance(entry.get("bios_keys"), list), f"inventory target {target!r} missing bios_keys list"
        assert isinstance(entry.get("telemetry_vars"), list), (
            f"inventory target {target!r} missing telemetry_vars list"
        )
        assert isinstance(entry.get("pack_steps"), list), f"inventory target {target!r} missing pack_steps list"
        assert entry["pack_steps"] == expected_steps_by_target.get(target, [])
        assert mapped_keys_by_target.get(target, set()).issubset(set(entry["bios_keys"]))
        assert isinstance(entry.get("highlightable"), bool), f"inventory target {target!r} missing highlightable bool"


def test_ddi_selector_brightness_and_contrast_targets_are_separate() -> None:
    ui_map = _load_yaml(UI_MAP_PATH)
    cockpit_elements = ui_map["cockpit_elements"]

    expected = {
        "left_mdi_brightness_selector": ("pnt_51", "LEFT_DDI_BRT_SELECT"),
        "left_mdi_brightness_control": ("pnt_52", "LEFT_DDI_BRT_CTL"),
        "left_mdi_contrast_control": ("pnt_53", "LEFT_DDI_CONT_CTL"),
        "right_mdi_brightness_selector": ("pnt_76", "RIGHT_DDI_BRT_SELECT"),
        "right_mdi_brightness_control": ("pnt_77", "RIGHT_DDI_BRT_CTL"),
        "right_mdi_contrast_control": ("pnt_78", "RIGHT_DDI_CONT_CTL"),
    }

    observed_dcs_ids: set[str] = set()
    for target, (dcs_id, bios_key) in expected.items():
        entry = cockpit_elements[target]
        assert entry["dcs_id"] == dcs_id
        assert bios_key in entry["aliases"]
        assert dcs_id not in observed_dcs_ids
        observed_dcs_ids.add(dcs_id)

    assert "LEFT_DDI_BRT_CTL" not in cockpit_elements["left_mdi_brightness_selector"]["aliases"]
    assert "LEFT_DDI_CONT_CTL" not in cockpit_elements["left_mdi_brightness_selector"]["aliases"]
    assert "RIGHT_DDI_BRT_CTL" not in cockpit_elements["right_mdi_brightness_selector"]["aliases"]
    assert "RIGHT_DDI_CONT_CTL" not in cockpit_elements["right_mdi_brightness_selector"]["aliases"]


def test_s08_display_power_targets_stay_on_ddi_selectors_not_potentiometers() -> None:
    pack = _load_yaml(PACK_PATH)
    s08 = next(step for step in pack["steps"] if step["id"] == "S08")

    targets = set(s08["ui_targets"])
    assert {"left_mdi_brightness_selector", "right_mdi_brightness_selector"}.issubset(targets)
    assert "left_mdi_brightness_control" not in targets
    assert "right_mdi_brightness_control" not in targets
    assert "left_mdi_contrast_control" not in targets
    assert "right_mdi_contrast_control" not in targets


def test_ui_map_interaction_policy_has_bilingual_entries() -> None:
    ui_map = _load_yaml(UI_MAP_PATH)
    policy = ui_map.get("interaction_policy")
    assert isinstance(policy, dict), "ui_map.yaml interaction_policy must be a mapping"
    assert set(policy.keys()) == _INTERACTION_POLICY_KEYS

    for key in sorted(_INTERACTION_POLICY_KEYS):
        entry = policy[key]
        assert isinstance(entry, dict), f"ui_map interaction_policy[{key!r}] must be a mapping"
        assert isinstance(entry.get("zh"), str) and entry["zh"].strip(), (
            f"ui_map interaction_policy[{key!r}].zh must be a non-empty string"
        )
        assert isinstance(entry.get("en"), str) and entry["en"].strip(), (
            f"ui_map interaction_policy[{key!r}].en must be a non-empty string"
        )


def test_pack_allowlist_matches_step_union_and_ui_map() -> None:
    pack = _load_yaml(PACK_PATH)
    ui_map = _load_yaml(UI_MAP_PATH)
    steps = pack.get("steps")
    assert isinstance(steps, list) and steps, "pack.yaml must have non-empty steps"

    cockpit_elements = ui_map.get("cockpit_elements")
    assert isinstance(cockpit_elements, dict) and cockpit_elements, "ui_map.yaml must define cockpit_elements"
    allowed_targets = set(cockpit_elements.keys())

    pack_ui_targets = pack.get("ui_targets")
    assert isinstance(pack_ui_targets, list) and pack_ui_targets, "pack.ui_targets must be a non-empty list"
    for idx, target in enumerate(pack_ui_targets):
        assert isinstance(target, str) and target, f"pack.ui_targets[{idx}] must be non-empty string"
        assert target in allowed_targets, f"pack.ui_targets[{idx}]={target!r} not found in ui_map"
    assert len(pack_ui_targets) == len(set(pack_ui_targets)), "pack.ui_targets must not contain duplicates"

    step_target_sequence: list[str] = []
    for idx, step in enumerate(steps):
        assert isinstance(step, dict), f"pack.steps[{idx}] must be mapping"
        ui_targets = step.get("ui_targets")
        assert isinstance(ui_targets, list), f"pack.steps[{idx}].ui_targets must be list"
        for target in ui_targets:
            if isinstance(target, str) and target:
                step_target_sequence.append(target)
    step_union = _dedupe_keep_order(step_target_sequence)
    assert set(step_union).issubset(set(pack_ui_targets)), (
        "pack.ui_targets must contain every deduped step ui target and may include extra dormant targets"
    )


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
