import os
import re
from pathlib import Path

import pytest
import yaml

from core.step_signal_metadata import STEP_EVIDENCE_REQUIREMENT_VALUES, STEP_OBSERVABILITY_VALUES


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
UI_MAP_PATH = BASE_DIR / "packs" / "fa18c_startup" / "ui_map.yaml"
DEFAULT_CLICKABLEDATA_PATH = BASE_DIR / "CockpitScripts" / "clickabledata.lua"
CLICKABLE_IDS_FIXTURE_PATH = BASE_DIR / "tests" / "fixtures" / "fa18c_clickable_ids.txt"
CLICKABLEDATA_ENV_VAR = "SIMTUTOR_FA18C_CLICKABLEDATA_PATH"
REQUIRED_STEP_IDS = tuple(f"S{i:02d}" for i in range(1, 26))
REQUIRED_OPERABLE_STEP_IDS = REQUIRED_STEP_IDS
_PNT_ID_PATTERN = re.compile(r"^pnt_[0-9]+(?:_[0-9]+)?$")
_CLICKABLE_ID_PATTERN = re.compile(r'^\s*elements\["(?P<id>pnt_[0-9_]+)"\]\s*=', re.MULTILINE)
_ALLOWED_MULTI_ACTION_DCS_IDS = {"pnt_124", "pnt_126"}


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
    for step_id in REQUIRED_OPERABLE_STEP_IDS:
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
    assert pack_ui_targets == step_union, "pack.ui_targets must exactly match deduped step ui_targets union order"


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
