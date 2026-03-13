from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator

from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from adapters.pack_gates import load_pack_gate_config
from adapters.step_inference import infer_step_id, load_pack_steps
from adapters.event_store.telemetry_writer import TelemetryWriter
from core.vars import VarResolver
from tools.build_coldstart_state_matrix import (
    build_coldstart_state_matrix_dataset,
    main,
)


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
TELEMETRY_MAP_PATH = BASE_DIR / "packs" / "fa18c_startup" / "telemetry_map.yaml"
BIOS_TO_UI_PATH = BASE_DIR / "packs" / "fa18c_startup" / "bios_to_ui.yaml"


def _load_bios_frame_schema() -> dict:
    schema_path = resources.files("simtutor.schemas.v2") / "dcs_bios_frame.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _recent_ui_targets_from_frames(frames: list[dict], mapper: BiosUiMapper) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for frame in frames[:-1]:
        delta = frame.get("delta")
        if not isinstance(delta, dict):
            continue
        for bios_key in delta.keys():
            for target in mapper.targets_for_key(str(bios_key)):
                if target in seen:
                    continue
                seen.add(target)
                out.append(target)
    return out


def test_build_coldstart_state_matrix_covers_all_steps_and_expected_inference(tmp_path: Path) -> None:
    output_dir = tmp_path / "coldstart_matrix"
    manifest = build_coldstart_state_matrix_dataset(
        output_dir=output_dir,
        pack_path=PACK_PATH,
        telemetry_map_path=TELEMETRY_MAP_PATH,
        bios_to_ui_path=BIOS_TO_UI_PATH,
        scenario_profile="airfield",
    )

    assert manifest["scenario_profile"] == "airfield"
    assert manifest["case_count"] == 50
    assert (output_dir / "matrix.json").exists()

    expected_step_ids = {f"S{i:02d}" for i in range(1, 26)}
    observed: dict[str, set[str]] = {}
    frame_schema = _load_bios_frame_schema()
    validator = Draft202012Validator(frame_schema)
    mapper = BiosUiMapper.from_yaml(BIOS_TO_UI_PATH)
    resolver = VarResolver.from_yaml(TELEMETRY_MAP_PATH)
    pack_steps = load_pack_steps(PACK_PATH)
    gate_cfg = load_pack_gate_config(PACK_PATH, scenario_profile="airfield")

    for case in manifest["cases"]:
        step_id = case["step_id"]
        observed.setdefault(step_id, set()).add(case["state_kind"])

        replay_path = output_dir / case["replay_input"]
        assert replay_path.exists(), replay_path
        frames = TelemetryWriter.load(replay_path)
        assert len(frames) == case["frame_count"]
        assert frames, case["case_id"]

        for frame in frames:
            validator.validate(frame)

        inferred_recent_targets = _recent_ui_targets_from_frames(frames, mapper)
        assert inferred_recent_targets == case["recent_ui_targets"]

        final_vars = resolver.resolve(frames[-1])
        result = infer_step_id(
            pack_steps,
            final_vars,
            inferred_recent_targets,
            precondition_gates=gate_cfg["precondition_gates"],
            completion_gates=gate_cfg["completion_gates"],
            pack_path=PACK_PATH,
            scenario_profile="airfield",
        )
        assert result.inferred_step_id == case["expected_inferred_step_id"]
        assert list(result.missing_conditions) == case["expected_missing_conditions"]

    assert set(observed.keys()) == expected_step_ids
    for step_id in sorted(expected_step_ids):
        assert observed[step_id] == {"blocked", "just_completed"}


def test_build_coldstart_state_matrix_cli_supports_carrier_profile(tmp_path: Path) -> None:
    output_dir = tmp_path / "carrier_matrix"
    code = main(
        [
            "--output-dir",
            str(output_dir),
            "--pack",
            str(PACK_PATH),
            "--telemetry-map",
            str(TELEMETRY_MAP_PATH),
            "--bios-to-ui",
            str(BIOS_TO_UI_PATH),
            "--scenario-profile",
            "carrier",
        ]
    )

    assert code == 0
    manifest = json.loads((output_dir / "matrix.json").read_text(encoding="utf-8"))
    assert manifest["scenario_profile"] == "carrier"
    s12_case = next(
        case
        for case in manifest["cases"]
        if case["step_id"] == "S12" and case["state_kind"] == "just_completed"
    )
    s23_case = next(
        case
        for case in manifest["cases"]
        if case["step_id"] == "S23" and case["state_kind"] == "just_completed"
    )

    resolver = VarResolver.from_yaml(TELEMETRY_MAP_PATH)
    s12_frames = TelemetryWriter.load(output_dir / s12_case["replay_input"])
    s23_frames = TelemetryWriter.load(output_dir / s23_case["replay_input"])
    s12_vars = resolver.resolve(s12_frames[-1])
    s23_vars = resolver.resolve(s23_frames[-1])

    assert s12_vars["ins_mode"] == 1
    assert 30 <= s23_vars["radar_altimeter_bug_value"] <= 60
