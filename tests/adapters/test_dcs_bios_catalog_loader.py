from __future__ import annotations

import json
from pathlib import Path

from adapters.dcs_bios.catalog_loader import build_catalog, load_control_reference, write_catalog_json


def test_catalog_loader_flattens_controls(tmp_path: Path) -> None:
    sample = {
        "TEST": {
            "SWITCH_1": {
                "identifier": "SWITCH_1",
                "category": "TEST",
                "description": "A switch",
                "control_type": "selector",
                "inputs": [{"interface": "set_state", "max_value": 1}],
                "outputs": [
                    {
                        "address": 100,
                        "mask": 255,
                        "shift_by": 0,
                        "max_value": 1,
                        "suffix": "",
                        "type": "integer",
                    }
                ],
            }
        }
    }
    input_path = tmp_path / "controls.json"
    input_path.write_text(json.dumps(sample), encoding="utf-8")

    data = load_control_reference(input_path)
    controls = build_catalog("TEST_AIRCRAFT", data)
    assert len(controls) == 1
    assert controls[0].identifier == "SWITCH_1"
    assert controls[0].value_type == "integer"

    output_path = tmp_path / "catalog.json"
    write_catalog_json("TEST_AIRCRAFT", controls, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["aircraft"] == "TEST_AIRCRAFT"
    assert payload["controls"][0]["id"] == "TEST_AIRCRAFT/SWITCH_1"

