from pathlib import Path
from uuid import uuid4

import yaml

from core.types_v2 import TelemetryFrame
from core.vars import VarResolver


def _tmp_dir() -> Path:
    base = Path("tests/.tmp_telemetry")
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_var_resolver_maps_bios_and_derived() -> None:
    mapping = {
        "vars": {
            "battery_on": "bios.BATTERY_SW == 2",
            "apu_on": "bios.APU_CONTROL_SW == 1",
            "power_available": "derived(vars.battery_on and vars.apu_on)",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    resolver = VarResolver.from_yaml(path)
    frame = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"BATTERY_SW": 2, "APU_CONTROL_SW": 1},
    )
    vars_out = resolver.resolve(frame)
    assert vars_out["battery_on"] is True
    assert vars_out["apu_on"] is True
    assert vars_out["power_available"] is True
