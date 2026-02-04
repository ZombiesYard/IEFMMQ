from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from core.types_v2 import TelemetryFrame
from core.vars import VarResolver, VarResolverError


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


def test_var_resolver_rejects_is_operator() -> None:
    """Test that 'is' comparison operator raises VarResolverError."""
    mapping = {
        "vars": {
            "invalid_check": "bios.VALUE is None",
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
        bios={"VALUE": 1},
    )
    
    with pytest.raises(VarResolverError, match="Unsupported comparison operator.*Is"):
        resolver.resolve(frame)


def test_var_resolver_rejects_is_not_operator() -> None:
    """Test that 'is not' comparison operator raises VarResolverError."""
    mapping = {
        "vars": {
            "invalid_check": "bios.VALUE is not None",
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
        bios={"VALUE": 1},
    )
    
    with pytest.raises(VarResolverError, match="Unsupported comparison operator.*IsNot"):
        resolver.resolve(frame)


def test_var_resolver_rejects_in_operator() -> None:
    """Test that 'in' comparison operator raises VarResolverError."""
    mapping = {
        "vars": {
            "invalid_check": "bios.VALUE in [1, 2, 3]",
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
        bios={"VALUE": 1},
    )
    
    with pytest.raises(VarResolverError, match="Unsupported comparison operator.*In"):
        resolver.resolve(frame)


def test_var_resolver_rejects_not_in_operator() -> None:
    """Test that 'not in' comparison operator raises VarResolverError."""
    mapping = {
        "vars": {
            "invalid_check": "bios.VALUE not in [4, 5, 6]",
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
        bios={"VALUE": 1},
    )
    
    with pytest.raises(VarResolverError, match="Unsupported comparison operator.*NotIn"):
        resolver.resolve(frame)
