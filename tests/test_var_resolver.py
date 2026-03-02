import json
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from core.types_v2 import TelemetryFrame
from core.vars import VarResolver, VarResolverError

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_TELEMETRY_MAP_PATH = REPO_ROOT / "packs" / "fa18c_startup" / "telemetry_map.yaml"
SAMPLE_FRAME_ONCE_PATH = REPO_ROOT / "artifacts" / "dcs_bios_frame_once.json"
SAMPLE_RAW_JSONL_PATH = REPO_ROOT / "logs" / "dcs_bios_raw_15s.jsonl"


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


def test_var_resolver_num_cast_and_missing_helper() -> None:
    mapping = {
        "vars": {
            "rpm_r": "derived(num(bios.IFEI_RPM_R))",
            "rpm_r_gte_25": "derived(vars.rpm_r >= 25)",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    resolver = VarResolver.from_yaml(path)

    frame_ok = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"IFEI_RPM_R": " 27"},
    )
    vars_ok = resolver.resolve(frame_ok)
    assert vars_ok["rpm_r"] == 27
    assert vars_ok["rpm_r_gte_25"] is True
    assert vars_ok["vars_source_missing"] == []

    frame_missing = TelemetryFrame(seq=2, t_wall=2.0, source="dcs_bios", bios={})
    vars_missing = resolver.resolve(frame_missing)
    assert vars_missing["rpm_r"] is None
    assert vars_missing["rpm_r_gte_25"] is False
    assert "rpm_r" in vars_missing["vars_source_missing"]
    assert "rpm_r_gte_25" in vars_missing["vars_source_missing"]


def test_var_resolver_pack_map_resolves_from_dcs_bios_frame_once() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)
    frame = json.loads(SAMPLE_FRAME_ONCE_PATH.read_text(encoding="utf-8"))

    vars_out = resolver.resolve(frame)
    required = ("rpm_r", "rpm_l", "temp_r", "ff_r", "oil_r", "noz_r", "rpm_r_gte_25")
    for key in required:
        assert key in vars_out

    assert isinstance(vars_out["rpm_r"], (int, float))
    assert isinstance(vars_out["temp_r"], (int, float))
    assert vars_out["rpm_r"] >= 0
    assert vars_out["temp_r"] >= 0
    assert vars_out["rpm_r_gte_25"] == (vars_out["rpm_r"] >= 25)
    assert isinstance(vars_out["vars_source_missing"], list)
    for key in ("ff_r", "oil_r", "noz_r"):
        assert (vars_out[key] is None) == (key in vars_out["vars_source_missing"])


def test_var_resolver_pack_battery_on_requires_switch_value_2() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)
    frame_on = TelemetryFrame(
        seq=2,
        t_wall=2.0,
        source="dcs_bios",
        bios={"BATTERY_SW": 2},
    )
    vars_on = resolver.resolve(frame_on)
    assert vars_on["battery_on"] is True

    frame_off = TelemetryFrame(
        seq=99,
        t_wall=99.0,
        source="dcs_bios",
        bios={"BATTERY_SW": 1},
    )
    vars_off = resolver.resolve(frame_off)
    assert vars_off["battery_on"] is False

    frame_override = TelemetryFrame(
        seq=100,
        t_wall=100.0,
        source="dcs_bios",
        bios={"BATTERY_SW": 0},
    )
    vars_override = resolver.resolve(frame_override)
    assert vars_override["battery_on"] is False


def test_var_resolver_pack_engine_crank_switch_three_position_mapping() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame_left = TelemetryFrame(
        seq=201,
        t_wall=201.0,
        source="dcs_bios",
        bios={"ENGINE_CRANK_SW": 0},
    )
    vars_left = resolver.resolve(frame_left)
    assert vars_left["engine_crank_left"] is True
    assert vars_left["engine_crank_right"] is False

    frame_off = TelemetryFrame(
        seq=202,
        t_wall=202.0,
        source="dcs_bios",
        bios={"ENGINE_CRANK_SW": 1},
    )
    vars_off = resolver.resolve(frame_off)
    assert vars_off["engine_crank_left"] is False
    assert vars_off["engine_crank_right"] is False

    frame_right = TelemetryFrame(
        seq=203,
        t_wall=203.0,
        source="dcs_bios",
        bios={"ENGINE_CRANK_SW": 2},
    )
    vars_right = resolver.resolve(frame_right)
    assert vars_right["engine_crank_left"] is False
    assert vars_right["engine_crank_right"] is True


def test_var_resolver_pack_throttle_not_off_flags_track_internal_throttle_axes() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame_off = TelemetryFrame(
        seq=301,
        t_wall=301.0,
        source="dcs_bios",
        bios={"INT_THROTTLE_RIGHT": 0, "INT_THROTTLE_LEFT": 0},
    )
    vars_off = resolver.resolve(frame_off)
    assert vars_off["throttle_r_not_off"] is False
    assert vars_off["throttle_l_not_off"] is False

    frame_idle = TelemetryFrame(
        seq=302,
        t_wall=302.0,
        source="dcs_bios",
        bios={"INT_THROTTLE_RIGHT": 1, "INT_THROTTLE_LEFT": 2},
    )
    vars_idle = resolver.resolve(frame_idle)
    assert vars_idle["throttle_r_not_off"] is True
    assert vars_idle["throttle_l_not_off"] is True


def test_var_resolver_pack_map_resolves_from_raw_jsonl_samples() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)
    lines = [line for line in SAMPLE_RAW_JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "expected non-empty bios raw jsonl sample"

    for line in lines[:30]:
        frame = json.loads(line)
        vars_out = resolver.resolve(frame)
        assert isinstance(vars_out["vars_source_missing"], list)
        for key in ("rpm_r", "rpm_l", "temp_r", "ff_r", "oil_r", "noz_r", "rpm_r_gte_25"):
            assert key in vars_out


def test_var_resolver_none_rule_counts_as_source_missing() -> None:
    mapping = {
        "vars": {
            "manual_placeholder": None,
            "manual_ready": "derived(vars.manual_placeholder == 1)",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    resolver = VarResolver.from_yaml(path)

    frame = TelemetryFrame(seq=1, t_wall=1.0, source="derived", bios={})
    vars_out = resolver.resolve(frame)

    assert vars_out["manual_placeholder"] is None
    assert vars_out["manual_ready"] is False
    assert "manual_placeholder" in vars_out["vars_source_missing"]
    assert "manual_ready" in vars_out["vars_source_missing"]
