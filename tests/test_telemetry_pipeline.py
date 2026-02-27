import json
from pathlib import Path

from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from adapters.telemetry_pipeline import TelemetryDebugCache, enrich_bios_observation
from core.types import Observation
from core.vars import VarResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = REPO_ROOT / "packs" / "fa18c_startup"


def _resolver() -> VarResolver:
    return VarResolver.from_yaml(PACK_DIR / "telemetry_map.yaml")


def _mapper() -> BiosUiMapper:
    return BiosUiMapper.from_yaml(PACK_DIR / "bios_to_ui.yaml", PACK_DIR / "ui_map.yaml")


def test_enrich_bios_observation_compacts_payload_and_keeps_metadata() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "schema_version": "v2",
            "seq": 42,
            "t_wall": 123.45,
            "aircraft": "FA-18C_hornet",
            "bios": {
                "BATTERY_SW": 2,
                "EXT_PWR_SW": 1,
                "L_GEN_SW": 1,
                "R_GEN_SW": 1,
                "ENGINE_CRANK_SW": 1,
                "IFEI_RPM_R": " 64",
                "IFEI_RPM_L": " 63",
                "IFEI_TEMP_R": "320",
                "IFEI_TEMP_L": "300",
                "IFEI_FF_R": "500",
                "IFEI_FF_L": "480",
                "IFEI_OIL_PRESS_R": "80",
                "IFEI_OIL_PRESS_L": "78",
                "EXT_NOZZLE_POS_R": 50010,
                "EXT_NOZZLE_POS_L": 49000,
            },
            "delta": {
                "BATTERY_SW": 2,
                "ENGINE_CRANK_SW": 1,
                "IFEI_RPM_R": " 64",
                "IFEI_RPM_L": " 63",
            },
        },
        metadata={"seq": 42, "gap": 0, "delta_count": 4, "from_addr": "127.0.0.1:5000"},
    )

    enriched = enrich_bios_observation(obs, _resolver(), mapper=_mapper())

    assert set(enriched.payload.keys()) == {"seq", "t_wall", "vars", "delta_summary", "recent_ui_targets"}
    assert "bios" not in enriched.payload
    assert "delta" not in enriched.payload
    assert enriched.payload["seq"] == 42
    assert enriched.payload["t_wall"] == 123.45
    assert enriched.payload["vars"]["rpm_r"] == 64
    assert enriched.payload["vars"]["rpm_r_gte_25"] is True
    assert enriched.payload["delta_summary"]["delta_count"] == 4
    assert "battery_switch" in enriched.payload["recent_ui_targets"]
    assert "eng_crank_switch" in enriched.payload["recent_ui_targets"]

    assert enriched.metadata["seq"] == 42
    assert enriched.metadata["gap"] == 0
    assert enriched.metadata["delta_count"] == 4
    assert isinstance(enriched.metadata.get("bios_hash"), str)
    assert len(enriched.metadata["bios_hash"]) == 64

    json.dumps(enriched.to_dict(), ensure_ascii=False)


def test_enrich_bios_observation_supports_tag_hook_and_debug_cache() -> None:
    obs = Observation(
        source="dcs_bios",
        tags=["live"],
        payload={
            "seq": 3,
            "t_wall": 3.0,
            "bios": {"BATTERY_SW": 2, "IFEI_RPM_R": " 10"},
            "delta": {"BATTERY_SW": 2},
        },
        metadata={"seq": 3, "gap": 1, "delta_count": 1},
    )
    cache = TelemetryDebugCache()

    def tag_hook(_obs: Observation, payload: dict) -> list[str]:
        if payload["vars"].get("battery_on"):
            return ["power_on"]
        return []

    enriched = enrich_bios_observation(
        obs,
        _resolver(),
        mapper=_mapper(),
        tag_hook=tag_hook,
        debug_cache=cache,
    )

    assert enriched.tags == ["live", "power_on"]
    assert cache.last_seq == 3
    assert cache.last_bios.get("BATTERY_SW") == 2
    assert cache.last_delta == {"BATTERY_SW": 2}
    assert cache.last_bios_hash == enriched.metadata["bios_hash"]
