from datetime import datetime, timezone
from pathlib import Path

import pytest

from adapters.pack_gates import evaluate_pack_gates, load_pack_gate_config


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"


def _obs_with_vars(**vars_map):
    return {
        "observation_id": "obs-1",
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "source": "unit",
        "payload": {"vars": vars_map},
        "version": "v1",
    }


def test_pack_gate_config_contains_s01_to_s11_for_precondition_and_completion() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    pre = cfg["precondition_gates"]
    comp = cfg["completion_gates"]

    for step_id in [f"S{i:02d}" for i in range(1, 12)]:
        assert step_id in pre, f"missing precondition gate config for {step_id}"
        assert step_id in comp, f"missing completion gate config for {step_id}"
        assert isinstance(pre[step_id], tuple)
        assert isinstance(comp[step_id], tuple)


def test_evaluate_pack_gates_reports_blocked_reason_code_for_s04_precondition() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(apu_ready=False)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    s04_pre = gates["S04.precondition"]

    assert s04_pre["status"] == "blocked"
    assert s04_pre["allowed"] is False
    assert s04_pre["reason_code"] == "s04_requires_apu_ready"
    assert isinstance(s04_pre["reason"], str) and s04_pre["reason"]


def test_evaluate_pack_gates_allows_s04_precondition_when_apu_ready_true() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(apu_ready=True)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    s04_pre = gates["S04.precondition"]

    assert s04_pre["status"] == "allowed"
    assert s04_pre["allowed"] is True
    assert s04_pre["reason_code"] == "ok"
    assert s04_pre["reason"] is None


def test_load_pack_gate_config_rejects_non_mapping_gate_block(tmp_path: Path) -> None:
    bad_pack = tmp_path / "bad_pack.yaml"
    bad_pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "precondition_gates: []\n"
        "completion_gates: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="precondition_gates must be a mapping"):
        load_pack_gate_config(bad_pack)


def test_load_pack_gate_config_rejects_rule_without_op(tmp_path: Path) -> None:
    bad_pack = tmp_path / "bad_pack_missing_op.yaml"
    bad_pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "precondition_gates:\n"
        "  S01:\n"
        "    - var: vars.power_available\n"
        "completion_gates: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"precondition_gates\.S01\[0\]\.op"):
        load_pack_gate_config(bad_pack)


def test_evaluate_pack_gates_accepts_generator_rules_iterable() -> None:
    precondition_gates = {
        "S99": ({"op": "flag_true", "var": "vars.apu_ready"} for _ in [0]),
    }
    completion_gates: dict[str, tuple[dict[str, object], ...]] = {}
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(apu_ready=False)],
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
    )

    assert gates["S99.precondition"]["status"] == "blocked"
    assert gates["S99.precondition"]["allowed"] is False
