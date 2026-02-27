from pathlib import Path
from dataclasses import FrozenInstanceError

import pytest

from adapters.delta_sanitizer import DeltaPolicy, DeltaPolicyError, DeltaSanitizer


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "packs" / "fa18c_startup" / "delta_policy.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def test_time_and_counter_noise_fields_are_filtered() -> None:
    policy = DeltaPolicy.from_yaml(POLICY_PATH)
    sanitizer = DeltaSanitizer(policy)

    out = sanitizer.sanitize_delta(
        {
            "IFEI_CLOCK_S": "01",
            "_UPDATE_COUNTER": 999,
            "EXT_STROBE_LIGHTS": 1,
            "BATTERY_SW": 2,
        },
        t_wall=1.0,
        seq=1,
    )

    assert out.kept == {"BATTERY_SW": 2}
    assert out.dropped_by_reason["blacklist"] == 3


def test_real_button_changes_are_not_filtered() -> None:
    policy = DeltaPolicy.from_yaml(POLICY_PATH)
    sanitizer = DeltaSanitizer(policy)

    out = sanitizer.sanitize_delta(
        {"BATTERY_SW": 2, "ENGINE_CRANK_SW": 1, "APU_CONTROL_SW": 1},
        t_wall=2.0,
        seq=2,
    )

    assert out.kept["BATTERY_SW"] == 2
    assert out.kept["ENGINE_CRANK_SW"] == 1
    assert out.kept["APU_CONTROL_SW"] == 1
    assert out.kept_count == 3


def test_rapid_jitter_is_debounced_and_last_value_kept() -> None:
    policy = DeltaPolicy(
        debounce_ms_by_key={"ENGINE_CRANK_SW": 300},
        max_changes_per_window=10,
    )
    sanitizer = DeltaSanitizer(policy)

    s1 = sanitizer.sanitize_delta({"ENGINE_CRANK_SW": 1}, t_wall=1.000, seq=1)
    s2 = sanitizer.sanitize_delta({"ENGINE_CRANK_SW": 2}, t_wall=1.100, seq=2)
    s3 = sanitizer.sanitize_delta({}, t_wall=1.200, seq=3)
    s4 = sanitizer.sanitize_delta({}, t_wall=1.350, seq=4)

    assert s1.kept == {"ENGINE_CRANK_SW": 1}
    assert s2.kept == {}
    assert s2.dropped_by_reason.get("debounce", 0) == 1
    assert s3.kept == {}
    assert s4.kept == {"ENGINE_CRANK_SW": 2}


def test_flush_uses_emit_time_for_debounce_window() -> None:
    policy = DeltaPolicy(
        debounce_ms_by_key={"ENGINE_CRANK_SW": 300},
        max_changes_per_window=10,
    )
    sanitizer = DeltaSanitizer(policy)

    sanitizer.sanitize_delta({"ENGINE_CRANK_SW": 1}, t_wall=1.000, seq=1)
    sanitizer.sanitize_delta({"ENGINE_CRANK_SW": 2}, t_wall=1.100, seq=2)
    sanitizer.sanitize_delta({}, t_wall=1.600, seq=3)  # flush pending value
    s4 = sanitizer.sanitize_delta({"ENGINE_CRANK_SW": 3}, t_wall=1.650, seq=4)

    assert s4.kept == {}
    assert s4.dropped_by_reason.get("debounce", 0) == 1


def test_flush_without_raw_delta_updates_raw_and_drop_counts_consistently() -> None:
    policy = DeltaPolicy(
        debounce_ms_by_key={"ENGINE_CRANK_SW": 300},
        max_changes_per_window=10,
    )
    sanitizer = DeltaSanitizer(policy)

    sanitizer.sanitize_delta({"ENGINE_CRANK_SW": 1}, t_wall=1.000, seq=1)
    sanitizer.sanitize_delta({"ENGINE_CRANK_SW": 2}, t_wall=1.100, seq=2)  # debounced, pending
    flushed = sanitizer.sanitize_delta({}, t_wall=1.500, seq=3)  # flush pending

    assert flushed.kept == {"ENGINE_CRANK_SW": 2}
    assert flushed.raw_count == 1
    assert flushed.kept_count == 1
    assert flushed.dropped_count == 0
    assert flushed.dropped_by_reason == {}


def test_invalid_keys_counted_in_dropped_total() -> None:
    policy = DeltaPolicy()
    sanitizer = DeltaSanitizer(policy)

    out = sanitizer.sanitize_delta({1: "x", "": "y"}, t_wall=1.0, seq=1)

    assert out.raw_count == 2
    assert out.kept_count == 0
    assert out.dropped_count == 2
    assert out.dropped_by_reason.get("invalid_key", 0) == 2


def test_from_yaml_missing_file_raises_delta_policy_error() -> None:
    missing = FIXTURE_DIR / "missing_delta_policy.yaml"
    with pytest.raises(DeltaPolicyError, match="read failed"):
        DeltaPolicy.from_yaml(missing)


def test_from_yaml_invalid_yaml_raises_delta_policy_error() -> None:
    invalid = FIXTURE_DIR / "delta_policy_invalid_yaml.yaml"
    with pytest.raises(DeltaPolicyError, match="contains invalid YAML"):
        DeltaPolicy.from_yaml(invalid)


def test_delta_policy_normalizes_mutable_inputs_to_immutable_fields() -> None:
    debounce = {"ENGINE_CRANK_SW": 300}
    epsilon = {"SAI_PITCH": 8.0}
    policy = DeltaPolicy(
        ignore_bios_prefixes=["A_", "B_"],
        ignore_bios_keys=["K1", "K2"],
        debounce_ms_by_key=debounce,
        epsilon_by_key=epsilon,
        important_bios_keys=["BATTERY_SW"],
    )

    debounce["ENGINE_CRANK_SW"] = 999
    epsilon["SAI_PITCH"] = 99.0

    assert policy.debounce_ms_for("ENGINE_CRANK_SW") == 300
    assert policy.epsilon_for("SAI_PITCH") == 8.0
    assert isinstance(policy.debounce_ms_by_key, tuple)
    assert isinstance(policy.epsilon_by_key, tuple)

    with pytest.raises(FrozenInstanceError):
        policy.max_changes_per_window = 999  # type: ignore[misc]


def test_delta_policy_prefix_order_does_not_affect_equality() -> None:
    p1 = DeltaPolicy(ignore_bios_prefixes=["SBY_", "IFEI_", "CLOCK_"])
    p2 = DeltaPolicy(ignore_bios_prefixes=["CLOCK_", "SBY_", "IFEI_"])

    assert p1 == p2
    assert p1.ignore_bios_prefixes == ("CLOCK_", "IFEI_", "SBY_")
