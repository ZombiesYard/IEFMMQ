from pathlib import Path

from adapters.delta_sanitizer import DeltaPolicy, DeltaSanitizer


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "packs" / "fa18c_startup" / "delta_policy.yaml"


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
