from datetime import datetime, timedelta, timezone

from core.gating import GatingEngine


def iso(dt):
    return dt.isoformat()


def make_obs(ts, payload=None, tags=None):
    return {
        "observation_id": "x",
        "timestamp": iso(ts),
        "source": "mock",
        "payload": payload or {},
        "tags": tags or [],
        "version": "v1",
    }


def test_var_gte_allows_when_met():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"rpm": 0.25})
    engine = GatingEngine([{"op": "var_gte", "var": "payload.rpm", "value": 0.2}])
    res = engine.evaluate([obs])
    assert res.allowed


def test_var_gte_blocks_when_low():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"rpm": 0.15})
    engine = GatingEngine([{"op": "var_gte", "var": "payload.rpm", "value": 0.2}])
    res = engine.evaluate([obs])
    assert not res.allowed
    assert "payload.rpm" in res.reason


def test_arg_in_range_blocks_out_of_bounds():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"temp": 650})
    engine = GatingEngine([{"op": "arg_in_range", "var": "payload.temp", "min": 190, "max": 590}])
    res = engine.evaluate([obs])
    assert not res.allowed
    assert "payload.temp" in res.reason


def test_arg_in_range_allows_when_in_bounds():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"temp": 400})
    engine = GatingEngine([{"op": "arg_in_range", "var": "payload.temp", "min": 190, "max": 590}])
    res = engine.evaluate([obs])
    assert res.allowed


def test_flag_true_passes():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"apu_ready": True})
    engine = GatingEngine([{"op": "flag_true", "var": "payload.apu_ready"}])
    res = engine.evaluate([obs])
    assert res.allowed


def test_flag_true_blocks_for_false_none_or_missing():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine = GatingEngine([{"op": "flag_true", "var": "payload.apu_ready"}])
    res_false = engine.evaluate([make_obs(now, payload={"apu_ready": False})])
    assert not res_false.allowed

    res_none = engine.evaluate([make_obs(now, payload={"apu_ready": None})])
    assert not res_none.allowed

    res_missing = engine.evaluate([make_obs(now, payload={})])
    assert not res_missing.allowed


def test_time_since_requires_elapsed_seconds():
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=6)
    history = [
        make_obs(t0, payload={}, tags=["apu_ready"]),
        make_obs(t1, payload={}, tags=[]),
    ]
    engine = GatingEngine([{"op": "time_since", "tag": "apu_ready", "at_least": 5}])
    res = engine.evaluate(history)
    assert res.allowed

    engine_tight = GatingEngine([{"op": "time_since", "tag": "apu_ready", "at_least": 10}])
    res2 = engine_tight.evaluate(history)
    assert not res2.allowed
    assert "time_since" in res2.reason


def test_time_since_ignores_latest_tag_only():
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    history = [make_obs(t0, payload={}, tags=["apu_ready"])]
    engine = GatingEngine([{"op": "time_since", "tag": "apu_ready", "at_least": 1}])
    res = engine.evaluate(history)
    assert not res.allowed
    assert "apu_ready" in res.reason


def test_evaluate_with_no_observations_returns_no():
    engine = GatingEngine([{"op": "flag_true", "var": "payload.apu_ready"}])
    res = engine.evaluate([])
    assert not res.allowed
    assert res.reason == "no observations"


def test_empty_rules_allow():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"rpm": 0.25})
    engine = GatingEngine([])
    res = engine.evaluate([obs])
    assert res.allowed


def test_multiple_rules_fail_on_first():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"rpm": 0.15, "temp": 650})
    rules = [
        {"op": "var_gte", "var": "payload.rpm", "value": 0.2},
        {"op": "arg_in_range", "var": "payload.temp", "min": 190, "max": 590},
    ]
    engine = GatingEngine(rules)
    res = engine.evaluate([obs])
    assert not res.allowed
    assert "payload.rpm" in res.reason


def test_unknown_operator_blocks():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"rpm": 0.25})
    engine = GatingEngine([{"op": "mystery_op", "var": "payload.rpm", "value": 0.2}])
    res = engine.evaluate([obs])
    assert not res.allowed
    assert "unknown op" in res.reason


def test_missing_var_path_blocks():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = make_obs(now, payload={"rpm": 0.25})
    engine = GatingEngine([{"op": "var_gte", "var": "payload.nonexistent_field", "value": 0.2}])
    res = engine.evaluate([obs])
    assert not res.allowed
    assert "missing" in res.reason

