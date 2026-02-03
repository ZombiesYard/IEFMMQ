from __future__ import annotations

import json

import pytest

from adapters.dcs.telemetry.codec import decode_dcs_observation, encode_dcs_observation
from adapters.dcs.telemetry.receiver import DcsTelemetryReceiver
from core.types_v2 import DcsObservation


def _sample_payload(seq: int = 1) -> dict:
    return {
        "schema_version": "v2",
        "seq": seq,
        "sim_time": 12.5,
        "aircraft": "FA-18C",
        "cockpit": {"speed": 250.0, "gear": "up"},
    }


def test_decode_and_validate_schema() -> None:
    payload = _sample_payload()
    data = json.dumps(payload).encode("utf-8")
    obs = decode_dcs_observation(data)
    assert obs.schema_version == "v2"
    assert obs.seq == 1
    assert obs.cockpit["speed"] == 250.0


def test_encode_round_trip() -> None:
    obs = DcsObservation(**_sample_payload())
    data = encode_dcs_observation(obs)
    decoded = decode_dcs_observation(data)
    assert decoded.seq == obs.seq
    assert decoded.aircraft == obs.aircraft


def test_receiver_drops_out_of_order() -> None:
    receiver = DcsTelemetryReceiver(host="127.0.0.1", port=0)
    try:
        obs1 = receiver._process_frame(DcsObservation(**_sample_payload(seq=2)), ("127.0.0.1", 1234))
        obs2 = receiver._process_frame(DcsObservation(**_sample_payload(seq=1)), ("127.0.0.1", 1234))
        assert obs1 is not None
        assert obs2 is None
    finally:
        receiver.close()

