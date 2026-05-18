from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core.live_replay_fixture import LiveReplayFixtureNotFound, extract_live_replay_fixture_from_jsonl
from simtutor.__main__ import main


def _event(
    kind: str,
    payload: dict[str, Any],
    *,
    related_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    vision_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": "v1",
        "event_id": f"event-{kind}",
        "timestamp": "2026-05-17T12:00:00+00:00",
        "kind": kind,
        "payload": payload,
        "related_id": related_id,
        "vision_refs": vision_refs or [],
        "metadata": metadata or {},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any] | str]) -> None:
    path.write_text(
        "".join(
            (row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sample_cycle_events(request_id: str = "req-288") -> list[dict[str, Any]]:
    help_cycle_id = request_id
    observation_id = "obs-1"
    request_payload = {
        "request_id": request_id,
        "observation_ref": observation_id,
        "message": "help",
        "context": {
            "candidate_steps": [
                {"step_id": "S03", "rank": 0, "confidence": 0.8},
                {"step_id": "S04", "rank": 1, "confidence": 0.5},
            ],
            "evidence_packet_summary": {
                "step_id": "S03",
                "telemetry_window_digest": {"frame_count": 2, "status": "ok"},
            },
            "evidence_snapshot": {
                "schema_version": "evidence_snapshot.v1",
                "snapshot_id": "snapshot-288",
                "source_observation_id": observation_id,
                "source_observation_seq": 1,
                "telemetry_window_first_seq": 1,
                "telemetry_window_latest_seq": 2,
                "candidate_generation_snapshot_id": "snapshot-288",
                "model_request_snapshot_id": "snapshot-288",
                "validator_snapshot_id": "snapshot-288",
                "final_decision_snapshot_id": "snapshot-288",
            },
            "snapshot_ids": {
                "candidate_generation": "snapshot-288",
                "model_request": "snapshot-288",
            },
            "vision_fact_summary": {"seen_fact_ids": ["apu_ready_light"], "frame_ids": ["frame-1"]},
        },
        "metadata": {
            "help_cycle_id": help_cycle_id,
            "vision_fact_status": "available",
            "vision_frame_ids": ["frame-1"],
            "evidence_snapshot": {
                "schema_version": "evidence_snapshot.v1",
                "snapshot_id": "snapshot-288",
                "source_observation_id": observation_id,
                "source_observation_seq": 1,
                "telemetry_window_first_seq": 1,
                "telemetry_window_latest_seq": 2,
                "candidate_generation_snapshot_id": "snapshot-288",
                "model_request_snapshot_id": "snapshot-288",
                "validator_snapshot_id": "snapshot-288",
                "final_decision_snapshot_id": "snapshot-288",
            },
        },
    }
    response_payload = {
        "in_reply_to": request_id,
        "status": "ok",
        "message": "Turn on APU.",
        "actions": [{"type": "overlay", "target": "apu_switch"}],
        "metadata": {
            "help_cycle_id": help_cycle_id,
            "generation_mode": "model",
            "final_overlay_targets": ["apu_switch"],
            "final_action_plan": {"step_id": "S03", "targets": ["apu_switch"], "source": "model"},
            "final_public_response": {
                "message": "Turn on APU.",
                "next": {"step_id": "S03"},
                "actions": [{"type": "overlay", "target": "apu_switch"}],
            },
            "harness_trace": {
                "schema_version": "v1",
                "evidence_snapshot": {
                    "schema_version": "evidence_snapshot.v1",
                    "snapshot_id": "snapshot-288",
                    "source_observation_id": observation_id,
                    "source_observation_seq": 1,
                    "telemetry_window_first_seq": 1,
                    "telemetry_window_latest_seq": 2,
                    "candidate_generation_snapshot_id": "snapshot-288",
                    "model_request_snapshot_id": "snapshot-288",
                    "validator_snapshot_id": "snapshot-288",
                    "final_decision_snapshot_id": "snapshot-288",
                },
                "snapshot_ids": {
                    "candidate_generation": "snapshot-288",
                    "model_request": "snapshot-288",
                    "validator": "snapshot-288",
                    "final_decision": "snapshot-288",
                },
                "evidence_packet_summary": {
                    "step_id": "S03",
                    "telemetry_window_digest": {"frame_count": 2, "status": "ok"},
                },
                "candidates": [{"step_id": "S03", "rank": 0}],
                "model_decision": {"step_id": "S03", "overlay_targets": ["apu_switch"]},
                "validator_result": {"rejected": False, "reasons": []},
                "repair_result": {"applied": False, "path": None},
                "final_action_plan": {"step_id": "S03", "targets": ["apu_switch"], "source": "model"},
                "final_overlay_targets": ["apu_switch"],
                "message_category": "model",
                "vlm_call": {"status": "called", "frame_ids": ["frame-1"]},
            },
        },
    }
    return [
        _event(
            "observation",
            {"observation_id": observation_id, "payload": {"seq": 1, "vars": {"apu_on": False}}},
            related_id=observation_id,
        ),
        _event("tutor_request", request_payload, related_id=request_id, metadata={"help_cycle_id": help_cycle_id}),
        _event(
            "overlay_dry_run",
            {"help_cycle_id": help_cycle_id, "target": "apu_switch"},
            related_id=request_id,
            metadata={"help_cycle_id": help_cycle_id},
        ),
        _event("tutor_response", response_payload, related_id=request_id, metadata={"help_cycle_id": help_cycle_id}),
    ]


def test_extract_live_replay_fixture_from_request_id_includes_harness_path(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    _write_jsonl(log_path, [_sample_cycle_events()[0], "{bad json", *_sample_cycle_events()[1:]])

    fixture = extract_live_replay_fixture_from_jsonl(log_path, request_id="req-288")

    assert fixture["schema_version"] == "live_help_replay_fixture.v1"
    assert fixture["request_id"] == "req-288"
    assert fixture["help_cycle_id"] == "req-288"
    assert fixture["source_log"]["malformed_lines"][0]["lineno"] == 2
    assert fixture["cycle"]["tutor_request"]["request_id"] == "req-288"
    assert fixture["cycle"]["tutor_response"]["in_reply_to"] == "req-288"
    assert fixture["context"]["evidence_packet_summary"]["step_id"] == "S03"
    assert fixture["context"]["telemetry_window_digest"] == {"frame_count": 2, "status": "ok"}
    assert fixture["context"]["evidence_snapshot"]["source_observation_seq"] == 1
    assert fixture["context"]["snapshot_ids"] == {
        "candidate_generation": "snapshot-288",
        "model_request": "snapshot-288",
        "validator": "snapshot-288",
        "final_decision": "snapshot-288",
    }
    assert fixture["harness"]["candidate_steps"][0]["step_id"] == "S03"
    assert fixture["harness"]["trace"]["validator_result"]["rejected"] is False
    assert fixture["expectations"] == {
        "expected_final_step_id": "S03",
        "expected_overlay_target_ids": ["apu_switch"],
        "vlm_call_status": "called",
        "llm_decision_status": "accepted",
        "final_response_source": "model",
    }


def test_extract_live_replay_fixture_missing_request_id_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    _write_jsonl(log_path, _sample_cycle_events(request_id="other-req"))

    with pytest.raises(LiveReplayFixtureNotFound, match="missing-req"):
        extract_live_replay_fixture_from_jsonl(log_path, request_id="missing-req")


def test_extract_live_replay_fixture_handles_cached_fallback_response(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    events = _sample_cycle_events(request_id="cached-req")
    response = events[-1]["payload"]
    response["status"] = "error"
    response["actions"] = []
    response["metadata"].update(
        {
            "provider": "fallback",
            "generation_mode": "fallback",
            "cached_response_reused": True,
            "final_overlay_targets": [],
            "final_action_plan": {"step_id": "S03", "targets": [], "source": "fallback"},
        }
    )
    response["metadata"]["harness_trace"].update(
        {
            "final_overlay_targets": [],
            "message_category": "fallback",
            "repair_result": {"applied": True, "path": "fallback_overlay"},
            "vlm_call": {"status": "not_required", "frame_ids": []},
        }
    )
    _write_jsonl(log_path, events)

    fixture = extract_live_replay_fixture_from_jsonl(log_path, request_id="cached-req")

    assert fixture["expectations"]["expected_overlay_target_ids"] == []
    assert fixture["expectations"]["vlm_call_status"] == "not_required"
    assert fixture["expectations"]["llm_decision_status"] == "repaired"
    assert fixture["expectations"]["final_response_source"] == "cache"
    assert fixture["cycle"]["tutor_response"]["metadata"]["cached_response_reused"] is True


def test_extract_live_replay_fixture_uses_response_metadata_when_trace_missing(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    events = _sample_cycle_events(request_id="legacy-req")
    response_metadata = events[-1]["payload"]["metadata"]
    response_metadata.pop("harness_trace")
    response_metadata.update(
        {
            "generation_mode": "repair",
            "final_overlay_targets": ["apu_switch"],
            "final_action_plan": {
                "step_id": "S03",
                "targets": ["apu_switch"],
                "source": "validator_action_hint",
            },
        }
    )
    _write_jsonl(log_path, events)

    fixture = extract_live_replay_fixture_from_jsonl(log_path, request_id="legacy-req")

    assert fixture["expectations"]["expected_final_step_id"] == "S03"
    assert fixture["expectations"]["expected_overlay_target_ids"] == ["apu_switch"]
    assert fixture["expectations"]["llm_decision_status"] == "repaired"
    assert fixture["expectations"]["final_response_source"] == "validator_action_hint"


def test_extract_live_replay_fixture_classifies_deterministic_fallback(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    events = _sample_cycle_events(request_id="fallback-overlay-req")
    response_metadata = events[-1]["payload"]["metadata"]
    response_metadata.update(
        {
            "generation_mode": "model",
            "fallback_overlay_used": True,
            "fallback_overlay_reason": "deterministic_step:S03",
        }
    )
    response_metadata["harness_trace"]["repair_result"] = {"applied": True, "path": "deterministic_step:S03"}
    _write_jsonl(log_path, events)

    fixture = extract_live_replay_fixture_from_jsonl(log_path, request_id="fallback-overlay-req")

    assert fixture["expectations"]["llm_decision_status"] == "repaired"
    assert fixture["expectations"]["final_response_source"] == "deterministic_step:S03"


def test_extract_live_replay_fixture_includes_matching_vision_fact_observation(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    events = _sample_cycle_events(request_id="vision-req")
    vision_fact_event = _event(
        "observation",
        {
            "observation_id": "vision-fact-1",
            "source": "vision_fact",
            "metadata": {"observation_kind": "vision_fact", "frame_ids": ["frame-1"]},
            "payload": {
                "frame_ids": ["frame-1"],
                "metadata": {"raw_llm_text": "{\"facts\": []}"},
                "facts": [{"fact_id": "apu_ready_light", "state": "seen"}],
            },
        },
        metadata={"observation_kind": "vision_fact"},
        vision_refs=["frame-1"],
    )
    _write_jsonl(log_path, [events[0], vision_fact_event, *events[1:]])

    fixture = extract_live_replay_fixture_from_jsonl(log_path, request_id="vision-req")

    vision_fact_observations = fixture["context"]["vision_fact_observations"]
    assert len(vision_fact_observations) == 1
    assert vision_fact_observations[0]["observation_id"] == "vision-fact-1"
    assert vision_fact_observations[0]["payload"]["metadata"]["raw_llm_text"] == "{\"facts\": []}"


def test_extract_live_replay_fixture_without_observation_ref_uses_nearby_observations(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    events = _sample_cycle_events(request_id="no-obs-ref")
    events[1]["payload"].pop("observation_ref")
    far_observations = [
        _event("observation", {"observation_id": f"far-{idx}", "payload": {"seq": idx}})
        for idx in range(8)
    ]
    _write_jsonl(log_path, [*far_observations, *events])

    fixture = extract_live_replay_fixture_from_jsonl(log_path, request_id="no-obs-ref")

    observation_ids = [item["observation_id"] for item in fixture["context"]["observations"]]
    assert "obs-1" in observation_ids
    assert "far-0" not in observation_ids
    assert len(observation_ids) < len(far_observations) + 1


def test_cli_extract_live_fixture_writes_into_output_dir(monkeypatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "live.jsonl"
    _write_jsonl(log_path, _sample_cycle_events())
    output_dir = tmp_path / "fixtures"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "extract-live-fixture",
            "--input",
            str(log_path),
            "--request-id",
            "req-288",
            "--output-dir",
            str(output_dir),
        ],
    )

    code = main()

    assert code == 0
    fixture_path = output_dir / "req-288.fixture.json"
    assert fixture_path.exists()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["request_id"] == "req-288"
    assert f"[EXTRACT_LIVE_FIXTURE] wrote {fixture_path}" in capsys.readouterr().out
