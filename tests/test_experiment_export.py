"""
End-to-end tests for experiment export pipeline.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import tempfile

from core.experiment_export import (
    SessionMeta,
    HelpCycleRecord,
    TimelineSnapshot,
    ExperimentExport,
    build_experiment_export,
)
from core.interaction_metrics import InteractionMetrics


# ── helpers ────────────────────────────────────────────────────────────

def _iso(ts: str) -> str:
    return ts


def _make_events_with_help_cycles() -> list[dict]:
    """Synthetic event log covering one session with two help cycles."""
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 5, 9, 10, 0, 5, tzinfo=timezone.utc)
    t2 = datetime(2026, 5, 9, 10, 0, 12, tzinfo=timezone.utc)
    t3 = datetime(2026, 5, 9, 10, 0, 20, tzinfo=timezone.utc)

    session_id = "test-session-001"
    cycle_1_id = "cycle-aaa-111"
    cycle_2_id = "cycle-bbb-222"

    return [
        # --- step activation ---
        {
            "kind": "step_activated",
            "payload": {"step_id": "S01"},
            "t_wall": 0.0,
            "session_id": session_id,
            "timestamp": t0.isoformat(),
        },
        # --- observation 1 ---
        {
            "kind": "observation",
            "payload": {"seq": 1, "t_wall": 0.5, "source": "telemetry"},
            "t_wall": 0.5,
            "session_id": session_id,
            "timestamp": t0.isoformat(),
        },
        # --- help cycle 1 ---
        {
            "kind": "tutor_request",
            "payload": {
                "intent": "ask_help",
                "context": {
                    "deterministic_step_hint": {
                        "inferred_step_id": "S01",
                        "missing_conditions": ["battery_on"],
                        "observability_status": "observable",
                        "requires_visual_confirmation": False,
                    },
                },
            },
            "t_wall": 5.0,
            "session_id": session_id,
            "related_id": cycle_1_id,
            "vision_refs": ["frame_001.png"],
            "metadata": {
                "help_cycle_id": cycle_1_id,
                "vision_status": "available",
                "vision_fact_status": "ok",
                "vision_used": True,
                "frame_id": "frame_001.png",
                "sync_delta_ms": 50,
                "vision_fact_summary": {"facts_seen": 3},
                "fused_step_id": "S01",
                "fused_missing_conditions": ["battery_on"],
                "layout_id": "v2",
                "generation_mode": "model",
            },
            "timestamp": t1.isoformat(),
        },
        {
            "kind": "overlay_dry_run",
            "payload": {"type": "highlight", "target": "battery_switch", "intent": "guide"},
            "t_wall": 5.2,
            "session_id": session_id,
            "metadata": {"help_cycle_id": cycle_1_id},
        },
        {
            "kind": "tutor_response",
            "payload": {
                "status": "ok",
                "message": "请先打开电池开关。",
                "actions": [
                    {"type": "highlight", "target": "battery_switch", "element_id": "pnt_301"}
                ],
            },
            "t_wall": 5.5,
            "session_id": session_id,
            "related_id": cycle_1_id,
            "vision_refs": ["frame_001.png"],
            "metadata": {
                "help_cycle_id": cycle_1_id,
                "generation_mode": "model",
                "vision_used": True,
                "frame_id": "frame_001.png",
                "sync_delta_ms": 50,
                "fused_step_id": "S01",
                "fused_missing_conditions": ["battery_on"],
                "layout_id": "v2",
                "help_response": {
                    "diagnosis": {"step_id": "S01", "error_category": "OM"},
                    "next": {"step_id": "S01"},
                    "overlay": {
                        "targets": ["battery_switch"],
                        "evidence": [
                            {
                                "target": "battery_switch",
                                "type": "gate",
                                "ref": "GATES.battery_on",
                                "quote": "Battery not yet on.",
                                "grounding_confidence": 0.95,
                            }
                        ],
                    },
                },
                "response_mapping": {
                    "executed": [{"target": "battery_switch", "type": "highlight"}],
                    "rejected": [],
                    "dropped": [],
                },
                "fallback_overlay_used": False,
                "fallback_overlay_reason": "not_needed",
                "observability_status": "observable",
                "requires_visual_confirmation": False,
                "scenario_profile": "airfield",
            },
            "timestamp": t2.isoformat(),
        },
        # --- step completed ---
        {
            "kind": "step_completed",
            "payload": {"step_id": "S01"},
            "t_wall": 8.0,
            "session_id": session_id,
            "timestamp": t2.isoformat(),
        },
        # --- step activation S02 ---
        {
            "kind": "step_activated",
            "payload": {"step_id": "S02"},
            "t_wall": 9.0,
            "session_id": session_id,
        },
        # --- help cycle 2 (with vision fallback) ---
        {
            "kind": "tutor_request",
            "payload": {
                "intent": "ask_help",
                "context": {
                    "deterministic_step_hint": {
                        "inferred_step_id": "S02",
                        "missing_conditions": ["left_ddi_on"],
                        "observability_status": "requires_visual",
                        "requires_visual_confirmation": True,
                    },
                },
            },
            "t_wall": 12.0,
            "session_id": session_id,
            "related_id": cycle_2_id,
            "vision_refs": [],
            "metadata": {
                "help_cycle_id": cycle_2_id,
                "vision_status": "sync_miss",
                "vision_fact_status": "vision_unavailable",
                "vision_used": False,
                "sync_delta_ms": None,
                "vision_fallback_reason": "vision_sync_miss",
                "fused_step_id": "S02",
                "fused_missing_conditions": ["left_ddi_on"],
                "layout_id": "v2",
                "generation_mode": "fallback",
            },
            "timestamp": t3.isoformat(),
        },
        {
            "kind": "overlay_dry_run",
            "payload": {"type": "highlight", "target": "left_mdi_brightness_selector", "intent": "guide"},
            "t_wall": 12.3,
            "session_id": session_id,
            "metadata": {"help_cycle_id": cycle_2_id},
        },
        {
            "kind": "overlay_rejected",
            "payload": {
                "help_cycle_id": cycle_2_id,
                "reason": "evidence_missing",
                "target": "left_mdi_brightness_selector",
            },
            "t_wall": 12.4,
            "session_id": session_id,
            "related_id": cycle_2_id,
            "metadata": {
                "help_cycle_id": cycle_2_id,
                "generation_mode": "fallback",
            },
        },
        {
            "kind": "tutor_response",
            "payload": {
                "status": "error",
                "message": "无法确认左DDI状态，请检查亮度旋钮。",
                "actions": [],
            },
            "t_wall": 12.5,
            "session_id": session_id,
            "related_id": cycle_2_id,
            "metadata": {
                "help_cycle_id": cycle_2_id,
                "generation_mode": "fallback",
                "vision_used": False,
                "vision_fallback_reason": "vision_sync_miss",
                "fused_step_id": "S02",
                "fused_missing_conditions": ["left_ddi_on"],
                "layout_id": "v2",
                "help_response": {
                    "diagnosis": {"step_id": "S02", "error_category": "OM"},
                    "next": {"step_id": "S02"},
                },
                "response_mapping": {
                    "executed": [],
                    "rejected": [{"target": "left_mdi_brightness_selector", "reason": "evidence_missing"}],
                    "dropped": [],
                },
                "response_mapping_failure_codes": ["vision_sync_miss"],
                "fallback_overlay_used": True,
                "fallback_overlay_reason": "vision_sync_miss",
                "observability_status": "requires_visual",
                "requires_visual_confirmation": True,
                "scenario_profile": "airfield",
            },
            "timestamp": t3.isoformat(),
        },
    ]


# ── tests ───────────────────────────────────────────────────────────────

def test_session_meta_roundtrip():
    meta = SessionMeta(
        participant_id="P01",
        session_id="sess-1",
        condition="with_tutor",
        group="novice",
        questionnaire_ref="questionnaires/P01_pre.yml",
        experimenter_notes="no issues",
        started_at="2026-05-09T10:00:00+00:00",
        ended_at="2026-05-09T10:30:00+00:00",
    )
    d = meta.to_dict()
    assert d["participant_id"] == "P01"
    assert d["condition"] == "with_tutor"
    assert d["questionnaire_ref"] == "questionnaires/P01_pre.yml"

    reloaded = SessionMeta.from_dict(d)
    assert reloaded.participant_id == "P01"
    assert reloaded.condition == "with_tutor"
    assert reloaded.questionnaire_ref == "questionnaires/P01_pre.yml"


def test_session_meta_from_partial():
    meta = SessionMeta.from_dict({"participant_id": "P02"})
    assert meta.participant_id == "P02"
    assert meta.condition == ""
    assert meta.questionnaire_ref is None


def test_build_export_with_help_cycles():
    events = _make_events_with_help_cycles()
    export = build_experiment_export(
        events,
        meta_overrides={
            "participant_id": "P01",
            "condition": "with_tutor",
            "group": "novice",
        },
    )

    # meta
    assert export.meta.participant_id == "P01"
    assert export.meta.condition == "with_tutor"
    assert export.meta.group == "novice"
    assert export.meta.session_id == "test-session-001"

    # help cycles
    assert len(export.help_cycles) == 2

    c1 = export.help_cycles[0]
    assert c1.cycle_index == 0
    assert c1.help_cycle_id == "cycle-aaa-111"
    assert c1.generation_mode == "model"
    assert c1.vision_used is True
    assert c1.sync_delta_ms == 50
    assert c1.frame_ids == ["frame_001.png"]
    assert c1.fused_step_id == "S01"
    assert c1.fused_missing_conditions == ["battery_on"]
    assert c1.overlay_targets == ["battery_switch"]
    assert c1.overlay_executed == 1
    assert c1.overlay_rejected == 0
    assert c1.overlay_dry_run_count == 1
    assert c1.response_status == "ok"
    assert c1.fallback_overlay_used is False
    assert c1.vision_fallback_reason is None

    c2 = export.help_cycles[1]
    assert c2.cycle_index == 1
    assert c2.help_cycle_id == "cycle-bbb-222"
    assert c2.generation_mode == "fallback"
    assert c2.vision_used is False
    assert c2.vision_fallback_reason == "vision_sync_miss"
    assert c2.fused_step_id == "S02"
    assert c2.fused_missing_conditions == ["left_ddi_on"]
    assert c2.overlay_targets == ["left_mdi_brightness_selector"]
    assert c2.overlay_executed == 0
    assert c2.overlay_rejected == 1
    assert c2.response_status == "error"
    assert c2.fallback_overlay_used is True
    assert c2.requires_visual_confirmation is True
    assert c2.observability_status == "requires_visual"

    # summary (interaction metrics)
    assert export.summary.help_requests == 2
    assert export.summary.highlight_requests >= 0

    # timeline
    assert len(export.timeline) >= 3  # at least: step_activated, help_cycle_1, step_completed
    # first snapshot should have S01 active
    snap0 = export.timeline[0]
    assert snap0.active_step_id == "S01"
    # find snapshot after first help cycle
    snap_ids = [sn.active_step_id for sn in export.timeline]
    assert "S02" in snap_ids or None in snap_ids

    # serialization
    d = export.to_dict()
    assert "meta" in d
    assert "summary" in d
    assert "help_cycles" in d
    assert "scoring" in d
    assert "timeline" in d


def test_build_export_no_help_cycles():
    events = [
        {
            "kind": "step_activated",
            "payload": {"step_id": "S01"},
            "t_wall": 0.0,
            "session_id": "sess-empty",
        },
        {
            "kind": "step_completed",
            "payload": {"step_id": "S01"},
            "t_wall": 5.0,
            "session_id": "sess-empty",
        },
    ]
    export = build_experiment_export(events)
    assert len(export.help_cycles) == 0
    assert export.meta.session_id == "sess-empty"
    assert export.summary.help_requests == 0


def test_build_export_empty_events():
    export = build_experiment_export([])
    assert len(export.help_cycles) == 0
    assert len(export.timeline) == 0
    assert export.meta.session_id == ""


def test_help_cycle_record_serialization():
    rec = HelpCycleRecord(
        cycle_index=0,
        help_cycle_id="test-id",
        trigger_wall_s=1.5,
        generation_mode="model",
        vision_used=True,
        sync_delta_ms=100,
        frame_ids=["f1.png", "f2.png"],
        fused_step_id="S03",
        fused_missing_conditions=["r_gen_on"],
        overlay_targets=["generator_right_switch"],
        overlay_executed=1,
        overlay_rejected=0,
    )
    d = rec.to_dict()
    assert d["help_cycle_id"] == "test-id"
    assert d["generation_mode"] == "model"
    assert d["overlay_targets"] == ["generator_right_switch"]


def test_experiment_export_to_json():
    events = _make_events_with_help_cycles()
    export = build_experiment_export(
        events,
        meta_overrides={
            "participant_id": "P03",
            "condition": "without_tutor",
            "group": "expert",
            "questionnaire_ref": "q/P03.yml",
        },
    )
    d = export.to_dict()
    json_str = json.dumps(d, ensure_ascii=False, sort_keys=True)
    reloaded = json.loads(json_str)

    assert reloaded["meta"]["participant_id"] == "P03"
    assert reloaded["meta"]["condition"] == "without_tutor"
    assert len(reloaded["help_cycles"]) == 2
    assert reloaded["help_cycles"][0]["help_cycle_id"] == "cycle-aaa-111"
    assert reloaded["help_cycles"][0]["generation_mode"] == "model"
    assert reloaded["help_cycles"][1]["generation_mode"] == "fallback"


def test_timeline_snapshot_fields():
    events = _make_events_with_help_cycles()
    export = build_experiment_export(events)
    for snap in export.timeline:
        assert isinstance(snap.wall_s, float)
        assert isinstance(snap.help_request_count, int)
        assert isinstance(snap.completed_step_ids, list)
        assert isinstance(snap.blocked_step_ids, list)
        # active_step_id may be None or str
        assert snap.active_step_id is None or isinstance(snap.active_step_id, str)
