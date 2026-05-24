"""
End-to-end tests for experiment export pipeline.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import simtutor.__main__ as simtutor_cli
from core.experiment_export import (
    ACTION_TIMELINE_CSV_FIELDS,
    HELP_CYCLES_CSV_FIELDS,
    SessionMeta,
    ActionTimelineRecord,
    HelpCycleRecord,
    STEP_CODING_CSV_FIELDS,
    TRIAL_SUMMARY_CSV_FIELDS,
    build_export_quality_report,
    build_experiment_export,
    build_file_sha256,
)
from simtutor.__main__ import _run_experiment_export


# ── helpers ────────────────────────────────────────────────────────────

def _make_events_with_help_cycles() -> list[dict]:
    """Synthetic event log covering one session with two help cycles.

    Event structure mirrors production (live_dcs._sanitize_*_payload_for_event):
    - event["metadata"] = normalized audit subset
    - event["payload"]["metadata"] = full request/response metadata (rich fields)
    """
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
            "source": "dcs_bios",
            "payload": {
                "seq": 1,
                "t_wall": 0.5,
                "source": "dcs_bios",
                "bios": {"BATTERY_SW": 2},
                "delta": {"BATTERY_SW": 2},
            },
            "metadata": {"seq": 1, "delta_count": 1},
            "t_wall": 0.5,
            "session_id": session_id,
            "timestamp": t0.isoformat(),
        },
        # --- help cycle 1 (model, vision available) ---
        {
            "kind": "tutor_request",
            "payload": {
                "intent": "ask_help",
                "context": {
                    "deterministic_step_hint": {
                        "inferred_step_id": "S01",
                        "missing_conditions": ["battery_on"],
                    },
                },
                "metadata": {
                    "help_cycle_id": cycle_1_id,
                    "vision_status": "available",
                    "vision_fact_status": "ok",
                    "vision_used": True,
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
                "metadata": {
                    "help_response": {
                        "diagnosis": {"step_id": "S01", "error_category": "OM"},
                        "next": {"step_id": "S01"},
                        "overlay": {
                            "targets": ["battery_switch"],
                            "evidence": [{"target": "battery_switch", "type": "gate", "ref": "GATES.battery_on"}],
                        },
                    },
                    "response_mapping": {
                        "rejected_targets": [],
                        "dropped_targets": [],
                        "overlay_rejected": False,
                    },
                    "generation_mode": "model",
                    "fallback_overlay_used": False,
                    "fallback_overlay_reason": "not_needed",
                    "observability_status": "observable",
                    "requires_visual_confirmation": False,
                    "scenario_profile": "airfield",
                },
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
        # --- help cycle 2 (fallback, vision sync miss) ---
        {
            "kind": "tutor_request",
            "payload": {
                "intent": "ask_help",
                "context": {
                    "deterministic_step_hint": {
                        "inferred_step_id": "S02",
                        "missing_conditions": ["left_ddi_on"],
                    },
                },
                "metadata": {
                    "help_cycle_id": cycle_2_id,
                    "vision_status": "sync_miss",
                    "vision_fact_status": "vision_unavailable",
                    "vision_used": False,
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
                "vision_fallback_reason": "vision_sync_miss",
                "fused_step_id": "S02",
                "fused_missing_conditions": ["left_ddi_on"],
                "layout_id": "v2",
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
                "failure_codes": ["evidence_missing"],
                "rejected_targets": ["left_mdi_brightness_selector"],
                "reasons": ["evidence_missing"],
            },
            "t_wall": 12.4,
            "session_id": session_id,
            "related_id": cycle_2_id,
            "metadata": {"help_cycle_id": cycle_2_id},
        },
        {
            "kind": "tutor_response",
            "payload": {
                "status": "error",
                "message": "无法确认左DDI状态，请检查亮度旋钮。",
                "actions": [],
                "metadata": {
                    "help_response": {
                        "diagnosis": {"step_id": "S02", "error_category": "OM"},
                        "next": {"step_id": "S02"},
                    },
                    "response_mapping": {
                        "rejected_targets": ["left_mdi_brightness_selector"],
                        "dropped_targets": [],
                        "overlay_rejected": True,
                    },
                    "response_mapping_failure_codes": ["vision_sync_miss"],
                    "generation_mode": "fallback",
                    "fallback_overlay_used": True,
                    "fallback_overlay_reason": "vision_sync_miss",
                    "observability_status": "requires_visual",
                    "requires_visual_confirmation": True,
                    "scenario_profile": "airfield",
                },
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
            },
            "timestamp": t3.isoformat(),
        },
    ]


def _make_events_with_action_timeline() -> list[dict]:
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 5, 9, 10, 0, 2, tzinfo=timezone.utc)
    t2 = datetime(2026, 5, 9, 10, 0, 4, tzinfo=timezone.utc)
    t3 = datetime(2026, 5, 9, 10, 0, 9, tzinfo=timezone.utc)
    t4 = datetime(2026, 5, 9, 10, 0, 12, tzinfo=timezone.utc)
    t5 = datetime(2026, 5, 9, 10, 0, 14, tzinfo=timezone.utc)
    t6 = datetime(2026, 5, 9, 10, 0, 20, tzinfo=timezone.utc)
    t7 = datetime(2026, 5, 9, 10, 0, 24, tzinfo=timezone.utc)

    return [
        {
            "kind": "step_activated",
            "payload": {"step_id": "S01"},
            "t_wall": 0.0,
            "timestamp": t0.isoformat(),
        },
        {
            "kind": "observation",
            "payload": {
                "observation_id": "obs-action-001",
                "timestamp": t1.isoformat(),
                "source": "dcs_bios",
                "payload": {
                    "seq": 1,
                    "t_wall": 2.0,
                    "delta_summary": {
                        "recent_key_changes_topk": [
                            {"key": "BATTERY_SW", "value": 2, "ui_targets": ["battery_switch"]},
                        ],
                    },
                    "recent_ui_targets": ["battery_switch"],
                },
                "metadata": {"seq": 1, "delta_count": 1},
            },
            "metadata": {"seq": 1, "delta_count": 1},
            "t_wall": 2.0,
            "timestamp": t1.isoformat(),
        },
        {
            "kind": "step_completed",
            "payload": {"step_id": "S01"},
            "t_wall": 4.0,
            "timestamp": t2.isoformat(),
        },
        {
            "kind": "step_activated",
            "payload": {"step_id": "S02"},
            "t_wall": 9.0,
            "timestamp": t3.isoformat(),
        },
        {
            "kind": "tutor_request",
            "payload": {
                "intent": "ask_help",
                "metadata": {"help_cycle_id": "cycle-s02", "fused_step_id": "S02"},
            },
            "metadata": {"help_cycle_id": "cycle-s02", "fused_step_id": "S02"},
            "related_id": "cycle-s02",
            "t_wall": 12.0,
            "timestamp": t4.isoformat(),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 2,
                "t_wall": 14.0,
                "source": "dcs_bios",
                "bios": {"BATTERY_SW": 2, "APU_CONTROL_SW": 1},
                "delta": {"APU_CONTROL_SW": 1},
            },
            "metadata": {"seq": 2, "delta_count": 1},
            "t_wall": 14.0,
            "timestamp": t5.isoformat(),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 3,
                "t_wall": 20.0,
                "source": "dcs_bios",
                "bios": {"BATTERY_SW": 2, "APU_CONTROL_SW": 1, "EXPERIMENTAL_RAW_KEY": 7},
                "delta": {"EXPERIMENTAL_RAW_KEY": 7},
            },
            "metadata": {"seq": 3, "delta_count": 1},
            "t_wall": 20.0,
            "timestamp": t6.isoformat(),
        },
        {
            "kind": "step_completed",
            "payload": {"step_id": "S03"},
            "t_wall": 24.0,
            "timestamp": t7.isoformat(),
        },
    ]


def _make_live_help_only_completion_events() -> list[dict]:
    t0 = datetime(2026, 5, 20, 14, 56, 0, tzinfo=timezone.utc)

    def ts(offset: int) -> str:
        return (t0 + timedelta(seconds=offset)).isoformat()

    def request(cycle_id: str, step_id: str, t_wall: float, missing: list[str]) -> dict:
        return {
            "kind": "tutor_request",
            "payload": {
                "intent": "help",
                "metadata": {
                    "help_cycle_id": cycle_id,
                    "fused_step_id": step_id,
                    "fused_missing_conditions": missing,
                },
            },
            "metadata": {
                "help_cycle_id": cycle_id,
                "fused_step_id": step_id,
                "fused_missing_conditions": missing,
            },
            "related_id": cycle_id,
            "t_wall": t_wall,
            "timestamp": ts(int(t_wall - 100.0)),
        }

    def response(
        cycle_id: str,
        fused_step_id: str,
        t_wall: float,
        *,
        message: str,
        category: str = "actionable",
        validation_reasons: list[str] | None = None,
    ) -> dict:
        return {
            "kind": "tutor_response",
            "payload": {
                "status": "ok",
                "message": message,
                "actions": [],
                "metadata": {
                    "help_cycle_id": cycle_id,
                    "generation_mode": "model",
                    "fused_step_id": fused_step_id,
                    "fused_missing_conditions": [],
                    "help_response": {
                        "diagnosis": {"step_id": fused_step_id, "error_category": "OM"},
                        "next": {"step_id": fused_step_id},
                        "overlay": {"targets": [], "evidence": []},
                    },
                    "final_public_instruction_category": category,
                    "harness_validation_reasons": validation_reasons or [],
                },
            },
            "metadata": {
                "help_cycle_id": cycle_id,
                "generation_mode": "model",
                "fused_step_id": fused_step_id,
                "fused_missing_conditions": [],
            },
            "related_id": cycle_id,
            "t_wall": t_wall,
            "timestamp": ts(int(t_wall - 100.0)),
        }

    return [
        {
            "kind": "observation",
            "source": "vision_frame_manifest",
            "payload": {
                "source": "vision_frame_manifest",
                "payload": {"source": "vision_frame_manifest"},
            },
            "t_wall": 10.0,
            "timestamp": ts(0),
        },
        {
            "kind": "observation",
            "source": "dcs_bios_raw",
            "payload": {
                "seq": 1,
                "t_wall": 100.0,
                "source": "dcs_bios_raw",
                "bios": {
                    "BATTERY_SW": 2,
                    "HOOK_LEVER": 1,
                    "HYD_IND_BRAKE": 42000,
                    "EXT_HOOK": 1000,
                },
                "delta": {
                    "BATTERY_SW": 2,
                    "HOOK_LEVER": 1,
                    "HYD_IND_BRAKE": 42000,
                    "EXT_HOOK": 1000,
                },
                "vars": {"battery_on": True},
            },
            "t_wall": 100.0,
            "timestamp": ts(0),
        },
        request("cycle-s01", "S01", 101.0, ["vars.battery_on==true"]),
        response("cycle-s01", "S01", 104.0, message="BATT 到 ON。"),
        request("cycle-s02", "S02", 120.0, ["vars.fire_test_a_complete==true"]),
        response("cycle-s02", "S02", 124.0, message="当前处于 S02。"),
        {
            "kind": "observation",
            "source": "dcs_bios_raw",
            "payload": {
                "seq": 2,
                "t_wall": 140.0,
                "source": "dcs_bios_raw",
                "bios": {"HYD_IND_BRAKE": 42001, "EXT_HOOK": 2000},
                "delta": {"HYD_IND_BRAKE": 42001, "EXT_HOOK": 2000},
            },
            "t_wall": 140.0,
            "timestamp": ts(40),
        },
        request("cycle-s33", "S32", 180.0, ["vars.standby_attitude_uncaged==true"]),
        response(
            "cycle-s33",
            "S33",
            184.0,
            message="姿态源选择器已经在 AUTO，S33 检查已满足；冷启动流程已完成，无需继续操作。",
            category="completed",
            validation_reasons=[
                "completion_gate_already_satisfied:S32",
                "completion_gate_already_satisfied:S33",
            ],
        ),
    ]


def _make_events_for_prescoring_candidates() -> list[dict]:
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)

    def ts(offset: int) -> str:
        return t0.replace(second=offset).isoformat()

    return [
        {
            "kind": "step_activated",
            "payload": {"step_id": "S01"},
            "t_wall": 0.0,
            "timestamp": ts(0),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 1,
                "t_wall": 1.0,
                "source": "dcs_bios",
                "bios": {"APU_CONTROL_SW": 1},
                "delta": {"APU_CONTROL_SW": 1},
                "vars": {"power_available": False},
            },
            "t_wall": 1.0,
            "timestamp": ts(1),
        },
        {
            "kind": "step_completed",
            "payload": {"step_id": "S01"},
            "t_wall": 2.0,
            "timestamp": ts(2),
        },
        {
            "kind": "step_activated",
            "payload": {"step_id": "S02"},
            "t_wall": 3.0,
            "timestamp": ts(3),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 2,
                "t_wall": 4.0,
                "source": "dcs_bios",
                "bios": {"EXPERIMENTAL_RAW_KEY": 7},
                "delta": {"EXPERIMENTAL_RAW_KEY": 7},
            },
            "t_wall": 4.0,
            "timestamp": ts(4),
        },
        {
            "kind": "step_completed",
            "payload": {"step_id": "S03"},
            "t_wall": 5.0,
            "timestamp": ts(5),
        },
        {
            "kind": "observation",
            "payload": {
                "procedure_hint": "S05",
                "tags": ["state_violation"],
            },
            "t_wall": 6.0,
            "timestamp": ts(6),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 3,
                "t_wall": 7.0,
                "source": "dcs_bios",
                "bios": {"RADALT_MIN_HEIGHT_PTR": 50000},
                "delta": {"RADALT_MIN_HEIGHT_PTR": 50000},
                "vars": {"radar_altimeter_bug_value": 500},
            },
            "t_wall": 7.0,
            "timestamp": ts(7),
        },
    ]


# ── tests ───────────────────────────────────────────────────────────────

def test_session_meta_roundtrip():
    meta = SessionMeta(
        trial_id="T01",
        study_id="study-alpha",
        participant_id="P01",
        session_id="sess-1",
        condition="with_tutor",
        group="novice",
        experimenter_id="E01",
        questionnaire_ref="questionnaires/P01_pre.yml",
        recording_ref="recordings/P01_T01.mp4",
        git_commit="abc123",
        git_dirty=False,
        pack_path="packs/fa18c_startup/pack.yaml",
        pack_hash="pack-sha",
        taxonomy_path="packs/fa18c_startup/taxonomy.yaml",
        taxonomy_hash="taxonomy-sha",
        ui_map_hash="ui-map-sha",
        bios_to_ui_hash="bios-ui-sha",
        model_provider="stub",
        model_name="ModelStub",
        vision_model_name="VisionStub",
        prompt_version="prompt-v1",
        prompt_hash="prompt-sha",
        scenario_profile="airfield",
        dcs_mission="cold-start.miz",
        dcs_aircraft="FA-18C",
        vr_setup="Quest 3",
        monitor_setup="native-viewports",
        raw_log_ref="raw_events.jsonl",
        raw_log_sha256="raw-sha",
        experimenter_notes="no issues",
        started_at="2026-05-09T10:00:00+00:00",
        ended_at="2026-05-09T10:30:00+00:00",
    )
    d = meta.to_dict()
    assert d["trial_id"] == "T01"
    assert d["study_id"] == "study-alpha"
    assert d["participant_id"] == "P01"
    assert d["condition"] == "with_tutor"
    assert d["questionnaire_ref"] == "questionnaires/P01_pre.yml"
    assert d["recording_ref"] == "recordings/P01_T01.mp4"
    assert d["git_dirty"] is False
    assert d["pack_hash"] == "pack-sha"
    assert d["raw_log_sha256"] == "raw-sha"

    reloaded = SessionMeta.from_dict(d)
    assert reloaded.trial_id == "T01"
    assert reloaded.participant_id == "P01"
    assert reloaded.condition == "with_tutor"
    assert reloaded.questionnaire_ref == "questionnaires/P01_pre.yml"
    assert reloaded.recording_ref == "recordings/P01_T01.mp4"
    assert reloaded.git_dirty is False
    assert reloaded.raw_log_ref == "raw_events.jsonl"


def test_session_meta_from_partial():
    meta = SessionMeta.from_dict({"participant_id": "P02"})
    assert meta.participant_id == "P02"
    assert meta.condition == ""
    assert meta.questionnaire_ref is None


def test_build_export_with_help_cycles():
    events = _make_events_with_help_cycles()
    root = _repo_root()
    export = build_experiment_export(
        events,
        meta_overrides={
            "trial_id": "T01",
            "participant_id": "P01",
            "condition": "with_tutor",
            "group": "novice",
        },
        pack_path=root / "packs/fa18c_startup/pack.yaml",
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
    assert "action_timeline" in d
    assert "step_coding" in d
    assert "trial_summary" in d
    assert "scoring" in d
    assert "timeline" in d


def test_build_export_includes_study_ready_tables():
    root = _repo_root()
    export = build_experiment_export(
        _make_events_with_help_cycles(),
        meta_overrides={
            "trial_id": "T01",
            "participant_id": "P01",
            "condition": "with_tutor",
        },
        pack_path=root / "packs/fa18c_startup/pack.yaml",
    )

    assert len(export.step_coding) == 33
    assert [row.StepID for row in export.step_coding][:3] == ["S01", "S02", "S03"]
    assert [row.StepID for row in export.step_coding][-1] == "S33"

    s01 = export.step_coding[0]
    assert s01.ParticipantID == "P01"
    assert s01.Condition == "with_tutor"
    assert s01.TrialID == "T01"
    assert s01.Phase == "P1"
    assert s01.Critical == "yes"
    assert s01.Performed == "yes"
    assert s01.Completed == "yes"
    assert s01.FirstHelpTime_sec == 5.0
    assert s01.HelpCount == 1
    assert s01.FirstOverlayTargets == "battery_switch"
    assert s01.LastOverlayTargets == "battery_switch"
    assert s01.FirstFusedStepID == "S01"
    assert s01.LastFusedStepID == "S01"
    assert "help_cycle:cycle-aaa-111" in s01.EvidenceRefs
    assert s01.Error_OM == ""
    assert s01.Error_CO == ""
    assert s01.CoderNotes == ""

    s02 = export.step_coding[1]
    assert s02.Performed == "yes"
    assert s02.Completed == "no"
    assert s02.HelpCount == 1
    assert s02.FirstHelpTime_sec == 12.0
    assert s02.FirstOverlayTargets == "left_mdi_brightness_selector"

    s33 = export.step_coding[-1]
    assert s33.Performed == "no"
    assert s33.Completed == "no"
    assert s33.HelpCount == 0

    assert len(export.trial_summary) == 1
    trial = export.trial_summary[0]
    assert trial.ParticipantID == "P01"
    assert trial.Condition == "with_tutor"
    assert trial.TrialID == "T01"
    assert trial.Completed == "no"
    assert trial.TaskTime_sec == 12.5
    assert trial.HelpRequests == 2
    assert trial.OverlayExecuted == 1
    assert trial.OverlayRejected == 1
    assert trial.FallbackCount == 1
    assert trial.CriticalStepsCompleted == 1
    assert trial.TotalStepsCompleted == 1
    assert trial.StepCompletionAccuracy == 0.030303


def test_study_ready_csv_contract_fields_are_frozen():
    assert ACTION_TIMELINE_CSV_FIELDS == [
        "ParticipantID", "Condition", "TrialID", "EventIndex", "Timestamp", "TWall", "Source",
        "RawKey", "RawValueBefore", "RawValueAfter", "Delta", "MappedTarget",
        "CandidateStepID", "ActiveStepID", "FusedStepID", "ExpectedForStep",
        "BeforeHelpCycleID", "AfterHelpCycleID", "NearestHelpCycleID", "SecondsSinceLastHelp",
        "StepCompletedByThisEvent", "GateViolationCandidate", "AutoCodingHint",
    ]
    assert HELP_CYCLES_CSV_FIELDS == [
        "cycle_index", "help_cycle_id", "trigger_wall_s", "generation_mode",
        "vision_used", "vision_status", "vision_fact_status", "vlm_call_status",
        "vision_fallback_reason", "sync_delta_ms",
        "frame_ids", "layout_id", "fused_step_id", "fused_missing_conditions", "model_next_step_id",
        "overlay_targets", "overlay_executed", "overlay_rejected",
        "overlay_dropped", "overlay_dry_run_count", "response_status", "fallback_overlay_used",
        "fallback_overlay_reason", "response_mapping_failure_codes",
        "observability_status", "requires_visual_confirmation",
        "scenario_profile",
    ]
    assert STEP_CODING_CSV_FIELDS == [
        "ParticipantID", "Condition", "TrialID", "StepID", "StepTitle", "Phase", "Critical",
        "Performed", "Completed", "FirstHelpTime_sec", "HelpCount", "FirstOverlayTargets",
        "LastOverlayTargets", "FirstFusedStepID", "LastFusedStepID", "EvidenceRefs",
        "Auto_Error_OM", "Auto_Error_CO", "Auto_Error_OR", "Auto_Error_PA", "Auto_Error_SV",
        "AutoConfidence", "AutoEvidenceRefs", "NeedsHumanReview",
        "Error_OM", "Error_CO", "Error_OR", "Error_PA", "Error_SV", "CoderID",
        "CoderNotes", "AutoCodingNotes",
    ]
    assert TRIAL_SUMMARY_CSV_FIELDS == [
        "ParticipantID", "Condition", "TrialID", "Completed", "TaskTime_sec", "HelpRequests",
        "LLMTriggers", "VLMCalls", "OverlayExecuted", "OverlayRejected", "FallbackCount",
        "CriticalStepsCompleted", "TotalStepsCompleted", "StepCompletionAccuracy",
    ]


def test_step_coding_prefills_error_candidates_for_human_review():
    root = _repo_root()
    export = build_experiment_export(
        _make_events_for_prescoring_candidates(),
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "with_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )
    by_step = {row.StepID: row for row in export.step_coding}

    assert by_step["S02"].Auto_Error_OM == "1"
    assert by_step["S02"].Auto_Error_CO == "1"
    assert by_step["S02"].NeedsHumanReview == "yes"
    assert "om:not_completed" in by_step["S02"].AutoEvidenceRefs
    assert "action:4:unmapped_raw_key" in by_step["S02"].AutoEvidenceRefs

    assert by_step["S03"].Completed == "yes"
    assert by_step["S03"].Auto_Error_OR == "0"
    assert by_step["S03"].Auto_Error_SV == "1"
    assert "action:1:precondition_gate:s03_requires_power_available" in by_step["S03"].AutoEvidenceRefs

    assert by_step["S05"].Auto_Error_SV == "1"
    assert "observation:6:state_violation" in by_step["S05"].AutoEvidenceRefs

    assert by_step["S31"].Auto_Error_PA == "1"
    assert "vars.radar_altimeter_bug_value=500" in by_step["S31"].AutoEvidenceRefs

    assert by_step["S01"].Auto_Error_OM == "0"
    assert by_step["S01"].NeedsHumanReview == "no"
    assert by_step["S01"].Error_OM == ""
    assert by_step["S01"].CoderID == ""


def test_order_prefill_requires_failed_gate_dependency(tmp_path: Path):
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "steps:\n"
        "  - id: S01\n"
        "    phase: P1\n"
        "    critical: false\n"
        "    completion_conditions: [Ready set.]\n"
        "    ui_targets: [ready_switch]\n"
        "  - id: S02\n"
        "    phase: P1\n"
        "    critical: false\n"
        "    completion_conditions: [Next set.]\n"
        "    ui_targets: [next_switch]\n"
        "precondition_gates:\n"
        "  S01: []\n"
        "  S02:\n"
        "    - op: flag_true\n"
        "      var: vars.ready\n"
        "      reason_code: s02_requires_ready\n"
        "completion_gates:\n"
        "  S01:\n"
        "    - op: flag_true\n"
        "      var: vars.ready\n"
        "      reason_code: s01_requires_ready\n"
        "  S02: []\n",
        encoding="utf-8",
    )
    ui_map = tmp_path / "ui_map.yaml"
    ui_map.write_text(
        "cockpit_elements:\n"
        "  ready_switch: {}\n"
        "  next_switch: {}\n",
        encoding="utf-8",
    )
    bios_to_ui = tmp_path / "bios_to_ui.yaml"
    bios_to_ui.write_text(
        "mappings:\n"
        "  NEXT_SW:\n"
        "    - next_switch\n",
        encoding="utf-8",
    )

    export = build_experiment_export(
        [
            {"kind": "step_activated", "payload": {"step_id": "S01"}, "t_wall": 0.0},
            {
                "kind": "observation",
                "source": "dcs_bios",
                "payload": {
                    "source": "dcs_bios",
                    "bios": {"NEXT_SW": 1},
                    "delta": {"NEXT_SW": 1},
                    "vars": {"ready": False},
                },
                "t_wall": 1.0,
            },
        ],
        pack_path=pack,
        bios_to_ui_path=bios_to_ui,
        ui_map_path=ui_map,
    )
    by_step = {row.StepID: row for row in export.step_coding}

    assert by_step["S02"].Auto_Error_SV == "1"
    assert by_step["S02"].Auto_Error_OR == "1"
    assert "action:1:out_of_order_before:S01" in by_step["S02"].AutoEvidenceRefs


def test_active_step_mismatch_without_blocked_gate_is_not_sv_or_order():
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        {"kind": "step_completed", "payload": {"step_id": "S01"}, "t_wall": 0.0, "timestamp": t0.isoformat()},
        {"kind": "step_activated", "payload": {"step_id": "S02"}, "t_wall": 1.0, "timestamp": t0.isoformat()},
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 1,
                "t_wall": 2.0,
                "source": "dcs_bios",
                "bios": {"APU_CONTROL_SW": 1},
                "delta": {"APU_CONTROL_SW": 1},
                "vars": {"power_available": True},
            },
            "t_wall": 2.0,
            "timestamp": t0.isoformat(),
        },
    ]
    root = _repo_root()
    export = build_experiment_export(
        events,
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )
    by_step = {row.StepID: row for row in export.step_coding}

    assert by_step["S03"].Auto_Error_SV == "0"
    assert by_step["S03"].Auto_Error_OR == "0"
    assert by_step["S02"].Auto_Error_CO == "1"


def test_parameter_prefill_uses_step_scoped_values_for_reversible_controls():
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 1,
                "t_wall": 1.0,
                "source": "dcs_bios",
                "bios": {"PROBE_SW": 0},
                "delta": {"PROBE_SW": 0},
                "vars": {"ext_refuel_probe_value": 65000},
            },
            "t_wall": 1.0,
            "timestamp": t0.isoformat(),
        },
        {"kind": "step_completed", "payload": {"step_id": "S20"}, "t_wall": 2.0, "timestamp": t0.isoformat()},
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 2,
                "t_wall": 3.0,
                "source": "dcs_bios",
                "bios": {"PROBE_SW": 1},
                "delta": {"PROBE_SW": 1},
                "vars": {"ext_refuel_probe_value": 0},
            },
            "t_wall": 3.0,
            "timestamp": t0.isoformat(),
        },
        {"kind": "step_completed", "payload": {"step_id": "S21"}, "t_wall": 4.0, "timestamp": t0.isoformat()},
    ]
    root = _repo_root()
    export = build_experiment_export(
        events,
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )
    by_step = {row.StepID: row for row in export.step_coding}

    assert by_step["S20"].Auto_Error_PA == "0"
    assert by_step["S21"].Auto_Error_PA == "0"


def test_parameter_prefill_reads_vars_only_telemetry_inside_step_window():
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 1,
                "t_wall": 1.0,
                "source": "dcs_bios",
                "bios": {"RADALT_MIN_HEIGHT_PTR": 50000},
                "delta": {"RADALT_MIN_HEIGHT_PTR": 50000},
            },
            "t_wall": 1.0,
            "timestamp": t0.isoformat(),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 2,
                "t_wall": 2.0,
                "source": "dcs_bios",
                "vars": {"radar_altimeter_bug_value": 500},
            },
            "t_wall": 2.0,
            "timestamp": t0.isoformat(),
        },
        {"kind": "step_completed", "payload": {"step_id": "S31"}, "t_wall": 3.0, "timestamp": t0.isoformat()},
    ]
    root = _repo_root()
    export = build_experiment_export(
        events,
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )
    by_step = {row.StepID: row for row in export.step_coding}

    assert by_step["S31"].Auto_Error_PA == "1"
    assert "vars.radar_altimeter_bug_value=500" in by_step["S31"].AutoEvidenceRefs


def test_parameter_prefill_honors_carrier_profile_overrides():
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
    events = [
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 1,
                "t_wall": 1.0,
                "source": "dcs_bios",
                "bios": {"RADALT_MIN_HEIGHT_PTR": 4000},
                "delta": {"RADALT_MIN_HEIGHT_PTR": 4000},
                "vars": {"radar_altimeter_bug_value": 40},
            },
            "t_wall": 1.0,
            "timestamp": t0.isoformat(),
        },
    ]
    root = _repo_root()
    export = build_experiment_export(
        events,
        meta_overrides={"scenario_profile": "carrier"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )
    by_step = {row.StepID: row for row in export.step_coding}

    assert by_step["S31"].Auto_Error_PA == "0"


def test_action_timeline_links_dcs_deltas_to_steps_and_help_cycles():
    root = _repo_root()
    export = build_experiment_export(
        _make_events_with_action_timeline(),
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "with_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    assert [row.RawKey for row in export.action_timeline] == [
        "BATTERY_SW",
        "APU_CONTROL_SW",
        "EXPERIMENTAL_RAW_KEY",
    ]

    expected = export.action_timeline[0]
    assert expected.ParticipantID == "P01"
    assert expected.Condition == "with_tutor"
    assert expected.TrialID == "T01"
    assert expected.EventIndex == 1
    assert expected.Source == "dcs_bios"
    assert expected.RawValueAfter == "2"
    assert expected.MappedTarget == "battery_switch"
    assert expected.CandidateStepID == "S01"
    assert expected.ActiveStepID == "S01"
    assert expected.ExpectedForStep == "yes"
    assert expected.BeforeHelpCycleID == "cycle-s02"
    assert expected.AfterHelpCycleID == ""
    assert expected.NearestHelpCycleID == "cycle-s02"
    assert expected.SecondsSinceLastHelp is None
    assert expected.StepCompletedByThisEvent == "yes"
    assert expected.GateViolationCandidate == "no"

    unrelated = export.action_timeline[1]
    assert unrelated.MappedTarget == "apu_switch"
    assert unrelated.CandidateStepID == "S03"
    assert unrelated.ActiveStepID == "S02"
    assert unrelated.ExpectedForStep == "no"
    assert unrelated.AfterHelpCycleID == "cycle-s02"
    assert unrelated.NearestHelpCycleID == "cycle-s02"
    assert unrelated.SecondsSinceLastHelp == 2.0
    assert unrelated.StepCompletedByThisEvent == "no"
    assert unrelated.GateViolationCandidate == "yes"
    assert "unexpected_for_active_step" in set(unrelated.AutoCodingHint.split(";"))

    unmapped = export.action_timeline[2]
    assert unmapped.RawKey == "EXPERIMENTAL_RAW_KEY"
    assert unmapped.RawValueAfter == "7"
    assert unmapped.MappedTarget == ""
    assert unmapped.CandidateStepID == ""
    assert "unmapped_raw_key" in set(unmapped.AutoCodingHint.split(";"))


def test_live_help_only_terminal_s33_infers_completion_and_task_time():
    root = _repo_root()
    export = build_experiment_export(
        _make_live_help_only_completion_events(),
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )
    by_step = {row.StepID: row for row in export.step_coding}

    assert export.trial_summary[0].Completed == "yes"
    assert export.trial_summary[0].TotalStepsCompleted == 33
    assert export.trial_summary[0].StepCompletionAccuracy == 1.0
    assert export.trial_summary[0].TaskTime_sec == 84.0

    assert by_step["S01"].Completed == "yes"
    assert "completed_inferred_from_progression" in by_step["S01"].AutoCodingNotes
    assert "progression_to:S02" in by_step["S01"].EvidenceRefs

    assert by_step["S33"].Completed == "yes"
    assert "completed_inferred_from_terminal_s33" in by_step["S33"].AutoCodingNotes
    assert "terminal_s33:cycle-s33" in by_step["S33"].EvidenceRefs


def test_task_time_does_not_stop_at_intermediate_completed_response():
    root = _repo_root()
    events = [
        {
            "kind": "observation",
            "source": "dcs_bios_raw",
            "payload": {
                "seq": 1,
                "t_wall": 0.0,
                "source": "dcs_bios_raw",
                "bios": {"BATTERY_SW": 2},
                "delta": {"BATTERY_SW": 2},
            },
            "t_wall": 0.0,
        },
        {
            "kind": "tutor_response",
            "payload": {
                "status": "ok",
                "message": "S08 already satisfied.",
                "metadata": {
                    "help_cycle_id": "cycle-s08",
                    "fused_step_id": "S08",
                    "help_response": {"next": {"step_id": "S08"}},
                    "final_public_instruction_category": "completed",
                },
            },
            "metadata": {"help_cycle_id": "cycle-s08", "fused_step_id": "S08"},
            "t_wall": 5.0,
        },
        {
            "kind": "observation",
            "source": "dcs_bios_raw",
            "payload": {
                "seq": 2,
                "t_wall": 20.0,
                "source": "dcs_bios_raw",
                "bios": {"APU_CONTROL_SW": 1},
                "delta": {"APU_CONTROL_SW": 1},
            },
            "t_wall": 20.0,
        },
    ]

    export = build_experiment_export(
        events,
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    assert export.trial_summary[0].TaskTime_sec == 20.0


def test_action_timeline_filters_passive_live_telemetry_but_keeps_mapped_actions():
    root = _repo_root()
    export = build_experiment_export(
        _make_live_help_only_completion_events(),
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    raw_keys = [row.RawKey for row in export.action_timeline]
    assert raw_keys == ["BATTERY_SW", "HOOK_LEVER"]
    assert export.action_timeline[0].MappedTarget == "battery_switch"
    assert export.action_timeline[0].CandidateStepID == "S01"
    assert export.action_timeline[1].MappedTarget == "arresting_hook_handle"
    assert export.action_timeline[1].CandidateStepID == "S24;S25"


def test_action_timeline_rejects_bios_mapping_with_unknown_ui_target():
    root = _repo_root()
    export_events = _make_events_with_action_timeline()

    try:
        build_experiment_export(
            export_events,
            pack_path=root / "packs/fa18c_startup/pack.yaml",
            bios_to_ui_path=root / "tests/adapters/fixtures/bios_to_ui_bad_target.yaml",
            ui_map_path=root / "tests/adapters/fixtures/ui_map_only_target.yaml",
        )
    except ValueError as exc:
        assert "unknown ui target" in str(exc)
    else:
        raise AssertionError("expected invalid bios_to_ui mapping to fail")


def test_action_timeline_record_serialization():
    rec = ActionTimelineRecord(
        ParticipantID="P01",
        Condition="with_tutor",
        TrialID="T01",
        EventIndex=3,
        RawKey="BATTERY_SW",
        RawValueAfter="2",
    )

    assert rec.to_dict()["ParticipantID"] == "P01"
    assert rec.to_dict()["RawKey"] == "BATTERY_SW"


def test_step_coding_uses_fused_step_as_help_cycle_owner():
    events = _make_events_with_help_cycles()
    first_response = events[4]["payload"]["metadata"]["help_response"]
    first_response["next"]["step_id"] = "S02"
    root = _repo_root()

    export = build_experiment_export(
        events,
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "with_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
    )

    s01 = export.step_coding[0]
    s02 = export.step_coding[1]
    assert s01.HelpCount == 1
    assert s01.FirstHelpTime_sec == 5.0
    assert s02.HelpCount == 1
    assert s02.FirstHelpTime_sec == 12.0


def test_step_coding_accepts_top_level_step_id_events(tmp_path: Path):
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "steps:\n"
        "  - id: S01\n"
        "    phase: P1\n"
        "    critical: \"false\"\n"
        "    completion_conditions: [Battery on.]\n",
        encoding="utf-8",
    )
    export = build_experiment_export(
        [
            {"type": "step_activated", "step_id": "S01", "t_wall": 0.0},
            {"type": "step_completed", "step_id": "S01", "t_wall": 4.0},
        ],
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "with_tutor"},
        pack_path=pack,
    )

    assert len(export.step_coding) == 1
    assert export.step_coding[0].Performed == "yes"
    assert export.step_coding[0].Completed == "yes"
    assert export.step_coding[0].Critical == "no"
    assert export.trial_summary[0].Completed == "yes"


def _make_observation_only_baseline_events() -> list[dict]:
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 1,
                "t_wall": 1.0,
                "source": "dcs_bios",
                "bios": {"BATTERY_SW": 2, "L_GEN_SW": 1, "R_GEN_SW": 1},
                "delta": {"BATTERY_SW": 2},
            },
            "metadata": {"seq": 1, "delta_count": 1},
            "t_wall": 1.0,
            "timestamp": t0.isoformat(),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 2,
                "t_wall": 3.0,
                "source": "dcs_bios",
                "bios": {
                    "BATTERY_SW": 2,
                    "L_GEN_SW": 1,
                    "R_GEN_SW": 1,
                    "APU_CONTROL_SW": 1,
                    "APU_READY_LT": 1,
                },
                "delta": {"APU_CONTROL_SW": 1},
            },
            "metadata": {"seq": 2, "delta_count": 1},
            "t_wall": 3.0,
            "timestamp": (t0 + timedelta(seconds=2)).isoformat(),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 3,
                "t_wall": 5.0,
                "source": "dcs_bios",
                "bios": {"PROBE_SW": 1, "EXT_REFUEL_PROBE": 0},
                "delta": {"PROBE_SW": 1, "EXT_REFUEL_PROBE": 0},
            },
            "metadata": {"seq": 3, "delta_count": 2},
            "t_wall": 5.0,
            "timestamp": (t0 + timedelta(seconds=4)).isoformat(),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 4,
                "t_wall": 8.0,
                "source": "dcs_bios",
                "bios": {"PROBE_SW": 1, "EXT_REFUEL_PROBE": 65535},
                "delta": {"EXT_REFUEL_PROBE": 65535},
            },
            "metadata": {"seq": 4, "delta_count": 1},
            "t_wall": 8.0,
            "timestamp": (t0 + timedelta(seconds=7)).isoformat(),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 5,
                "t_wall": 11.0,
                "source": "dcs_bios",
                "bios": {"PROBE_SW": 1, "EXT_REFUEL_PROBE": 0},
                "delta": {"EXT_REFUEL_PROBE": 0},
            },
            "metadata": {"seq": 5, "delta_count": 1},
            "t_wall": 11.0,
            "timestamp": (t0 + timedelta(seconds=10)).isoformat(),
        },
    ]


def test_without_tutor_export_marks_passive_telemetry_gate_completions() -> None:
    root = _repo_root()
    export = build_experiment_export(
        _make_observation_only_baseline_events(),
        meta_overrides={"trial_id": "T01", "participant_id": "P_BASE", "condition": "without_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    rows = {row.StepID: row for row in export.step_coding}
    assert rows["S01"].Completed == "yes"
    assert rows["S03"].Completed == "yes"
    assert rows["S20"].Completed == "yes"
    assert rows["S21"].Completed == "yes"
    assert "completed_from_passive_gate" in rows["S01"].AutoCodingNotes
    assert "passive_gate:S01:s01_requires_battery_on" in rows["S01"].EvidenceRefs
    assert "action:BATTERY_SW" in rows["S01"].EvidenceRefs
    assert "telemetry_var:vars.ext_refuel_probe_value" in rows["S20"].EvidenceRefs

    trial = export.trial_summary[0]
    assert trial.HelpRequests == 0
    assert trial.LLMTriggers == 0
    assert trial.VLMCalls == 0
    assert trial.OverlayExecuted == 0
    assert trial.TotalStepsCompleted >= 4
    assert trial.StepCompletionAccuracy > 0


def test_without_tutor_export_preserves_logged_latch_vars_over_resolver_recompute() -> None:
    root = _repo_root()
    t0 = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
    base_vars = {
        "fire_test_a_complete": False,
        "fire_test_b_complete": False,
        "obogs_ready": False,
        "standby_altimeter_set": False,
        "vars_source_missing": [
            "fire_test_a_complete",
            "fire_test_b_complete",
            "obogs_ready",
            "standby_altimeter_set",
        ],
    }
    complete_vars = {
        "fire_test_a_complete": True,
        "fire_test_b_complete": True,
        "obogs_ready": True,
        "standby_altimeter_set": True,
        "vars_source_missing": [
            "fire_test_a_complete",
            "fire_test_b_complete",
            "obogs_ready",
            "standby_altimeter_set",
        ],
    }
    bios_that_cannot_rederive_logged_completions = {
        "FIRE_TEST_SW": 1,
        "OBOGS_SW": 0,
        "OXY_FLOW": 0,
        "STBY_PRESS_SET_0": 0,
        "STBY_PRESS_SET_1": 0,
        "STBY_PRESS_SET_2": 0,
    }
    events = [
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 1,
                "t_wall": 1.0,
                "source": "dcs_bios",
                "bios": bios_that_cannot_rederive_logged_completions,
                "vars": base_vars,
            },
            "metadata": {"seq": 1, "delta_count": 0},
            "t_wall": 1.0,
            "timestamp": t0.isoformat(),
        },
        {
            "kind": "observation",
            "source": "dcs_bios",
            "payload": {
                "seq": 2,
                "t_wall": 2.0,
                "source": "dcs_bios",
                "bios": bios_that_cannot_rederive_logged_completions,
                "vars": complete_vars,
            },
            "metadata": {"seq": 2, "delta_count": 0},
            "t_wall": 2.0,
            "timestamp": (t0 + timedelta(seconds=1)).isoformat(),
        },
    ]

    export = build_experiment_export(
        events,
        meta_overrides={"trial_id": "T01", "participant_id": "P_BASE", "condition": "without_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    rows = {row.StepID: row for row in export.step_coding}
    assert rows["S02"].Completed == "yes"
    assert "passive_gate:S02:s02_requires_fire_test_a" in rows["S02"].EvidenceRefs
    assert "telemetry_var:vars.fire_test_a_complete" in rows["S02"].EvidenceRefs
    assert "telemetry_var:vars.fire_test_b_complete" in rows["S02"].EvidenceRefs
    assert rows["S14"].Completed == "yes"
    assert "telemetry_var:vars.obogs_ready" in rows["S14"].EvidenceRefs
    assert rows["S30"].Completed == "yes"
    assert "telemetry_var:vars.standby_altimeter_set" in rows["S30"].EvidenceRefs

    trial = export.trial_summary[0]
    assert trial.HelpRequests == 0
    assert trial.LLMTriggers == 0
    assert trial.VLMCalls == 0
    assert trial.OverlayExecuted == 0
    assert trial.OverlayRejected == 0
    assert trial.FallbackCount == 0


def test_passive_export_does_not_complete_mapped_action_when_gate_is_unsatisfied() -> None:
    root = _repo_root()
    event = _make_observation_only_baseline_events()[0]
    event["payload"]["bios"] = {"BATTERY_SW": 2}
    export = build_experiment_export(
        [event],
        meta_overrides={"trial_id": "T01", "participant_id": "P_BASE", "condition": "without_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    rows = {row.StepID: row for row in export.step_coding}
    assert rows["S01"].Completed == "no"
    assert export.trial_summary[0].TotalStepsCompleted == 0


def test_passive_export_does_not_complete_retraction_from_initial_latch_state() -> None:
    root = _repo_root()
    event = _make_observation_only_baseline_events()[2]
    export = build_experiment_export(
        [event],
        meta_overrides={"trial_id": "T01", "participant_id": "P_BASE", "condition": "without_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    rows = {row.StepID: row for row in export.step_coding}
    assert rows["S20"].Completed == "no"
    assert rows["S21"].Completed == "no"


def test_passive_export_assigns_shared_action_to_earliest_uncompleted_candidate() -> None:
    root = _repo_root()
    event = {
        "kind": "observation",
        "source": "dcs_bios",
        "payload": {
            "seq": 1,
            "t_wall": 1.0,
            "source": "dcs_bios",
            "bios": {"FLAP_SW": 2},
            "delta": {"FLAP_SW": 2},
        },
        "metadata": {"seq": 1, "delta_count": 1},
        "t_wall": 1.0,
        "timestamp": datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
    }

    export = build_experiment_export(
        [event],
        meta_overrides={"trial_id": "T01", "participant_id": "P_BASE", "condition": "without_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    rows = {row.StepID: row for row in export.step_coding}
    assert rows["S16"].Completed == "yes"
    assert rows["S27"].Completed == "no"


def test_passive_export_respects_completion_gate_profile_overrides() -> None:
    root = _repo_root()
    event = {
        "kind": "observation",
        "source": "dcs_bios",
        "payload": {
            "seq": 1,
            "t_wall": 1.0,
            "source": "dcs_bios",
            "bios": {"RADALT_MIN_HEIGHT_PTR": 4000},
            "delta": {"RADALT_MIN_HEIGHT_PTR": 4000},
        },
        "metadata": {"seq": 1, "delta_count": 1},
        "t_wall": 1.0,
        "timestamp": datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
    }

    airfield = build_experiment_export(
        [event],
        meta_overrides={
            "trial_id": "T01",
            "participant_id": "P_BASE",
            "condition": "without_tutor",
            "scenario_profile": "airfield",
        },
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )
    carrier = build_experiment_export(
        [event],
        meta_overrides={
            "trial_id": "T01",
            "participant_id": "P_BASE",
            "condition": "without_tutor",
            "scenario_profile": "carrier",
        },
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    airfield_rows = {row.StepID: row for row in airfield.step_coding}
    carrier_rows = {row.StepID: row for row in carrier.step_coding}
    assert airfield_rows["S31"].Completed == "no"
    assert carrier_rows["S31"].Completed == "yes"
    assert "passive_gate:S31:s31_requires_radalt_bug_carrier_40" in carrier_rows["S31"].EvidenceRefs


def test_with_tutor_export_does_not_use_passive_baseline_inference() -> None:
    root = _repo_root()
    export = build_experiment_export(
        _make_observation_only_baseline_events(),
        meta_overrides={"trial_id": "T01", "participant_id": "P_HELP", "condition": "with_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
    )

    rows = {row.StepID: row for row in export.step_coding}
    assert rows["S01"].Completed == "no"
    assert rows["S03"].Completed == "no"
    assert export.trial_summary[0].TotalStepsCompleted == 0


def test_trial_summary_counts_vlm_calls_once_per_help_cycle():
    events = _make_events_with_help_cycles()
    events[9]["metadata"]["vlm_call_status"] = "called"
    events[10]["payload"]["metadata"]["vlm_call_status"] = "called"
    root = _repo_root()

    export = build_experiment_export(
        events,
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "with_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
    )

    assert export.trial_summary[0].VLMCalls == 1
    assert export.help_cycles[1].vlm_call_status == "called"


def test_trial_summary_honors_explicit_non_called_vlm_status():
    events = _make_events_with_help_cycles()
    events[4]["payload"]["metadata"]["vlm_call_status"] = "not_required"
    root = _repo_root()

    export = build_experiment_export(
        events,
        meta_overrides={"trial_id": "T01", "participant_id": "P01", "condition": "with_tutor"},
        pack_path=root / "packs/fa18c_startup/pack.yaml",
    )

    assert export.trial_summary[0].VLMCalls == 0
    assert export.help_cycles[0].vlm_call_status == "not_required"


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


def test_build_file_sha256(tmp_path: Path):
    payload = b"experiment provenance\n"
    path = tmp_path / "raw.jsonl"
    path.write_bytes(payload)

    assert build_file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_quality_gate_strict_reports_missing_critical_metadata():
    report = build_export_quality_report(
        SessionMeta(participant_id="P01"),
        events=[],
        strict=True,
        pack_path=None,
        taxonomy_path=None,
        raw_log_copied=False,
        expected_help_cycle_rows=0,
        actual_help_cycle_rows=None,
        help_cycle_csv_headers=None,
    )

    assert report.passed is False
    assert any("trial_id" in error for error in report.errors)
    assert any("study_id" in error for error in report.errors)
    assert any("raw_log" in error for error in report.errors)


def test_quality_gate_strict_requires_model_prompt_and_dcs_metadata(tmp_path: Path):
    root = _repo_root()
    raw_log = tmp_path / "raw.jsonl"
    raw_log.write_text("{}\n", encoding="utf-8")
    meta = SessionMeta(
        trial_id="T01",
        study_id="study-alpha",
        participant_id="P01",
        condition="with_tutor",
        group="novice",
        experimenter_id="E01",
        questionnaire_ref="questionnaires/P01_pre.yml",
        recording_ref="recordings/P01_T01.mp4",
        git_commit="abc123",
        git_dirty=False,
        pack_path=str(root / "packs/fa18c_startup/pack.yaml"),
        pack_hash=build_file_sha256(root / "packs/fa18c_startup/pack.yaml"),
        taxonomy_path=str(root / "packs/fa18c_startup/taxonomy.yaml"),
        taxonomy_hash=build_file_sha256(root / "packs/fa18c_startup/taxonomy.yaml"),
        ui_map_hash=build_file_sha256(root / "packs/fa18c_startup/ui_map.yaml"),
        bios_to_ui_hash=build_file_sha256(root / "packs/fa18c_startup/bios_to_ui.yaml"),
        raw_log_ref="raw_events.jsonl",
        raw_log_sha256=build_file_sha256(raw_log),
    )

    report = build_export_quality_report(
        meta,
        events=[],
        strict=True,
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        taxonomy_path=root / "packs/fa18c_startup/taxonomy.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        raw_log_copied=True,
        expected_help_cycle_rows=0,
        actual_help_cycle_rows=0,
        help_cycle_csv_headers=HELP_CYCLES_CSV_FIELDS,
    )

    assert report.passed is False
    assert any("model_provider" in error for error in report.errors)
    assert any("prompt_version or prompt_hash" in error for error in report.errors)
    assert any("dcs_mission" in error for error in report.errors)
    assert any("vr_setup or monitor_setup" in error for error in report.errors)


def test_quality_gate_rejects_hash_mismatch():
    root = _repo_root()
    meta = SessionMeta(
        trial_id="T01",
        study_id="study-alpha",
        participant_id="P01",
        condition="with_tutor",
        group="novice",
        experimenter_id="E01",
        questionnaire_ref="questionnaires/P01_pre.yml",
        recording_ref="recordings/P01_T01.mp4",
        git_commit="abc123",
        git_dirty=False,
        pack_path=str(root / "packs/fa18c_startup/pack.yaml"),
        pack_hash="wrong-pack-hash",
        taxonomy_path=str(root / "packs/fa18c_startup/taxonomy.yaml"),
        taxonomy_hash=build_file_sha256(root / "packs/fa18c_startup/taxonomy.yaml"),
        ui_map_hash=build_file_sha256(root / "packs/fa18c_startup/ui_map.yaml"),
        bios_to_ui_hash=build_file_sha256(root / "packs/fa18c_startup/bios_to_ui.yaml"),
        model_provider="stub",
        model_name="ModelStub",
        vision_model_name="VisionStub",
        prompt_version="prompt-v1",
        scenario_profile="airfield",
        dcs_mission="cold-start.miz",
        dcs_aircraft="FA-18C",
        vr_setup="Quest 3",
        monitor_setup="native-viewports",
        raw_log_ref="raw_events.jsonl",
        raw_log_sha256="raw-sha",
    )

    report = build_export_quality_report(
        meta,
        events=[],
        strict=True,
        pack_path=root / "packs/fa18c_startup/pack.yaml",
        taxonomy_path=root / "packs/fa18c_startup/taxonomy.yaml",
        ui_map_path=root / "packs/fa18c_startup/ui_map.yaml",
        bios_to_ui_path=root / "packs/fa18c_startup/bios_to_ui.yaml",
        raw_log_copied=True,
        expected_help_cycle_rows=0,
        actual_help_cycle_rows=0,
        help_cycle_csv_headers=HELP_CYCLES_CSV_FIELDS,
    )

    assert report.passed is False
    assert any("pack_hash" in error and "does not match" in error for error in report.errors)


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_experiment_export_cli_freezes_metadata_and_copies_raw_log(tmp_path: Path):
    raw_log = tmp_path / "events.jsonl"
    _write_jsonl(raw_log, _make_events_with_help_cycles())
    output_dir = tmp_path / "exports"
    root = _repo_root()

    args = argparse.Namespace(
        file=str(raw_log),
        participant_id="P01",
        trial_id="T01",
        study_id="study-alpha",
        condition="with_tutor",
        group="novice",
        experimenter_id="E01",
        questionnaire="questionnaires/P01_pre.yml",
        recording_ref="recordings/P01_T01.mp4",
        notes="no issues",
        output_dir=str(output_dir),
        scoring=None,
        pack=str(root / "packs/fa18c_startup/pack.yaml"),
        taxonomy=str(root / "packs/fa18c_startup/taxonomy.yaml"),
        ui_map=str(root / "packs/fa18c_startup/ui_map.yaml"),
        bios_to_ui=str(root / "packs/fa18c_startup/bios_to_ui.yaml"),
        model_provider="stub",
        model_name="ModelStub",
        vision_model_name="VisionStub",
        prompt_version="prompt-v1",
        prompt_hash=None,
        scenario_profile="airfield",
        dcs_mission="cold-start.miz",
        dcs_aircraft="FA-18C",
        vr_setup="Quest 3",
        monitor_setup="native-viewports",
        git_commit="abc123",
        git_dirty=False,
        strict=True,
        overwrite=False,
        copy_raw_log=True,
    )

    assert _run_experiment_export(args) == 0

    trial_dir = output_dir / "P01" / "T01"
    session = json.loads((trial_dir / "session.json").read_text(encoding="utf-8"))
    meta = session["meta"]
    assert meta["trial_id"] == "T01"
    assert meta["study_id"] == "study-alpha"
    assert meta["git_commit"] == "abc123"
    assert meta["git_dirty"] is False
    assert meta["pack_hash"] == build_file_sha256(root / "packs/fa18c_startup/pack.yaml")
    assert meta["taxonomy_hash"] == build_file_sha256(root / "packs/fa18c_startup/taxonomy.yaml")
    assert meta["ui_map_hash"] == build_file_sha256(root / "packs/fa18c_startup/ui_map.yaml")
    assert meta["bios_to_ui_hash"] == build_file_sha256(root / "packs/fa18c_startup/bios_to_ui.yaml")
    assert meta["raw_log_ref"] == "raw_events.jsonl"
    assert meta["raw_log_sha256"] == build_file_sha256(raw_log)
    assert build_file_sha256(trial_dir / "raw_events.jsonl") == build_file_sha256(raw_log)
    assert session["quality_gate"]["passed"] is True
    assert (trial_dir / "step_coding.csv").exists()
    assert (trial_dir / "trial_summary.csv").exists()
    assert (trial_dir / "action_timeline.csv").exists()
    with (trial_dir / "action_timeline.csv").open("r", newline="", encoding="utf-8") as f:
        action_rows = list(csv.DictReader(f))
    assert list(action_rows[0].keys()) == ACTION_TIMELINE_CSV_FIELDS
    with (trial_dir / "step_coding.csv").open("r", newline="", encoding="utf-8") as f:
        step_rows = list(csv.DictReader(f))
    assert len(step_rows) == 33
    assert list(step_rows[0].keys()) == STEP_CODING_CSV_FIELDS
    assert step_rows[0]["StepID"] == "S01"
    assert step_rows[0]["Completed"] == "yes"
    assert step_rows[1]["StepID"] == "S02"
    assert step_rows[1]["HelpCount"] == "1"
    with (trial_dir / "trial_summary.csv").open("r", newline="", encoding="utf-8") as f:
        trial_rows = list(csv.DictReader(f))
    assert len(trial_rows) == 1
    assert list(trial_rows[0].keys()) == TRIAL_SUMMARY_CSV_FIELDS
    assert trial_rows[0]["ParticipantID"] == "P01"
    assert trial_rows[0]["TotalStepsCompleted"] == "1"
    with (trial_dir / "help_cycles.csv").open("r", newline="", encoding="utf-8") as f:
        help_rows = list(csv.DictReader(f))
    assert "vision_fact_status" in help_rows[0]
    assert "vlm_call_status" in help_rows[0]
    assert "fallback_overlay_reason" in help_rows[0]
    assert "response_mapping_failure_codes" in help_rows[0]
    assert "frame_ids" in help_rows[0]
    assert "layout_id" in help_rows[0]
    assert help_rows[0]["frame_ids"] == "frame_001.png"
    assert help_rows[1]["response_mapping_failure_codes"] == "vision_sync_miss"

    # A second export to the same participant/trial directory must not
    # silently replace study data unless explicitly allowed.
    assert _run_experiment_export(args) == 1
    args.overwrite = True
    assert _run_experiment_export(args) == 0


def test_experiment_export_cli_writes_action_timeline_rows(tmp_path: Path):
    raw_log = tmp_path / "events.jsonl"
    _write_jsonl(raw_log, _make_events_with_action_timeline())
    output_dir = tmp_path / "exports"
    root = _repo_root()

    args = argparse.Namespace(
        file=str(raw_log),
        participant_id="P01",
        trial_id="T01",
        study_id="study-alpha",
        condition="with_tutor",
        group="novice",
        experimenter_id=None,
        questionnaire=None,
        recording_ref=None,
        notes=None,
        output_dir=str(output_dir),
        scoring=None,
        pack=str(root / "packs/fa18c_startup/pack.yaml"),
        taxonomy=str(root / "packs/fa18c_startup/taxonomy.yaml"),
        ui_map=str(root / "packs/fa18c_startup/ui_map.yaml"),
        bios_to_ui=str(root / "packs/fa18c_startup/bios_to_ui.yaml"),
        model_provider=None,
        model_name=None,
        vision_model_name=None,
        prompt_version=None,
        prompt_hash=None,
        scenario_profile=None,
        dcs_mission=None,
        dcs_aircraft=None,
        vr_setup=None,
        monitor_setup=None,
        git_commit="abc123",
        git_dirty=False,
        strict=False,
        overwrite=False,
        copy_raw_log=True,
    )

    assert _run_experiment_export(args) == 0

    trial_dir = output_dir / "P01" / "T01"
    with (trial_dir / "action_timeline.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert [row["RawKey"] for row in rows] == ["BATTERY_SW", "APU_CONTROL_SW", "EXPERIMENTAL_RAW_KEY"]
    assert rows[0]["MappedTarget"] == "battery_switch"
    assert rows[0]["CandidateStepID"] == "S01"
    assert rows[1]["SecondsSinceLastHelp"] == "2.0"
    assert rows[1]["GateViolationCandidate"] == "yes"
    assert "unexpected_for_active_step" in set(rows[1]["AutoCodingHint"].split(";"))
    assert rows[2]["RawValueAfter"] == "7"
    assert rows[2]["MappedTarget"] == ""
    assert "unmapped_raw_key" in set(rows[2]["AutoCodingHint"].split(";"))


def test_real_smoke_live_help_log_export_and_analysis_regression(tmp_path: Path):
    root = _repo_root()
    raw_log = root / "logs/live_help_harness_smoke_20260520_145436.jsonl"
    if not raw_log.exists():
        pytest.skip(f"smoke regression log not available: {raw_log}")

    output_dir = tmp_path / "exports"
    args = argparse.Namespace(
        file=str(raw_log),
        participant_id="P_SMOKE_20260520",
        trial_id="T01",
        study_id="smoke_validation_20260520",
        condition="tutor",
        group="internal_smoke",
        experimenter_id="yz",
        questionnaire="questionnaires/smoke_placeholder.json",
        recording_ref=str(raw_log),
        notes=None,
        output_dir=str(output_dir),
        scoring=None,
        pack=str(root / "packs/fa18c_startup/pack.yaml"),
        taxonomy=str(root / "packs/fa18c_startup/taxonomy.yaml"),
        ui_map=str(root / "packs/fa18c_startup/ui_map.yaml"),
        bios_to_ui=str(root / "packs/fa18c_startup/bios_to_ui.yaml"),
        model_provider="openai_compat",
        model_name="qwen36_27b",
        vision_model_name="simtutor-vision",
        prompt_version="harness_v04",
        prompt_hash=None,
        scenario_profile="airfield",
        dcs_mission="live_smoke",
        dcs_aircraft="FA-18C_hornet",
        vr_setup="live_dcs_vr_or_composite",
        monitor_setup="fa18c_composite_panel_v2",
        git_commit="abc123",
        git_dirty=False,
        strict=True,
        overwrite=False,
        copy_raw_log=True,
    )

    assert _run_experiment_export(args) == 0

    trial_dir = output_dir / "P_SMOKE_20260520" / "T01"
    quality_gate = json.loads((trial_dir / "quality_gate.json").read_text(encoding="utf-8"))
    assert quality_gate["passed"] is True

    with (trial_dir / "trial_summary.csv").open("r", newline="", encoding="utf-8") as f:
        trial = list(csv.DictReader(f))[0]
    assert trial["Completed"] == "yes"
    assert int(trial["TotalStepsCompleted"]) >= 30
    assert float(trial["StepCompletionAccuracy"]) >= 0.9
    assert 0 < float(trial["TaskTime_sec"]) < 3600

    with (trial_dir / "step_coding.csv").open("r", newline="", encoding="utf-8") as f:
        steps = {row["StepID"]: row for row in csv.DictReader(f)}
    assert steps["S20"]["Completed"] == "yes"
    assert "completed_inferred_from_progression" in steps["S20"]["AutoCodingNotes"]
    assert steps["S33"]["Completed"] == "yes"
    assert "completed_inferred_from_terminal_s33" in steps["S33"]["AutoCodingNotes"]

    with (trial_dir / "action_timeline.csv").open("r", newline="", encoding="utf-8") as f:
        action_rows = list(csv.DictReader(f))
    action_keys = {row["RawKey"] for row in action_rows}
    assert len(action_rows) < 1000
    assert "BATTERY_SW" in action_keys
    assert "HYD_IND_BRAKE" not in action_keys
    assert "HYD_IND_LEFT" not in action_keys
    assert "EXT_REFUEL_PROBE" not in action_keys
    assert "EXT_HOOK" not in action_keys
    assert "IFEI_TEMP_R" not in action_keys

    analysis_dir = tmp_path / "analysis"
    analyze_args = argparse.Namespace(
        input_dir=str(output_dir),
        output_dir=str(analysis_dir),
        no_figures=True,
    )
    assert simtutor_cli._run_experiment_analyze(analyze_args) == 0
    with (analysis_dir / "condition_summary.csv").open("r", newline="", encoding="utf-8") as f:
        condition = list(csv.DictReader(f))[0]
    assert condition["CompletionRate"] == "1.0"
    assert float(condition["MeanStepCompletionAccuracy"]) >= 0.9
    assert 0 < float(condition["MeanTaskTime_sec"]) < 3600


def test_experiment_export_cli_overwrite_replaces_stale_managed_outputs(tmp_path: Path):
    raw_log = tmp_path / "events.jsonl"
    _write_jsonl(raw_log, _make_events_with_help_cycles())
    output_dir = tmp_path / "exports"
    root = _repo_root()
    args = argparse.Namespace(
        file=str(raw_log),
        participant_id="P01",
        trial_id="T01",
        study_id="study-alpha",
        condition="with_tutor",
        group="novice",
        experimenter_id="E01",
        questionnaire="questionnaires/P01_pre.yml",
        recording_ref="recordings/P01_T01.mp4",
        notes="no issues",
        output_dir=str(output_dir),
        scoring=None,
        pack=str(root / "packs/fa18c_startup/pack.yaml"),
        taxonomy=str(root / "packs/fa18c_startup/taxonomy.yaml"),
        ui_map=str(root / "packs/fa18c_startup/ui_map.yaml"),
        bios_to_ui=str(root / "packs/fa18c_startup/bios_to_ui.yaml"),
        model_provider="stub",
        model_name="ModelStub",
        vision_model_name="VisionStub",
        prompt_version="prompt-v1",
        prompt_hash=None,
        scenario_profile="airfield",
        dcs_mission="cold-start.miz",
        dcs_aircraft="FA-18C",
        vr_setup="Quest 3",
        monitor_setup="native-viewports",
        git_commit="abc123",
        git_dirty=False,
        strict=True,
        overwrite=False,
        copy_raw_log=True,
    )
    assert _run_experiment_export(args) == 0

    _write_jsonl(raw_log, [
        {
            "kind": "step_activated",
            "payload": {"step_id": "S01"},
            "t_wall": 0.0,
            "session_id": "sess-no-help",
        }
    ])
    args.overwrite = True
    assert _run_experiment_export(args) == 0

    trial_dir = output_dir / "P01" / "T01"
    with (trial_dir / "help_cycles.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1
    assert build_file_sha256(trial_dir / "raw_events.jsonl") == build_file_sha256(raw_log)


def test_experiment_export_cli_overwrite_preserves_old_export_when_preflight_fails(tmp_path: Path):
    raw_log = tmp_path / "events.jsonl"
    _write_jsonl(raw_log, _make_events_with_help_cycles())
    output_dir = tmp_path / "exports"
    root = _repo_root()
    args = argparse.Namespace(
        file=str(raw_log),
        participant_id="P01",
        trial_id="T01",
        study_id="study-alpha",
        condition="with_tutor",
        group="novice",
        experimenter_id="E01",
        questionnaire="questionnaires/P01_pre.yml",
        recording_ref="recordings/P01_T01.mp4",
        notes="no issues",
        output_dir=str(output_dir),
        scoring=None,
        pack=str(root / "packs/fa18c_startup/pack.yaml"),
        taxonomy=str(root / "packs/fa18c_startup/taxonomy.yaml"),
        ui_map=str(root / "packs/fa18c_startup/ui_map.yaml"),
        bios_to_ui=str(root / "packs/fa18c_startup/bios_to_ui.yaml"),
        model_provider="stub",
        model_name="ModelStub",
        vision_model_name="VisionStub",
        prompt_version="prompt-v1",
        prompt_hash=None,
        scenario_profile="airfield",
        dcs_mission="cold-start.miz",
        dcs_aircraft="FA-18C",
        vr_setup="Quest 3",
        monitor_setup="native-viewports",
        git_commit="abc123",
        git_dirty=False,
        strict=True,
        overwrite=False,
        copy_raw_log=True,
    )
    assert _run_experiment_export(args) == 0

    trial_dir = output_dir / "P01" / "T01"
    old_session = json.loads((trial_dir / "session.json").read_text(encoding="utf-8"))
    args.overwrite = True
    args.study_id = None
    assert _run_experiment_export(args) == 1

    preserved_session = json.loads((trial_dir / "session.json").read_text(encoding="utf-8"))
    assert preserved_session["meta"]["study_id"] == old_session["meta"]["study_id"]
    assert (trial_dir / "raw_events.jsonl").exists()


def test_experiment_export_cli_overwrite_preserves_old_export_when_staging_write_fails(
    tmp_path: Path,
    monkeypatch,
):
    raw_log = tmp_path / "events.jsonl"
    _write_jsonl(raw_log, _make_events_with_help_cycles())
    output_dir = tmp_path / "exports"
    root = _repo_root()
    args = argparse.Namespace(
        file=str(raw_log),
        participant_id="P01",
        trial_id="T01",
        study_id="study-alpha",
        condition="with_tutor",
        group="novice",
        experimenter_id="E01",
        questionnaire="questionnaires/P01_pre.yml",
        recording_ref="recordings/P01_T01.mp4",
        notes="no issues",
        output_dir=str(output_dir),
        scoring=None,
        pack=str(root / "packs/fa18c_startup/pack.yaml"),
        taxonomy=str(root / "packs/fa18c_startup/taxonomy.yaml"),
        ui_map=str(root / "packs/fa18c_startup/ui_map.yaml"),
        bios_to_ui=str(root / "packs/fa18c_startup/bios_to_ui.yaml"),
        model_provider="stub",
        model_name="ModelStub",
        vision_model_name="VisionStub",
        prompt_version="prompt-v1",
        prompt_hash=None,
        scenario_profile="airfield",
        dcs_mission="cold-start.miz",
        dcs_aircraft="FA-18C",
        vr_setup="Quest 3",
        monitor_setup="native-viewports",
        git_commit="abc123",
        git_dirty=False,
        strict=True,
        overwrite=False,
        copy_raw_log=True,
    )
    assert _run_experiment_export(args) == 0

    trial_dir = output_dir / "P01" / "T01"
    old_session_text = (trial_dir / "session.json").read_text(encoding="utf-8")
    old_raw_hash = build_file_sha256(trial_dir / "raw_events.jsonl")

    def _fail_copy(*_args, **_kwargs):
        raise OSError("simulated copy failure")

    args.overwrite = True
    monkeypatch.setattr(simtutor_cli.shutil, "copy2", _fail_copy)

    assert _run_experiment_export(args) == 1
    assert (trial_dir / "session.json").read_text(encoding="utf-8") == old_session_text
    assert build_file_sha256(trial_dir / "raw_events.jsonl") == old_raw_hash
