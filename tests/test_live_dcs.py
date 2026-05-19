from __future__ import annotations

import builtins
import json
import math
import socket
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any
import time
from uuid import UUID

import pytest
import yaml

from core.types_v2 import VisionFact, VisionFactObservation, VisionObservation
from adapters.action_executor import OverlayActionExecutor
from adapters.dcs.overlay.sender import DcsOverlaySender
from adapters.openai_compat_model import OpenAICompatModel
from adapters.step_inference import StepInferenceResult
from adapters.vision_sync import HelpCycleVisionSelection
from adapters.vision_fact_extractor import VisionFactExtractionResult
from adapters.source_chunk_refs import build_source_chunk_ref
from core.help_failure import ALLOWLIST_FAIL, EVIDENCE_FAIL
from core.types import Observation, TutorRequest, TutorResponse
from live_dcs import (
    CompositeHelpTrigger,
    LiveDcsTutorLoop,
    ReplayBiosReceiver,
    StdinHelpTrigger,
    UdpHelpTrigger,
    _build_model_from_args,
    _build_observation_source_from_args,
    _build_harness_trace_metadata,
    _build_procedural_action_hint,
    build_arg_parser,
    _build_vision_fact_extractor_from_model,
    _build_vision_port_from_args,
    _emit_vision_observation_event,
    _emit_vision_fact_observation_event,
    _is_help_trigger_payload,
    _load_overlay_allowlist,
    _load_step_signal_profiles,
    _normalize_cached_response_metadata,
    _path_like_to_uri,
    _prefer_navigation_target_from_vision_context,
    _emit_multi_target_overlay_config_warning,
    _resolve_overlay_step_id,
    _resolve_step_overlay_allowlist,
    _sanitize_request_payload_for_event,
    _sanitize_response_payload_for_event,
    _sanitize_policy_error_for_user,
    _telemetry_window_signature,
    _text_claims_step_complete,
)
from simtutor.schemas import validate_instance
from tools.index_docs import build_index
from tests._fakes import FakeClient
from tests.adapters.socket_stubs import DummySocket


def _bios_frame(seq: int, t_wall: float, *, apu_switch: int) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "seq": seq,
        "t_wall": t_wall,
        "aircraft": "FA-18C_hornet",
        "bios": {
            "BATTERY_SW": 2,
            "L_GEN_SW": 1,
            "R_GEN_SW": 1,
            "FIRE_TEST_SW": 0,
            "APU_CONTROL_SW": apu_switch,
            "APU_READY_LT": 0,
            "ENGINE_CRANK_SW": 1,
        },
        "delta": {"APU_CONTROL_SW": apu_switch},
    }


def _bios_frame_fire_test_b_pre(seq: int, t_wall: float) -> dict[str, Any]:
    """Pre-frame to latch fire test B before the main interaction frame."""
    return {
        "schema_version": "v2",
        "seq": seq,
        "t_wall": t_wall,
        "aircraft": "FA-18C_hornet",
        "bios": {
            "BATTERY_SW": 2,
            "L_GEN_SW": 1,
            "R_GEN_SW": 1,
            "FIRE_TEST_SW": 2,
            "APU_CONTROL_SW": 0,
            "APU_READY_LT": 0,
            "ENGINE_CRANK_SW": 1,
        },
        "delta": {},
    }


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        self.calls.append({"observation": observation, "request": request})
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id if request else None,
            message="Turn on APU.",
            actions=[],
            explanations=["Turn on APU."],
            metadata={
                "provider": "mock_qwen",
                "help_response": {
                    "diagnosis": {"step_id": "S02", "error_category": "OM"},
                    "next": {"step_id": "S03"},
                    "overlay": {
                        "targets": ["apu_switch"],
                        "evidence": [
                            {
                                "target": "apu_switch",
                                "type": "delta",
                                "ref": "RECENT_UI_TARGETS.apu_switch",
                                "quote": "Recent delta shows APU switch activity.",
                            }
                        ],
                    },
                    "explanations": ["Turn on APU."],
                    "confidence": 0.9,
                },
            },
        )


class MultiTargetHelpResponseModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id if request else None,
            message="Hold FCS BIT and press PB5 together.",
            actions=[],
            explanations=["Hold FCS BIT and press PB5 together."],
            metadata={
                "provider": "fake_llm",
                "generation_mode": "model",
                "help_response": {
                    "diagnosis": {"step_id": "S19", "error_category": "OM"},
                    "next": {"step_id": "S19"},
                    "overlay": {
                        "targets": ["fcs_bit_switch", "right_mdi_pb5"],
                        "evidence": [
                            {
                                "target": "fcs_bit_switch",
                                "type": "delta",
                                "ref": "RECENT_UI_TARGETS.fcs_bit_switch",
                                "quote": "Recent delta shows the FCS BIT switch interaction.",
                                "grounding_confidence": 0.92,
                            },
                            {
                                "target": "right_mdi_pb5",
                                "type": "delta",
                                "ref": "RECENT_UI_TARGETS.right_mdi_pb5",
                                "quote": "Recent delta shows the right DDI PB5 interaction.",
                                "grounding_confidence": 0.9,
                            },
                        ],
                    },
                    "explanations": ["Hold FCS BIT and press PB5 together."],
                    "confidence": 0.91,
                },
            },
        )


class FailingModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        raise RuntimeError("model unavailable")


class _TriggerOnce:
    def __init__(self) -> None:
        self._fired = False

    def poll(self) -> bool:
        if self._fired:
            return False
        self._fired = True
        return True


class _DelayedObservationSource:
    def __init__(self, observation: Observation) -> None:
        self._observation = observation
        self._calls = 0
        self.is_exhausted = False

    def get_observation(self) -> Observation | None:
        self._calls += 1
        if self._calls == 1:
            return None
        if self._calls == 2:
            return self._observation
        self.is_exhausted = True
        return None


class SequencedGenerationModeModel:
    def __init__(self) -> None:
        self.calls = 0

    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        self.calls += 1
        if self.calls == 1:
            mode = "model"
            status = "ok"
            message = "Model native response."
            actions = [
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "apu_switch",
                    "element_id": _apu_element_id_from_ui_map(),
                }
            ]
        elif self.calls == 2:
            mode = "repair"
            status = "ok"
            message = "Locally repaired response."
            actions = [
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "apu_switch",
                    "element_id": _apu_element_id_from_ui_map(),
                }
            ]
        else:
            mode = "fallback"
            status = "error"
            message = "Fallback: likely stuck at S03."
            actions = []
        return TutorResponse(
            status=status,
            in_reply_to=request.request_id if request else None,
            message=message,
            actions=actions,
            explanations=[message],
            metadata={
                "provider": "mock_qwen" if mode != "fallback" else "fallback",
                "generation_mode": mode,
            },
        )


class RecordingExecutor:
    def __init__(self, *, include_dry_run: bool = False, dry_run: bool = False) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.include_dry_run = include_dry_run
        self.dry_run = dry_run

    def execute_actions(self, actions):
        actions_list = [dict(item) for item in actions if isinstance(item, dict)]
        self.calls.append(actions_list)
        report = {"executed": actions_list, "rejected": [], "dropped": [], "dry_run": []}
        if self.include_dry_run:
            report["dry_run"] = [{"target": item.get("target")} for item in actions_list]
        return report

    def close(self) -> None:  # pragma: no cover
        return


class RecordingTutorTextSender:
    def __init__(self, *, result: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result or {
            "status": "ok",
            "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        }

    def send_text(
        self,
        text: str,
        *,
        display_time_s: float = 12.0,
        clear_view: bool = False,
        expect_ack: bool = True,
        cmd_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "text": text,
                "display_time_s": display_time_s,
                "clear_view": clear_view,
                "expect_ack": expect_ack,
                "cmd_id": cmd_id,
            }
        )
        payload = dict(self.result)
        payload.setdefault("text", text)
        payload.setdefault("display_time_s", display_time_s)
        payload.setdefault("clear_view", clear_view)
        return payload

    def close(self) -> None:  # pragma: no cover
        return


def _make_evented_overlay_executor(monkeypatch, events: list[dict[str, Any]], *, session_id: str) -> OverlayActionExecutor:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sink = lambda event: events.append(event.to_dict())
    return OverlayActionExecutor(
        sender=DcsOverlaySender(
            auto_clear=False,
            ack_enabled=False,
            session_id=session_id,
            event_sink=sink,
        ),
        session_id=session_id,
        event_sink=sink,
        max_targets=1,
    )


def _make_multi_target_overlay_executor(
    monkeypatch,
    events: list[dict[str, Any]],
    *,
    session_id: str,
) -> OverlayActionExecutor:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sink = lambda event: events.append(event.to_dict())
    return OverlayActionExecutor(
        sender=DcsOverlaySender(
            auto_clear=False,
            ack_enabled=False,
            session_id=session_id,
            event_sink=sink,
        ),
        session_id=session_id,
        event_sink=sink,
        max_targets=2,
    )


def _make_multi_target_overlay_executor_with_auto_clear(
    monkeypatch,
    events: list[dict[str, Any]],
    *,
    session_id: str,
) -> OverlayActionExecutor:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sink = lambda event: events.append(event.to_dict())
    return OverlayActionExecutor(
        sender=DcsOverlaySender(
            auto_clear=True,
            ack_enabled=False,
            session_id=session_id,
            event_sink=sink,
        ),
        session_id=session_id,
        event_sink=sink,
        max_targets=2,
    )


class QueryOnlyKnowledge:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        self.calls.append({"text": text, "k": k})
        return [
            {
                "doc_id": "manual",
                "section": "S02",
                "page_or_heading": "S02",
                "snippet": "Complete FIRE TEST A and FIRE TEST B before APU start.",
                "snippet_id": "manual_s02_1",
                "score": 1.0,
            }
        ]


class FailingKnowledge:
    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        raise RuntimeError("knowledge backend down")


class EmptyKnowledge:
    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        return []


class NonSerializableKnowledge:
    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": Path("manual.md"),
                "section": Path("S02"),
                "page_or_heading": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "snippet": {"text": "Complete FIRE TEST A/B before APU start"},
                "snippet_id": Path("manual_s02_1"),
                "score": 0.9,
                "unexpected": {"nested": True},
            }
        ]


class MetaKnowledge:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve_with_meta(
        self,
        query: str,
        top_k: int = 5,
        *,
        step_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.calls.append({"query": query, "top_k": top_k, "step_id": step_id})
        snippets = [
            {
                "doc_id": "meta_manual",
                "section": "S02",
                "page_or_heading": "S02",
                "snippet": "Complete FIRE TEST A/B before APU.",
                "snippet_id": "meta_s02_1",
                "score": 1.0,
            }
        ]
        meta = {
            "cache_hit": True,
            "grounding_missing": False,
            "grounding_reason": None,
            "snippet_ids": ["meta_s02_1"],
            "index_path": "meta://store",
        }
        return snippets, meta

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        raise AssertionError("query() should not be called when retrieve_with_meta() is available")


class NonSerializableMetaKnowledge:
    def retrieve_with_meta(
        self,
        query: str,
        top_k: int = 5,
        *,
        step_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        snippets = [
            {
                "doc_id": Path("meta_manual.md"),
                "section": {"unexpected": "mapping"},
                "page_or_heading": datetime(2026, 1, 2, tzinfo=timezone.utc),
                "snippet": {"text": "Complete FIRE TEST A/B before APU"},
                "snippet_id": Path("meta_s02_1"),
                "score": float("inf"),
                "extra": {"nested": True},
            }
        ]
        meta = {
            "cache_hit": "yes",
            "grounding_missing": 0,
            "grounding_reason": {"unexpected": "mapping"},
            "snippet_ids": [Path("meta_s02_1"), {"nested": True}],
            "index_path": Path("meta_store/index.json"),
            "grounding_error_type": {"err": "Type"},
        }
        return snippets, meta

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        raise AssertionError("query() should not be called when retrieve_with_meta() is available")


class PolicyMixedKnowledge:
    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": "fa18c_startup_master",
                "section": "Master Step Table",
                "page_or_heading": "Master Step Table",
                "snippet": "S03 APU switch to ON and wait for green APU READY light.",
                "snippet_id": "fa18c_startup_master_1",
                "score": 1.0,
            },
            {
                "doc_id": "DCS FA-18C Early Access Guide EN",
                "section": "INTRODUCTION",
                "page_or_heading": 2,
                "snippet": "This PDF snippet should be rejected by policy.",
                "snippet_id": "DCS FA-18C Early Access Guide EN_1",
                "score": 0.9,
            },
        ]


class PolicyMixedKnowledgeWithDistinctChunkId:
    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": "fa18c_startup_master",
                "section": "Master Step Table",
                "page_or_heading": "Master Step Table",
                "snippet": "S03 APU switch to ON and wait for green APU READY light.",
                "snippet_id": "custom_alias_for_chunk",
                "chunk_id": "fa18c_startup_master_1",
                "score": 1.0,
            }
        ]


class PolicyRejectAllKnowledge:
    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": "DCS FA-18C Early Access Guide EN",
                "section": "INTRODUCTION",
                "page_or_heading": 2,
                "snippet": "This PDF snippet should be rejected by policy.",
                "snippet_id": "DCS FA-18C Early Access Guide EN_1",
                "score": 0.9,
            }
        ]


def _write_replay(path: Path, frames: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(frame, ensure_ascii=False) + "\n" for frame in frames)
    path.write_text(text, encoding="utf-8")


def _apu_element_id_from_ui_map() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    ui_map_path = repo_root / "packs" / "fa18c_startup" / "ui_map.yaml"
    ui_map = yaml.safe_load(ui_map_path.read_text(encoding="utf-8"))
    return str(ui_map["cockpit_elements"]["apu_switch"]["dcs_id"])


def _default_pack_path() -> Path:
    return Path(__file__).resolve().parents[1] / "packs" / "fa18c_startup" / "pack.yaml"


def _default_policy_path() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge_source_policy.yaml"


class OutOfAllowlistTargetModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id if request else None,
            message="Check switch.",
            actions=[],
            explanations=["Check switch."],
            metadata={
                "provider": "mock_qwen",
                "help_response": {
                    "diagnosis": {"step_id": "S02", "error_category": "OM"},
                    "next": {"step_id": "S03"},
                    "overlay": {
                        "targets": ["battery_switch"],
                        "evidence": [
                            {
                                "target": "battery_switch",
                                "type": "var",
                                "ref": "VARS.battery_on",
                                "quote": "Battery power state indicates this control path.",
                            }
                        ],
                    },
                    "explanations": ["Check switch."],
                    "confidence": 0.9,
                },
            },
        )


class MixedAllowlistTargetsModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id if request else None,
            message="Check allowed switch first.",
            actions=[],
            explanations=["Check allowed switch first."],
            metadata={
                "provider": "mock_qwen",
                "help_response": {
                    "diagnosis": {"step_id": "S02", "error_category": "OM"},
                    "next": {"step_id": "S03"},
                    "overlay": {
                        "targets": ["apu_switch", "battery_switch"],
                        "evidence": [
                            {
                                "target": "apu_switch",
                                "type": "delta",
                                "ref": "RECENT_UI_TARGETS.apu_switch",
                                "quote": "Recent delta points to APU switch.",
                            },
                            {
                                "target": "battery_switch",
                                "type": "var",
                                "ref": "VARS.battery_on",
                                "quote": "Battery var confirms control context.",
                            },
                        ],
                    },
                    "explanations": ["Check allowed switch first."],
                    "confidence": 0.9,
                },
            },
        )


class MixedAllowlistTargetsWithInvalidEvidenceItemModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id if request else None,
            message="Check allowed switch first.",
            actions=[],
            explanations=["Check allowed switch first."],
            metadata={
                "provider": "mock_qwen",
                "help_response": {
                    "diagnosis": {"step_id": "S02", "error_category": "OM"},
                    "next": {"step_id": "S03"},
                    "overlay": {
                        "targets": ["apu_switch", "battery_switch"],
                        "evidence": [
                            {
                                "target": "apu_switch",
                                "type": "delta",
                                "ref": "RECENT_UI_TARGETS.apu_switch",
                                "quote": "Recent delta points to APU switch.",
                            },
                            "invalid-evidence-item",
                            {
                                "target": "battery_switch",
                                "type": "var",
                                "ref": "VARS.battery_on",
                                "quote": "Battery var confirms control context.",
                            },
                        ],
                    },
                    "explanations": ["Check allowed switch first."],
                    "confidence": 0.9,
                },
            },
        )


def test_live_loop_offline_single_sample_runs_help_response_and_actions(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_one.jsonl"
    _write_replay(replay_path, [
        _bios_frame_fire_test_b_pre(0, 9.0),
        _bios_frame(1, 10.0, apu_switch=0),
    ])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    assert stats["frames"] == 2
    assert stats["help_cycles"] == 1
    assert stats["model_calls"] == 1
    assert len(model.calls) == 1
    request = model.calls[0]["request"]
    assert request is not None
    assert request.intent == "help"
    assert "candidate_steps" in request.context
    candidate_steps = request.context["candidate_steps"]
    assert isinstance(candidate_steps, list)
    assert candidate_steps[0]["source"] == "deterministic"
    assert candidate_steps[0]["role"] == "candidate_not_authoritative"
    assert candidate_steps[0]["step_id"]
    assert "supporting_evidence_refs" in candidate_steps[0]
    assert "proposed_next_action_target_ids" in candidate_steps[0]
    assert "recent_deltas" in request.context
    assert "recent_actions" in request.context
    assert "deterministic_step_hint" in request.context
    assert "gates" in request.context
    assert request.context["pack_path"].endswith("packs/fa18c_startup/pack.yaml")
    assert request.context["telemetry_map_path"].endswith("packs/fa18c_startup/telemetry_map.yaml")
    hint = request.context["deterministic_step_hint"]
    assert isinstance(hint, dict)
    assert hint.get("inferred_step_id")
    assert isinstance(hint.get("requires_visual_confirmation"), bool)
    assert hint.get("step_ui_targets") == ["apu_switch"]
    harness_spec = hint.get("step_harness_spec")
    assert isinstance(harness_spec, dict)
    assert harness_spec["step_id"] == hint["inferred_step_id"]
    assert harness_spec["allowed_overlay_targets"] == ["apu_switch"]
    gates = request.context["gates"]
    assert isinstance(gates, dict)
    assert "S03.completion" in gates
    assert gates["S03.completion"]["status"] in {"allowed", "blocked"}
    assert request.metadata["prompt_hash"]
    assert request.context["evidence_packet_summary"]["telemetry_status"] in {
        "nominal",
        "low_confidence_bootstrap",
    }
    telemetry_window_digest = request.context["state_harness"]["telemetry_window_digest"]
    assert telemetry_window_digest["frame_count"] >= 1
    assert telemetry_window_digest["first_seq"] == 0
    assert telemetry_window_digest["latest_seq"] == 1
    assert "telemetry_window_digest" in request.context["evidence_packet_summary"]
    assert isinstance(request.context["evidence_packet_summary"]["blocked_gate_count"], int)
    assert request.metadata["evidence_packet_summary"] == request.context["evidence_packet_summary"]
    evidence_snapshot = request.context["evidence_snapshot"]
    assert evidence_snapshot["source_observation_seq"] == 1
    assert evidence_snapshot["telemetry_window_latest_seq"] == 1
    assert evidence_snapshot["candidate_generation_snapshot_id"]
    assert (
        request.metadata["evidence_snapshot"]["model_request_snapshot_id"]
        == evidence_snapshot["model_request_snapshot_id"]
    )
    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    assert tutor_request_payload["context"]["evidence_packet_summary"] == request.context["evidence_packet_summary"]
    assert tutor_request_payload["metadata"]["evidence_packet_summary"] == request.context["evidence_packet_summary"]
    assert tutor_request_payload["context"]["evidence_snapshot"]["source_observation_seq"] == 1
    assert tutor_request_payload["metadata"]["evidence_snapshot"]["source_observation_seq"] == 1
    assert (
        tutor_request_payload["context"]["snapshot_ids"]["candidate_generation"]
        == evidence_snapshot["candidate_generation_snapshot_id"]
    )
    assert tutor_request_payload["context"]["state_harness"]["telemetry_window_digest"]["first_seq"] == 0
    assert request.context["overlay_target_allowlist"] == ["apu_switch"]

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    response_meta = tutor_response_payload["metadata"]
    trace = response_meta["harness_trace"]
    assert trace["schema_version"] == "v1"
    assert trace["evidence_packet_summary"] == request.context["evidence_packet_summary"]
    assert trace["evidence_snapshot"] == request.context["evidence_snapshot"]
    assert trace["snapshot_ids"]["candidate_generation"] == evidence_snapshot["candidate_generation_snapshot_id"]
    assert trace["snapshot_ids"]["model_request"] == evidence_snapshot["model_request_snapshot_id"]
    assert trace["snapshot_ids"]["validator"] == evidence_snapshot["validator_snapshot_id"]
    assert trace["snapshot_ids"]["final_decision"] == evidence_snapshot["final_decision_snapshot_id"]
    assert trace["final_action_plan"]["evidence_snapshot_id"] == evidence_snapshot["final_decision_snapshot_id"]
    assert trace["candidates"][0]["step_id"] == candidate_steps[0]["step_id"]
    assert trace["model_decision"]["step_id"] == "S03"
    assert trace["final_overlay_targets"] == ["apu_switch"]
    assert trace["vlm_call"]["status"] in {"not_required", "skipped"}
    assert response_meta["final_overlay_targets"] == ["apu_switch"]
    assert response_meta["vlm_call_status"] == trace["vlm_call"]["status"]

    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    assert executor.calls[0][0]["type"] == "overlay"
    assert executor.calls[0][0]["target"] == "apu_switch"
    assert "evidence_snapshot" not in executor.calls[0][0]
    assert "snapshot_ids" not in executor.calls[0][0]


def test_harness_trace_marks_response_hint_superseded_when_snapshot_differs() -> None:
    request = TutorRequest(
        request_id="cycle-315",
        actor="learner",
        intent="help",
        message="help",
        observation_ref="obs-new",
        context={
            "candidate_steps": [{"step_id": "S09", "source": "deterministic"}],
            "evidence_packet_summary": {"telemetry_status": "nominal"},
            "evidence_snapshot": {
                "schema_version": "evidence_snapshot.v1",
                "snapshot_id": "snapshot-new",
                "source_observation_id": "obs-new",
                "source_observation_seq": 20,
                "telemetry_window_latest_seq": 20,
                "candidate_generation_snapshot_id": "snapshot-new",
                "model_request_snapshot_id": "snapshot-new",
                "validator_snapshot_id": "snapshot-new",
                "final_decision_snapshot_id": "snapshot-new",
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "missing_conditions": ["vars.comm1_freq_134_000==true"],
            },
        },
    )
    response = TutorResponse(
        status="ok",
        message="Tune COMM1.",
        actions=[{"type": "overlay", "target": "ufc_key_1"}],
        explanations=["Tune COMM1."],
        metadata={
            "generation_mode": "model",
            "deterministic_step_hint": {
                "inferred_step_id": "S08",
                "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
            },
            "harness_action_plan": {
                "step_id": "S09",
                "targets": ["ufc_key_1"],
                "source": "model",
                "evidence_snapshot_id": "snapshot-old",
            },
            "help_response": {
                "next": {"step_id": "S09"},
                "overlay": {"targets": ["ufc_key_1"], "evidence": []},
            },
        },
    )

    trace = _build_harness_trace_metadata(
        request=request,
        response=response,
        vision_selection=HelpCycleVisionSelection(
            status="vision_not_required",
            observation_ref=None,
            observation_seq=20,
            observation_t_wall_s=20.0,
            observation_t_wall_ms=20000,
            trigger_wall_ms=20000,
            sync_window_ms=250,
            vision_used=False,
            frame_id=None,
            sync_status="not_required",
            sync_delta_ms=None,
            frame_stale=None,
            frame_ids=[],
            selected_frames=[],
            pre_trigger_frame=None,
            trigger_frame=None,
            sync_miss_reason=None,
        ),
        vision_fact_context={"status": "vision_not_required", "vision_fact_summary": {}, "vision_facts": []},
    )

    assert response.metadata["deterministic_step_hint"]["superseded_by_snapshot"] == "snapshot-new"
    assert response.metadata["evidence_snapshot_consistency"]["deterministic_hint_matches_request"] is False
    assert response.metadata["evidence_snapshot_consistency"]["final_action_plan_matches_snapshot"] is False
    assert (
        response.metadata["evidence_snapshot_consistency"]["final_action_plan_superseded_by_snapshot"]
        == "snapshot-new"
    )
    assert trace["evidence_snapshot_consistency"]["deterministic_hint_matches_request"] is False
    assert trace["final_action_plan"]["evidence_snapshot_id"] == "snapshot-new"


def test_harness_trace_uses_empty_snapshot_ids_when_snapshot_is_absent() -> None:
    request = TutorRequest(
        request_id="legacy-cycle",
        actor="learner",
        intent="help",
        message="help",
        context={
            "candidate_steps": [{"step_id": "S03", "source": "deterministic"}],
            "deterministic_step_hint": {"inferred_step_id": "S03", "missing_conditions": []},
        },
    )
    response = TutorResponse(
        status="ok",
        message="Start APU.",
        actions=[{"type": "overlay", "target": "apu_switch"}],
        explanations=["Start APU."],
        metadata={
            "generation_mode": "model",
            "harness_action_plan": {"step_id": "S03", "targets": ["apu_switch"], "source": "model"},
            "help_response": {"next": {"step_id": "S03"}, "overlay": {"targets": ["apu_switch"]}},
        },
    )

    trace = _build_harness_trace_metadata(
        request=request,
        response=response,
        vision_selection=HelpCycleVisionSelection(
            status="vision_not_required",
            observation_ref=None,
            observation_seq=None,
            observation_t_wall_s=None,
            observation_t_wall_ms=None,
            trigger_wall_ms=1000,
            sync_window_ms=250,
            vision_used=False,
            frame_id=None,
            sync_status="not_required",
            sync_delta_ms=None,
            frame_stale=None,
            frame_ids=[],
            selected_frames=[],
            pre_trigger_frame=None,
            trigger_frame=None,
            sync_miss_reason=None,
        ),
        vision_fact_context={"status": "vision_not_required", "vision_fact_summary": {}, "vision_facts": []},
    )

    assert trace["snapshot_ids"] == {}
    assert response.metadata["snapshot_ids"] == {}
    assert "evidence_snapshot_id" not in trace["final_action_plan"]


def test_live_loop_telemetry_window_uses_canonical_vars_not_raw_delta_keys(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_fcs_bit_window.jsonl"
    frame1 = _bios_frame(1, 10.0, apu_switch=0)
    frame2 = _bios_frame(2, 11.0, apu_switch=0)
    frame1["bios"]["FCS_BIT_SW"] = 0
    frame1["delta"]["FCS_BIT_SW"] = 0
    frame2["bios"]["FCS_BIT_SW"] = 1
    frame2["delta"]["FCS_BIT_SW"] = 1
    _write_replay(replay_path, [frame1, frame2])

    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=model,
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="en",
    )
    try:
        loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    request = model.calls[0]["request"]
    assert any(
        candidate["source"] == "telemetry_window" and candidate["step_id"] == "S19"
        for candidate in request.context["candidate_steps"]
    )
    digest = request.context["state_harness"]["telemetry_window_digest"]
    assert any(item["var"] == "fcs_bit_switch_up" for item in digest["changed_vars"])


def test_tutor_request_event_sanitizes_telemetry_window_digest_nested_values() -> None:
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "state_harness": {
                "telemetry_window_digest": {
                    "frame_count": 2,
                    "latest_seq": 9,
                    "unexpected": "drop-me",
                    "changed_vars": [
                        {
                            "var": "debug_text",
                            "first_value": "A" * 200,
                            "last_value": "B" * 200,
                            "extra": "drop-me",
                        }
                    ],
                    "contradictions": ["recent telemetry transition conflicts with current blocked gate"],
                }
            }
        },
    )

    payload = _sanitize_request_payload_for_event(request)
    digest = payload["context"]["state_harness"]["telemetry_window_digest"]

    assert digest["frame_count"] == 2
    assert "unexpected" not in digest
    assert "extra" not in digest["changed_vars"][0]
    assert len(digest["changed_vars"][0]["first_value"]) <= 83
    assert len(digest["changed_vars"][0]["last_value"]) <= 83


def test_telemetry_window_signature_ignores_seq_churn_but_keeps_semantic_changes() -> None:
    base = {
        "telemetry_window_digest": {
            "latest_seq": 1,
            "latest_t_wall": 10.0,
            "changed_vars": [
                {
                    "var": "fcs_bit_switch_up",
                    "first_value": False,
                    "last_value": True,
                    "transition_count": 1,
                    "latest_transition_age_s": 0.0,
                }
            ],
            "contradictions": [],
        }
    }
    churned = {
        "telemetry_window_digest": {
            **base["telemetry_window_digest"],
            "latest_seq": 99,
            "latest_t_wall": 99.0,
        }
    }
    changed = {
        "telemetry_window_digest": {
            **base["telemetry_window_digest"],
            "changed_vars": [
                {
                    "var": "fcs_bit_switch_up",
                    "first_value": True,
                    "last_value": False,
                    "transition_count": 1,
                    "latest_transition_age_s": 0.0,
                }
            ],
        }
    }

    assert _telemetry_window_signature(base) == _telemetry_window_signature(churned)
    assert _telemetry_window_signature(base) != _telemetry_window_signature(changed)


def test_live_loop_sends_final_tutor_message_to_dcs_text_channel(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_one.jsonl"
    _write_replay(replay_path, [
        _bios_frame_fire_test_b_pre(0, 9.0),
        _bios_frame(1, 10.0, apu_switch=0),
    ])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    tutor_text_sender = RecordingTutorTextSender()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        tutor_text_sender=tutor_text_sender,
        tutor_text_display_time_s=9.0,
        tutor_text_clear_view=True,
        cooldown_s=5.0,
        lang="zh",
        event_sink=events.append,
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    assert stats["help_cycles"] == 1
    assert tutor_text_sender.calls == [
        {
            "text": "当前处于 S03。请左键点击 APU 开关将其拨到 ON，然后等待绿色 APU READY 灯亮起。",
            "display_time_s": 9.0,
            "clear_view": True,
            "expect_ack": True,
            "cmd_id": None,
        }
    ]
    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    assert tutor_response_payload["metadata"]["dcs_tutor_text"]["status"] == "ok"


def test_live_loop_records_tutor_text_failure_in_response_metadata(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_one.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    tutor_text_sender = RecordingTutorTextSender(
        result={
            "status": "failed",
            "failure_class": "ack_timeout",
            "reason": "timed out waiting for DCS tutor text ack",
        }
    )
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        tutor_text_sender=tutor_text_sender,
        cooldown_s=5.0,
        lang="zh",
        event_sink=events.append,
    )
    try:
        stats = loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert stats["help_cycles"] == 1
    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    assert tutor_response_payload["metadata"]["dcs_tutor_text"]["status"] == "failed"
    assert tutor_response_payload["metadata"]["dcs_tutor_text"]["failure_class"] == "ack_timeout"


def _make_tutor_text_loop(*, sender=None, lang="zh", display_time_s=12.0, clear_view=False):
    loop = LiveDcsTutorLoop.__new__(LiveDcsTutorLoop)
    loop.tutor_text_sender = sender
    loop.lang = lang
    loop.tutor_text_display_time_s = display_time_s
    loop.tutor_text_clear_view = clear_view
    return loop


def _make_tutor_response(*, message=None, status="ok", in_reply_to="req-1"):
    return TutorResponse(
        status=status,
        in_reply_to=in_reply_to,
        message=message,
        actions=[],
        explanations=[],
        metadata={},
    )


def test_send_tutor_text_skips_when_message_is_none() -> None:
    sender = RecordingTutorTextSender()
    response = _make_tutor_response(message=None)
    loop = _make_tutor_text_loop(sender=sender)

    loop._send_tutor_text(response)

    assert response.metadata["dcs_tutor_text"]["status"] == "skipped"
    assert response.metadata["dcs_tutor_text"]["reason"] == "empty_message"
    assert sender.calls == []


def test_send_tutor_text_skips_when_message_is_empty_string() -> None:
    sender = RecordingTutorTextSender()
    response = _make_tutor_response(message="")
    loop = _make_tutor_text_loop(sender=sender)

    loop._send_tutor_text(response)

    assert response.metadata["dcs_tutor_text"]["status"] == "skipped"
    assert response.metadata["dcs_tutor_text"]["reason"] == "empty_message"
    assert sender.calls == []


def test_send_tutor_text_skips_when_message_is_whitespace_only() -> None:
    sender = RecordingTutorTextSender()
    response = _make_tutor_response(message="   ")
    loop = _make_tutor_text_loop(sender=sender)

    loop._send_tutor_text(response)

    assert response.metadata["dcs_tutor_text"]["status"] == "skipped"
    assert response.metadata["dcs_tutor_text"]["reason"] == "empty_message"
    assert sender.calls == []


def test_send_tutor_text_records_sender_unavailable_when_none() -> None:
    response = _make_tutor_response(message="Hello")
    loop = _make_tutor_text_loop(sender=None)

    loop._send_tutor_text(response)

    assert response.metadata["dcs_tutor_text"]["status"] == "skipped"
    assert response.metadata["dcs_tutor_text"]["reason"] == "sender_unavailable"


def test_send_tutor_text_handles_non_mapping_sender_result() -> None:
    class NonMappingSender:
        def send_text(self, text, **kwargs):
            return ["not", "a", "mapping"]

    response = _make_tutor_response(message="Hello world")
    loop = _make_tutor_text_loop(sender=NonMappingSender())

    loop._send_tutor_text(response)

    assert response.metadata["dcs_tutor_text"]["status"] == "failed"
    assert response.metadata["dcs_tutor_text"]["failure_class"] == "invalid_sender_result"


def test_send_tutor_text_catches_sender_exception() -> None:
    class ThrowingSender:
        def send_text(self, text, **kwargs):
            raise RuntimeError("socket unexpectedly closed")

    response = _make_tutor_response(message="Hello world")
    loop = _make_tutor_text_loop(sender=ThrowingSender())

    loop._send_tutor_text(response)

    assert response.metadata["dcs_tutor_text"]["status"] == "failed"
    assert response.metadata["dcs_tutor_text"]["failure_class"] == "sender_exception"
    assert "RuntimeError" in response.metadata["dcs_tutor_text"]["reason"]


def test_live_auto_help_uses_help_action_wall_time_for_live_vision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_one.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class _Notifier:
        def __init__(self) -> None:
            self.calls = 0

        def notify_help(self) -> None:
            self.calls += 1

    notifier = _Notifier()
    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-live-trigger-time",
        vision_mode="live",
    )
    monkeypatch.setattr("live_dcs.time.time", lambda: 1772872445.25)
    try:
        stats = loop.run(max_frames=1, auto_help_on_first_frame=True, help_capture_notifier=notifier)
    finally:
        loop.close()

    assert stats["help_cycles"] == 1
    assert notifier.calls == 1
    request = model.calls[0]["request"]
    assert request.context["vision"]["trigger_wall_ms"] == 1772872445250


def test_live_loop_records_grounding_snippet_ids_when_index_available(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_grounding.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    doc = tmp_path / "fa18c_startup_master.md"
    doc.write_text(
        "# S02\nF/A-18C Cold Start MVP subset checklist.\nComplete FIRE TEST A and FIRE TEST B before APU start. Use the fire_test_switch.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    build_index([str(doc)], str(index_path))

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_index_path=index_path,
        rag_top_k=3,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert req_meta["grounding_missing"] is False
    assert req_meta["grounding_reason"] is None
    assert req_meta["grounding_snippet_ids"] == ["fa18c_startup_master_0"]
    rag_topk = tutor_request_payload["context"]["rag_topk"]
    assert rag_topk
    assert rag_topk[0]["snippet_id"]
    assert rag_topk[0]["doc_id"] == "fa18c_startup_master"

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    prompt_build = tutor_response_payload["metadata"]["prompt_build"]
    assert prompt_build["grounding_missing"] is False
    assert prompt_build["rag_snippet_ids"] == req_meta["grounding_snippet_ids"]


def test_live_loop_redacts_user_message_and_rag_snippet_text_in_request_event(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_request_redaction.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    doc = tmp_path / "manual.md"
    doc.write_text(
        "# S02\nConnect to https://api.example.com:8443/v1 with api_key=sk-test before APU start.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    build_index([str(doc)], str(index_path))

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_index_path=index_path,
        rag_top_k=3,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    assert tutor_request_payload["message"] == "[REDACTED_USER_MESSAGE]"
    assert tutor_request_payload["context"]["grounding_query"] == "[REDACTED_GROUNDING_QUERY]"
    rag_topk = tutor_request_payload["context"]["rag_topk"]
    assert rag_topk
    assert "snippet" not in rag_topk[0]
    assert rag_topk[0]["snippet_id"] == "manual_0"


def test_live_loop_redacts_help_response_quotes_in_response_event(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_response_redaction.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    help_response = tutor_response_payload["metadata"]["help_response"]
    evidence = help_response["overlay"]["evidence"]
    assert evidence[0]["quote"] == "[REDACTED_SOURCE_QUOTE]"


def test_sanitize_request_payload_for_event_preserves_empty_string_message() -> None:
    request = TutorRequest(message="", metadata={"help_cycle_id": "cycle-1"})

    payload = _sanitize_request_payload_for_event(request)

    assert payload["message"] == ""


def test_sanitize_request_payload_for_event_does_not_call_to_dict(monkeypatch) -> None:
    request = TutorRequest(message="help", context={"grounding_query": "q"})

    monkeypatch.setattr(TutorRequest, "to_dict", lambda self: (_ for _ in ()).throw(AssertionError("unexpected")))

    payload = _sanitize_request_payload_for_event(request)

    assert payload["message"] == "[REDACTED_USER_MESSAGE]"
    assert payload["context"]["grounding_query"] == "[REDACTED_GROUNDING_QUERY]"


def test_sanitize_request_payload_for_event_summarizes_vision_context() -> None:
    request = TutorRequest(
        context={
            "vision": {
                "status": "available",
                "observation_ref": "obs-1",
                "frame_id": "frame-1",
                "frame_ids": ["frame-1", "frame-0"],
                "sync_status": "matched_past",
                "sync_delta_ms": -50,
                "selected_frames": [
                    {
                        "frame_id": "frame-1",
                        "image_uri": "file:///tmp/frame-1.png",
                        "source_image_path": "/tmp/frame-1.png",
                    }
                ],
                "trigger_frame": {
                    "frame_id": "frame-1",
                    "image_uri": "file:///tmp/frame-1.png",
                },
                "pre_trigger_frame": {
                    "frame_id": "frame-0",
                    "source_image_path": "/tmp/frame-0.png",
                },
            }
        }
    )

    payload = _sanitize_request_payload_for_event(request)

    assert payload["context"]["vision"] == {
        "status": "available",
        "observation_ref": "obs-1",
        "frame_id": "frame-1",
        "sync_status": "matched_past",
        "sync_delta_ms": -50,
        "frame_ids": ["frame-1", "frame-0"],
    }


def test_sanitize_request_payload_for_event_drops_unallowlisted_context_fields() -> None:
    request = TutorRequest(
        context={
            "scenario_profile": "carrier",
            "grounding_reason": "index_missing",
            "grounding_query": "q",
            "rag_topk": [{"snippet_id": "manual_0", "snippet": "secret"}],
            "vision_fact_summary": {
                "status": "available",
                "seen_fact_ids": ["fact-1"],
                "source_image_path": "/tmp/frame.png",
            },
            "vars": {"api_key": "secret"},
            "recent_deltas": [{"ui_target": "apu_switch"}],
            "unexpected_nested": {"token": "secret"},
        }
    )

    payload = _sanitize_request_payload_for_event(request)

    assert payload["context"] == {
        "scenario_profile": "carrier",
        "grounding_reason": "index_missing",
        "grounding_query": "[REDACTED_GROUNDING_QUERY]",
        "rag_topk": [{"snippet_id": "manual_0"}],
        "vision_fact_summary": {
            "status": "available",
            "seen_fact_ids": ["fact-1"],
        },
    }


def test_sanitize_response_payload_for_event_drops_raw_llm_text_fields() -> None:
    response = TutorResponse(
        metadata={
            "raw_llm_text": "{\"secret\":true}",
            "raw_llm_text_attempts": ["{\"secret\":true}"],
            "help_response": {
                "diagnosis": {"step_id": "S02", "error_category": "OM"},
                "next": {"step_id": "S03"},
                "overlay": {"targets": [], "evidence": []},
                "explanations": ["ok"],
                "confidence": 0.8,
            },
            "model_raw_help_response": {
                "diagnosis": {"step_id": "S02", "error_category": "OM"},
                "next": {"step_id": "S03"},
                "overlay": {"targets": [], "evidence": []},
                "explanations": ["raw token=abc123"],
                "confidence": 0.8,
            },
            "final_public_response": {
                "message": "Visit https://api.example.com/v1/chat.",
                "explanations": ["Set token=abc123."],
                "next": {"step_id": "S02"},
            },
        }
    )

    payload = _sanitize_response_payload_for_event(response, lang="en")

    assert "raw_llm_text" not in payload["metadata"]
    assert "raw_llm_text_attempts" not in payload["metadata"]
    assert payload["metadata"]["model_raw_help_response"]["explanations"] == ["raw token=[REDACTED_SECRET]"]
    assert payload["metadata"]["final_public_response"]["message"] == "Visit [REDACTED_URL]."
    assert payload["metadata"]["final_public_response"]["explanations"] == ["Set token=[REDACTED_SECRET]."]


def test_sanitize_response_payload_for_event_summarizes_prompt_build() -> None:
    response = TutorResponse(
        metadata={
            "prompt_build": {
                "grounding_missing": False,
                "rag_snippet_ids": ["manual_0"],
                "allowed_evidence_refs": ["RAG.manual_0"],
                "EVIDENCE_SOURCES": {
                    "RAG_SNIPPETS": [{"id": "manual_0", "snippet": "secret token=abc"}],
                },
                "full_prompt": "system: api.example.com/v1?token=abc",
            }
        }
    )

    payload = _sanitize_response_payload_for_event(response, lang="en")

    assert payload["metadata"]["prompt_build"] == {
        "allowed_evidence_refs": ["RAG.manual_0"],
        "rag_snippet_ids": ["manual_0"],
        "grounding_missing": False,
    }


def test_sanitize_response_payload_for_event_does_not_call_to_dict(monkeypatch) -> None:
    response = TutorResponse(
        message="Visit https://api.example.com/v1/chat.",
        explanations=["Set token=abc123."],
        actions=[{"type": "overlay", "target": "apu_switch"}],
        metadata={"error": "connect api.example.com/v1 failed"},
    )

    monkeypatch.setattr(TutorResponse, "to_dict", lambda self: (_ for _ in ()).throw(AssertionError("unexpected")))

    payload = _sanitize_response_payload_for_event(response, lang="en")

    assert payload["message"] == "Visit [REDACTED_URL]."
    assert payload["explanations"] == ["Set token=[REDACTED_SECRET]."]
    assert payload["actions"] == [{"type": "overlay", "target": "apu_switch"}]


def test_live_loop_help_cycle_id_links_request_response_and_overlay_events(monkeypatch, tmp_path: Path) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)

    replay_path = tmp_path / "bios_help_cycle_trace.jsonl"
    _write_replay(
        replay_path,
        [
            _bios_frame(1, 10.0, apu_switch=0),
            _bios_frame(2, 10.4, apu_switch=1),
            _bios_frame(3, 10.8, apu_switch=0),
        ],
    )

    source = ReplayBiosReceiver(replay_path)
    model = SequencedGenerationModeModel()
    events = []
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False, event_sink=events.append)
    executor = OverlayActionExecutor(sender=sender, event_sink=events.append, max_targets=1)
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=0.0,
        lang="en",
    )
    try:
        stats = loop.run(max_frames=3, auto_help_every_n_frames=1)
    finally:
        loop.close()

    assert stats["help_cycles"] == 3
    grouped: dict[str, list[Any]] = {}
    for event in events:
        if event.kind not in {
            "tutor_request",
            "tutor_response",
            "overlay_requested",
            "overlay_applied",
            "overlay_failed",
            "overlay_rejected",
            "overlay_dry_run",
        }:
            continue
        help_cycle_id = event.metadata.get("help_cycle_id")
        assert isinstance(help_cycle_id, str) and help_cycle_id
        grouped.setdefault(help_cycle_id, []).append(event)

    assert len(grouped) == 3
    observed_generation_modes: list[str] = []
    overlay_event_kinds: set[str] = set()
    for help_cycle_id, cycle_events in grouped.items():
        request_events = [event for event in cycle_events if event.kind == "tutor_request"]
        response_events = [event for event in cycle_events if event.kind == "tutor_response"]
        assert len(request_events) == 1
        assert len(response_events) == 1
        request_payload = request_events[0].payload
        response_payload = response_events[0].payload
        assert request_payload["metadata"]["help_cycle_id"] == help_cycle_id
        assert response_payload["metadata"]["help_cycle_id"] == help_cycle_id
        generation_mode = response_payload["metadata"]["generation_mode"]
        observed_generation_modes.append(generation_mode)
        harness_trace = response_payload["metadata"]["harness_trace"]
        assert harness_trace["schema_version"] == "v1"
        assert harness_trace["final_overlay_targets"] == [
            action["target"]
            for action in response_payload["actions"]
            if isinstance(action, dict) and isinstance(action.get("target"), str)
        ]
        assert harness_trace["validator_result"]["fallback_overlay_reason"]
        assert response_payload["metadata"]["message_category"] == harness_trace["message_category"]
        assert response_events[0].metadata["generation_mode"] == generation_mode
        for event in cycle_events:
            if event.kind.startswith("overlay_"):
                overlay_event_kinds.add(event.kind)
                assert event.payload["help_cycle_id"] == help_cycle_id
                assert event.metadata["generation_mode"] == generation_mode
                assert event.metadata["final_overlay_targets"]
                assert "harness_trace" not in event.metadata

    assert observed_generation_modes == ["model", "repair", "fallback"]
    assert "overlay_requested" in overlay_event_kinds


def test_live_loop_marks_grounding_missing_when_index_absent(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_no_index.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_index_path=tmp_path / "missing_index.json",
        rag_top_k=3,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert req_meta["grounding_missing"] is True
    assert req_meta["grounding_reason"] == "index_missing"
    assert req_meta["grounding_snippet_ids"] == []
    assert tutor_request_payload["context"]["grounding_reason"] == "index_missing"
    assert tutor_request_payload["context"]["grounding_query"] == "[REDACTED_GROUNDING_QUERY]"
    assert tutor_request_payload["context"]["rag_topk"] == []

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    prompt_build = tutor_response_payload["metadata"]["prompt_build"]
    assert prompt_build["grounding_missing"] is True
    assert prompt_build["rag_snippet_ids"] == []


def test_live_loop_rejects_missing_policy_in_cold_start_production_mode(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_policy_missing.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    missing_policy_path = tmp_path / "dir with spaces" / "missing_knowledge_source_policy.yaml"

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    try:
        with pytest.raises(ValueError) as exc_info:
            loop = LiveDcsTutorLoop(
                source=source,
                model=model,
                action_executor=executor,
                cooldown_s=5.0,
                lang="en",
                rag_top_k=0,
                cold_start_production=True,
                knowledge_source_policy_path=missing_policy_path,
            )
            loop.close()
        message = str(exc_info.value)
        assert "cold-start production requires valid knowledge source policy" in message
        assert "knowledge source policy read failed" in message
        assert "missing_knowledge_source_policy.yaml" in message
        assert str(missing_policy_path) not in message
    finally:
        source.close()


def test_live_loop_rejects_missing_default_policy_in_cold_start_production_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_policy_default_missing.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    monkeypatch.setattr("live_dcs._default_knowledge_source_policy_path", lambda: tmp_path / "missing_default.yaml")

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    try:
        with pytest.raises(ValueError) as exc_info:
            loop = LiveDcsTutorLoop(
                source=source,
                model=model,
                action_executor=executor,
                cooldown_s=5.0,
                lang="en",
                rag_top_k=0,
                cold_start_production=True,
            )
            loop.close()
        message = str(exc_info.value)
        assert "default policy file" in message
        assert "not found" in message
        assert "Provide --knowledge-source-policy explicitly" in message
        assert str(tmp_path / "missing_default.yaml") not in message
    finally:
        source.close()


def test_live_loop_normalizes_relative_knowledge_index_path(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_relative_index.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    (tmp_path / "index.json").write_text('{"documents":[]}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        rag_top_k=0,
        knowledge_index_path=Path("index.json"),
    )
    try:
        assert loop.knowledge_index_path == (tmp_path / "index.json").resolve()
    finally:
        loop.close()


def test_live_loop_cold_start_production_prints_policy_summary(tmp_path: Path, capsys) -> None:
    replay_path = tmp_path / "bios_policy_summary.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        rag_top_k=0,
        cold_start_production=True,
        knowledge_source_policy_path=_default_policy_path(),
    )
    try:
        loop.run(max_frames=1)
    finally:
        loop.close()
    out = capsys.readouterr().out
    assert "当前仅使用 cold-start 白名单块" in out


def test_live_loop_applies_policy_filter_in_cold_start_production_mode(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_policy_filter.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=PolicyMixedKnowledge(),
        rag_top_k=2,
        cold_start_production=True,
        knowledge_source_policy_path=_default_policy_path(),
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    rag_topk = tutor_request_payload["context"]["rag_topk"]
    assert len(rag_topk) == 1
    assert rag_topk[0]["doc_id"] == "fa18c_startup_master"
    assert rag_topk[0]["snippet_id"] == "fa18c_startup_master_1"
    assert req_meta["grounding_policy_id"] == "fa18c_cold_start_whitelist_v1"
    assert req_meta["grounding_policy_version"] == "v2"
    assert req_meta["grounding_policy_filtered_out_count"] == 1
    assert req_meta["source_chunk_refs"] == ["fa18c_startup_master/fa18c_startup_master_1:1-56"]


def test_live_loop_applies_policy_filter_when_policy_path_provided_without_cold_start(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_policy_filter_non_cold_start.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=PolicyMixedKnowledge(),
        rag_top_k=2,
        cold_start_production=False,
        knowledge_source_policy_path=_default_policy_path(),
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    rag_topk = tutor_request_payload["context"]["rag_topk"]
    assert len(rag_topk) == 1
    assert rag_topk[0]["doc_id"] == "fa18c_startup_master"
    assert req_meta["grounding_policy_id"] == "fa18c_cold_start_whitelist_v1"
    assert req_meta["grounding_policy_version"] == "v2"
    assert req_meta["grounding_policy_filtered_out_count"] == 1
    assert req_meta["source_chunk_refs"] == ["fa18c_startup_master/fa18c_startup_master_1:1-56"]


def test_live_loop_marks_policy_filtered_all_as_grounding_missing_and_logs_chunk_refs(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_policy_filtered_all.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=PolicyRejectAllKnowledge(),
        rag_top_k=2,
        cold_start_production=False,
        knowledge_source_policy_path=_default_policy_path(),
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert tutor_request_payload["context"]["rag_topk"] == []
    assert req_meta["grounding_missing"] is True
    assert req_meta["grounding_reason"] == "policy_filtered_all"
    assert req_meta["source_chunk_refs"] == []
    assert req_meta["grounding_policy_filtered_out_count"] == 1


def test_live_loop_prefers_chunk_id_when_building_source_chunk_refs_for_policy_filtered_snippets(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_policy_chunk_id.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=PolicyMixedKnowledgeWithDistinctChunkId(),
        rag_top_k=2,
        cold_start_production=False,
        knowledge_source_policy_path=_default_policy_path(),
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert req_meta["source_chunk_refs"] == ["fa18c_startup_master/fa18c_startup_master_1:1-56"]


def test_build_source_chunk_ref_falls_back_to_chunk_without_line_range() -> None:
    assert build_source_chunk_ref(
        {
            "doc_id": "fa18c_startup_master",
            "snippet_id": "custom_alias",
            "chunk_id": "fa18c_startup_master_1",
        }
    ) == "fa18c_startup_master/fa18c_startup_master_1"


def test_sanitize_policy_error_handles_unc_paths_with_spaces() -> None:
    unc_path = r"\\server\share\folder with spaces\missing_policy.yaml"
    message = f"knowledge source policy read failed: {unc_path}"
    sanitized = _sanitize_policy_error_for_user(message, path_hints=[unc_path])
    assert "missing_policy.yaml" in sanitized
    assert unc_path not in sanitized


def test_live_loop_surfaces_index_load_error_type_in_grounding_metadata(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_bad_index.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    bad_index = tmp_path / "bad_index.json"
    bad_index.write_text("{invalid json", encoding="utf-8")

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_index_path=bad_index,
        rag_top_k=3,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert req_meta["grounding_missing"] is True
    assert req_meta["grounding_reason"] == "index_load_error"
    assert isinstance(req_meta["grounding_error_type"], str) and req_meta["grounding_error_type"]


def test_live_loop_accepts_query_only_knowledge_port(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_query_only_knowledge.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    knowledge = QueryOnlyKnowledge()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=knowledge,
        rag_top_k=2,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(knowledge.calls) == 1
    assert knowledge.calls[0]["k"] == 2
    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert req_meta["grounding_missing"] is False
    assert req_meta["grounding_reason"] is None
    assert req_meta["grounding_snippet_ids"] == ["manual_s02_1"]
    assert "grounding_index_path" not in req_meta
    assert "grounding_query" not in req_meta
    assert tutor_request_payload["context"]["grounding_reason"] is None
    assert tutor_request_payload["context"]["grounding_query"] == "[REDACTED_GROUNDING_QUERY]"
    assert tutor_request_payload["context"]["rag_topk"][0]["snippet_id"] == "manual_s02_1"


def test_live_loop_degrades_when_knowledge_adapter_raises(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_knowledge_error.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=FailingKnowledge(),
        rag_top_k=2,
    )
    try:
        stats = loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert stats["help_cycles"] == 1
    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert req_meta["grounding_missing"] is True
    assert req_meta["grounding_reason"] == "knowledge_retrieve_error"
    assert req_meta["grounding_error_type"] == "RuntimeError"
    assert tutor_request_payload["context"]["rag_topk"] == []
    assert tutor_request_payload["context"]["grounding_reason"] == "knowledge_retrieve_error"


def test_live_loop_uses_effective_grounding_reason_when_no_snippets_returned(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_no_snippets.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=EmptyKnowledge(),
        rag_top_k=2,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert req_meta["grounding_missing"] is True
    assert req_meta["grounding_reason"] == "no_rag_snippets"
    assert req_meta["grounding_missing_requested"] is False
    assert req_meta["grounding_reason_requested"] is None


def test_live_loop_normalizes_query_only_snippets_to_json_safe_scalars(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_query_snippet_normalize.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=NonSerializableKnowledge(),
        rag_top_k=2,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    rag_topk = tutor_request_payload["context"]["rag_topk"]
    assert len(rag_topk) == 1
    first = rag_topk[0]
    assert set(first.keys()) <= {"doc_id", "section", "page_or_heading", "snippet_id", "score"}
    assert isinstance(first["doc_id"], str)
    assert isinstance(first["section"], str)
    assert isinstance(first["page_or_heading"], str)
    assert first["snippet_id"] == "snippet_0"
    assert req_meta["grounding_snippet_ids"] == ["snippet_0"]
    assert req_meta["grounding_reason"] is None
    json.dumps(tutor_request_payload, ensure_ascii=False)


def test_live_loop_prefers_retrieve_with_meta_protocol_when_available(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_meta_knowledge.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    knowledge = MetaKnowledge()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=knowledge,
        rag_top_k=2,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(knowledge.calls) == 1
    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    assert req_meta["grounding_missing"] is False
    assert req_meta["grounding_cache_hit"] is True
    assert "grounding_index_path" not in req_meta
    assert req_meta["grounding_snippet_ids"] == ["meta_s02_1"]


def test_live_loop_normalizes_retrieve_with_meta_payloads_to_json_safe_scalars(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_meta_nonserializable.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        event_sink=events.append,
        cooldown_s=5.0,
        lang="en",
        knowledge_adapter=NonSerializableMetaKnowledge(),
        rag_top_k=2,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_request_payload = next(event.payload for event in events if event.kind == "tutor_request")
    req_meta = tutor_request_payload["metadata"]
    rag_topk = tutor_request_payload["context"]["rag_topk"]
    assert len(rag_topk) == 1
    first = rag_topk[0]
    assert set(first.keys()) <= {"doc_id", "section", "page_or_heading", "snippet_id", "score"}
    assert first["snippet_id"] == "snippet_0"
    assert isinstance(first["doc_id"], str)
    assert isinstance(first["section"], str)
    assert isinstance(first["page_or_heading"], str)
    assert "grounding_index_path" not in req_meta
    assert req_meta["grounding_reason_requested"] is None
    assert isinstance(req_meta["grounding_error_type"], str)
    json.dumps(tutor_request_payload, ensure_ascii=False)


def test_live_loop_does_not_initialize_local_knowledge_when_rag_disabled(tmp_path: Path, monkeypatch) -> None:
    replay_path = tmp_path / "bios_rag_disabled.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()

    def _raise_local_knowledge(*_args, **_kwargs):
        raise AssertionError("LocalKnowledgeAdapter should not be initialized when rag_top_k=0")

    monkeypatch.setattr("live_dcs.LocalKnowledgeAdapter", _raise_local_knowledge)
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        rag_top_k=0,
    )
    try:
        stats = loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert stats["help_cycles"] == 1


def test_live_loop_reuses_cached_result_for_same_state_within_cooldown(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_two.jsonl"
    _write_replay(
        replay_path,
        [
            _bios_frame(1, 10.0, apu_switch=0),
            _bios_frame(2, 10.1, apu_switch=0),
        ],
    )

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=30.0,
        lang="en",
        event_sink=events.append,
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=1)
    finally:
        loop.close()

    assert stats["frames"] == 2
    assert stats["help_cycles"] == 2
    assert stats["model_calls"] == 1
    assert stats["cache_hits"] == 1
    assert len(model.calls) == 1
    assert len(executor.calls) == 2
    tutor_request_payloads = [event.payload for event in events if event.kind == "tutor_request"]
    assert len(tutor_request_payloads) == 2
    tutor_response_payloads = [event.payload for event in events if event.kind == "tutor_response"]
    assert len(tutor_response_payloads) == 2
    assert tutor_response_payloads[0]["response_id"] != tutor_response_payloads[1]["response_id"]
    first_request_meta = tutor_request_payloads[0]["metadata"]
    second_request_meta = tutor_request_payloads[1]["metadata"]
    first_response_meta = tutor_response_payloads[0]["metadata"]
    second_response_meta = tutor_response_payloads[1]["metadata"]
    assert first_response_meta["generation_prompt_hash"] == first_request_meta["prompt_hash"]
    assert second_response_meta["generation_prompt_hash"] == first_request_meta["prompt_hash"]
    assert second_response_meta["prompt_hash"] == first_request_meta["prompt_hash"]
    assert second_response_meta["request_prompt_hash"] == second_request_meta["prompt_hash"]
    assert second_response_meta["request_prompt_tokens_est"] == second_request_meta["prompt_tokens_est"]
    assert second_response_meta["request_prompt_trimmed"] == second_request_meta["prompt_trimmed"]
    assert first_response_meta["harness_trace"]["schema_version"] == "v1"
    assert second_response_meta["harness_trace"]["schema_version"] == "v1"
    assert first_response_meta["help_cycle_id"] != second_response_meta["help_cycle_id"]
    assert first_response_meta["harness_trace"]["final_overlay_targets"] == ["apu_switch"]
    assert second_response_meta["harness_trace"]["final_overlay_targets"] == ["apu_switch"]
    assert executor.calls[0][0]["help_cycle_id"] == first_response_meta["help_cycle_id"]
    assert executor.calls[1][0]["help_cycle_id"] == second_response_meta["help_cycle_id"]
    assert "harness_trace" not in executor.calls[0][0]
    assert "harness_trace" not in executor.calls[1][0]


def test_live_loop_filters_help_overlay_targets_by_request_allowlist(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_allowlist_filter.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    ui_targets:\n"
        "      - apu_switch\n",
        encoding="utf-8",
    )

    source = ReplayBiosReceiver(replay_path)
    model = OutOfAllowlistTargetModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        pack_path=pack,
        ui_map_path=Path(_default_pack_path()).parent / "ui_map.yaml",
        event_sink=events.append,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    assert executor.calls[0][0]["target"] == "apu_switch"
    tutor_response_payloads = [event.payload for event in events if event.kind == "tutor_response"]
    assert len(tutor_response_payloads) == 1
    assert tutor_response_payloads[0]["metadata"].get("vision_fallback_reason") is None
    response_mapping = tutor_response_payloads[0]["metadata"]["response_mapping"]
    assert response_mapping["rejected_targets_by_request_allowlist"] == ["battery_switch"]
    assert "overlay_target_not_in_request_allowlist" in response_mapping["mapping_errors"]
    assert tutor_response_payloads[0]["metadata"]["response_mapping_failure_codes"] == [ALLOWLIST_FAIL]
    assert tutor_response_payloads[0]["metadata"]["response_mapping_failure_code"] == ALLOWLIST_FAIL
    assert tutor_response_payloads[0]["metadata"]["response_mapping_failure_stage"] == "response_mapping"
    assert tutor_response_payloads[0]["metadata"]["fallback_overlay_used"] is True
    assert tutor_response_payloads[0]["metadata"]["fallback_overlay_reason"] == "validator_repair"
    assert tutor_response_payloads[0]["metadata"]["harness_validator_fallback_reason"].startswith("deterministic_step:")


def test_resolve_step_overlay_allowlist_treats_tuple_hint_blockers_as_hard() -> None:
    allowlist = _resolve_step_overlay_allowlist(
        "S03",
        step_fallback_profiles={
            "S03": {
                "ui_targets": ["apu_switch", "battery_switch"],
            }
        },
        overlay_allowset={"apu_switch", "battery_switch"},
        default_allowlist=["battery_switch", "apu_switch"],
        deterministic_hint={
            "missing_conditions": (),
            "gate_blockers": ({"ref": "GATES.S03.precondition", "reason": "blocked"},),
        },
    )

    assert allowlist == ["apu_switch", "battery_switch"]


def test_resolve_step_overlay_allowlist_uses_tuple_recent_ui_targets_for_partial_observability() -> None:
    allowlist = _resolve_step_overlay_allowlist(
        "S03",
        step_fallback_profiles={
            "S03": {
                "ui_targets": [],
            }
        },
        overlay_allowset={"apu_switch", "battery_switch"},
        default_allowlist=["battery_switch", "apu_switch"],
        deterministic_hint={
            "observability_status": "partial",
            "recent_ui_targets": ("apu_switch",),
        },
    )

    assert allowlist == ["apu_switch"]


def test_resolve_step_overlay_allowlist_returns_empty_for_overlay_disabled_step() -> None:
    allowlist = _resolve_step_overlay_allowlist(
        "S05",
        step_fallback_profiles={
            "S05": {
                "ui_targets": [],
                "overlay_enabled": False,
            }
        },
        overlay_allowset={"apu_switch", "battery_switch"},
        default_allowlist=["battery_switch", "apu_switch"],
        deterministic_hint={
            "observability_status": "observable",
            "recent_ui_targets": ("apu_switch",),
        },
    )

    assert allowlist == []


def test_safe_fallback_overlay_respects_overlay_disabled_with_declared_targets(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_overlay_disabled.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    observability: observable\n"
        "    overlay_enabled: false\n"
        "    evidence_requirements: [delta]\n"
        "    ui_targets: [apu_switch]\n"
        "precondition_gates:\n"
        "  S01: []\n"
        "completion_gates:\n"
        "  S01: []\n",
        encoding="utf-8",
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
        pack_path=pack,
        ui_map_path=Path(_default_pack_path()).parent / "ui_map.yaml",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {},
                "gates": {},
                "recent_deltas": [{"ui_target": "apu_switch", "k": "APU_CONTROL_SW"}],
                "overlay_target_allowlist": [],
                "deterministic_step_hint": {
                    "inferred_step_id": "S01",
                    "overlay_step_id": "S01",
                    "missing_conditions": ["vars.apu_on==true"],
                    "recent_ui_targets": ["apu_switch"],
                    "step_evidence_requirements": ["delta"],
                },
            },
        )
        fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)
    finally:
        loop.close()

    assert fallback_help_obj is None
    assert fallback_reason == "overlay_disabled:S01"


def test_safe_fallback_overlay_reports_unsupported_when_overlay_disabled_has_no_targets(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_overlay_disabled_no_targets.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    observability: observable\n"
        "    overlay_enabled: false\n"
        "    evidence_requirements: [delta]\n"
        "    ui_targets: []\n"
        "precondition_gates:\n"
        "  S01: []\n"
        "completion_gates:\n"
        "  S01: []\n",
        encoding="utf-8",
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
        pack_path=pack,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {},
                "gates": {},
                "recent_deltas": [],
                "overlay_target_allowlist": [],
                "deterministic_step_hint": {
                    "inferred_step_id": "S01",
                    "overlay_step_id": "S01",
                    "missing_conditions": ["vars.apu_on==true"],
                    "step_evidence_requirements": ["delta"],
                },
            },
        )
        fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)
    finally:
        loop.close()

    assert fallback_help_obj is None
    assert fallback_reason == "unsupported_step:S01"


def test_live_loop_allowlist_filter_keeps_actions_for_remaining_targets_with_evidence(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_allowlist_partial_filter.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    ui_targets:\n"
        "      - apu_switch\n",
        encoding="utf-8",
    )

    source = ReplayBiosReceiver(replay_path)
    model = MixedAllowlistTargetsModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        pack_path=pack,
        ui_map_path=Path(_default_pack_path()).parent / "ui_map.yaml",
        event_sink=events.append,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    assert executor.calls[0][0]["target"] == "apu_switch"
    assert executor.calls[0][0]["element_id"] == _apu_element_id_from_ui_map()
    tutor_response_payloads = [event.payload for event in events if event.kind == "tutor_response"]
    assert len(tutor_response_payloads) == 1
    assert tutor_response_payloads[0]["metadata"]["failure_code"] == ALLOWLIST_FAIL
    response_mapping = tutor_response_payloads[0]["metadata"]["response_mapping"]
    assert response_mapping["rejected_targets_by_request_allowlist"] == ["battery_switch"]
    assert "overlay_target_not_in_request_allowlist" in response_mapping["mapping_errors"]
    assert response_mapping.get("overlay_rejected") is not True
    reasons = response_mapping.get("overlay_rejected_reasons", [])
    assert "evidence_target_not_in_overlay_targets" not in reasons
    rejected_payloads = [event.payload for event in events if event.kind == "overlay_rejected"]
    assert len(rejected_payloads) == 1
    assert rejected_payloads[0]["failure_code"] == ALLOWLIST_FAIL


def test_live_loop_emits_overlay_rejected_event_for_evidence_failure(tmp_path: Path) -> None:
    class EvidenceFailModel:
        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Need more evidence.",
                actions=[],
                explanations=["Need more evidence."],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S03", "error_category": "OM"},
                        "next": {"step_id": "S03"},
                        "overlay": {"targets": ["apu_switch"], "evidence": []},
                        "explanations": ["Need more evidence."],
                        "confidence": 0.4,
                    },
                },
            )

    replay_path = tmp_path / "bios_evidence_rejected.jsonl"
    _write_replay(replay_path, [
        _bios_frame_fire_test_b_pre(0, 9.0),
        _bios_frame(1, 10.0, apu_switch=0),
    ])

    source = ReplayBiosReceiver(replay_path)
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=EvidenceFailModel(),
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        event_sink=events.append,
    )
    try:
        loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    assert tutor_response_payload["metadata"].get("vision_fallback_reason") is None
    rejected_payload = next(event.payload for event in events if event.kind == "overlay_rejected")
    assert rejected_payload["failure_code"] == EVIDENCE_FAIL
    assert rejected_payload["overlay_rejected"] is True


def test_live_loop_allowlist_filter_drops_non_mapping_evidence_items(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_allowlist_drop_invalid_evidence.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    ui_targets:\n"
        "      - apu_switch\n",
        encoding="utf-8",
    )

    source = ReplayBiosReceiver(replay_path)
    model = MixedAllowlistTargetsWithInvalidEvidenceItemModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        pack_path=pack,
        ui_map_path=Path(_default_pack_path()).parent / "ui_map.yaml",
        event_sink=events.append,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    assert executor.calls[0][0]["target"] == "apu_switch"
    tutor_response_payloads = [event.payload for event in events if event.kind == "tutor_response"]
    assert len(tutor_response_payloads) == 1
    response_mapping = tutor_response_payloads[0]["metadata"]["response_mapping"]
    reasons = response_mapping.get("overlay_rejected_reasons", [])
    assert "invalid_overlay_evidence_item" not in reasons
    assert response_mapping.get("overlay_rejected") is not True


def test_live_loop_dry_run_overlay_prints_planned_actions(tmp_path: Path, capsys) -> None:
    replay_path = tmp_path / "bios_dry_run.jsonl"
    _write_replay(replay_path, [
        _bios_frame_fire_test_b_pre(0, 9.0),
        _bios_frame(1, 11.0, apu_switch=0),
    ])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor(include_dry_run=True)
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        dry_run_overlay=True,
    )
    try:
        loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    assert len(executor.calls) == 0
    out = capsys.readouterr().out
    assert "dry_run_actions" in out
    assert "apu_switch" in out


def test_live_loop_dry_run_overlay_uses_executor_when_executor_is_dry_run(tmp_path: Path, capsys) -> None:
    replay_path = tmp_path / "bios_dry_run_exec.jsonl"
    _write_replay(replay_path, [
        _bios_frame_fire_test_b_pre(0, 9.0),
        _bios_frame(1, 11.5, apu_switch=0),
    ])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor(include_dry_run=True, dry_run=True)
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        dry_run_overlay=True,
    )
    try:
        loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    out = capsys.readouterr().out
    assert "dry_run_actions" in out
    assert "apu_switch" in out


def test_replay_receiver_streams_and_only_parses_on_demand(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_streaming.jsonl"
    replay_path.write_text(
        json.dumps(_bios_frame(1, 12.0, apu_switch=0), ensure_ascii=False)
        + "\n"
        + "{bad-json}\n",
        encoding="utf-8",
    )

    source = ReplayBiosReceiver(replay_path)
    try:
        first = source.get_observation()
        assert first is not None
        assert first.payload["seq"] == 1

        with pytest.raises(ValueError, match="invalid JSON"):
            source.get_observation()
        assert source.is_exhausted is True
        assert source._fh.closed is True
    finally:
        source.close()


def test_replay_receiver_skips_non_mapping_json_values(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_non_mapping_values.jsonl"
    replay_path.write_text(
        "[]\n"
        + json.dumps(_bios_frame(7, 20.0, apu_switch=1), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    source = ReplayBiosReceiver(replay_path)
    try:
        obs = source.get_observation()
        assert obs is not None
        assert obs.payload["seq"] == 7
    finally:
        source.close()


def test_replay_receiver_speed_realtime_paces_by_t_wall(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_speed_realtime.jsonl"
    _write_replay(
        replay_path,
        [
            _bios_frame(1, 100.0, apu_switch=0),
            _bios_frame(2, 100.3, apu_switch=1),
        ],
    )

    sleeps: list[float] = []
    fake_now = [1000.0]

    def _fake_monotonic() -> float:
        return fake_now[0]

    def _fake_sleep(value: float) -> None:
        sleep_s = float(value)
        sleeps.append(sleep_s)
        fake_now[0] += sleep_s

    monkeypatch.setattr("live_dcs.time.monotonic", _fake_monotonic)
    monkeypatch.setattr("live_dcs.time.sleep", _fake_sleep)

    source = ReplayBiosReceiver(replay_path, speed=1.0)
    try:
        first = source.get_observation()
        second = source.get_observation()
        assert first is not None
        assert second is not None
        assert second.payload["seq"] == 2
    finally:
        source.close()

    assert sleeps == pytest.approx([0.3], rel=1e-3, abs=1e-3)


def test_replay_receiver_speed_zero_disables_pacing(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_speed_zero.jsonl"
    _write_replay(
        replay_path,
        [
            _bios_frame(1, 200.0, apu_switch=0),
            _bios_frame(2, 201.0, apu_switch=1),
        ],
    )

    sleeps: list[float] = []
    monkeypatch.setattr("live_dcs.time.sleep", lambda value: sleeps.append(float(value)))

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    try:
        assert source.get_observation() is not None
        assert source.get_observation() is not None
    finally:
        source.close()

    assert sleeps == []


def test_stdin_help_trigger_reader_does_not_enqueue_after_stop_set_during_input(
    monkeypatch,
) -> None:
    trigger = StdinHelpTrigger()

    def _fake_input() -> str:
        trigger._stop.set()
        return "help"

    monkeypatch.setattr(builtins, "input", _fake_input)
    trigger._reader()
    assert trigger.poll() is False


def test_udp_help_trigger_receives_help_datagram(monkeypatch) -> None:
    class FakeDatagramSocket:
        _registry: dict[tuple[str, int], "FakeDatagramSocket"] = {}
        _next_port = 40000

        def __init__(self, *_args, **_kwargs) -> None:
            self._timeout = 0.0
            self._bound = ("127.0.0.1", 0)
            self._recv_queue: list[tuple[bytes, tuple[str, int]]] = []

        def settimeout(self, timeout: float) -> None:
            self._timeout = float(timeout)

        def bind(self, addr: tuple[str, int]) -> None:
            host, port = addr
            if port == 0:
                port = FakeDatagramSocket._next_port
                FakeDatagramSocket._next_port += 1
            self._bound = (host, port)
            FakeDatagramSocket._registry[self._bound] = self

        def getsockname(self) -> tuple[str, int]:
            return self._bound

        def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
            deadline = time.time() + self._timeout
            while True:
                if self._recv_queue:
                    return self._recv_queue.pop(0)
                if time.time() >= deadline:
                    raise socket.timeout()
                time.sleep(0.001)

        def sendto(self, payload: bytes, addr: tuple[str, int]) -> None:
            target = FakeDatagramSocket._registry[addr]
            target._recv_queue.append((payload, self._bound))

        def close(self) -> None:
            FakeDatagramSocket._registry.pop(self._bound, None)

    monkeypatch.setattr("live_dcs.socket.socket", lambda *_args, **_kwargs: FakeDatagramSocket())
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: FakeDatagramSocket())

    trigger = UdpHelpTrigger(host="127.0.0.1", port=0, timeout=0.05)
    trigger.start()
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(b"help", ("127.0.0.1", trigger.bound_port))
        sender.close()

        deadline = time.time() + 1.0
        fired = False
        while time.time() < deadline:
            if trigger.poll():
                fired = True
                break
            time.sleep(0.01)
        assert fired is True
    finally:
        trigger.close()


def test_udp_help_payload_rejects_empty_and_accepts_explicit_help() -> None:
    assert _is_help_trigger_payload("  ") is False
    assert _is_help_trigger_payload("help") is True
    assert _is_help_trigger_payload('{"intent":"help"}') is True


def test_composite_help_trigger_does_not_drain_all_triggers_in_one_poll() -> None:
    class QueueTrigger:
        def __init__(self, queued: int) -> None:
            self.queued = queued

        def poll(self) -> bool:
            if self.queued <= 0:
                return False
            self.queued -= 1
            return True

    first = QueueTrigger(queued=1)
    second = QueueTrigger(queued=1)
    trigger = CompositeHelpTrigger([first, second])

    assert trigger.poll() is True
    assert first.queued == 0
    # second should remain queued for next loop iteration
    assert second.queued == 1
    assert trigger.poll() is True
    assert second.queued == 0


def test_load_overlay_allowlist_raises_when_pack_ui_targets_contains_unknown_target(tmp_path: Path) -> None:
    ui_map = tmp_path / "ui_map.yaml"
    pack = tmp_path / "pack.yaml"
    ui_map.write_text(
        "version: v1\n"
        "cockpit_elements:\n"
        "  apu_switch:\n"
        "    dcs_id: pnt_375\n",
        encoding="utf-8",
    )
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "ui_targets:\n"
        "  - not_in_ui_map\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        _load_overlay_allowlist(pack, ui_map)
    message = str(exc_info.value)
    assert "pack.ui_targets[0]='not_in_ui_map'" in message
    assert "not found in ui_map" in message


def test_load_overlay_allowlist_raises_when_pack_ui_targets_mixes_valid_and_unknown(tmp_path: Path) -> None:
    ui_map = tmp_path / "ui_map.yaml"
    pack = tmp_path / "pack.yaml"
    ui_map.write_text(
        "version: v1\n"
        "cockpit_elements:\n"
        "  apu_switch:\n"
        "    dcs_id: pnt_375\n"
        "  battery_switch:\n"
        "    dcs_id: pnt_404\n",
        encoding="utf-8",
    )
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "ui_targets:\n"
        "  - apu_switch\n"
        "  - typo_target\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        _load_overlay_allowlist(pack, ui_map)
    message = str(exc_info.value)
    assert "pack.ui_targets[1]='typo_target'" in message
    assert "not found in ui_map" in message


def test_load_overlay_allowlist_uses_step_ui_targets_union_when_top_level_missing(tmp_path: Path) -> None:
    ui_map = tmp_path / "ui_map.yaml"
    pack = tmp_path / "pack.yaml"
    ui_map.write_text(
        "version: v1\n"
        "cockpit_elements:\n"
        "  apu_switch:\n"
        "    dcs_id: pnt_375\n"
        "  battery_switch:\n"
        "    dcs_id: pnt_404\n"
        "  fire_test_switch:\n"
        "    dcs_id: pnt_331\n",
        encoding="utf-8",
    )
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    ui_targets:\n"
        "      - battery_switch\n"
        "  - id: S02\n"
        "    ui_targets:\n"
        "      - apu_switch\n"
        "  - id: S03\n"
        "    ui_targets: []\n",
        encoding="utf-8",
    )

    allowlist = _load_overlay_allowlist(pack, ui_map)
    assert allowlist == ["apu_switch", "battery_switch"]


def test_load_step_signal_profiles_parses_valid_step_metadata(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    observability: observable\n"
        "    evidence_requirements: [var, gate]\n"
        "    ui_targets: [battery_switch, battery_switch]\n"
        "  - id: S02\n"
        "    observability: unknown\n"
        "    evidence_requirements: [visual, rag]\n",
        encoding="utf-8",
    )

    profiles = _load_step_signal_profiles(pack)
    assert profiles["S01"]["observability"] == "observable"
    assert profiles["S01"]["observability_status"] == "observable"
    assert profiles["S01"]["evidence_requirements"] == ["var", "gate"]
    assert profiles["S01"]["ui_targets"] == ["battery_switch"]
    assert profiles["S01"]["requires_visual_confirmation"] is False
    assert profiles["S02"]["observability"] == "unobservable"
    assert profiles["S02"]["observability_status"] == "unobservable"
    assert profiles["S02"]["evidence_requirements"] == ["visual", "rag"]
    assert profiles["S02"]["requires_visual_confirmation"] is True


def test_real_fa18c_pack_marks_s05_as_bios_observable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pack_path = repo_root / "packs" / "fa18c_startup" / "pack.yaml"

    profiles = _load_step_signal_profiles(pack_path)

    assert profiles["S05"]["observability"] == "observable"
    assert profiles["S05"]["observability_status"] == "observable"
    assert profiles["S05"]["evidence_requirements"] == ["var", "gate"]
    assert profiles["S05"]["ui_targets"] == []
    assert profiles["S05"]["overlay_enabled"] is False
    assert profiles["S05"]["requires_visual_confirmation"] is False


def test_real_fa18c_pack_marks_non_display_steps_as_bios_observable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pack_path = repo_root / "packs" / "fa18c_startup" / "pack.yaml"

    profiles = _load_step_signal_profiles(pack_path)

    for step_id in (
        "S11",
        "S12",
        "S13",
        "S14",
        "S16",
        "S20",
        "S21",
        "S22",
        "S23",
        "S24",
        "S25",
        "S26",
        "S27",
        "S28",
        "S29",
        "S31",
        "S32",
        "S33",
    ):
        assert profiles[step_id]["observability"] == "observable"
        assert profiles[step_id]["observability_status"] == "observable"
        assert profiles[step_id]["requires_visual_confirmation"] is False

    assert profiles["S11"]["ui_targets"] == []
    assert profiles["S11"]["overlay_enabled"] is False
    assert profiles["S15"]["requires_visual_confirmation"] is True
    assert profiles["S18"]["requires_visual_confirmation"] is True
    assert profiles["S08"]["ui_targets"] == [
        "left_mdi_brightness_selector",
        "right_mdi_brightness_selector",
        "ampcd_off_brightness_knob",
        "hud_symbology_brightness_knob",
        "left_mdi_pb18",
        "left_mdi_pb15",
        "right_mdi_pb18",
        "right_mdi_pb5",
    ]
    assert profiles["S15"]["ui_targets"] == ["fcs_reset_button", "left_mdi_pb18", "left_mdi_pb15"]
    assert profiles["S18"]["ui_targets"] == ["right_mdi_pb18", "right_mdi_pb5"]
    assert profiles["S19"]["ui_targets"] == ["fcs_bit_switch", "right_mdi_pb5"]


def test_real_fa18c_pack_marks_non_display_partial_steps_as_non_visual() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pack_path = repo_root / "packs" / "fa18c_startup" / "pack.yaml"

    profiles = _load_step_signal_profiles(pack_path)

    for step_id in ("S17", "S30"):
        assert profiles[step_id]["observability"] == "partial"
        assert profiles[step_id]["observability_status"] == "partial"
        assert profiles[step_id]["requires_visual_confirmation"] is False

    for step_id in ("S02", "S07"):
        assert profiles[step_id]["observability"] == "observable"
        assert profiles[step_id]["observability_status"] == "observable"
        assert profiles[step_id]["requires_visual_confirmation"] is False

    assert profiles["S02"]["evidence_requirements"] == ["var", "delta", "gate"]
    assert profiles["S02"]["ui_targets"] == ["fire_test_switch"]
    assert profiles["S02"]["overlay_enabled"] is True
    assert profiles["S07"]["evidence_requirements"] == ["var", "delta", "gate"]
    assert profiles["S07"]["ui_targets"] == ["lights_test_button"]
    assert profiles["S07"]["overlay_enabled"] is True
    assert profiles["S09"]["observability"] == "observable"
    assert profiles["S09"]["observability_status"] == "observable"
    assert profiles["S09"]["requires_visual_confirmation"] is False
    assert profiles["S09"]["evidence_requirements"] == ["var", "delta", "rag"]


def test_load_step_signal_profiles_rejects_invalid_observability(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    observability: maybe\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"pack\.steps\[0\]\.observability must be one of"):
        _load_step_signal_profiles(pack)


def test_load_step_signal_profiles_rejects_invalid_evidence_requirement(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    observability: observable\n"
        "    evidence_requirements: [var, visual, unsupported]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"pack\.steps\[0\]\.evidence_requirements\[2\] must be one of"):
        _load_step_signal_profiles(pack)


def test_load_step_signal_profiles_rejects_invalid_ui_target(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    observability: observable\n"
        "    ui_targets: [battery_switch, '']\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"pack\.steps\[0\]\.ui_targets\[1\] must be non-empty string"):
        _load_step_signal_profiles(pack)


def test_live_loop_counts_model_attempt_when_model_raises(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_model_error.jsonl"
    _write_replay(replay_path, [
        _bios_frame_fire_test_b_pre(0, 9.0),
        _bios_frame(1, 13.0, apu_switch=0),
    ])

    source = ReplayBiosReceiver(replay_path)
    model = FailingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    assert stats["help_cycles"] == 1
    assert stats["model_calls"] == 1
    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    assert executor.calls[0][0]["type"] == "overlay"
    assert executor.calls[0][0]["target"] == "apu_switch"


def test_live_loop_uses_safe_fallback_overlay_when_model_response_is_error(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_model_error_fallback_overlay.jsonl"
    _write_replay(replay_path, [
        _bios_frame_fire_test_b_pre(0, 9.0),
        _bios_frame(1, 19.0, apu_switch=0),
    ])

    source = ReplayBiosReceiver(replay_path)
    model = FailingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        event_sink=events.append,
    )
    try:
        loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    action = executor.calls[0][0]
    assert action["type"] == "overlay"
    assert action["target"] == "apu_switch"

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    meta = tutor_response_payload["metadata"]
    assert meta["fallback_overlay_used"] is True
    assert isinstance(meta["fallback_overlay_reason"], str)
    assert meta["fallback_overlay_reason"] == "emergency_presentation_fallback"
    assert meta["final_action_plan"]["source"] == "emergency_presentation_fallback"


def test_live_loop_replaces_rejected_future_step_overlay_with_safe_current_step_overlay(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_wrong_future_overlay.jsonl"
    _write_replay(replay_path, [
        _bios_frame_fire_test_b_pre(0, 9.0),
        _bios_frame(1, 19.0, apu_switch=0),
    ])

    class WrongFutureStepModel:
        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Move ENG CRANK RIGHT.",
                actions=[],
                explanations=["Move ENG CRANK RIGHT."],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S04", "error_category": "OM"},
                        "next": {"step_id": "S04"},
                        "overlay": {
                            "targets": ["eng_crank_switch"],
                            "evidence": [
                                {
                                    "target": "eng_crank_switch",
                                    "type": "delta",
                                    "ref": "RECENT_UI_TARGETS.eng_crank_switch",
                                    "quote": "Recent delta shows ENG CRANK switch activity.",
                                }
                            ],
                        },
                        "explanations": ["Move ENG CRANK RIGHT."],
                        "confidence": 0.82,
                    },
                },
            )

    source = ReplayBiosReceiver(replay_path)
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=WrongFutureStepModel(),
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
        event_sink=events.append,
    )
    try:
        loop.run(max_frames=2, auto_help_every_n_frames=2)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    assert executor.calls[0][0]["target"] == "apu_switch"

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    meta = tutor_response_payload["metadata"]
    assert meta["fallback_overlay_used"] is True
    assert meta["fallback_overlay_reason"] == "validator_action_hint"
    assert meta["presentation_fallback_reason"] == "deterministic_step:S03"
    response_mapping = meta["response_mapping"]
    assert response_mapping["rejected_targets_by_request_allowlist"] == ["eng_crank_switch"]
    assert "overlay_target_not_in_request_allowlist" in response_mapping["mapping_errors"]
    assert tutor_response_payload["message"] == (
        "You are on S03. Left-click the APU switch to ON, then wait for the green APU READY light."
    )
    assert tutor_response_payload["explanations"] == [tutor_response_payload["message"]]
    assert meta["fallback_message"] == "Please operate apu_switch first."
    assert "Please operate apu_switch first." in meta["fallback_explanations"]
    assert meta["model_raw_help_response"]["next"]["step_id"] == "S04"
    assert meta["final_public_response"]["message"] == tutor_response_payload["message"]


def test_safe_fallback_overlay_is_pack_driven_for_s01_s25(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_pack_driven_fallback.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    loop = LiveDcsTutorLoop(
        source=source,
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        step_meta: dict[str, dict[str, Any]] = {}
        for step in loop.pack_steps:
            if not isinstance(step, dict):
                continue
            step_id = step.get("id")
            if not isinstance(step_id, str) or not step_id:
                continue
            raw_targets = step.get("ui_targets")
            targets: list[str] = []
            if isinstance(raw_targets, list):
                targets = [item for item in raw_targets if isinstance(item, str) and item]
            step_meta[step_id] = {"targets": targets}

        vars_map: dict[str, Any] = {}
        for gate_map in (loop.precondition_gates, loop.completion_gates):
            for rules in gate_map.values():
                if not isinstance(rules, (list, tuple)):
                    continue
                for rule in rules:
                    if not isinstance(rule, dict):
                        continue
                    raw_var = rule.get("var")
                    if not isinstance(raw_var, str) or not raw_var:
                        continue
                    var_name = raw_var
                    if var_name.startswith("payload.vars."):
                        var_name = var_name[len("payload.vars.") :]
                    elif var_name.startswith("vars."):
                        var_name = var_name[len("vars.") :]
                    elif var_name.startswith("payload.") and "." in var_name:
                        var_name = var_name[len("payload.") :]
                    vars_map[var_name] = 0

        verifiable_types = {"var", "gate", "delta"}
        for step_id in [f"S{i:02d}" for i in range(1, 26)]:
            raw_targets = step_meta.get(step_id, {}).get("targets", [])
            step_targets = [
                target for target in raw_targets if isinstance(target, str) and target in set(loop.overlay_allowlist)
            ]
            recent_deltas = [{"ui_target": target, "k": f"DUMMY_{idx}"} for idx, target in enumerate(step_targets)]
            hint: dict[str, Any] = {
                "inferred_step_id": step_id,
                "missing_conditions": [],
                "gate_blockers": [{"ref": f"GATES.{step_id}.precondition", "reason": "blocked"}],
                "recent_ui_targets": list(step_targets),
            }
            profile = loop.step_signal_profiles.get(step_id, {})
            requirements = profile.get("evidence_requirements")
            if isinstance(requirements, list):
                hint["step_evidence_requirements"] = [
                    item for item in requirements if isinstance(item, str) and item
                ]
            request = TutorRequest(
                actor="learner",
                intent="help",
                message="help",
                context={
                    "vars": dict(vars_map),
                    "gates": {
                        f"{step_id}.precondition": {
                            "status": "blocked",
                            "reason_code": "test_blocked",
                            "reason": "blocked in test",
                        },
                        f"{step_id}.completion": {
                            "status": "blocked",
                            "reason_code": "test_blocked",
                            "reason": "blocked in test",
                        },
                    },
                    "recent_deltas": recent_deltas,
                    "overlay_target_allowlist": list(loop.overlay_allowlist),
                    "deterministic_step_hint": hint,
                },
            )
            fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)
            can_verify = True
            if isinstance(requirements, list) and requirements:
                can_verify = any(item in verifiable_types for item in requirements if isinstance(item, str))
            expected_overlay = bool(step_targets) and can_verify
            if fallback_reason.startswith("evidence_conflict:"):
                assert fallback_help_obj is None, step_id
                continue
            if expected_overlay:
                assert isinstance(fallback_help_obj, dict), step_id
                assert fallback_reason == f"deterministic_step:{step_id}"
                overlay = fallback_help_obj["overlay"]
                assert isinstance(overlay, dict)
                assert len(overlay["targets"]) == 1
                assert overlay["targets"][0] in step_targets
            else:
                assert fallback_help_obj is None, step_id
    finally:
        loop.close()


def test_safe_fallback_overlay_prefers_hud_target_for_s08_hud_missing(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_pack_driven_fallback_s08.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {
                    "left_ddi_on": True,
                    "right_ddi_on": True,
                    "mpcd_on": True,
                    "hud_on": False,
                },
                "gates": {
                    "S08.completion": {
                        "status": "blocked",
                        "reason_code": "s08_requires_hud_on",
                        "reason": "HUD must be powered.",
                    }
                },
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "missing_conditions": ["vars.hud_on==true"],
                    "gate_blockers": [{"ref": "GATES.S08.completion", "reason": "HUD must be powered."}],
                    "recent_ui_targets": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                },
            },
        )

        fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)

        assert fallback_reason == "deterministic_step:S08"
        assert isinstance(fallback_help_obj, dict)
        assert fallback_help_obj["overlay"]["targets"] == ["hud_symbology_brightness_knob"]
    finally:
        loop.close()


def test_safe_fallback_overlay_prefers_ampcd_when_s08_ddis_are_powered(tmp_path: Path) -> None:
    """When both DDI selector vars are true and only AMPCD remains missing,
    S08 fallback should no longer regress to a DDI selector.
    """
    replay_path = tmp_path / "bios_s08_ddi_before_ampcd.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {
                    "left_ddi_on": True,
                    "right_ddi_on": True,
                    "mpcd_on": False,
                },
                "gates": {
                    "S08.completion": {
                        "status": "blocked",
                        "reason_code": "s08_requires_mpcd_on",
                        "reason": "MPCD must be powered.",
                    }
                },
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "missing_conditions": ["vars.mpcd_on==true"],
                    "gate_blockers": [{"ref": "GATES.S08.completion", "reason": "MPCD must be powered."}],
                    "recent_ui_targets": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                },
            },
        )

        fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)

        assert fallback_reason == "deterministic_step:S08"
        assert isinstance(fallback_help_obj, dict)
        assert fallback_help_obj["overlay"]["targets"] == ["ampcd_off_brightness_knob"], (
            f"Expected ampcd_off_brightness_knob, got {fallback_help_obj['overlay']['targets']}"
        )
    finally:
        loop.close()


def test_safe_fallback_overlay_highlights_all_four_display_power_controls_for_s08(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s08_four_display_power_targets.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=4,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {
                    "left_ddi_on": True,
                    "right_ddi_on": False,
                    "mpcd_on": False,
                    "hud_on": False,
                },
                "gates": {
                    "S08.completion": {
                        "status": "blocked",
                        "reason_code": "s08_requires_displays_on",
                        "reason": "Displays must be powered.",
                    }
                },
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": [
                        "vars.right_ddi_on==true",
                        "vars.mpcd_on==true",
                        "vars.hud_on==true",
                    ],
                    "gate_blockers": [
                        {
                            "ref": "GATES.S08.completion",
                            "reason": "Displays must be powered.",
                        }
                    ],
                    "recent_ui_targets": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                },
            },
        )

        fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)

        assert fallback_reason == "deterministic_step:S08"
        assert isinstance(fallback_help_obj, dict)
        assert fallback_help_obj["overlay"]["targets"] == [
            "right_mdi_brightness_selector",
            "ampcd_off_brightness_knob",
            "hud_symbology_brightness_knob",
        ]
    finally:
        loop.close()


def test_normalize_observable_text_only_response_rewrites_bad_visual_excuse(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_observable_text_rewrite.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "deterministic_step_hint": {
                    "inferred_step_id": "S05",
                    "missing_conditions": ["vars.rpm_r>=25"],
                    "observability_status": "observable",
                    "requires_visual_confirmation": False,
                }
            },
        )
        response = TutorResponse(
            status="ok",
            message="当前步骤为 S05。由于缺乏变量证据且视觉不可用，无法确认具体操作目标。",
            actions=[],
            explanations=["由于缺乏变量证据且视觉不可用，无法确认具体操作目标。"],
            metadata={},
        )

        loop._normalize_observable_text_only_response(response, request)

        assert response.metadata["observable_text_rewritten"] is True
        assert response.message == "降级提示：你大概率卡在 S05，请先完成该步骤的未满足条件。"
        assert response.explanations == ["降级提示：你大概率卡在 S05，请先完成该步骤的未满足条件。"]
        assert "vars.rpm_r" not in response.message
    finally:
        loop.close()


def test_normalize_observable_text_only_response_rewrites_visual_analysis_unavailable(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_observable_visual_analysis_text_rewrite.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
                    "observability_status": "observable",
                    "requires_visual_confirmation": False,
                }
            },
        )
        response = TutorResponse(
            status="ok",
            message="虽然视觉分析不可用，但根据任务流程提示，系统推断您应处于此阶段。",
            actions=[],
            explanations=["虽然视觉分析不可用，但根据任务流程提示，系统推断您应处于此阶段。"],
            metadata={},
        )

        loop._normalize_observable_text_only_response(response, request)

        assert response.metadata["observable_text_rewritten"] is True
        assert response.message == "降级提示：你大概率卡在 S08，请先完成该步骤的未满足条件。"
    finally:
        loop.close()


def test_annotate_response_audit_metadata_sanitizes_public_predicates(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_public_predicate_sanitize.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        response = TutorResponse(
            status="ok",
            message="当前 S04 尚未完成，请先满足：vars.right_engine_nominal_start_params==true。",
            explanations=["需要 vars.rpm_r>=25 后继续。"],
            metadata={},
        )

        loop._annotate_response_audit_metadata(response)

        assert "vars." not in response.message
        assert all("vars." not in item for item in response.explanations)
        assert "vars." not in response.metadata["final_public_response"]["message"]
        assert all("vars." not in item for item in response.metadata["final_public_response"]["explanations"])
    finally:
        loop.close()


def test_should_use_deterministic_overlay_fallback_for_observable_text_only_step(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_overlay_fallback_observable.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "missing_conditions": ["vars.hud_on==true"],
                    "observability_status": "observable",
                    "requires_visual_confirmation": False,
                }
            },
        )
        response = TutorResponse(
            status="ok",
            message="HUD is not powered yet.",
            actions=[],
            explanations=["HUD is not powered yet."],
            metadata={},
        )

        assert loop._should_use_deterministic_overlay_fallback(response, request, {}) is True
    finally:
        loop.close()


def test_safe_fallback_overlay_prefers_left_ddi_menu_navigation_for_s08_fcs_page_missing(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_pack_driven_fallback_s08_fcs_page.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "allowed"},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_fact_summary": {
                    "seen_fact_ids": ["bit_root_page_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible"],
                    "uncertain_fact_ids": ["tac_page_visible"],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
                    "gate_blockers": [],
                    "recent_ui_targets": ["lights_test_button"],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                },
                "rag_topk": [],
            },
        )

        fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)

        assert fallback_reason == "deterministic_step:S08"
        assert isinstance(fallback_help_obj, dict)
        assert fallback_help_obj["overlay"]["targets"] == ["left_mdi_brightness_selector"]
    finally:
        loop.close()


def test_safe_fallback_overlay_prefers_left_ddi_fcs_button_when_menu_page_visible(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_pack_driven_fallback_s08_fcs_button.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "allowed"},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_fact_summary": {
                    "seen_fact_ids": ["bit_root_page_visible", "supt_page_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
                    "gate_blockers": [],
                    "recent_ui_targets": ["lights_test_button"],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                },
                "rag_topk": [],
            },
        )

        fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)

        assert fallback_reason == "deterministic_step:S08"
        assert isinstance(fallback_help_obj, dict)
        assert fallback_help_obj["overlay"]["targets"] == ["left_mdi_pb15"]
    finally:
        loop.close()


def test_prefer_navigation_target_from_vision_context_returns_left_pb15_for_s08_menu_fcs_state() -> None:
    targets = _prefer_navigation_target_from_vision_context(
        inferred_step_id="S08",
        missing_conditions=["vision_facts.fcs_page_visible==seen"],
        context={
            "vision_fact_summary": {
                "seen_fact_ids": ["bit_root_page_visible", "supt_page_visible"],
                "not_seen_fact_ids": ["fcs_page_visible"],
                "uncertain_fact_ids": [],
            }
        },
        allowed_targets=["left_mdi_pb18", "left_mdi_pb15", "left_mdi_brightness_selector"],
    )

    assert targets == ["left_mdi_pb15"]


def test_prefer_navigation_target_from_vision_context_returns_left_pb18_for_s08_tac_page_before_supt() -> None:
    targets = _prefer_navigation_target_from_vision_context(
        inferred_step_id="S08",
        missing_conditions=["vision_facts.fcs_page_visible==seen"],
        context={
            "vision_fact_summary": {
                "seen_fact_ids": ["bit_root_page_visible", "tac_page_visible"],
                "not_seen_fact_ids": ["fcs_page_visible"],
                "uncertain_fact_ids": ["supt_page_visible"],
            }
        },
        allowed_targets=["left_mdi_pb18", "left_mdi_pb15", "left_mdi_brightness_selector"],
    )

    assert targets == ["left_mdi_pb18"]


def test_prefer_navigation_target_from_vision_context_does_not_use_uncertain_tac_supt_as_navigation_evidence() -> None:
    targets = _prefer_navigation_target_from_vision_context(
        inferred_step_id="S08",
        missing_conditions=["vision_facts.fcs_page_visible==seen"],
        context={
            "vision_fact_summary": {
                "seen_fact_ids": ["bit_root_page_visible"],
                "not_seen_fact_ids": ["fcs_page_visible"],
                "uncertain_fact_ids": ["tac_page_visible", "supt_page_visible"],
            }
        },
        allowed_targets=["left_mdi_pb18", "left_mdi_pb15", "left_mdi_brightness_selector"],
    )

    assert targets == ["left_mdi_brightness_selector"]


def test_prefer_navigation_target_from_vision_context_returns_left_pb15_for_explicit_fcs_button_fact() -> None:
    targets = _prefer_navigation_target_from_vision_context(
        inferred_step_id="S08",
        missing_conditions=["vision_facts.fcs_page_visible==seen"],
        context={
            "vision_fact_summary": {
                "seen_fact_ids": ["bit_root_page_visible", "supt_page_visible"],
                "not_seen_fact_ids": ["fcs_page_visible"],
                "uncertain_fact_ids": [],
            }
        },
        allowed_targets=["left_mdi_pb18", "left_mdi_pb15", "left_mdi_brightness_selector"],
    )

    assert targets == ["left_mdi_pb15"]


def test_visual_action_hint_does_not_override_existing_model_action(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_visual_action_hint_override_s08.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "allowed"},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": ["bit_root_page_visible", "tac_page_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible"],
                    "uncertain_fact_ids": ["supt_page_visible"],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                    "visual_action_hint": {
                        "target": "left_mdi_pb18",
                        "reason": "Press PB18 first to reach SUPT.",
                    },
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message=(
                "The Left DDI is currently on the TAC (Tactical) page, not the required "
                "FCS (Flight Control System) page for Step S08. Press PB15 to enter FCS."
            ),
            actions=[
                {
                    "kind": "highlight",
                    "target": "left_mdi_pb15",
                    "element_id": "pnt_68",
                    "duration_ms": 2500,
                }
            ],
            explanations=[
                "The Left DDI is currently on the TAC (Tactical) page, not the required FCS page. Press PB15 next."
            ],
            metadata={},
        )

        override_used, override_reason = loop._apply_action_hint_overlay_override(response, request)

        assert override_used is False
        assert override_reason == "missing_override_target"
        assert response.actions
        assert response.actions[0]["target"] == "left_mdi_pb15"
        assert "Press PB15 to enter FCS" in response.message
        assert response.metadata == {}
    finally:
        loop.close()


def test_harness_validation_repairs_s08_selector_to_supt_visual_hint(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s08_visual_hint_repairs_selector_to_pb15.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "blocked", "reason": "FCS page is not visible."},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_facts": [
                    {"fact_id": "supt_page_visible", "state": "seen", "source_frame_id": "frame-supt"},
                    {"fact_id": "bit_root_page_visible", "state": "seen", "source_frame_id": "frame-bit"},
                    {"fact_id": "fcs_page_visible", "state": "not_seen", "source_frame_id": "frame-fcs"},
                ],
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": ["bit_root_page_visible", "supt_page_visible"],
                    "fresh_fact_ids": ["bit_root_page_visible", "supt_page_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["visual", "gate"],
                    "visual_action_hint": {
                        "target": "left_mdi_pb15",
                        "reason": "SUPT is visible; press PB15 to enter FCS.",
                    },
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="Use the left MDI to navigate to FCS.",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "left_mdi_brightness_selector",
                    "element_id": "pnt_51",
                }
            ],
            explanations=["Use the left MDI to navigate to FCS."],
            metadata={
                "help_response": {
                    "diagnosis": {"step_id": "S08", "error_category": "OM"},
                    "next": {"step_id": "S08"},
                    "overlay": {
                        "targets": ["left_mdi_brightness_selector"],
                        "evidence": [
                            {
                                "target": "left_mdi_brightness_selector",
                                "type": "visual",
                                "ref": "VISION_FACTS.supt_page_visible@frame-supt",
                                "quote": "SUPT visible.",
                                "grounding_confidence": 0.9,
                            }
                        ],
                    },
                    "explanations": ["Use the left MDI to navigate to FCS."],
                }
            },
        )

        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "validator_action_hint"
        assert [action["target"] for action in response.actions] == ["left_mdi_pb15"]
        assert response.metadata["harness_validator_fallback_reason"] == "deterministic_step:S08"
        assert response.metadata["s08_visual_hint_repair_applied"] is True
        assert response.metadata["rejected_model_target"] == "left_mdi_brightness_selector"
        assert response.metadata["visual_hint_target"] == "left_mdi_pb15"
        assert response.metadata["final_action_plan_source"] == "validator_action_hint"
        assert response.metadata["help_response"]["overlay"]["evidence"][0]["ref"] == (
            "VISION_FACTS.supt_page_visible@frame-supt"
        )
    finally:
        loop.close()


def test_harness_validation_repairs_s08_selector_to_tac_visual_hint(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s08_visual_hint_repairs_selector_to_pb18.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "blocked", "reason": "FCS page is not visible."},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_facts": [
                    {"fact_id": "tac_page_visible", "state": "seen", "source_frame_id": "frame-tac"},
                    {"fact_id": "bit_root_page_visible", "state": "seen", "source_frame_id": "frame-bit"},
                    {"fact_id": "fcs_page_visible", "state": "not_seen", "source_frame_id": "frame-fcs"},
                ],
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": ["bit_root_page_visible", "tac_page_visible"],
                    "fresh_fact_ids": ["bit_root_page_visible", "tac_page_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible"],
                    "uncertain_fact_ids": ["supt_page_visible"],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["visual", "gate"],
                    "visual_action_hint": {
                        "target": "left_mdi_pb18",
                        "reason": "TAC is visible; press PB18 to reach SUPT.",
                    },
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="Use the left MDI to navigate to FCS.",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "left_mdi_brightness_selector",
                    "element_id": "pnt_51",
                }
            ],
            explanations=["Use the left MDI to navigate to FCS."],
            metadata={
                "help_response": {
                    "diagnosis": {"step_id": "S08", "error_category": "OM"},
                    "next": {"step_id": "S08"},
                    "overlay": {
                        "targets": ["left_mdi_brightness_selector"],
                        "evidence": [
                            {
                                "target": "left_mdi_brightness_selector",
                                "type": "visual",
                                "ref": "VISION_FACTS.tac_page_visible@frame-tac",
                                "quote": "TAC visible.",
                                "grounding_confidence": 0.9,
                            }
                        ],
                    },
                    "explanations": ["Use the left MDI to navigate to FCS."],
                }
            },
        )

        used, _reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert [action["target"] for action in response.actions] == ["left_mdi_pb18"]
        assert response.metadata["s08_visual_hint_repair_applied"] is True
        assert response.metadata["rejected_model_target"] == "left_mdi_brightness_selector"
        assert response.metadata["visual_hint_target"] == "left_mdi_pb18"
        assert response.metadata["help_response"]["overlay"]["evidence"][0]["ref"] == (
            "VISION_FACTS.tac_page_visible@frame-tac"
        )
    finally:
        loop.close()


def test_harness_validation_repairs_s08_selector_from_tac_evidence_ref(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s08_visual_evidence_repairs_selector_to_pb18.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "blocked", "reason": "FCS page is not visible."},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_facts": [
                    {"fact_id": "tac_page_visible", "state": "seen", "source_frame_id": "frame-tac"},
                    {"fact_id": "fcs_page_visible", "state": "not_seen", "source_frame_id": "frame-fcs"},
                ],
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": ["tac_page_visible"],
                    "fresh_fact_ids": ["tac_page_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["visual", "gate"],
                    "action_hint": {"target": "left_mdi_brightness_selector"},
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="The left DDI is showing TAC; navigate toward the FCS page.",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "left_mdi_brightness_selector",
                    "element_id": "pnt_51",
                }
            ],
            explanations=["The left DDI is showing TAC; navigate toward the FCS page."],
            metadata={
                "diagnosis": {"step_id": "S08", "error_category": "CO"},
                "next": {"step_id": "S08"},
                "help_response": {
                    "diagnosis": {"step_id": "S08", "error_category": "CO"},
                    "next": {"step_id": "S08"},
                    "overlay": {
                        "targets": ["left_mdi_brightness_selector"],
                        "evidence": [
                            {
                                "target": "left_mdi_brightness_selector",
                                "type": "visual",
                                "ref": "VISION_FACTS.tac_page_visible@frame-tac",
                                "quote": "TAC visible.",
                                "grounding_confidence": 0.9,
                            }
                        ],
                    },
                    "explanations": ["The left DDI is showing TAC; navigate toward the FCS page."],
                }
            },
        )

        used, _reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert [action["target"] for action in response.actions] == ["left_mdi_pb18"]
        assert response.metadata["s08_visual_hint_repair_applied"] is True
        assert response.metadata["rejected_model_target"] == "left_mdi_brightness_selector"
        assert response.metadata["visual_hint_target"] == "left_mdi_pb18"
        assert response.metadata["visual_hint_evidence_ref"] == "VISION_FACTS.tac_page_visible@frame-tac"
    finally:
        loop.close()


def test_harness_validation_keeps_s08_power_hint_when_tac_seen_but_left_ddi_power_missing(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_s08_visual_evidence_seen_keeps_power_hint.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "blocked", "reason": "Left DDI must be powered."},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_facts": [
                    {"fact_id": "tac_page_visible", "state": "seen", "source_frame_id": "frame-tac"},
                    {"fact_id": "fcs_page_visible", "state": "not_seen", "source_frame_id": "frame-fcs"},
                ],
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": ["tac_page_visible"],
                    "fresh_fact_ids": ["tac_page_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": ["vars.left_ddi_on==true"],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["visual", "gate"],
                    "action_hint": {"target": "left_mdi_brightness_selector"},
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="The left DDI is showing TAC; navigate toward the FCS page.",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "left_mdi_brightness_selector",
                    "element_id": "pnt_51",
                }
            ],
            explanations=["The left DDI is showing TAC; navigate toward the FCS page."],
            metadata={
                "diagnosis": {"step_id": "S08", "error_category": "CO"},
                "next": {"step_id": "S08"},
                "help_response": {
                    "diagnosis": {"step_id": "S08", "error_category": "CO"},
                    "next": {"step_id": "S08"},
                    "overlay": {
                        "targets": ["left_mdi_brightness_selector"],
                        "evidence": [
                            {
                                "target": "left_mdi_brightness_selector",
                                "type": "visual",
                                "ref": "VISION_FACTS.tac_page_visible@frame-tac",
                                "quote": "TAC visible.",
                                "grounding_confidence": 0.9,
                            }
                        ],
                    },
                    "explanations": ["The left DDI is showing TAC; navigate toward the FCS page."],
                }
            },
        )

        _used, _reason = loop._apply_harness_validation_action_plan(response, request)

        assert [action["target"] for action in response.actions] == ["left_mdi_brightness_selector"]
        assert "s08_visual_hint_repair_applied" not in response.metadata
        assert "visual_hint_target" not in response.metadata
    finally:
        loop.close()


def test_harness_validation_does_not_trust_s08_tac_evidence_ref_when_fact_not_seen(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_s08_visual_evidence_not_seen_keeps_power_hint.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "blocked", "reason": "Left DDI must be powered."},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_facts": [
                    {"fact_id": "tac_page_visible", "state": "not_seen", "source_frame_id": "frame-tac"},
                    {"fact_id": "fcs_page_visible", "state": "not_seen", "source_frame_id": "frame-fcs"},
                ],
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": [],
                    "fresh_fact_ids": [],
                    "not_seen_fact_ids": ["tac_page_visible", "fcs_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": ["vars.left_ddi_on==true"],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["visual", "gate"],
                    "action_hint": {"target": "left_mdi_brightness_selector"},
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="The left DDI is showing TAC; navigate toward the FCS page.",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "left_mdi_brightness_selector",
                    "element_id": "pnt_51",
                    "evidence_refs": ["VISION_FACTS.tac_page_visible@frame-tac"],
                }
            ],
            explanations=["The left DDI is showing TAC; navigate toward the FCS page."],
            metadata={
                "diagnosis": {"step_id": "S08", "error_category": "CO"},
                "next": {"step_id": "S08"},
                "help_response": {
                    "diagnosis": {"step_id": "S08", "error_category": "CO"},
                    "next": {"step_id": "S08"},
                    "overlay": {
                        "targets": ["left_mdi_brightness_selector"],
                        "evidence": [
                            {
                                "target": "left_mdi_brightness_selector",
                                "type": "visual",
                                "ref": "VISION_FACTS.tac_page_visible@frame-tac",
                                "quote": "TAC visible.",
                                "grounding_confidence": 0.9,
                            }
                        ],
                    },
                    "explanations": ["The left DDI is showing TAC; navigate toward the FCS page."],
                }
            },
        )

        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "s08_unconfirmed_visual_evidence_filtered"
        assert [action["target"] for action in response.actions] == ["left_mdi_brightness_selector"]
        assert "s08_visual_hint_repair_applied" not in response.metadata
        assert "visual_hint_target" not in response.metadata
        assert response.metadata["s08_unconfirmed_visual_evidence_filtered"] is True
        help_response = response.metadata["help_response"]
        evidence = help_response["overlay"]["evidence"]
        assert all(item.get("ref") != "VISION_FACTS.tac_page_visible@frame-tac" for item in evidence)
        assert response.message is not None
        assert "not powered" in response.message
        assert all(
            "VISION_FACTS.tac_page_visible" not in ref
            for action in response.actions
            for ref in action.get("evidence_refs", [])
        )
    finally:
        loop.close()


def test_harness_validation_does_not_trust_s08_visual_action_hint_when_fact_not_seen(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_s08_visual_action_hint_not_seen_keeps_selector.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S08.completion", "status": "blocked", "reason": "FCS page is not visible."},
                    {"gate_id": "S08.precondition", "status": "allowed"},
                ],
                "vision_facts": [
                    {"fact_id": "tac_page_visible", "state": "not_seen", "source_frame_id": "frame-tac"},
                    {"fact_id": "fcs_page_visible", "state": "not_seen", "source_frame_id": "frame-fcs"},
                ],
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": [],
                    "fresh_fact_ids": [],
                    "not_seen_fact_ids": ["tac_page_visible", "fcs_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["visual", "gate"],
                    "action_hint": {"target": "left_mdi_brightness_selector"},
                    "visual_action_hint": {
                        "target": "left_mdi_pb18",
                        "reason": "TAC is visible; press PB18 to reach SUPT.",
                    },
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="The left DDI is showing TAC; navigate toward the FCS page.",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "left_mdi_brightness_selector",
                    "element_id": "pnt_51",
                }
            ],
            explanations=["The left DDI is showing TAC; navigate toward the FCS page."],
            metadata={
                "diagnosis": {"step_id": "S08", "error_category": "CO"},
                "next": {"step_id": "S08"},
                "help_response": {
                    "diagnosis": {"step_id": "S08", "error_category": "CO"},
                    "next": {"step_id": "S08"},
                    "overlay": {
                        "targets": ["left_mdi_brightness_selector"],
                        "evidence": [
                            {
                                "target": "left_mdi_brightness_selector",
                                "type": "visual",
                                "ref": "VISION_FACTS.tac_page_visible@frame-tac",
                                "quote": "TAC visible.",
                                "grounding_confidence": 0.9,
                            }
                        ],
                    },
                    "explanations": ["The left DDI is showing TAC; navigate toward the FCS page."],
                }
            },
        )

        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "s08_unconfirmed_visual_evidence_filtered"
        assert [action["target"] for action in response.actions] == ["left_mdi_brightness_selector"]
        assert "s08_visual_hint_repair_applied" not in response.metadata
        assert "visual_hint_target" not in response.metadata
        assert response.metadata["s08_unconfirmed_visual_evidence_filtered"] is True
        help_response = response.metadata["help_response"]
        evidence = help_response["overlay"]["evidence"]
        assert all(item.get("ref") != "VISION_FACTS.tac_page_visible@frame-tac" for item in evidence)
    finally:
        loop.close()


def test_s08_dual_visual_missing_overrides_model_left_nav_to_right_display_recovery(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s08_dual_visual_missing_model_left_nav.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=4,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": {
                    "S08.completion": {
                        "status": "blocked",
                        "reason_code": "s08_requires_visual_pages",
                        "reason": "FCS and BIT pages are not confirmed.",
                    },
                },
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": ["tac_page_visible", "hsi_page_visible", "hsi_map_layer_visible"],
                    "fresh_fact_ids": ["tac_page_visible", "hsi_page_visible", "hsi_map_layer_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible", "bit_root_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": [
                        "vision_facts.fcs_page_visible==seen",
                        "vision_facts.bit_root_page_visible==seen",
                    ],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                    "visual_action_hint": {"target": "left_mdi_pb18"},
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="左 DDI 当前显示 TAC 页面，请先按 PB18 切到 SUPT。",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "left_mdi_pb18",
                    "element_id": "pnt_72",
                }
            ],
            explanations=["左 DDI 当前显示 TAC 页面，请先按 PB18 切到 SUPT。"],
            metadata={"help_response": {"next": {"step_id": "S08"}, "diagnosis": {"step_id": "S08"}}},
        )

        override_used, override_reason = loop._apply_s08_visual_recovery_overlay_override(response, request)

        assert override_used is True
        assert override_reason == "validator_repair"
        assert response.metadata["presentation_fallback_reason"] == "deterministic_step:S08"
        assert [action["target"] for action in response.actions][:3] == [
            "right_mdi_brightness_selector",
            "left_mdi_pb18",
            "right_mdi_pb18",
        ]
        assert "右 DDI" in response.message
        assert response.metadata["s08_visual_recovery_overlay_override_used"] is True
    finally:
        loop.close()


def test_s08_visual_uncertain_overrides_model_ampcd_to_ddi_recovery(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s08_visual_uncertain_model_ampcd.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=4,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": {
                    "S08.completion": {
                        "status": "blocked",
                        "reason_code": "s08_requires_mpcd_on",
                        "reason": "MPCD must be powered.",
                    },
                },
                "vision_fact_summary": {
                    "status": "extractor_failed",
                    "seen_fact_ids": [],
                    "fresh_fact_ids": [],
                    "not_seen_fact_ids": ["fcsmc_final_go_result_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": ["vars.mpcd_on==true"],
                    "gate_blockers": [
                        {"ref": "GATES.S08.completion", "reason": "MPCD must be powered."}
                    ],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="当前 MPCD 未通电。请先将 AMPCD 亮度旋钮向上点亮。",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "ampcd_off_brightness_knob",
                    "element_id": "pnt_203",
                }
            ],
            explanations=["当前 MPCD 未通电。请先将 AMPCD 亮度旋钮向上点亮。"],
            metadata={"help_response": {"next": {"step_id": "S08"}, "diagnosis": {"step_id": "S08"}}},
        )

        override_used, override_reason = loop._apply_s08_visual_recovery_overlay_override(response, request)

        assert override_used is False
        assert override_reason == "ampcd_allowed_after_ddi_power"
        assert [action["target"] for action in response.actions] == ["ampcd_off_brightness_knob"]
        assert "AMPCD" in response.message
        assert "s08_visual_recovery_overlay_override_used" not in response.metadata
    finally:
        loop.close()


def test_harness_conflict_guardrail_rejects_early_step_model_answer(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_harness_conflict_guardrail.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=4,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "state_harness": {
                    "conflicts": ["early_step_from_telemetry_vs_late_display_from_vlm"],
                    "vision_evidence": {
                        "late_display_anchors": [
                            "tac_page_visible",
                            "bit_root_page_visible",
                            "hsi_page_visible",
                        ],
                        "visual_candidate_steps": ["S08", "S09"],
                    },
                },
                "vision_fact_summary": {
                    "status": "available",
                    "seen_fact_ids": ["tac_page_visible", "bit_root_page_visible", "hsi_page_visible"],
                    "fresh_fact_ids": ["tac_page_visible", "bit_root_page_visible", "hsi_page_visible"],
                    "not_seen_fact_ids": ["fcs_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "vision_facts": [
                    {
                        "fact_id": "tac_page_visible",
                        "state": "seen",
                        "source_frame_id": "frame-001",
                    },
                ],
                "deterministic_step_hint": {
                    "inferred_step_id": "S01",
                    "overlay_step_id": "S01",
                    "missing_conditions": ["vars.battery_on==true"],
                    "gate_blockers": [{"ref": "GATES.S01.completion", "reason": "Battery must be on."}],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="当前 S01 尚未完成，请先将 BATT 开关拨到 ON（右键）。",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "battery_switch",
                    "element_id": "pnt_404",
                }
            ],
            explanations=["当前 S01 尚未完成，请先将 BATT 开关拨到 ON（右键）。"],
            metadata={
                "help_response": {
                    "diagnosis": {"step_id": "S01", "error_category": "OM"},
                    "next": {"step_id": "S01"},
                    "overlay": {"targets": ["battery_switch"], "evidence": []},
                    "explanations": ["当前 S01 尚未完成，请先将 BATT 开关拨到 ON（右键）。"],
                },
                "diagnosis": {"step_id": "S01", "error_category": "OM"},
                "next": {"step_id": "S01"},
            },
        )

        guardrail_used, guardrail_reason = loop._apply_harness_conflict_guardrail(response, request)

        assert guardrail_used is True
        assert guardrail_reason == "early_step_from_telemetry_vs_late_display_from_vlm"
        assert response.actions
        assert response.actions[0]["target"] == "left_mdi_pb18"
        assert response.metadata["harness_conflict_detected"] is True
        assert response.metadata["harness_guardrail_applied"] is True
        assert response.metadata["rejected_model_step_id"] == "S01"
        assert "电瓶" not in response.message
    finally:
        loop.close()


def test_manual_throttle_guidance_rewrites_s11_throttle_reference_to_keyboard_text(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_manual_throttle_guidance_s11.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S11.completion", "status": "blocked"},
                    {"gate_id": "S11.precondition", "status": "allowed"},
                ],
                "deterministic_step_hint": {
                    "inferred_step_id": "S11",
                    "overlay_step_id": "S11",
                    "missing_conditions": ["vars.throttle_l_not_off==true"],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate"],
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="Move the left throttle out of OFF.",
            actions=[
                {
                    "kind": "highlight",
                    "target": "throttle_quadrant_reference",
                    "element_id": "pnt_504",
                    "duration_ms": 2500,
                }
            ],
            explanations=["Move the left throttle to IDLE."],
            metadata={},
        )

        rewritten, reason = loop._rewrite_manual_throttle_guidance_response(response, request)

        assert rewritten is True
        assert reason == "manual_throttle_keyboard_guidance"
        assert response.actions == []
        assert "Right Alt+Home" in response.message
        assert "throttle_quadrant_reference" not in response.message
        assert response.explanations == [response.message]
        assert response.metadata["manual_throttle_guidance_rewritten"] is True
        assert response.metadata["manual_throttle_guidance_step_id"] == "S11"
    finally:
        loop.close()


def test_action_hint_overlay_override_rewrites_s19_fcsmc_step_to_fcs_bit_switch(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s18_fcsmc_action_hint_override.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="en",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": [
                    {"gate_id": "S19.completion", "status": "allowed"},
                    {"gate_id": "S19.precondition", "status": "allowed"},
                ],
                "vision_fact_summary": {
                    "status": "uncertain",
                    "seen_fact_ids": ["fcs_page_visible", "fcsmc_page_visible"],
                    "not_seen_fact_ids": ["bit_root_page_visible"],
                    "uncertain_fact_ids": ["fcsmc_final_go_result_visible"],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S19",
                    "overlay_step_id": "S19",
                    "missing_conditions": ["vision_facts.fcsmc_final_go_result_visible==seen"],
                    "gate_blockers": [],
                    "observability_status": "partial",
                    "requires_visual_confirmation": True,
                    "step_evidence_requirements": ["delta", "gate", "visual"],
                    "action_hint": {
                        "target": "fcs_bit_switch",
                        "reason": "Hold the FCS BIT switch up while pressing PB5.",
                    },
                },
                "rag_topk": [],
            },
        )
        response = TutorResponse(
            status="ok",
            message="Return to the BIT root page first.",
            actions=[
                {
                    "kind": "highlight",
                    "target": "right_mdi_pb18",
                    "element_id": "pnt_96",
                    "duration_ms": 2500,
                }
            ],
            explanations=["Press PB18 to return to the BIT root page."],
            metadata={},
        )

        override_used, override_reason = loop._apply_action_hint_overlay_override(response, request)

        assert override_used is True
        assert override_reason == "validator_action_hint"
        assert response.actions
        assert response.actions[0]["target"] == "fcs_bit_switch"
        assert "Hold the FCS BIT switch up" in response.message
        assert "PB5" in response.message
        assert response.metadata["action_hint_overlay_override_used"] is True
        assert response.metadata["action_hint_overlay_override_target"] == "fcs_bit_switch"
        assert response.metadata["action_hint_overlay_override_kind"] == "action_hint"
        assert response.metadata["action_hint_overlay_override_fallback_reason"] == "deterministic_step:S19"
    finally:
        loop.close()


def test_build_procedural_action_hint_for_s09_starts_with_comm1_pull() -> None:
    hint = _build_procedural_action_hint(
        inferred_step_id="S09",
        vars_selected={"comm1_freq_134_000": False},
        allowed_targets=["ufc_comm1_channel_selector_pull", "ufc_key_1", "ufc_ent_button"],
    )

    assert hint == {
        "target": "ufc_comm1_channel_selector_pull",
        "reason": "Pull the UFC COMM1 channel selector to open preset 1 in the scratchpad before entering 134.000.",
    }


def test_live_loop_rewrites_s03_apu_on_wait_for_ready_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s03_apu_on_wait_ready.jsonl"
    _write_replay(
        replay_path,
        [
            {
                "schema_version": "v2",
                "seq": 1,
                "t_wall": 10.0,
                "aircraft": "FA-18C_hornet",
                "bios": {
                    "BATTERY_SW": 2,
                    "L_GEN_SW": 1,
                    "R_GEN_SW": 1,
                    "APU_CONTROL_SW": 1,
                    "APU_READY_LT": 0,
                },
                "delta": {"APU_CONTROL_SW": 1},
            }
        ],
    )

    class WrongApuOnModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="请左键点击 APU 开关将其打开。",
                actions=[],
                explanations=["请打开 APU 开关。"],
                metadata={
                    "provider": "fake_llm",
                    "help_response": {
                        "diagnosis": {"step_id": "S03", "error_category": "OM"},
                        "next": {"step_id": "S03"},
                        "overlay": {
                            "targets": ["apu_switch"],
                            "evidence": [
                                {
                                    "target": "apu_switch",
                                    "type": "var",
                                    "ref": "VARS.apu_ready",
                                    "quote": "APU READY is not yet true.",
                                    "grounding_confidence": 0.8,
                                }
                            ],
                        },
                        "explanations": ["请打开 APU 开关。"],
                        "confidence": 0.8,
                    },
                },
            )

    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(
            inferred_step_id="S03",
            missing_conditions=("vars.apu_start_support_complete==true",),
        ),
    )
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=WrongApuOnModel(),
        action_executor=RecordingExecutor(),
        lang="zh",
    )
    try:
        obs = loop.source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert "APU READY" in response.message
    assert "等待" in response.message
    assert "打开 APU 开关" not in response.message


def test_live_loop_clears_s05_throttle_reference_overlay_when_idle_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s05_throttle_keyboard_only.jsonl"
    _write_replay(
        replay_path,
        [
            {
                "schema_version": "v2",
                "seq": 1,
                "t_wall": 10.0,
                "aircraft": "FA-18C_hornet",
                "bios": {
                    "BATTERY_SW": 2,
                    "L_GEN_SW": 1,
                    "R_GEN_SW": 1,
                    "IFEI_RPM_R": 26,
                    "INT_THROTTLE_RIGHT": 0,
                },
                "delta": {"IFEI_RPM_R": 26},
            }
        ],
    )

    class WrongThrottleModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="请操作油门区域。",
                actions=[],
                explanations=["请操作油门区域。"],
                metadata={
                    "provider": "fake_llm",
                    "help_response": {
                        "diagnosis": {"step_id": "S05", "error_category": "OM"},
                        "next": {"step_id": "S05"},
                        "overlay": {
                            "targets": ["throttle_quadrant_reference"],
                            "evidence": [
                                {
                                    "target": "throttle_quadrant_reference",
                                    "type": "var",
                                    "ref": "VARS.throttle_r_idle_complete",
                                    "quote": "Right throttle idle is not complete.",
                                    "grounding_confidence": 0.8,
                                }
                            ],
                        },
                        "explanations": ["请操作油门区域。"],
                        "confidence": 0.8,
                    },
                },
            )

    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(
            inferred_step_id="S05",
            missing_conditions=("vars.throttle_r_idle_complete==true",),
        ),
    )
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=WrongThrottleModel(),
        action_executor=executor,
        lang="zh",
    )
    try:
        obs = loop.source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.actions == []
    assert report["executed"] == []
    assert "Right Shift+Home" in response.message
    assert response.metadata["manual_throttle_guidance_rewritten"] is True


def test_safe_fallback_overlay_uses_s08_left_and_right_page_navigation_when_visual_pages_missing(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_s08_visual_pages_missing_multi.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=2,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "gates": {
                    "S08.completion": {
                        "status": "blocked",
                        "reason_code": "s08_requires_pages",
                        "reason": "FCS and BIT pages are not confirmed.",
                    },
                },
                "vision_fact_summary": {
                    "status": "extractor_failed",
                    "seen_fact_ids": [],
                    "not_seen_fact_ids": ["fcs_page_visible", "bit_root_page_visible"],
                    "uncertain_fact_ids": [],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S08",
                    "overlay_step_id": "S08",
                    "missing_conditions": [
                        "vision_facts.fcs_page_visible==seen",
                        "vision_facts.bit_root_page_visible==seen",
                    ],
                    "gate_blockers": [],
                    "observability_status": "observable",
                    "step_evidence_requirements": ["var", "gate", "delta"],
                },
                "rag_topk": [],
            },
        )

        fallback_help_obj, fallback_reason = loop._build_safe_fallback_overlay_help_obj(request)
    finally:
        loop.close()

    assert fallback_reason == "deterministic_step:S08"
    assert isinstance(fallback_help_obj, dict)
    assert fallback_help_obj["overlay"]["targets"] == ["right_mdi_brightness_selector", "left_mdi_pb18"]


def test_build_procedural_action_hint_for_s09_advances_through_ufc_entry_sequence() -> None:
    allowed = [
        "ufc_comm1_channel_selector_pull",
        "ufc_key_1",
        "ufc_key_3",
        "ufc_key_4",
        "ufc_key_0",
        "ufc_ent_button",
    ]

    assert _build_procedural_action_hint(
        inferred_step_id="S09",
        vars_selected={
            "comm1_freq_134_000": False,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "305.000",
        },
        allowed_targets=allowed,
    )["target"] == "ufc_key_1"
    assert _build_procedural_action_hint(
        inferred_step_id="S09",
        vars_selected={
            "comm1_freq_134_000": False,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "     .1",
        },
        allowed_targets=allowed,
    )["target"] == "ufc_key_3"
    assert _build_procedural_action_hint(
        inferred_step_id="S09",
        vars_selected={
            "comm1_freq_134_000": False,
            "ufc_comm1_pull_pressed": True,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "    .13",
        },
        allowed_targets=allowed,
    )["target"] == "ufc_key_4"
    assert _build_procedural_action_hint(
        inferred_step_id="S09",
        vars_selected={
            "comm1_freq_134_000": False,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "    .13",
        },
        allowed_targets=allowed,
    )["target"] == "ufc_key_4"
    assert _build_procedural_action_hint(
        inferred_step_id="S09",
        vars_selected={
            "comm1_freq_134_000": False,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "   .134",
        },
        allowed_targets=allowed,
    )["target"] == "ufc_key_0"
    assert _build_procedural_action_hint(
        inferred_step_id="S09",
        vars_selected={
            "comm1_freq_134_000": False,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "134.000",
        },
        allowed_targets=allowed,
    )["target"] == "ufc_ent_button"
    assert _build_procedural_action_hint(
        inferred_step_id="S09",
        vars_selected={
            "comm1_freq_134_000": True,
            "ufc_scratchpad_string_1_display": " 1",
            "ufc_scratchpad_string_2_display": "--",
            "ufc_scratchpad_number_display": " 134.000",
        },
        allowed_targets=allowed,
    ) is None


def test_build_procedural_action_hint_for_s14_prefers_obogs_control_before_flow() -> None:
    allowed = ["obogs_control_switch", "obogs_flow_knob"]

    assert _build_procedural_action_hint(
        inferred_step_id="S14",
        vars_selected={"obogs_switch_on": False, "obogs_flow_on": True},
        allowed_targets=allowed,
    ) == {
        "target": "obogs_control_switch",
        "reason": "OBOGS control is not yet ON; switch OBOGS on first.",
    }
    assert _build_procedural_action_hint(
        inferred_step_id="S14",
        vars_selected={"obogs_switch_on": True, "obogs_flow_on": False},
        allowed_targets=allowed,
    ) == {
        "target": "obogs_flow_knob",
        "reason": "OBOGS control is already ON, but FLOW is not yet ON; set the OXY FLOW knob next.",
    }
    assert _build_procedural_action_hint(
        inferred_step_id="S14",
        vars_selected={"obogs_switch_on": True, "obogs_flow_on": True},
        allowed_targets=allowed,
    ) is None


def test_build_procedural_action_hint_for_s18_prefers_right_ddi_pb5() -> None:
    allowed = ["right_mdi_pb18", "right_mdi_pb5"]

    assert _build_procedural_action_hint(
        inferred_step_id="S18",
        vars_selected={"fcs_bit_switch_up": False},
        allowed_targets=allowed,
    ) == {
        "target": "right_mdi_pb5",
        "reason": "On the right DDI BIT FAILURES page, press PB5 to enter the FCS-MC BIT page before holding the FCS BIT switch.",
    }
    assert _build_procedural_action_hint(
        inferred_step_id="S18",
        vars_selected={"fcs_bit_switch_up": True},
        allowed_targets=allowed,
    ) == {
        "target": "right_mdi_pb5",
        "reason": "On the right DDI BIT FAILURES page, press PB5 to enter the FCS-MC BIT page before holding the FCS BIT switch.",
    }


def test_build_procedural_action_hint_for_s19_prefers_fcs_bit_switch_on_fcsmc_page() -> None:
    allowed = ["fcs_bit_switch", "right_mdi_pb5"]

    assert _build_procedural_action_hint(
        inferred_step_id="S19",
        vars_selected={"fcs_bit_switch_up": False},
        allowed_targets=allowed,
        vision_fact_summary={"seen_fact_ids": ["fcsmc_page_visible"]},
    ) == {
        "targets": ["fcs_bit_switch", "right_mdi_pb5"],
        "reason": "Hold the FCS BIT switch up (Y) while pressing Right DDI PB5 to start the FCS BIT.",
    }
    assert _build_procedural_action_hint(
        inferred_step_id="S19",
        vars_selected={"fcs_bit_switch_up": False},
        allowed_targets=allowed,
        vision_fact_summary={"seen_fact_ids": ["fcsmc_final_go_result_visible"]},
    ) is None


def test_build_procedural_action_hint_for_s12_prompts_ampcd_pb19_after_ins_mode_set() -> None:
    allowed = ["ins_mode_knob", "ampcd_pb19"]

    assert _build_procedural_action_hint(
        inferred_step_id="S12",
        vars_selected={"ins_mode_cv_or_gnd": True, "ins_fast_align_complete": False},
        allowed_targets=allowed,
    ) == {
        "target": "ampcd_pb19",
        "reason": "INS mode is set for alignment; press AMPCD PB19 to start the fast alignment self-test.",
    }


def test_build_procedural_action_hint_for_split_four_down_steps_uses_only_current_step_target() -> None:
    cases = {
        "S20": ("refuel_probe_switch", "Extend the refueling probe for the four-down check."),
        "S21": ("refuel_probe_switch", "Retract the refueling probe after confirming extension."),
        "S22": ("launch_bar_switch", "Extend the launch bar for the four-down check."),
        "S23": ("launch_bar_switch", "Retract the launch bar after confirming extension."),
        "S24": ("arresting_hook_handle", "Lower the arresting hook for the four-down check."),
        "S25": ("arresting_hook_handle", "Raise the arresting hook after confirming it is down."),
        "S26": ("pitot_heater_switch", "Turn pitot heat ON."),
        "S27": ("flap_switch", "Move the flap switch to AUTO."),
    }

    for step_id, (target, reason) in cases.items():
        assert _build_procedural_action_hint(
            inferred_step_id=step_id,
            vars_selected={"probe_cycle_complete": True, "pitot_heat_on": False},
            allowed_targets=list({target, "refuel_probe_switch", "launch_bar_switch", "arresting_hook_handle", "pitot_heater_switch", "flap_switch"}),
            step_interacted_targets=["refuel_probe_switch", "launch_bar_switch"],
        ) == {"target": target, "reason": reason}


def test_procedural_guidance_rewrite_mentions_s09_frequency_134(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s09_frequency_rewrite.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
        max_overlay_targets=2,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {"comm1_freq_134_000": False},
                "deterministic_step_hint": {
                    "inferred_step_id": "S09",
                    "missing_conditions": ["vars.comm1_freq_134_000==true"],
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="请设置 COMM1。",
            actions=[],
            explanations=["请设置 COMM1。"],
            metadata={},
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s09_comm1_frequency_guidance"
        assert "134.000" in response.message
        assert "COMM1" in response.message
        assert "1-3-4-0-0-0" not in response.message
    finally:
        loop.close()


def test_procedural_guidance_rewrite_waits_without_highlight_when_s19_in_test(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s19_in_test_wait.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
        max_overlay_targets=2,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vision_fact_summary": {
                    "seen_fact_ids": ["fcsmc_page_visible", "fcsmc_in_test_visible"],
                    "fresh_fact_ids": ["fcsmc_in_test_visible"],
                    "not_seen_fact_ids": ["fcsmc_final_go_result_visible"],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S19",
                    "missing_conditions": ["vision_facts.fcsmc_final_go_result_visible==seen"],
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="请保持 FCS BIT 开关向上等待测试完成。",
            actions=[{"kind": "highlight", "target": "fcs_bit_switch"}],
            explanations=["请保持 FCS BIT 开关向上等待测试完成。"],
            metadata={"next": {"step_id": "S19"}},
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s19_fcs_bit_in_test_wait"
        assert response.actions == []
        assert "无需继续保持" in response.message
        assert "等待最终 GO" in response.message
    finally:
        loop.close()


def test_procedural_guidance_does_not_wait_on_s19_intermediate_without_in_test(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s19_intermediate_not_wait.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
        max_overlay_targets=2,
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vision_fact_summary": {
                    "seen_fact_ids": ["fcsmc_page_visible", "fcsmc_intermediate_result_visible"],
                    "fresh_fact_ids": ["fcsmc_intermediate_result_visible"],
                    "not_seen_fact_ids": ["fcsmc_in_test_visible", "fcsmc_final_go_result_visible"],
                },
                "gates": [
                    {"gate_id": "S19.completion", "status": "blocked"},
                    {"gate_id": "S19.precondition", "status": "allowed"},
                ],
                "deterministic_step_hint": {
                    "inferred_step_id": "S19",
                    "missing_conditions": ["vision_facts.fcsmc_final_go_result_visible==seen"],
                    "step_evidence_requirements": ["gate", "visual"],
                    "action_hint": {"targets": ["fcs_bit_switch", "right_mdi_pb5"]},
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="FCS BIT 已经开始运行，请松开并等待最终 GO。",
            actions=[],
            explanations=["FCS BIT 已经开始运行，请松开并等待最终 GO。"],
            metadata={
                "next": {"step_id": "S19"},
                "help_response": {
                    "diagnosis": {"step_id": "S19", "error_category": "CO"},
                    "next": {"step_id": "S19"},
                    "overlay": {"targets": [], "evidence": []},
                    "explanations": ["FCS BIT 已经开始运行，请松开并等待最终 GO。"],
                },
            },
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s19_intermediate_requires_bit_start"
        assert response.actions[0]["target"] == "fcs_bit_switch"
        assert response.actions[1]["target"] == "right_mdi_pb5"
        assert "启动测试" in response.message
        assert "等待最终 GO" not in response.message
    finally:
        loop.close()


def test_s19_final_go_fresh_fact_suppresses_s19_fallback(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s19_final_go_fresh.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vision_fact_summary": {
                    "seen_fact_ids": ["fcsmc_page_visible"],
                    "fresh_fact_ids": ["fcsmc_final_go_result_visible"],
                    "not_seen_fact_ids": ["fcsmc_final_go_result_visible"],
                },
                "vars": {"probe_cycle_complete": True},
                "gates": [
                    {"gate_id": "S20.completion", "status": "blocked"},
                    {"gate_id": "S20.precondition", "status": "allowed"},
                ],
                "deterministic_step_hint": {
                    "inferred_step_id": "S19",
                    "missing_conditions": ["vision_facts.fcsmc_final_go_result_visible==seen"],
                    "observability_status": "partial",
                    "requires_visual_confirmation": True,
                    "action_hint": {"targets": ["fcs_bit_switch", "right_mdi_pb5"]},
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="继续按住 FCS BIT。",
            actions=[],
            explanations=["继续按住 FCS BIT。"],
            metadata={
                "next": {"step_id": "S19"},
                "help_response": {
                    "diagnosis": {"step_id": "S19", "error_category": "CO"},
                    "next": {"step_id": "S19"},
                    "overlay": {"targets": ["fcs_bit_switch"], "evidence": []},
                    "explanations": ["继续按住 FCS BIT。"],
                },
            },
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s19_final_go_complete"
        assert response.actions
        assert response.metadata["next"]["step_id"] == "S20"
        assert response.metadata["help_response"]["next"]["step_id"] == "S20"
        assert response.metadata["help_response"]["diagnosis"]["step_id"] == "S20"
        assert response.metadata["help_response"]["overlay"]["targets"] == []
        assert response.metadata["s19_final_go_guardrail_applied"] is True
        assert response.metadata["s19_final_go_s20_overlay_applied"] is True
        assert response.actions[0]["target"] in {
            "refuel_probe_switch",
            "launch_bar_switch",
            "arresting_hook_handle",
            "pitot_heater_switch",
        }
        assert loop._should_use_deterministic_overlay_fallback(response, request, None) is False
    finally:
        loop.close()


def test_s19_final_go_trace_preserves_raw_model_step_before_guardrail(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s19_final_go_trace.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="en",
    )
    try:
        request = TutorRequest(
            request_id="cycle-s19-final-go",
            actor="learner",
            intent="help",
            message="help",
            context={
                "candidate_steps": [
                    {"step_id": "S19", "source": "deterministic"},
                    {"step_id": "S20", "source": "visual_completion"},
                ],
                "evidence_packet_summary": {"vision_status": "available"},
                "vision_fact_summary": {
                    "seen_fact_ids": ["fcsmc_final_go_result_visible"],
                    "fresh_fact_ids": [],
                    "not_seen_fact_ids": [],
                },
                "vars": {"probe_cycle_complete": True},
                "gates": [
                    {"gate_id": "S20.completion", "status": "blocked"},
                    {"gate_id": "S20.precondition", "status": "allowed"},
                ],
                "deterministic_step_hint": {
                    "inferred_step_id": "S19",
                    "missing_conditions": ["vision_facts.fcsmc_final_go_result_visible==seen"],
                    "observability_status": "partial",
                    "requires_visual_confirmation": True,
                    "action_hint": {"targets": ["fcs_bit_switch", "right_mdi_pb5"]},
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="Keep holding FCS BIT.",
            actions=[],
            explanations=["Keep holding FCS BIT."],
            metadata={
                "generation_mode": "model",
                "next": {"step_id": "S19"},
                "help_response": {
                    "diagnosis": {"step_id": "S19", "error_category": "CO"},
                    "next": {"step_id": "S19"},
                    "overlay": {"targets": ["fcs_bit_switch"], "evidence": []},
                    "explanations": ["Keep holding FCS BIT."],
                },
            },
        )

        loop._capture_model_raw_help_response(response)
        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)
        loop._annotate_response_audit_metadata(response)
        trace = _build_harness_trace_metadata(
            request=request,
            response=response,
            vision_selection=HelpCycleVisionSelection(
                status="available",
                observation_ref=None,
                observation_seq=None,
                observation_t_wall_s=None,
                observation_t_wall_ms=10000,
                trigger_wall_ms=10000,
                sync_window_ms=100,
                vision_used=True,
                frame_id="frame-s19-final-go",
                sync_status="matched_exact",
                sync_delta_ms=0,
                frame_stale=False,
                frame_ids=["frame-s19-final-go"],
                selected_frames=[],
                pre_trigger_frame=None,
                trigger_frame=None,
                sync_miss_reason=None,
            ),
            vision_fact_context={
                "status": "available",
                "vision_fact_summary": {
                    "seen_fact_ids": ["fcsmc_final_go_result_visible"],
                    "fresh_fact_ids": [],
                    "frame_ids": ["frame-s19-final-go"],
                },
                "vision_facts": [],
                "metadata": {"extractor_used": True},
            },
        )

        assert rewritten is True
        assert reason == "s19_final_go_complete"
        assert response.metadata["next"]["step_id"] == "S20"
        assert response.metadata["model_raw_help_response"]["next"]["step_id"] == "S19"
        assert trace["model_decision"]["step_id"] == "S19"
        assert trace["final_action_plan"]["step_id"] == "S20"
    finally:
        loop.close()


def test_procedural_guidance_waits_without_highlight_for_s21_probe_retracting(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s21_probe_retracting.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {
                    "probe_switch_value": 1,
                    "ext_refuel_probe_value": 5606,
                    "probe_retracted": False,
                },
                "state_harness": {
                    "telemetry_window_digest": {
                        "frame_count": 2,
                        "changed_vars": [
                            {
                                "var": "ext_refuel_probe_value",
                                "first_value": 6200,
                                "last_value": 5606,
                                "transition_count": 1,
                            }
                        ],
                    }
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S21",
                    "missing_conditions": ["vars.ext_refuel_probe_value in [0,5000]"],
                    "action_hint": {"target": "refuel_probe_switch"},
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="当前 S21 尚未完成，请先满足：vars.ext_refuel_probe_value in [0,5000]。",
            actions=[{"kind": "highlight", "target": "refuel_probe_switch"}],
            explanations=["当前 S21 尚未完成，请先满足：vars.ext_refuel_probe_value in [0,5000]。"],
            metadata={
                "next": {"step_id": "S21"},
                "diagnosis": {"step_id": "S21", "error_category": "CO"},
                "help_response": {
                    "diagnosis": {"step_id": "S21", "error_category": "CO"},
                    "next": {"step_id": "S21"},
                    "overlay": {"targets": ["refuel_probe_switch"], "evidence": []},
                    "explanations": ["当前 S21 尚未完成，请先满足：vars.ext_refuel_probe_value in [0,5000]。"],
                },
            },
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s21_refuel_probe_retracting_wait"
        assert response.actions == []
        assert "正在收起" in response.message
        assert "等待" in response.message
        assert "vars.ext_refuel_probe_value" not in response.message
        assert all("vars.ext_refuel_probe_value" not in item for item in response.explanations)
        assert response.metadata["help_response"]["overlay"]["targets"] == []
        assert all(
            "vars.ext_refuel_probe_value" not in item
            for item in response.metadata["help_response"]["explanations"]
        )
        planned, plan_reason = loop._apply_harness_validation_action_plan(response, request)
        assert planned is False
        assert plan_reason == "refuel_probe_motion_wait_already_rewritten"
        assert loop._should_use_deterministic_overlay_fallback(response, request, None) is False
    finally:
        loop.close()


def test_procedural_guidance_waits_without_highlight_for_s20_probe_extending(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s20_probe_extending.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {
                    "probe_switch_value": 0,
                    "ext_refuel_probe_value": 12000,
                    "probe_extended": False,
                },
                "state_harness": {
                    "telemetry_window_digest": {
                        "frame_count": 2,
                        "changed_vars": [
                            {
                                "var": "ext_refuel_probe_value",
                                "first_value": 8000,
                                "last_value": 12000,
                                "transition_count": 1,
                            }
                        ],
                    }
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S20",
                    "missing_conditions": ["vars.ext_refuel_probe_value in [60000,65535]"],
                    "action_hint": {"target": "refuel_probe_switch"},
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="当前 S20 尚未完成，请先操作 refuel_probe_switch。",
            actions=[{"kind": "highlight", "target": "refuel_probe_switch"}],
            explanations=["当前 S20 尚未完成，请先操作 refuel_probe_switch。"],
            metadata={"next": {"step_id": "S20"}},
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s20_refuel_probe_extending_wait"
        assert response.actions == []
        assert "正在伸出" in response.message
        assert "等待" in response.message
        assert "vars.ext_refuel_probe_value" not in response.message
        assert "refuel_probe_switch" not in response.message
        assert all("vars.ext_refuel_probe_value" not in item for item in response.explanations)
        assert all("refuel_probe_switch" not in item for item in response.explanations)
        assert all(
            "vars.ext_refuel_probe_value" not in item
            for item in response.metadata["help_response"]["explanations"]
        )
        planned, plan_reason = loop._apply_harness_validation_action_plan(response, request)
        assert planned is False
        assert plan_reason == "refuel_probe_motion_wait_already_rewritten"
        assert loop._should_use_deterministic_overlay_fallback(response, request, None) is False
    finally:
        loop.close()


def test_procedural_guidance_advances_to_s22_when_s21_probe_retracted(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s21_probe_retracted.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {
                    "probe_switch_value": 1,
                    "ext_refuel_probe_value": 4652,
                    "probe_retracted": True,
                },
                "gates": {
                    "S22.completion": {
                        "status": "blocked",
                        "step_id": "S22",
                        "reason_code": "s22_requires_launch_bar_extended",
                        "reason": "Launch bar must be extended.",
                    }
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S21",
                    "missing_conditions": ["vars.ext_refuel_probe_value in [0,5000]"],
                    "action_hint": {"target": "refuel_probe_switch"},
                    "gate_blockers": [
                        {
                            "ref": "S22.completion",
                            "reason_code": "s22_requires_launch_bar_extended",
                            "reason": "Launch bar must be extended.",
                        }
                    ],
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="当前 S21 尚未完成，请先满足：vars.ext_refuel_probe_value in [0,5000]。",
            actions=[{"kind": "highlight", "target": "refuel_probe_switch"}],
            explanations=["当前 S21 尚未完成，请先满足：vars.ext_refuel_probe_value in [0,5000]。"],
            metadata={"next": {"step_id": "S21"}},
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s21_refuel_probe_retracted_complete"
        assert response.metadata["next"]["step_id"] == "S22"
        assert response.metadata["diagnosis"]["step_id"] == "S22"
        assert response.metadata["help_response"]["next"]["step_id"] == "S22"
        assert response.metadata["help_response"]["overlay"]["targets"] == []
        assert response.metadata["refuel_probe_completion_s22_overlay_applied"] is True
        assert response.actions
        assert response.actions[0]["target"] == "launch_bar_switch"
        assert "vars.ext_refuel_probe_value" not in response.message
        assert all("vars.ext_refuel_probe_value" not in item for item in response.explanations)
        assert all(
            "vars.ext_refuel_probe_value" not in item
            for item in response.metadata["help_response"]["explanations"]
        )
        overridden, override_reason = loop._apply_action_hint_overlay_override(response, request)
        assert overridden is False
        assert override_reason == "refuel_probe_motion_wait_already_rewritten"
        assert response.actions[0]["target"] == "launch_bar_switch"
    finally:
        loop.close()


def test_procedural_guidance_advances_to_s21_when_s20_probe_extended(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s20_probe_extended.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {
                    "probe_switch_value": 0,
                    "ext_refuel_probe_value": 65000,
                    "probe_extended": True,
                },
                "gates": {
                    "S21.completion": {
                        "status": "blocked",
                        "step_id": "S21",
                        "reason_code": "s21_requires_probe_retracted",
                        "reason": "Refueling probe must be fully retracted.",
                    }
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S20",
                    "missing_conditions": ["vars.ext_refuel_probe_value in [60000,65535]"],
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="当前 S20 尚未完成，请先操作 refuel_probe_switch。",
            actions=[{"kind": "highlight", "target": "refuel_probe_switch"}],
            explanations=["当前 S20 尚未完成，请先操作 refuel_probe_switch。"],
            metadata={"next": {"step_id": "S20"}},
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s20_refuel_probe_extended_complete"
        assert response.metadata["next"]["step_id"] == "S21"
        assert response.metadata["diagnosis"]["step_id"] == "S21"
        assert response.metadata["refuel_probe_completion_s21_overlay_applied"] is True
        assert response.actions
        assert response.actions[0]["target"] == "refuel_probe_switch"
        assert "vars.ext_refuel_probe_value" not in response.message
        assert all("vars.ext_refuel_probe_value" not in item for item in response.explanations)
        assert all(
            "vars.ext_refuel_probe_value" not in item
            for item in response.metadata["help_response"]["explanations"]
        )
        assert "尚未完成" not in response.message
    finally:
        loop.close()


def test_procedural_guidance_rewrite_mentions_s09_frequency_134(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s09_frequency_rewrite.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vars": {"comm1_freq_134_000": False},
                "deterministic_step_hint": {
                    "inferred_step_id": "S09",
                    "missing_conditions": ["vars.comm1_freq_134_000==true"],
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="请设置 COMM1。",
            actions=[],
            explanations=["请设置 COMM1。"],
            metadata={},
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s09_comm1_frequency_guidance"
        assert "134.000" in response.message
        assert "COMM1" in response.message
        assert "1-3-4-0-0-0" not in response.message
    finally:
        loop.close()


def test_procedural_guidance_rewrite_waits_without_highlight_when_s19_in_test(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s19_in_test_wait.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "vision_fact_summary": {
                    "seen_fact_ids": ["fcsmc_page_visible", "fcsmc_in_test_visible"],
                    "fresh_fact_ids": ["fcsmc_in_test_visible"],
                    "not_seen_fact_ids": ["fcsmc_final_go_result_visible"],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S19",
                    "missing_conditions": ["vision_facts.fcsmc_final_go_result_visible==seen"],
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="请保持 FCS BIT 开关向上等待测试完成。",
            actions=[{"kind": "highlight", "target": "fcs_bit_switch"}],
            explanations=["请保持 FCS BIT 开关向上等待测试完成。"],
            metadata={"next": {"step_id": "S19"}},
        )

        rewritten, reason = loop._rewrite_procedural_guidance_response(response, request)

        assert rewritten is True
        assert reason == "s19_fcs_bit_in_test_wait"
        assert response.actions == []
        assert "无需继续保持" in response.message
        assert "等待最终 GO" in response.message
    finally:
        loop.close()


def test_action_hint_overlay_override_uses_split_s26_pitot_target() -> None:
    response = TutorResponse(
        message="Extend the refuel probe.",
        explanations=["Extend the refuel probe."],
        actions=[
            {
                "type": "overlay",
                "intent": "highlight",
                "target": "refuel_probe_switch",
                "element_id": "pnt_341",
            }
        ],
        metadata={
            "next": {"step_id": "S26"},
            "diagnosis": {"step_id": "S26", "error_category": "OM"},
        },
    )
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "overlay_target_allowlist": ["refuel_probe_switch", "launch_bar_switch"],
            "gates": [
                {"gate_id": "S26.completion", "status": "blocked"},
                {"gate_id": "S26.precondition", "status": "allowed"},
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S26",
                "overlay_step_id": "S26",
                "requires_visual_confirmation": False,
                "step_evidence_requirements": ["gate", "rag", "delta"],
                "action_hint": {"target": "pitot_heater_switch", "reason": "Turn pitot heat ON."},
            },
            "rag_topk": [],
        },
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(Path("/dev/null")),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        request.context["overlay_target_allowlist"] = list(loop.overlay_allowlist)

        used, reason = loop._apply_action_hint_overlay_override(response, request)

        assert used is True
        assert reason == "validator_action_hint"
        assert response.actions[0]["target"] == "pitot_heater_switch"
        assert response.metadata["action_hint_overlay_override_target"] == "pitot_heater_switch"
        assert response.message == "Turn pitot heat ON."
    finally:
        loop.close()


def test_harness_validation_action_plan_records_s26_repair_metadata() -> None:
    response = TutorResponse(
        message="Extend the refuel probe.",
        explanations=["Extend the refuel probe."],
        actions=[
            {
                "type": "overlay",
                "intent": "highlight",
                "target": "refuel_probe_switch",
                "element_id": "pnt_341",
            }
        ],
        metadata={
            "next": {"step_id": "S26"},
            "diagnosis": {"step_id": "S26", "error_category": "OM"},
            "help_response": {
                "diagnosis": {"step_id": "S26", "error_category": "OM"},
                "next": {"step_id": "S26"},
                "overlay": {
                    "targets": ["refuel_probe_switch"],
                    "evidence": [
                        {
                            "target": "refuel_probe_switch",
                            "type": "gate",
                            "ref": "GATES.S26.completion",
                            "quote": "blocked",
                        }
                    ],
                },
                "explanations": ["Extend the refuel probe."],
            },
        },
    )
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "overlay_target_allowlist": ["refuel_probe_switch", "pitot_heater_switch"],
            "gates": [
                {"gate_id": "S26.completion", "status": "blocked"},
                {"gate_id": "S26.precondition", "status": "allowed"},
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S26",
                "overlay_step_id": "S26",
                "requires_visual_confirmation": False,
                "step_evidence_requirements": ["gate"],
                "action_hint": {"target": "pitot_heater_switch", "reason": "Turn pitot heat ON."},
            },
            "rag_topk": [],
        },
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(Path("/dev/null")),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=2,
    )
    try:
        request.context["overlay_target_allowlist"] = list(loop.overlay_allowlist)

        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "validator_action_hint"
        assert response.actions[0]["target"] == "pitot_heater_switch"
        assert response.metadata["validator_rejected"] is True
        assert response.metadata["repair_applied"] is True
        assert response.metadata["final_action_plan_source"] == "validator_action_hint"
        assert response.metadata["harness_validator_fallback_reason"] == "deterministic_step:S26"
    finally:
        loop.close()


def _annotate_test_final_metadata(
    loop: LiveDcsTutorLoop,
    response: TutorResponse,
    request: TutorRequest,
) -> None:
    loop._annotate_response_audit_metadata(response)
    _build_harness_trace_metadata(
        request=request,
        response=response,
        vision_selection=HelpCycleVisionSelection(
            status="vision_not_required",
            observation_ref=None,
            observation_seq=None,
            observation_t_wall_s=None,
            observation_t_wall_ms=None,
            trigger_wall_ms=None,
            sync_window_ms=None,
            vision_used=False,
            frame_id=None,
            sync_status=None,
            sync_delta_ms=None,
            frame_stale=False,
            frame_ids=[],
            selected_frames=[],
            pre_trigger_frame=None,
            trigger_frame=None,
            sync_miss_reason=None,
        ),
        vision_fact_context={
            "status": "vision_not_required",
            "vision_fact_summary": {"status": "vision_not_required"},
            "vision_facts": [],
        },
    )


def test_harness_validation_action_plan_uses_s08_all_displays_off_targets() -> None:
    response = TutorResponse(
        message="Turn on the left DDI.",
        explanations=["Turn on the left DDI."],
        actions=[
            {
                "type": "overlay",
                "intent": "highlight",
                "target": "left_mdi_brightness_selector",
                "element_id": "pnt_51",
            }
        ],
        metadata={
            "next": {"step_id": "S08"},
            "diagnosis": {"step_id": "S08", "error_category": "OM"},
            "help_response": {
                "diagnosis": {"step_id": "S08", "error_category": "OM"},
                "next": {"step_id": "S08"},
                "overlay": {
                    "targets": ["left_mdi_brightness_selector"],
                    "evidence": [
                        {
                            "target": "left_mdi_brightness_selector",
                            "type": "gate",
                            "ref": "GATES.S08.completion",
                            "quote": "blocked",
                        }
                    ],
                },
                "explanations": ["Turn on the left DDI."],
            },
        },
    )
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "overlay_target_allowlist": [],
            "gates": [
                {"gate_id": "S08.completion", "status": "blocked", "reason": "Displays must be powered."},
                {"gate_id": "S08.precondition", "status": "allowed"},
            ],
            "vars": {
                "left_ddi_on": False,
                "right_ddi_on": False,
                "mpcd_on": False,
                "hud_on": False,
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S08",
                "overlay_step_id": "S08",
                "missing_conditions": [
                    "vars.left_ddi_on==true",
                    "vars.right_ddi_on==true",
                    "vars.mpcd_on==true",
                    "vars.hud_on==true",
                ],
                "step_evidence_requirements": ["var", "gate"],
            },
            "rag_topk": [],
        },
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(Path("/dev/null")),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=4,
    )
    try:
        request.context["overlay_target_allowlist"] = list(loop.overlay_allowlist)

        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "state_action_planner"
        assert [action["target"] for action in response.actions] == [
            "left_mdi_brightness_selector",
            "right_mdi_brightness_selector",
            "ampcd_off_brightness_knob",
            "hud_symbology_brightness_knob",
        ]
        assert "DDI" in response.message
        assert "页面" not in response.message
        assert response.metadata["final_action_plan_source"] == "state_action_planner"
        _annotate_test_final_metadata(loop, response, request)
        assert response.metadata["final_overlay_targets"] == [
            "left_mdi_brightness_selector",
            "right_mdi_brightness_selector",
            "ampcd_off_brightness_knob",
            "hud_symbology_brightness_knob",
        ]
        assert response.metadata["final_public_response"]["actions"][0]["target"] == "left_mdi_brightness_selector"
    finally:
        loop.close()


def test_harness_validation_action_plan_uses_s09_initial_numeric_sequence_targets() -> None:
    response = TutorResponse(
        message="Pull COMM1.",
        explanations=["Pull COMM1."],
        actions=[
            {
                "type": "overlay",
                "intent": "highlight",
                "target": "ufc_comm1_channel_selector_pull",
                "element_id": "pnt_301",
            }
        ],
        metadata={
            "next": {"step_id": "S09"},
            "diagnosis": {"step_id": "S09", "error_category": "OM"},
            "help_response": {
                "diagnosis": {"step_id": "S09", "error_category": "OM"},
                "next": {"step_id": "S09"},
                "overlay": {
                    "targets": ["ufc_comm1_channel_selector_pull"],
                    "evidence": [
                        {
                            "target": "ufc_comm1_channel_selector_pull",
                            "type": "gate",
                            "ref": "GATES.S09.completion",
                            "quote": "blocked",
                        }
                    ],
                },
                "explanations": ["Pull COMM1."],
            },
        },
    )
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "overlay_target_allowlist": [],
            "gates": [
                {"gate_id": "S09.completion", "status": "blocked", "reason": "COMM1 frequency is not 134.000."},
                {"gate_id": "S09.precondition", "status": "allowed"},
            ],
            "vars": {
                "comm1_freq_134_000": False,
                "ufc_comm1_pull_pressed": True,
                "ufc_scratchpad_string_1_display": "1-",
                "ufc_scratchpad_string_2_display": "-",
                "ufc_scratchpad_number_display": "305.000",
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "overlay_step_id": "S09",
                "missing_conditions": ["vars.comm1_freq_134_000==true"],
                "step_evidence_requirements": ["var", "gate"],
            },
            "rag_topk": [],
        },
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(Path("/dev/null")),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=4,
    )
    try:
        request.context["overlay_target_allowlist"] = list(loop.overlay_allowlist)

        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "state_action_planner"
        assert [action["target"] for action in response.actions] == [
            "ufc_key_1",
            "ufc_key_3",
            "ufc_key_4",
            "ufc_key_0",
        ]
        assert "134.000" in response.message
        assert "1-3-4-0-0-0" in response.message
        assert "ENT" in response.message
        assert response.metadata["help_response"]["overlay"]["targets"] == [
            "ufc_key_1",
            "ufc_key_3",
            "ufc_key_4",
            "ufc_key_0",
        ]
        _annotate_test_final_metadata(loop, response, request)
        assert response.metadata["final_overlay_targets"] == [
            "ufc_key_1",
            "ufc_key_3",
            "ufc_key_4",
            "ufc_key_0",
        ]
        assert response.metadata["final_public_response"]["message"] == response.message
        assert response.metadata["final_public_response"]["actions"][0]["target"] == "ufc_key_1"
    finally:
        loop.close()


def test_harness_validation_action_plan_s09_empty_scratchpad_mentions_selector_if_needed() -> None:
    response = TutorResponse(
        message="Pull COMM1.",
        explanations=["Pull COMM1."],
        actions=[
            {
                "type": "overlay",
                "intent": "highlight",
                "target": "ufc_comm1_channel_selector_pull",
                "element_id": "pnt_301",
            }
        ],
        metadata={
            "next": {"step_id": "S09"},
            "diagnosis": {"step_id": "S09", "error_category": "OM"},
            "help_response": {
                "diagnosis": {"step_id": "S09", "error_category": "OM"},
                "next": {"step_id": "S09"},
                "overlay": {
                    "targets": ["ufc_comm1_channel_selector_pull"],
                    "evidence": [
                        {
                            "target": "ufc_comm1_channel_selector_pull",
                            "type": "gate",
                            "ref": "GATES.S09.completion",
                            "quote": "blocked",
                        }
                    ],
                },
                "explanations": ["Pull COMM1."],
            },
        },
    )
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "overlay_target_allowlist": [],
            "gates": [
                {"gate_id": "S09.completion", "status": "blocked", "reason": "COMM1 frequency is not 134.000."},
                {"gate_id": "S09.precondition", "status": "allowed"},
            ],
            "vars": {"comm1_freq_134_000": False},
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "overlay_step_id": "S09",
                "missing_conditions": ["vars.comm1_freq_134_000==true"],
                "step_evidence_requirements": ["var", "gate"],
            },
            "rag_topk": [],
        },
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(Path("/dev/null")),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
        max_overlay_targets=4,
    )
    try:
        request.context["overlay_target_allowlist"] = list(loop.overlay_allowlist)

        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "state_action_planner"
        assert [action["target"] for action in response.actions] == ["ufc_comm1_channel_selector_pull"]
        assert "COMM1" in response.message
        assert "134.000" in response.message
        assert "1-3-4-0-0-0" not in response.message
        _annotate_test_final_metadata(loop, response, request)
        assert response.metadata["final_overlay_targets"] == ["ufc_comm1_channel_selector_pull"]
        assert response.metadata["final_public_response"]["message"] == response.message
    finally:
        loop.close()


def test_harness_validation_action_plan_waits_without_highlight_for_s20_probe_moving() -> None:
    response = TutorResponse(
        message="Extend the refuel probe.",
        explanations=["Extend the refuel probe."],
        actions=[
            {
                "type": "overlay",
                "intent": "highlight",
                "target": "refuel_probe_switch",
                "element_id": "pnt_probe",
            }
        ],
        metadata={
            "next": {"step_id": "S20"},
            "diagnosis": {"step_id": "S20", "error_category": "OM"},
            "help_response": {
                "diagnosis": {"step_id": "S20", "error_category": "OM"},
                "next": {"step_id": "S20"},
                "overlay": {
                    "targets": ["refuel_probe_switch"],
                    "evidence": [
                        {
                            "target": "refuel_probe_switch",
                            "type": "gate",
                            "ref": "GATES.S20.completion",
                            "quote": "blocked",
                        }
                    ],
                },
                "explanations": ["Extend the refuel probe."],
            },
        },
    )
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "vars": {
                "probe_switch_value": 0,
                "ext_refuel_probe_value": 12000,
                "probe_extended": False,
            },
            "telemetry_window_frames": [
                {"seq": 1, "t_wall": 1.0, "vars": {"ext_refuel_probe_value": 8000}},
                {"seq": 2, "t_wall": 2.0, "vars": {"ext_refuel_probe_value": 12000}},
            ],
            "overlay_target_allowlist": ["refuel_probe_switch"],
            "gates": {"S20.completion": {"status": "blocked"}},
            "deterministic_step_hint": {
                "inferred_step_id": "S20",
                "overlay_step_id": "S20",
                "missing_conditions": ["vars.ext_refuel_probe_value in [60000,65535]"],
            },
        },
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(Path("/dev/null")),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "state_action_planner_wait"
        assert response.actions == []
        assert response.metadata["harness_action_plan"]["text_only"] is True
        assert response.metadata["final_action_plan_source"] == "state_action_planner_wait"
        assert response.metadata["help_response"]["overlay"]["targets"] == []
        assert "正在伸出" in response.message
        assert "manual_throttle_guidance_original_message" not in response.metadata
        assert "manual_throttle_guidance_original_explanations" not in response.metadata
    finally:
        loop.close()


def test_harness_validation_action_plan_rewrites_manual_throttle_to_text_only() -> None:
    response = TutorResponse(
        message="Highlight throttle.",
        explanations=["Highlight throttle."],
        actions=[
            {
                "type": "overlay",
                "intent": "highlight",
                "target": "throttle_quadrant_reference",
                "element_id": "pnt_throttle",
            }
        ],
        metadata={
            "next": {"step_id": "S11"},
            "diagnosis": {"step_id": "S11", "error_category": "OM"},
            "help_response": {
                "diagnosis": {"step_id": "S11", "error_category": "OM"},
                "next": {"step_id": "S11"},
                "overlay": {
                    "targets": ["throttle_quadrant_reference"],
                    "evidence": [
                        {
                            "target": "throttle_quadrant_reference",
                            "type": "gate",
                            "ref": "GATES.S11.completion",
                            "quote": "blocked",
                        }
                    ],
                },
                "explanations": ["Highlight throttle."],
            },
        },
    )
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "overlay_target_allowlist": ["throttle_quadrant_reference"],
            "gates": [{"gate_id": "S11.completion", "status": "blocked"}],
            "deterministic_step_hint": {
                "inferred_step_id": "S11",
                "overlay_step_id": "S11",
                "missing_conditions": ["vars.throttle_l_idle_complete==true"],
            },
        },
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(Path("/dev/null")),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        used, reason = loop._apply_harness_validation_action_plan(response, request)

        assert used is True
        assert reason == "manual_throttle_keyboard_guidance"
        assert response.actions == []
        assert "Right Alt+Home" in response.message
        assert response.metadata["validator_rejected"] is True
        assert response.metadata["repair_applied"] is True
        assert response.metadata["final_action_plan_source"] == "validator_text_only_guidance"
    finally:
        loop.close()


def test_resolve_overlay_step_id_does_not_advance_partial_visual_hold_steps() -> None:
    assert _resolve_overlay_step_id(
        "S18",
        missing_conditions=[],
        candidate_steps=["S18", "S19"],
        step_order_index={"S18": 0, "S19": 1},
        observability_status="partial",
        requires_visual_confirmation=True,
    ) == "S18"


def test_resolve_overlay_step_id_advances_to_next_step_when_current_step_has_no_missing_conditions() -> None:
    candidate_steps = ["S08", "S09", "S10"]
    step_order_index = {step_id: idx for idx, step_id in enumerate(candidate_steps)}

    assert _resolve_overlay_step_id(
        "S08",
        missing_conditions=[],
        candidate_steps=candidate_steps,
        step_order_index=step_order_index,
    ) == "S09"
    assert _resolve_overlay_step_id(
        "S08",
        missing_conditions=["vision_facts.bit_root_page_visible==seen"],
        candidate_steps=candidate_steps,
        step_order_index=step_order_index,
    ) == "S08"


def test_live_loop_pack_step_without_ui_targets_degrades_to_safe_text(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack_no_step_targets.yaml"
    pack_path.write_text(
        "pack_id: test_fallback_pack\n"
        "version: v1\n"
        "title: Test Fallback Pack\n"
        "precondition_gates:\n"
        "  S01: []\n"
        "completion_gates:\n"
        "  S01: []\n"
        "steps:\n"
        "  - id: S01\n"
        "    phase: P1\n"
        "    observability: unknown\n"
        "    evidence_requirements: [visual, rag]\n"
        "    completion_conditions:\n"
        "      - \"visual check only\"\n",
        encoding="utf-8",
    )

    replay_path = tmp_path / "bios_model_error_no_step_target.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 21.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = FailingModel()
    executor = RecordingExecutor()
    events = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        pack_path=pack_path,
        cooldown_s=5.0,
        lang="en",
        event_sink=events.append,
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    assert executor.calls[0] == []

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    assert tutor_response_payload["actions"] == []
    assert isinstance(tutor_response_payload.get("message"), str)
    assert tutor_response_payload["message"].startswith("Fallback:")
    meta = tutor_response_payload["metadata"]
    assert meta["fallback_overlay_used"] is False
    assert isinstance(meta["fallback_overlay_reason"], str)
    assert meta["fallback_overlay_reason"].startswith("unsupported_step:S01")


def test_live_loop_cache_key_ignores_numeric_churn_when_discrete_state_unchanged(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_numeric_churn.jsonl"
    frame1 = _bios_frame(1, 15.0, apu_switch=0)
    frame2 = _bios_frame(2, 15.1, apu_switch=0)
    frame1["bios"]["IFEI_RPM_R"] = "10"
    frame1["delta"]["IFEI_RPM_R"] = "10"
    frame2["bios"]["IFEI_RPM_R"] = "11"
    frame2["delta"]["IFEI_RPM_R"] = "11"
    _write_replay(replay_path, [frame1, frame2])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=30.0,
        lang="en",
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=1)
    finally:
        loop.close()

    assert stats["help_cycles"] == 2
    assert stats["model_calls"] == 1
    assert stats["cache_hits"] == 1


def test_live_loop_does_not_cache_error_response_and_retries_model_within_cooldown(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_retry_on_error.jsonl"
    _write_replay(
        replay_path,
        [
            _bios_frame(1, 16.0, apu_switch=0),
            _bios_frame(2, 16.1, apu_switch=0),
        ],
    )

    source = ReplayBiosReceiver(replay_path)
    model = FailingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=30.0,
        lang="en",
    )
    try:
        stats = loop.run(max_frames=2, auto_help_every_n_frames=1)
    finally:
        loop.close()

    assert stats["help_cycles"] == 2
    assert stats["model_calls"] == 2
    assert stats["cache_hits"] == 0


def test_live_loop_keeps_gate_blockers_out_of_missing_conditions_for_grounding_query(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_gate_blockers_for_hint.jsonl"
    frame = _bios_frame(1, 18.0, apu_switch=1)
    frame["bios"]["APU_READY_LT"] = 1
    frame["bios"]["ENGINE_CRANK_SW"] = 1
    frame["bios"]["IFEI_RPM_R"] = 22
    frame["delta"]["IFEI_RPM_R"] = 22
    _write_replay(replay_path, [frame])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        cooldown_s=5.0,
        lang="en",
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(model.calls) == 1
    request = model.calls[0]["request"]
    assert request is not None
    hint = request.context["deterministic_step_hint"]
    missing_conditions = hint["missing_conditions"]
    gate_blockers = hint["gate_blockers"]

    assert isinstance(missing_conditions, list)
    assert isinstance(gate_blockers, list)
    assert all(not item.startswith("GATES.") for item in missing_conditions if isinstance(item, str))
    inferred_step_id = hint.get("inferred_step_id")
    assert isinstance(inferred_step_id, str) and inferred_step_id
    if gate_blockers:
        assert all(
            isinstance(item, dict) and isinstance(item.get("ref"), str) and item.get("ref", "").startswith("GATES.")
            for item in gate_blockers
        )
        expected_refs = {
            f"GATES.{inferred_step_id}.precondition",
            f"GATES.{inferred_step_id}.completion",
        }
        assert any(
            isinstance(item, dict) and item.get("ref") in expected_refs
            for item in gate_blockers
        )


def test_live_dcs_cli_log_raw_llm_text_can_disable_env_default(monkeypatch) -> None:
    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "1")
    parser = build_arg_parser()
    args = parser.parse_args(["--no-log-raw-llm-text"])
    assert args.log_raw_llm_text is False


def test_live_dcs_cli_log_raw_llm_text_can_enable_when_env_default_off(monkeypatch) -> None:
    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "0")
    parser = build_arg_parser()
    args = parser.parse_args(["--log-raw-llm-text"])
    assert args.log_raw_llm_text is True


def test_live_dcs_openai_compat_multimodal_flag_keeps_main_help_text_only(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-provider",
            "openai_compat",
            "--model-base-url",
            "http://127.0.0.1:8000",
            "--model-name",
            "simtutor-base",
            "--model-enable-multimodal",
            "--vision-saved-games-dir",
            str(tmp_path),
        ]
    )

    model = _build_model_from_args(args)

    assert isinstance(model, OpenAICompatModel)
    assert model.enable_multimodal is True
    assert model.enable_help_multimodal is False


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("true", True),
        ("false", False),
    ],
)
def test_live_dcs_cli_log_raw_llm_text_reads_common_boolean_env_values(
    monkeypatch,
    env_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", env_value)
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.log_raw_llm_text is expected


def test_live_dcs_cli_log_raw_llm_text_invalid_env_falls_back_false_with_warning(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SIMTUTOR_LOG_RAW_LLM_TEXT", "abc")
    with caplog.at_level("WARNING"):
        parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.log_raw_llm_text is False
    assert any(
        "SIMTUTOR_LOG_RAW_LLM_TEXT" in record.message and "Invalid boolean environment value" in record.message
        for record in caplog.records
    )


def test_live_dcs_cli_print_model_io_can_disable_env_default(monkeypatch) -> None:
    monkeypatch.setenv("SIMTUTOR_PRINT_MODEL_IO", "1")
    parser = build_arg_parser()
    args = parser.parse_args(["--no-print-model-io"])
    assert args.print_model_io is False


def test_live_dcs_cli_print_model_io_can_enable_when_env_default_off(monkeypatch) -> None:
    monkeypatch.setenv("SIMTUTOR_PRINT_MODEL_IO", "0")
    parser = build_arg_parser()
    args = parser.parse_args(["--print-model-io"])
    assert args.print_model_io is True


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("true", True),
        ("false", False),
    ],
)
def test_live_dcs_cli_print_model_io_reads_common_boolean_env_values(
    monkeypatch,
    env_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("SIMTUTOR_PRINT_MODEL_IO", env_value)
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.print_model_io is expected


def test_live_dcs_cli_print_model_io_invalid_env_falls_back_false_with_warning(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SIMTUTOR_PRINT_MODEL_IO", "abc")
    with caplog.at_level("WARNING"):
        parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.print_model_io is False
    assert any(
        "SIMTUTOR_PRINT_MODEL_IO" in record.message and "Invalid boolean environment value" in record.message
        for record in caplog.records
    )


def test_live_dcs_warns_when_multi_target_overlay_uses_single_slot_config(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "Saved Games" / "DCS" / "Scripts" / "SimTutor" / "SimTutorConfig.lua"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "return {\n"
        "    overlay = {\n"
        "        hilite_id = 9101,\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )
    events: list[Any] = []

    warning = _emit_multi_target_overlay_config_warning(
        max_overlay_targets=2,
        config_path=config_path,
        event_sink=events.append,
    )

    captured = capsys.readouterr()
    assert warning is not None
    assert "[LIVE_DCS] WARNING:" in captured.out
    assert "only 1 DCS highlight slot" in captured.out
    assert len(events) == 1
    assert events[0].kind == "system"
    assert events[0].payload["event"] == "overlay_config_warning"
    assert events[0].payload["max_overlay_targets"] == 2


def test_live_dcs_cli_cold_start_production_reads_env_default(monkeypatch) -> None:
    monkeypatch.setenv("SIMTUTOR_COLD_START_PRODUCTION", "true")
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.cold_start_production is True


def test_live_dcs_cli_cold_start_production_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("SIMTUTOR_COLD_START_PRODUCTION", "true")
    parser = build_arg_parser()
    args = parser.parse_args(["--no-cold-start-production"])
    assert args.cold_start_production is False

    args = parser.parse_args(["--cold-start-production"])
    assert args.cold_start_production is True


def test_live_dcs_cli_model_max_tokens_reads_env_default(monkeypatch) -> None:
    monkeypatch.setenv("SIMTUTOR_MODEL_MAX_TOKENS", "256")
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.model_max_tokens == 256


def test_live_dcs_cli_model_max_tokens_can_be_overridden() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--model-max-tokens", "128"])
    assert args.model_max_tokens == 128


def test_live_dcs_cli_model_max_tokens_invalid_env_falls_back_to_zero(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SIMTUTOR_MODEL_MAX_TOKENS", "")
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.model_max_tokens == 0
    assert "SIMTUTOR_MODEL_MAX_TOKENS" in caplog.text


def test_live_dcs_cli_model_max_tokens_rejects_negative_value() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-max-tokens", "-1"])


def test_live_dcs_cli_scenario_profile_defaults_airfield_and_accepts_carrier() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.scenario_profile == "airfield"

    args = parser.parse_args(["--scenario-profile", "carrier"])
    assert args.scenario_profile == "carrier"


def test_live_dcs_cli_parses_windows_global_help_trigger_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--global-help-hotkey",
            "X1",
            "--global-help-modifiers",
            "Ctrl+Shift",
            "--global-help-cooldown-ms",
            "800",
        ]
    )
    assert args.global_help_hotkey == "X1"
    assert args.global_help_modifiers == "Ctrl+Shift"
    assert args.global_help_cooldown_ms == 800


def test_run_help_cycle_ignores_nan_trigger_wall_and_uses_observation_time() -> None:
    class _SingleObservationSource:
        def close(self) -> None:
            return

    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=_SingleObservationSource(),
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-nan-trigger",
        vision_mode="replay",
    )
    try:
        loop._ingest_observation(
            Observation(
                source="mock",
                payload={
                    "seq": 1,
                    "t_wall": 10.0,
                    "vars": {
                        "battery_on": True,
                        "l_gen_on": True,
                        "r_gen_on": True,
                    },
                },
            )
        )
        response, _report = loop.run_help_cycle(trigger_t_wall=float("nan"))
    finally:
        loop.close()

    assert response is not None
    assert len(model.calls) == 1
    request = model.calls[0]["request"]
    assert request.context["vision"]["trigger_wall_ms"] == 10000


def test_live_loop_preserves_pending_help_until_first_observation() -> None:
    observation = Observation(source="dcs_bios_raw", payload=_bios_frame(1, 10.0, apu_switch=0))
    source = _DelayedObservationSource(observation)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-pending-help",
    )
    try:
        stats = loop.run(max_frames=1, help_trigger=_TriggerOnce(), idle_sleep_s=0.0)
    finally:
        loop.close()

    assert stats["frames"] == 1
    assert len(model.calls) == 1


def test_live_loop_stabilizes_inference_without_power_reset(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_sticky_inference.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-sticky-inference",
    )
    try:
        later = loop._stabilize_live_inference(
            StepInferenceResult(inferred_step_id="S08", missing_conditions=("vars.hud_on==true",)),
            {"battery_on": True, "power_available": True},
        )
        regressed = loop._stabilize_live_inference(
            StepInferenceResult(inferred_step_id="S03", missing_conditions=("vars.apu_start_support_complete==true",)),
            {"battery_on": True, "power_available": True},
        )
        reset = loop._stabilize_live_inference(
            StepInferenceResult(inferred_step_id="S01", missing_conditions=("vars.battery_on==true",)),
            {"battery_on": False, "power_available": False},
        )
    finally:
        loop.close()

    assert later.inferred_step_id == "S08"
    assert regressed.inferred_step_id == "S08"
    assert regressed.missing_conditions == ("vars.hud_on==true",)
    assert reset.inferred_step_id == "S01"


def test_live_inference_latches_refuel_probe_cycle_after_observed_s20_completion(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_refuel_probe_latch.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-refuel-probe-latch",
    )
    try:
        extended = loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S21",
                missing_conditions=("vars.ext_refuel_probe_value in [0,5000]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": True,
                "probe_retracted": False,
                "ext_refuel_probe_value": 65535,
                "launch_bar_switch_value": 0,
            },
        )
        retracted = loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S20",
                missing_conditions=("vars.ext_refuel_probe_value in [60000,65535]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": False,
                "probe_retracted": True,
                "ext_refuel_probe_value": 0,
                "launch_bar_switch_value": 0,
            },
        )
    finally:
        loop.close()

    assert extended.inferred_step_id == "S21"
    assert retracted.inferred_step_id == "S22"
    assert retracted.inferred_step_id != "S20"


def test_live_inference_does_not_skip_s20_from_initial_retracted_probe(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_refuel_probe_initial_retracted.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-refuel-probe-initial",
    )
    try:
        result = loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S20",
                missing_conditions=("vars.ext_refuel_probe_value in [60000,65535]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": False,
                "probe_retracted": True,
                "ext_refuel_probe_value": 0,
                "launch_bar_switch_value": 0,
            },
        )
    finally:
        loop.close()

    assert result.inferred_step_id == "S20"
    assert result.missing_conditions == ("vars.ext_refuel_probe_value in [60000,65535]",)


def test_live_inference_ignores_probe_motion_seen_before_s20(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_refuel_probe_early_motion.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-refuel-probe-early",
    )
    try:
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S19",
                missing_conditions=("vision_facts.fcsmc_final_go_result_visible==seen",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": True,
                "probe_retracted": False,
                "ext_refuel_probe_value": 65535,
            },
        )
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S19",
                missing_conditions=("vision_facts.fcsmc_final_go_result_visible==seen",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": False,
                "probe_retracted": True,
                "ext_refuel_probe_value": 0,
            },
        )
        result = loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S20",
                missing_conditions=("vars.ext_refuel_probe_value in [60000,65535]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": False,
                "probe_retracted": True,
                "ext_refuel_probe_value": 0,
            },
        )
    finally:
        loop.close()

    assert result.inferred_step_id == "S20"
    assert loop._refuel_probe_s20_latched_complete is False
    assert loop._refuel_probe_s21_latched_complete is False


def test_live_inference_requires_probe_to_remain_retracted_for_s21_latch(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_refuel_probe_reextended.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-refuel-probe-reextended",
    )
    try:
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S21",
                missing_conditions=("vars.ext_refuel_probe_value in [0,5000]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": True,
                "probe_retracted": False,
                "ext_refuel_probe_value": 65535,
            },
        )
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S20",
                missing_conditions=("vars.ext_refuel_probe_value in [60000,65535]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": False,
                "probe_retracted": True,
                "ext_refuel_probe_value": 0,
            },
        )
        reextended = loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S20",
                missing_conditions=("vars.ext_refuel_probe_value in [60000,65535]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "probe_extended": True,
                "probe_retracted": False,
                "ext_refuel_probe_value": 65535,
            },
        )
    finally:
        loop.close()

    assert reextended.inferred_step_id == "S21"
    assert loop._refuel_probe_s20_latched_complete is True
    assert loop._refuel_probe_s21_latched_complete is False


def test_live_build_request_serializes_refuel_probe_latch_history(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_refuel_probe_latch_request.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-refuel-probe-latch-request",
    )
    try:
        extended_vars = {
            "battery_on": True,
            "power_available": True,
            "probe_extended": True,
            "probe_retracted": False,
            "ext_refuel_probe_value": 65535,
            "launch_bar_switch_value": 0,
        }
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S21",
                missing_conditions=("vars.ext_refuel_probe_value in [0,5000]",),
            ),
            extended_vars,
        )
        extended_obs = Observation(
            source="mock",
            payload={"seq": 1, "t_wall": 10.0, "vars": extended_vars},
        )
        extended_vision = loop._build_vision_selection(observation=extended_obs, trigger_t_wall=10.0)
        extended_request, _prompt_meta, extended_state_key = loop._build_request(
            extended_obs,
            vision_selection=extended_vision,
            vision_fact_context=loop._extract_vision_fact_context(vision_selection=extended_vision),
        )

        retracted_vars = {
            "battery_on": True,
            "power_available": True,
            "probe_extended": False,
            "probe_retracted": True,
            "probe_cycle_complete": True,
            "ext_refuel_probe_value": 0,
            "launch_bar_switch_value": 0,
        }
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S20",
                missing_conditions=("vars.ext_refuel_probe_value in [60000,65535]",),
            ),
            retracted_vars,
        )
        retracted_obs = Observation(
            source="mock",
            payload={"seq": 2, "t_wall": 20.0, "vars": retracted_vars},
        )
        retracted_vision = loop._build_vision_selection(observation=retracted_obs, trigger_t_wall=20.0)
        retracted_request, _prompt_meta, retracted_state_key = loop._build_request(
            retracted_obs,
            vision_selection=retracted_vision,
            vision_fact_context=loop._extract_vision_fact_context(vision_selection=retracted_vision),
        )
    finally:
        loop.close()

    assert extended_request.context["refuel_probe_completion_latches"] == {
        "s20_latched_complete": True,
        "s21_latched_complete": False,
    }
    assert retracted_request.context["refuel_probe_completion_latches"] == {
        "s20_latched_complete": True,
        "s21_latched_complete": True,
    }
    assert retracted_state_key != extended_state_key


def test_live_build_request_serializes_four_down_transition_latch_history(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_four_down_latch_request.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-four-down-latch-request",
    )
    try:
        loop._refuel_probe_s20_latched_complete = True
        loop._refuel_probe_s21_latched_complete = True
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S22",
                missing_conditions=("vars.launch_bar_switch_value in [1,1]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "launch_bar_switch_value": 0,
                "hook_handle_value": 1,
            },
        )
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S22",
                missing_conditions=("vars.launch_bar_switch_value in [1,1]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "launch_bar_switch_value": 1,
                "hook_handle_value": 1,
            },
        )
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S23",
                missing_conditions=("vars.launch_bar_switch_value in [0,0]",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "launch_bar_switch_value": 0,
                "hook_handle_value": 1,
            },
        )
        before_hook_latch_obs = Observation(
            source="mock",
            payload={
                "seq": 1,
                "t_wall": 10.0,
                "vars": {
                    "battery_on": True,
                    "power_available": True,
                    "launch_bar_switch_value": 0,
                    "hook_handle_value": 1,
                },
            },
        )
        before_hook_latch_vision = loop._build_vision_selection(
            observation=before_hook_latch_obs,
            trigger_t_wall=10.0,
        )
        before_hook_latch_request, _prompt_meta, before_hook_latch_state_key = loop._build_request(
            before_hook_latch_obs,
            vision_selection=before_hook_latch_vision,
            vision_fact_context=loop._extract_vision_fact_context(vision_selection=before_hook_latch_vision),
        )

        hook_initial_down = loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S24",
                missing_conditions=("session.hook_down_transition_observed==true",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "launch_bar_switch_value": 0,
                "hook_handle_value": 1,
            },
        )
        loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S24",
                missing_conditions=("session.hook_down_transition_observed==true",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "launch_bar_switch_value": 0,
                "hook_handle_value": 0,
            },
        )
        hook_transition_down = loop._stabilize_live_inference(
            StepInferenceResult(
                inferred_step_id="S24",
                missing_conditions=("session.hook_down_transition_observed==true",),
            ),
            {
                "battery_on": True,
                "power_available": True,
                "launch_bar_switch_value": 0,
                "hook_handle_value": 0,
            },
        )

        hook_cold_loop = LiveDcsTutorLoop(
            source=ReplayBiosReceiver(replay_path),
            model=FailingModel(),
            action_executor=RecordingExecutor(),
            session_id="sess-four-down-hook-cold",
        )
        try:
            hook_cold_s25 = hook_cold_loop._stabilize_live_inference(
                StepInferenceResult(
                    inferred_step_id="S25",
                    missing_conditions=("vars.hook_handle_value in [1,1]",),
                ),
                {
                    "battery_on": True,
                    "power_available": True,
                    "launch_bar_switch_value": 0,
                    "hook_handle_value": 0,
                },
            )
        finally:
            hook_cold_loop.close()
        launch_bar_cold_loop = LiveDcsTutorLoop(
            source=ReplayBiosReceiver(replay_path),
            model=FailingModel(),
            action_executor=RecordingExecutor(),
            session_id="sess-four-down-launch-cold",
        )
        try:
            launch_bar_cold_s23 = launch_bar_cold_loop._stabilize_live_inference(
                StepInferenceResult(
                    inferred_step_id="S23",
                    missing_conditions=("vars.launch_bar_switch_value in [0,0]",),
                ),
                {
                    "battery_on": True,
                    "power_available": True,
                    "launch_bar_switch_value": 1,
                },
            )
        finally:
            launch_bar_cold_loop.close()

        obs = Observation(
            source="mock",
            payload={
                "seq": 1,
                "t_wall": 10.0,
                "vars": {
                    "battery_on": True,
                    "power_available": True,
                    "launch_bar_switch_value": 0,
                    "hook_handle_value": 0,
                },
            },
        )
        vision_selection = loop._build_vision_selection(observation=obs, trigger_t_wall=10.0)
        request, _prompt_meta, _state_key = loop._build_request(
            obs,
            vision_selection=vision_selection,
            vision_fact_context=loop._extract_vision_fact_context(vision_selection=vision_selection),
        )
    finally:
        loop.close()

    assert hook_initial_down.inferred_step_id == "S24"
    assert hook_cold_s25.inferred_step_id == "S24"
    assert launch_bar_cold_s23.inferred_step_id == "S22"
    assert hook_transition_down.inferred_step_id == "S25"
    assert before_hook_latch_request.context["four_down_completion_latches"]["s24_latched_complete"] is False
    assert request.context["four_down_completion_latches"] == {
        "s20_latched_complete": True,
        "s21_latched_complete": True,
        "s22_latched_complete": True,
        "s23_latched_complete": True,
        "s24_latched_complete": True,
        "s25_latched_complete": False,
    }
    assert before_hook_latch_state_key != _state_key


def test_live_dcs_cli_parses_raw_bios_source_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--bios-source",
            "raw",
            "--raw-bios-aircraft",
            "FA-18C_hornet",
            "--raw-bios-port",
            "5010",
        ]
    )
    assert args.bios_source == "raw"
    assert args.raw_bios_aircraft == "FA-18C_hornet"
    assert args.raw_bios_port == 5010


def test_build_observation_source_from_args_uses_raw_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--bios-source",
            "raw",
            "--raw-bios-aircraft",
            "FA-18C_hornet",
            "--raw-bios-host",
            "239.255.50.10",
            "--raw-bios-port",
            "5010",
        ]
    )
    captured: dict[str, Any] = {}

    class FakeRawReceiver:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("live_dcs.DcsBiosRawReceiver", FakeRawReceiver)

    source = _build_observation_source_from_args(args)

    assert isinstance(source, FakeRawReceiver)
    assert captured["host"] == "239.255.50.10"
    assert captured["port"] == 5010
    assert captured["aircraft"] == "FA-18C_hornet"


def test_build_observation_source_from_args_requires_aircraft_for_raw() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--bios-source", "raw"])

    with pytest.raises(ValueError, match="--raw-bios-aircraft"):
        _build_observation_source_from_args(args)


def test_live_dcs_main_wires_tutor_text_sender_into_loop(monkeypatch, tmp_path: Path) -> None:
    import live_dcs

    captured: dict[str, Any] = {}

    class FakeStore:
        def __init__(self, *_args, **_kwargs) -> None:
            return

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def append(self, _event) -> None:
            return

    class FakeTutorTextSender:
        def __init__(self, **kwargs) -> None:
            captured["sender_kwargs"] = dict(kwargs)

        def close(self) -> None:
            return

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            captured["loop_kwargs"] = dict(kwargs)

        def run(self, **_kwargs) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            return

    monkeypatch.setattr("live_dcs.JsonlEventStore", FakeStore)
    monkeypatch.setattr("live_dcs.OverlayActionExecutor", lambda **_kwargs: object())
    monkeypatch.setattr("live_dcs._build_observation_source_from_args", lambda _args: object())
    monkeypatch.setattr("live_dcs._build_model_from_args", lambda _args: object())
    monkeypatch.setattr("live_dcs._build_vision_port_from_args", lambda _args, mode: (None, None, None, None))
    monkeypatch.setattr("live_dcs.DcsTutorTextSender", FakeTutorTextSender)
    monkeypatch.setattr("live_dcs.LiveDcsTutorLoop", FakeLoop)

    code = live_dcs.main(["--output", str(tmp_path / "events.jsonl"), "--duration", "0"])

    assert code == 0
    assert captured["sender_kwargs"] == {
        "host": "127.0.0.1",
        "port": 7783,
        "timeout": 0.5,
        "enabled": True,
    }
    assert isinstance(captured["loop_kwargs"]["tutor_text_sender"], FakeTutorTextSender)


def test_live_dcs_main_resolves_existing_output_path_and_logs_metadata(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import live_dcs

    requested_output = tmp_path / "events.jsonl"
    requested_output.write_text("previous run\n", encoding="utf-8")
    captured: dict[str, Any] = {"events": []}

    class FakeStore:
        def __init__(self, path, *_args, mode: str = "a", **_kwargs) -> None:
            attempted = captured.setdefault("attempted_paths", [])
            attempted.append(Path(path))
            assert mode == "x"
            if len(attempted) <= 2:
                raise FileExistsError(path)
            captured["store_path"] = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def append(self, event) -> None:
            captured["events"].append(event)

    class FakeTutorTextSender:
        def __init__(self, **_kwargs) -> None:
            return

        def close(self) -> None:
            return

    class FakeLoop:
        def __init__(self, **_kwargs) -> None:
            return

        def run(self, **_kwargs) -> dict[str, Any]:
            return {"help_cycles": 0}

        def close(self) -> None:
            return

    monkeypatch.setattr("live_dcs.JsonlEventStore", FakeStore)
    monkeypatch.setattr("live_dcs.OverlayActionExecutor", lambda **_kwargs: object())
    monkeypatch.setattr("live_dcs._build_observation_source_from_args", lambda _args: object())
    monkeypatch.setattr("live_dcs._build_model_from_args", lambda _args: object())
    monkeypatch.setattr("live_dcs._build_vision_port_from_args", lambda _args, mode: (None, None, None, None))
    monkeypatch.setattr("live_dcs.DcsTutorTextSender", FakeTutorTextSender)
    monkeypatch.setattr("live_dcs.LiveDcsTutorLoop", FakeLoop)

    code = live_dcs.main(["--output", str(requested_output), "--duration", "0"])

    assert code == 0
    resolved_output = captured["store_path"]
    assert resolved_output != requested_output
    assert resolved_output.parent == requested_output.parent
    assert resolved_output.name.startswith("events_")
    assert len(captured["attempted_paths"]) == 3
    assert requested_output.read_text(encoding="utf-8") == "previous run\n"
    startup_events = [event for event in captured["events"] if event.kind == "system"]
    assert startup_events
    assert startup_events[0].payload["event"] == "live_dcs_runtime_log"
    assert startup_events[0].payload["requested_output_path"] == str(requested_output)
    assert startup_events[0].payload["resolved_output_path"] == str(resolved_output)
    assert startup_events[0].metadata["requested_output_path"] == str(requested_output)
    assert startup_events[0].metadata["resolved_output_path"] == str(resolved_output)
    out = capsys.readouterr().out
    assert f"[LIVE_DCS] resolved output path: {resolved_output}" in out
    assert f"[LIVE_DCS] wrote events to {resolved_output}" in out


def test_live_loop_request_context_and_metadata_include_scenario_profile(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_scenario_profile.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path)
    model = RecordingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        scenario_profile="carrier",
        lang="en",
    )
    try:
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(model.calls) == 1
    request = model.calls[0]["request"]
    assert request is not None
    assert request.context["scenario_profile"] == "carrier"
    assert request.metadata["scenario_profile"] == "carrier"
    assert request.context["deterministic_step_hint"]["scenario_profile"] == "carrier"


def test_normalize_cached_response_metadata_normalizes_fallback_reason() -> None:
    missing_reason: dict[str, Any] = {}
    _normalize_cached_response_metadata(missing_reason)
    assert missing_reason["fallback_overlay_reason"] == "not_needed"

    none_reason: dict[str, Any] = {"fallback_overlay_reason": None}
    _normalize_cached_response_metadata(none_reason)
    assert none_reason["fallback_overlay_reason"] == "not_needed"

    explicit_reason: dict[str, Any] = {"fallback_overlay_reason": "deterministic_step:S01"}
    _normalize_cached_response_metadata(explicit_reason)
    assert explicit_reason["fallback_overlay_reason"] == "deterministic_step:S01"


def test_build_vision_port_from_args_rejects_session_id_with_path_separators() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--vision-saved-games-dir",
            "/tmp/saved games",
            "--vision-session-id",
            "../escape",
        ]
    )

    with pytest.raises(ValueError, match="--vision-session-id"):
        _build_vision_port_from_args(args, mode="live")


def test_build_vision_port_from_args_rejects_drive_qualified_session_id() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--vision-saved-games-dir",
            "/tmp/saved games",
            "--vision-session-id",
            "C:escape",
        ]
    )

    with pytest.raises(ValueError, match="--vision-session-id"):
        _build_vision_port_from_args(args, mode="live")


def test_build_vision_port_from_args_rejects_channel_with_path_separators() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--vision-saved-games-dir",
            "/tmp/saved games",
            "--vision-session-id",
            "sess-live",
            "--vision-channel",
            "nested/channel",
        ]
    )

    with pytest.raises(ValueError, match="--vision-channel"):
        _build_vision_port_from_args(args, mode="live")


def test_build_vision_port_from_args_live_zero_trigger_wait_uses_mode_default() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--vision-saved-games-dir",
            "/tmp/saved games",
            "--vision-session-id",
            "sess-live",
        ]
    )

    _vision_port, _session_id, _sync_window_ms, trigger_wait_ms = _build_vision_port_from_args(args, mode="live")
    assert trigger_wait_ms is None


def test_live_loop_help_cycle_includes_selected_vision_frames_in_request_and_events(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_with_vision.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class StaticVisionPort:
        def start(self, session_id: str) -> None:
            assert session_id == "sess-live"

        def poll(self) -> list[VisionObservation]:
            return [
                VisionObservation(
                    frame_id="1772872444950_000122",
                    source="vision_test",
                    capture_wall_ms=1772872444950,
                    frame_seq=122,
                    layout_id="fa18c_composite_panel_v2",
                    channel="composite_panel",
                    image_uri="/tmp/1772872444950_000122.png",
                ),
                VisionObservation(
                    frame_id="1772872445010_000123",
                    source="vision_test",
                    capture_wall_ms=1772872445010,
                    frame_seq=123,
                    layout_id="fa18c_composite_panel_v2",
                    channel="composite_panel",
                    image_uri="/tmp/1772872445010_000123.png",
                ),
            ]

        def stop(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    executor = RecordingExecutor()
    events: list[dict[str, Any]] = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        session_id="sess-live",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-live",
        vision_mode="live",
        event_sink=lambda event: events.append(event.to_dict()),
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=1772872445.0)
    finally:
        loop.close()

    assert response is not None
    assert len(model.calls) == 1
    request = model.calls[0]["request"]
    vision = request.context["vision"]
    assert vision["status"] == "available"
    assert vision["observation_ref"] == request.observation_ref
    assert vision["observation_t_wall_ms"] == 10000
    assert vision["trigger_wall_ms"] == 1772872445000
    assert vision["frame_id"] == "1772872444950_000122"
    assert vision["sync_status"] == "matched_past"
    assert vision["sync_delta_ms"] == -50
    assert vision["frame_stale"] is True
    assert vision["frame_ids"] == ["1772872444950_000122", "1772872445010_000123"]
    assert vision["pre_trigger_frame"]["frame_id"] == "1772872444950_000122"
    assert vision["trigger_frame"]["frame_id"] == "1772872445010_000123"
    tutor_request = next(event for event in events if event["kind"] == "tutor_request")
    tutor_response = next(event for event in events if event["kind"] == "tutor_response")
    assert tutor_request["vision_refs"] == ["1772872444950_000122", "1772872445010_000123"]
    assert tutor_response["vision_refs"] == ["1772872444950_000122", "1772872445010_000123"]


def test_live_loop_emits_vision_observation_events_with_attachments(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_with_vision_events.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    artifact_dir = tmp_path / "vision"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    first_artifact = artifact_dir / "1772872444950_000122_vlm.png"
    first_source = artifact_dir / "1772872444950_000122.png"
    second_artifact = artifact_dir / "1772872445010_000123_vlm.png"
    second_source = artifact_dir / "1772872445010_000123.png"

    class StaticVisionPort:
        def start(self, session_id: str) -> None:
            assert session_id == "sess-live"

        def poll(self) -> list[VisionObservation]:
            return [
                VisionObservation(
                    frame_id="1772872444950_000122",
                    source="vision_test",
                    capture_wall_ms=1772872444950,
                    frame_seq=122,
                    layout_id="fa18c_composite_panel_v2",
                    channel="composite_panel",
                    image_uri=str(first_artifact),
                    source_image_path=str(first_source),
                ),
                VisionObservation(
                    frame_id="1772872445010_000123",
                    source="vision_test",
                    capture_wall_ms=1772872445010,
                    frame_seq=123,
                    layout_id="fa18c_composite_panel_v2",
                    channel="composite_panel",
                    image_uri=str(second_artifact),
                    source_image_path=str(second_source),
                ),
            ]

        def stop(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    events: list[dict[str, Any]] = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-live",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-live",
        vision_mode="live",
        event_sink=lambda event: events.append(event.to_dict()),
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        loop.run_help_cycle(trigger_t_wall=1772872445.0)
    finally:
        loop.close()

    vision_events = [
        event
        for event in events
        if event["kind"] == "observation"
        and event.get("metadata", {}).get("observation_kind") == "vision"
    ]
    assert [event["vision_refs"] for event in vision_events] == [
        ["1772872444950_000122"],
        ["1772872445010_000123"],
    ]
    first_payload = vision_events[0]["payload"]
    validate_instance(vision_events[0], "event")
    validate_instance(first_payload, "observation")
    assert first_payload["attachments"] == [
        first_artifact.resolve().as_uri(),
        first_source.resolve().as_uri(),
    ]
    assert first_payload["payload"]["frame_id"] == "1772872444950_000122"
    assert first_payload["payload"]["observation_ref"] == first_payload["observation_id"]


def test_path_like_to_uri_normalizes_windows_drive_paths() -> None:
    raw_path = r"C:\SimTutor\frames\1772872444950_000122.png"
    assert _path_like_to_uri(raw_path) == PureWindowsPath(raw_path).as_uri()


def test_emit_vision_observation_event_replaces_invalid_observation_ref_with_uuid(tmp_path: Path) -> None:
    artifact_path = tmp_path / "vision" / "1772872444950_000122_vlm.png"
    source_path = tmp_path / "vision" / "1772872444950_000122.png"
    events: list[dict[str, Any]] = []

    _emit_vision_observation_event(
        observation=VisionObservation(
            frame_id="1772872444950_000122",
            source="vision_test",
            observation_ref="not-a-uuid",
            capture_wall_ms=1772872444950,
            frame_seq=122,
            layout_id="fa18c_composite_panel_v2",
            channel="composite_panel",
            image_uri=str(artifact_path),
            source_image_path=str(source_path),
        ),
        event_sink=lambda event: events.append(event.to_dict()),
        fallback_session_id="sess-live",
    )

    assert len(events) == 1
    payload = events[0]["payload"]
    validate_instance(events[0], "event")
    validate_instance(payload, "observation")
    assert payload["payload"]["observation_ref"] == payload["observation_id"]
    assert str(UUID(payload["observation_id"])) == payload["observation_id"]


def test_emit_vision_observation_event_uses_tutor_session_id_over_payload_session_id(tmp_path: Path) -> None:
    artifact_path = tmp_path / "vision" / "1772872444950_000122_vlm.png"
    source_path = tmp_path / "vision" / "1772872444950_000122.png"
    events: list[dict[str, Any]] = []

    _emit_vision_observation_event(
        observation=VisionObservation(
            frame_id="1772872444950_000122",
            source="vision_test",
            session_id="vision-sidecar-session",
            capture_wall_ms=1772872444950,
            frame_seq=122,
            layout_id="fa18c_composite_panel_v2",
            channel="composite_panel",
            image_uri=str(artifact_path),
            source_image_path=str(source_path),
        ),
        event_sink=lambda event: events.append(event.to_dict()),
        fallback_session_id="sess-live",
    )

    assert len(events) == 1
    assert events[0]["session_id"] == "sess-live"
    assert events[0]["payload"]["payload"]["session_id"] == "vision-sidecar-session"


def test_emit_vision_fact_observation_event_uses_frame_refs() -> None:
    events: list[dict[str, Any]] = []

    _emit_vision_fact_observation_event(
        observation=VisionFactObservation(
            session_id="vision-sidecar-session",
            trigger_wall_ms=1772872445000,
            frame_ids=["1772872444950_000122", "1772872445010_000123"],
            facts=[
                VisionFact(
                    fact_id="supt_page_visible",
                    state="seen",
                    source_frame_id="1772872445010_000123",
                    expires_after_ms=2000,
                    evidence_note="SUPT page evidence visible on the left DDI.",
                )
            ],
        ),
        event_sink=lambda event: events.append(event.to_dict()),
        fallback_session_id="sess-live",
    )

    assert len(events) == 1
    assert events[0]["vision_refs"] == ["1772872444950_000122", "1772872445010_000123"]
    assert events[0]["metadata"]["observation_kind"] == "vision_fact"
    assert events[0]["payload"]["metadata"]["observation_kind"] == "vision_fact"


def test_live_loop_records_vision_fact_context_and_event(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_with_vision_facts.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class StaticVisionPort:
        def start(self, session_id: str) -> None:
            assert session_id == "sess-live"

        def poll(self) -> list[VisionObservation]:
            return [
                VisionObservation(
                    frame_id="1772872444950_000122",
                    source="vision_test",
                    capture_wall_ms=1772872444950,
                    frame_seq=122,
                    layout_id="fa18c_composite_panel_v2",
                    channel="composite_panel",
                    image_uri=str(tmp_path / "1772872444950_000122.png"),
                ),
                VisionObservation(
                    frame_id="1772872445010_000123",
                    source="vision_test",
                    capture_wall_ms=1772872445010,
                    frame_seq=123,
                    layout_id="fa18c_composite_panel_v2",
                    channel="composite_panel",
                    image_uri=str(tmp_path / "1772872445010_000123.png"),
                ),
            ]

        def stop(self) -> None:
            return

    class StaticVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            assert session_id == "sess-live"
            assert vision["frame_ids"] == ["1772872444950_000122", "1772872445010_000123"]
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=["1772872445010_000123"],
                        facts=[
                            VisionFact(
                                fact_id="supt_page_visible",
                                state="seen",
                                source_frame_id="1772872445010_000123",
                                expires_after_ms=2000,
                                evidence_note="SUPT page visible on the left DDI.",
                            )
                        ],
                    ),
                },
            )()

        def close(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    executor = RecordingExecutor()
    events: list[dict[str, Any]] = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        session_id="sess-live",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-live",
        vision_mode="live",
        vision_fact_extractor=StaticVisionFactExtractor(),
        event_sink=lambda event: events.append(event.to_dict()),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=1772872445.0)
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    assert request.context["vision_fact_summary"]["status"] == "available"
    assert request.context["vision_fact_summary"]["seen_fact_ids"] == ["supt_page_visible"]
    assert request.metadata["vision_fact_status"] == "available"
    assert response.metadata["vision_fact_status"] == "available"
    assert response.metadata["vision_fact_summary"]["seen_fact_ids"] == ["supt_page_visible"]
    fact_events = [
        event
        for event in events
        if event["kind"] == "observation"
        and event.get("metadata", {}).get("observation_kind") == "vision_fact"
    ]
    assert len(fact_events) == 1
    assert fact_events[0]["vision_refs"] == ["1772872445010_000123"]


def test_live_loop_records_vision_fact_raw_json_in_event(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_with_vision_fact_raw_json.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    raw_llm_text = (
        '{"facts":[{"fact_id":"supt_page_visible","state":"seen",'
        '"evidence_note":"SUPT page visible on the left DDI."}]}'
    )

    class StaticVisionPort:
        def start(self, session_id: str) -> None:
            assert session_id == "sess-live-raw-json"

        def poll(self) -> list[VisionObservation]:
            return [
                VisionObservation(
                    frame_id="1772872445010_000123",
                    source="vision_test",
                    capture_wall_ms=1772872445010,
                    frame_seq=123,
                    layout_id="fa18c_composite_panel_v2",
                    channel="composite_panel",
                    image_uri=str(tmp_path / "1772872445010_000123.png"),
                ),
            ]

        def stop(self) -> None:
            return

    class StaticVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {"raw_llm_text": raw_llm_text},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=["1772872445010_000123"],
                        facts=[
                            VisionFact(
                                fact_id="supt_page_visible",
                                state="seen",
                                source_frame_id="1772872445010_000123",
                                expires_after_ms=2000,
                                evidence_note="SUPT page visible on the left DDI.",
                            )
                        ],
                        metadata={"raw_llm_text": raw_llm_text},
                    ),
                },
            )()

        def close(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    executor = RecordingExecutor()
    events: list[dict[str, Any]] = []
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        session_id="sess-live-raw-json",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-live-raw-json",
        vision_mode="live",
        vision_fact_extractor=StaticVisionFactExtractor(),
        event_sink=lambda event: events.append(event.to_dict()),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=1772872445.01)
    finally:
        loop.close()

    assert response is not None
    fact_events = [
        event
        for event in events
        if event["kind"] == "observation"
        and event.get("metadata", {}).get("observation_kind") == "vision_fact"
    ]
    assert len(fact_events) == 1
    assert fact_events[0]["payload"]["payload"]["metadata"]["raw_llm_text"] == raw_llm_text


def test_live_loop_marks_vision_fact_unavailable_without_extractor(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_without_vision_facts.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-no-vision-facts",
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=1772872445.0)
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    assert request.metadata["vision_fact_status"] == "vision_unavailable"
    assert request.context["vision_fact_summary"]["status"] == "vision_unavailable"


def test_live_loop_skips_vision_fact_extractor_for_non_visual_step(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_non_visual_step_skip_vlm.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-skip-vlm"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000123",
                    capture_wall_ms=10000,
                    frame_seq=123,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000123.png"),
                )
            ]

    class FailingIfCalledVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):  # pragma: no cover
            del vision, session_id, trigger_wall_ms
            raise AssertionError("VLM extractor should not be called for non-visual steps")

        def close(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-skip-vlm",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-skip-vlm",
        vision_mode="replay",
        vision_fact_extractor=FailingIfCalledVisionFactExtractor(),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S22", [])
    loop._sticky_inference_step_id = "S08"
    loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
    loop._last_inferred_step_id = "S08"
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
        stats = loop.stats.to_dict()
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    assert request.metadata["vision_fact_status"] == "vision_not_required"
    assert request.metadata["vision_fact_active_step_ids"] == ["S22"]
    assert request.context["vision_fact_summary"]["status"] == "vision_not_required"
    assert response.metadata["vision_fact_status"] == "vision_not_required"
    assert response.metadata["vision_fallback_reason"] is None
    assert stats["vision_cycles"] == 0


def test_live_loop_ignores_stale_s19_visual_facts_for_s20(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s20_ignores_stale_s19_vlm.jsonl"
    frame = _bios_frame(1, 10.0, apu_switch=1)
    frame["bios"]["EXT_REFUEL_PROBE_SW"] = 0
    frame["delta"]["EXT_REFUEL_PROBE_SW"] = 0
    _write_replay(replay_path, [frame])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-s20-no-vlm"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000321",
                    capture_wall_ms=10000,
                    frame_seq=321,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000321.png"),
                )
            ]

    class FailingIfCalledVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):  # pragma: no cover
            del vision, session_id, trigger_wall_ms
            raise AssertionError("VLM extractor should not be called for S20")

        def close(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-s20-no-vlm",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-s20-no-vlm",
        vision_mode="replay",
        vision_fact_extractor=FailingIfCalledVisionFactExtractor(),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult(
        "S20",
        ("vars.ext_refuel_probe_value>=60000",),
    )
    loop._last_inferred_step_id = "S19"
    loop._sticky_inference_step_id = "S19"
    loop._sticky_inference_missing_conditions = ("vars.ext_refuel_probe_value>=60000",)
    loop._vision_fact_snapshot = {
        "fcsmc_final_go_result_visible": {
            "fact_id": "fcsmc_final_go_result_visible",
            "state": "seen",
            "source_frame_id": "old-s19-frame",
            "sticky": True,
            "expires_after_ms": 600000,
            "observed_at_wall_ms": 9000,
            "expires_at_wall_ms": 609000,
            "evidence_note": "Stale FCS-MC GO result from S19.",
        },
        "fcs_page_visible": {
            "fact_id": "fcs_page_visible",
            "state": "seen",
            "source_frame_id": "old-s08-frame",
            "sticky": False,
            "expires_after_ms": 600000,
            "observed_at_wall_ms": 9000,
            "expires_at_wall_ms": 609000,
            "evidence_note": "Stale S08 page.",
        },
    }
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
        stats = loop.stats.to_dict()
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    assert request.metadata["vision_fact_status"] == "vision_not_required"
    assert request.metadata["vision_fact_active_step_ids"] == ["S20"]
    assert request.context["vision_facts"] == []
    assert request.context["vision_fact_summary"]["seen_fact_ids"] == []
    assert request.context["state_harness"]["vision_evidence"]["late_display_anchors"] == []
    assert not any(
        item["source"] in {"sticky_state", "visual_anchor"}
        for item in request.context["candidate_steps"]
    )
    assert response.metadata["vlm_call_status"] == "not_required"
    assert response.metadata["harness_trace"]["vlm_call"]["ignored_fact_count"] == 2
    assert response.metadata["harness_trace"]["vlm_call"]["sticky_fact_count"] == 1
    assert response.metadata["harness_trace"]["vlm_call"]["extractor_called"] is False
    assert stats["vision_cycles"] == 0


def test_live_loop_advances_satisfied_s19_visual_hold_before_vlm_gate(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s20_satisfied_s19_visual_hold.jsonl"
    frame = _bios_frame(1, 10.0, apu_switch=1)
    frame["bios"]["EXT_REFUEL_PROBE_SW"] = 0
    frame["delta"]["EXT_REFUEL_PROBE_SW"] = 0
    _write_replay(replay_path, [frame])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-s19-hold-satisfied"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000654",
                    capture_wall_ms=10000,
                    frame_seq=654,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000654.png"),
                )
            ]

    class FailingIfCalledVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):  # pragma: no cover
            del vision, session_id, trigger_wall_ms
            raise AssertionError("VLM extractor should not be called after S19 visual hold is satisfied")

        def close(self) -> None:
            return

    class RequestHintModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            self.calls.append({"observation": observation, "request": request})
            hint = request.context["deterministic_step_hint"]
            step_id = hint["inferred_step_id"]
            target = hint.get("action_hint", {}).get("target") or hint["step_ui_targets"][0]
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id,
                message=f"Operate {target}.",
                actions=[],
                explanations=[f"Operate {target}."],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": step_id, "error_category": "OM"},
                        "next": {"step_id": step_id},
                        "overlay": {
                            "targets": [target],
                            "evidence": [
                                {
                                    "target": target,
                                    "type": "gate",
                                    "ref": f"GATES.{step_id}.completion",
                                    "quote": "Current gate remains incomplete.",
                                    "grounding_confidence": 0.95,
                                }
                            ],
                        },
                        "explanations": [f"Operate {target}."],
                    },
                },
            )

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RequestHintModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-s19-hold-satisfied",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-s19-hold-satisfied",
        vision_mode="replay",
        vision_fact_extractor=FailingIfCalledVisionFactExtractor(),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult(
        "S19",
        ("vision_facts.fcsmc_page_visible==seen",),
    )
    loop._last_inferred_step_id = "S19"
    loop._sticky_inference_step_id = "S19"
    loop._sticky_inference_missing_conditions = ("vision_facts.fcsmc_page_visible==seen",)
    loop._vision_fact_snapshot = {
        "fcsmc_final_go_result_visible": {
            "fact_id": "fcsmc_final_go_result_visible",
            "state": "seen",
            "source_frame_id": "old-s19-frame",
            "sticky": True,
            "expires_after_ms": 600000,
            "observed_at_wall_ms": 9000,
            "expires_at_wall_ms": 609000,
        }
    }
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
        stats = loop.stats.to_dict()
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    assert request.metadata["vision_fact_status"] == "vision_not_required"
    assert request.metadata["vision_fact_active_step_ids"] == ["S20"]
    assert request.context["vision_facts"] == []
    assert request.context["deterministic_step_hint"]["inferred_step_id"] == "S20"
    assert response.metadata["final_action_plan"]["step_id"] == "S20"
    assert response.actions[0]["target"] == "refuel_probe_switch"
    assert response.metadata["vlm_call_status"] == "not_required"
    assert response.metadata["harness_trace"]["vlm_call"]["extractor_called"] is False
    assert stats["vision_cycles"] == 0


def test_live_loop_keeps_unresolved_visual_sticky_hold_for_regressed_preliminary_step(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_visual_sticky_regression.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-sticky-visual-regression",
    )
    try:
        loop._sticky_inference_step_id = "S19"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcsmc_final_go_result_visible==seen",)

        regressed = loop._active_step_ids_for_vision_facts(StepInferenceResult("S17", ()))
        advanced = loop._active_step_ids_for_vision_facts(StepInferenceResult("S20", ()))
    finally:
        loop.close()

    assert regressed == ["S17", "S19"]
    assert advanced == ["S20"]


def test_live_loop_advances_satisfied_s19_sticky_hold_for_regressed_preliminary_step(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s19_sticky_satisfied_regression.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-sticky-visual-satisfied-regression",
    )
    try:
        loop._last_inferred_step_id = "S19"
        loop._sticky_inference_step_id = "S19"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcsmc_final_go_result_visible==seen",)
        loop._vision_fact_snapshot = {
            "fcsmc_final_go_result_visible": {
                "fact_id": "fcsmc_final_go_result_visible",
                "state": "seen",
                "source_frame_id": "old-s19-frame",
                "sticky": True,
                "observed_at_wall_ms": 9000,
                "expires_at_wall_ms": 609000,
            }
        }

        active = loop._active_step_ids_for_vision_facts(
            StepInferenceResult("S17", ()),
            now_wall_ms=10000,
        )
    finally:
        loop.close()

    assert active == ["S20"]


def test_live_loop_suppresses_stale_visual_preliminary_behind_progress(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_stale_visual_preliminary.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-stale-visual-preliminary",
    )
    try:
        loop._last_inferred_step_id = "S16"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()))
    finally:
        loop.close()

    assert active == ["S16"]


def test_live_loop_floors_stale_preliminary_to_non_visual_progress(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_non_visual_progress_floor.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-non-visual-progress-floor",
    )
    try:
        loop._last_inferred_step_id = "S14"

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()))
    finally:
        loop.close()

    assert active == ["S14"]


def test_live_loop_preserves_visual_priority_progress_floor_with_nonvisual_hold(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_visual_progress_floor.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-visual-progress-floor",
    )
    try:
        loop._last_inferred_step_id = "S15"
        loop._sticky_inference_step_id = "S15"
        loop._sticky_inference_missing_conditions = ("vars.fcs_reset_complete==true",)

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()))
    finally:
        loop.close()

    assert active == ["S15"]


def test_live_loop_advances_past_visual_progress_without_visual_hold(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_completed_visual_progress.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-completed-visual-progress",
    )
    try:
        loop._last_inferred_step_id = "S19"
        loop._sticky_inference_step_id = "S19"
        loop._sticky_inference_missing_conditions = ("vars.ext_refuel_probe_value>=60000",)

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()))
    finally:
        loop.close()

    assert active == ["S20"]


def test_live_loop_treats_completed_s08_page_navigation_as_s09_for_vision_gate(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_completed_s08_navigation.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s08-nav-complete",
    )
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
        loop.recent_ring.add_delta({"LEFT_DDI_PB_15": 1}, t_wall=10.1, seq=2)
        loop._vision_fact_snapshot = {
            "bit_root_page_visible": {
                "fact_id": "bit_root_page_visible",
                "state": "seen",
                "source_frame_id": "old-s08-frame",
                "sticky": False,
                "observed_at_wall_ms": 10000,
            },
            "supt_page_visible": {
                "fact_id": "supt_page_visible",
                "state": "seen",
                "source_frame_id": "old-s08-frame",
                "sticky": False,
                "observed_at_wall_ms": 10000,
            }
        }

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()), now_wall_ms=10000)
    finally:
        loop.close()

    assert active == ["S09"]


def test_live_loop_does_not_complete_s08_navigation_from_pb15_before_supt_fact(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_early_pb15_s08_navigation.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-early-pb15-s08-nav",
    )
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
        loop.recent_ring.add_delta({"LEFT_DDI_PB_15": 1}, t_wall=9.0, seq=1)
        loop._vision_fact_snapshot = {
            "bit_root_page_visible": {
                "fact_id": "bit_root_page_visible",
                "state": "seen",
                "source_frame_id": "old-s08-frame",
                "sticky": False,
                "observed_at_wall_ms": 10000,
            },
            "supt_page_visible": {
                "fact_id": "supt_page_visible",
                "state": "seen",
                "source_frame_id": "old-s08-frame",
                "sticky": False,
                "observed_at_wall_ms": 10000,
            },
        }

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()), now_wall_ms=11000)
    finally:
        loop.close()

    assert active == ["S08"]


def test_live_loop_does_not_complete_s08_navigation_from_pb15_release_after_supt_fact(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_pb15_release_s08_navigation.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-pb15-release-s08-nav",
    )
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
        loop.recent_ring.add_delta({"LEFT_DDI_PB_15": 1}, t_wall=9.0, seq=1)
        loop.recent_ring.add_delta({"LEFT_DDI_PB_15": 0}, t_wall=10.1, seq=2)
        loop._vision_fact_snapshot = {
            "bit_root_page_visible": {
                "fact_id": "bit_root_page_visible",
                "state": "seen",
                "source_frame_id": "old-s08-frame",
                "sticky": False,
                "observed_at_wall_ms": 10000,
            },
            "supt_page_visible": {
                "fact_id": "supt_page_visible",
                "state": "seen",
                "source_frame_id": "old-s08-frame",
                "sticky": False,
                "observed_at_wall_ms": 10000,
            },
        }

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()), now_wall_ms=11000)
    finally:
        loop.close()

    assert active == ["S08"]


def test_live_loop_does_not_complete_s08_navigation_without_bit_root(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_incomplete_s08_navigation.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-incomplete-s08-nav",
    )
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
        loop._step_interacted_targets = {"left_mdi_pb15"}
        loop._vision_fact_snapshot = {
            "supt_page_visible": {
                "fact_id": "supt_page_visible",
                "state": "seen",
                "source_frame_id": "old-s08-frame",
                "sticky": False,
            }
        }

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()), now_wall_ms=10000)
    finally:
        loop.close()

    assert active == ["S08"]


def test_live_loop_does_not_complete_s08_navigation_from_expired_visual_fact(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_expired_s08_navigation.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-expired-s08-nav",
    )
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
        loop._step_interacted_targets = {"left_mdi_pb15"}
        loop.recent_ring.add_delta({"LEFT_DDI_PB_15": 1}, t_wall=9.5, seq=1)
        loop._vision_fact_snapshot = {
            "bit_root_page_visible": {
                "fact_id": "bit_root_page_visible",
                "state": "seen",
                "source_frame_id": "fresh-bit-frame",
                "sticky": False,
                "observed_at_wall_ms": 9500,
                "expires_at_wall_ms": 12000,
            },
            "supt_page_visible": {
                "fact_id": "supt_page_visible",
                "state": "seen",
                "source_frame_id": "expired-s08-frame",
                "sticky": False,
                "observed_at_wall_ms": 8000,
                "expires_at_wall_ms": 9000,
            }
        }

        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S08", ()), now_wall_ms=10000)
    finally:
        loop.close()

    assert active == ["S08"]


def test_live_inference_advances_sticky_s09_when_comm1_is_complete(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_sticky_s09_complete.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s09-sticky-complete",
        lang="zh",
    )
    try:
        loop._sticky_inference_step_id = "S09"
        loop._sticky_inference_missing_conditions = ("vars.comm1_freq_134_000==true",)
        loop._last_inferred_step_id = "S09"

        stabilized = loop._stabilize_live_inference(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
            {
                "battery_on": True,
                "power_available": True,
                "left_ddi_on": True,
                "right_ddi_on": True,
                "mpcd_on": True,
                "hud_on": True,
                "right_engine_nominal_start_params": True,
                "comm1_freq_134_000": True,
                "engine_crank_left_complete": False,
            },
            recent_ui_targets=[],
        )
    finally:
        loop.close()

    assert stabilized.inferred_step_id == "S10"
    assert "vars.comm1_freq_134_000==true" not in stabilized.missing_conditions


def test_live_inference_advances_sticky_s09_when_comm1_value_is_complete(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_sticky_s09_value_complete.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s09-sticky-value-complete",
        lang="zh",
    )
    try:
        loop._sticky_inference_step_id = "S09"
        loop._sticky_inference_missing_conditions = ("vars.comm1_freq_134_000==true",)
        loop._last_inferred_step_id = "S09"

        stabilized = loop._stabilize_live_inference(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
            {
                "battery_on": True,
                "power_available": True,
                "left_ddi_on": True,
                "right_ddi_on": True,
                "mpcd_on": True,
                "hud_on": True,
                "right_engine_nominal_start_params": True,
                "comm1_freq_value": 13400,
                "engine_crank_left_complete": False,
            },
            recent_ui_targets=[],
        )
    finally:
        loop.close()

    assert stabilized.inferred_step_id == "S10"
    assert "vars.comm1_freq_134_000==true" not in stabilized.missing_conditions


def test_live_inference_advances_sticky_s10_when_left_engine_is_complete(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_sticky_s10_left_engine_complete.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s10-sticky-left-complete",
        lang="zh",
    )
    try:
        loop._sticky_inference_step_id = "S10"
        loop._sticky_inference_missing_conditions = ("vars.engine_crank_left_complete==true",)
        loop._last_inferred_step_id = "S10"

        stabilized = loop._stabilize_live_inference(
            StepInferenceResult("S10", ("vars.engine_crank_left_complete==true",)),
            {
                "battery_on": True,
                "power_available": True,
                "left_ddi_on": True,
                "right_ddi_on": True,
                "mpcd_on": True,
                "hud_on": True,
                "right_engine_nominal_start_params": True,
                "comm1_freq_134_000": True,
                "engine_crank_left": False,
                "engine_crank_left_complete": True,
                "rpm_l": 64,
                "rpm_l_gte_25": True,
                "rpm_l_gte_60": True,
                "left_engine_nominal_start_params": True,
                "left_engine_idle_ready": True,
                "throttle_l_not_off": True,
                "ins_mode": 0,
                "ins_fast_align_complete": False,
            },
            recent_ui_targets=[],
        )
    finally:
        loop.close()

    assert stabilized.inferred_step_id == "S12"
    assert "vars.engine_crank_left_complete==true" not in stabilized.missing_conditions


def test_live_inference_treats_stable_left_engine_params_as_s10_complete(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_sticky_s10_left_engine_stable_params.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s10-sticky-left-stable-params",
        lang="zh",
    )
    try:
        loop._sticky_inference_step_id = "S10"
        loop._sticky_inference_missing_conditions = ("vars.engine_crank_left_complete==true",)
        loop._last_inferred_step_id = "S10"

        stabilized = loop._stabilize_live_inference(
            StepInferenceResult("S10", ("vars.engine_crank_left_complete==true",)),
            {
                "battery_on": True,
                "power_available": True,
                "left_ddi_on": True,
                "right_ddi_on": True,
                "mpcd_on": True,
                "hud_on": True,
                "right_engine_nominal_start_params": True,
                "comm1_freq_134_000": True,
                "engine_crank_left": False,
                "engine_crank_left_complete": False,
                "rpm_l": 64,
                "rpm_l_gte_25": True,
                "rpm_l_gte_60": True,
                "left_engine_nominal_start_params": True,
                "left_engine_idle_ready": True,
                "throttle_l_not_off": True,
                "ins_mode": 0,
                "ins_fast_align_complete": False,
            },
            recent_ui_targets=[],
        )
    finally:
        loop.close()

    assert stabilized.inferred_step_id == "S12"
    assert "vars.engine_crank_left_complete==true" not in stabilized.missing_conditions


def test_live_loop_ignores_stale_s19_visual_facts_for_s21(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s21_ignores_stale_s19_vlm.jsonl"
    frame = _bios_frame(1, 10.0, apu_switch=1)
    frame["bios"]["EXT_REFUEL_PROBE_SW"] = 65535
    frame["delta"]["EXT_REFUEL_PROBE_SW"] = 65535
    _write_replay(replay_path, [frame])

    class FailingIfCalledVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):  # pragma: no cover
            del vision, session_id, trigger_wall_ms
            raise AssertionError("VLM extractor should not be called for S21")

        def close(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-s21-no-vlm",
        vision_fact_extractor=FailingIfCalledVisionFactExtractor(),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult(
        "S21",
        ("vars.ext_refuel_probe_value==0",),
    )
    loop._last_inferred_step_id = "S19"
    loop._sticky_inference_step_id = "S19"
    loop._sticky_inference_missing_conditions = ("vars.ext_refuel_probe_value==0",)
    loop._vision_fact_snapshot = {
        "fcsmc_final_go_result_visible": {
            "fact_id": "fcsmc_final_go_result_visible",
            "state": "seen",
            "source_frame_id": "old-s19-frame",
            "sticky": True,
            "expires_after_ms": 600000,
            "observed_at_wall_ms": 9000,
            "expires_at_wall_ms": 609000,
            "evidence_note": "Stale FCS-MC GO result from S19.",
        }
    }
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    assert request.metadata["vision_fact_status"] == "vision_not_required"
    assert request.metadata["vision_fact_active_step_ids"] == ["S21"]
    assert request.context["vision_facts"] == []
    assert not any(
        item["source"] in {"sticky_state", "visual_anchor"}
        for item in request.context["candidate_steps"]
    )
    assert response.metadata["vlm_call_status"] == "not_required"
    assert response.metadata["harness_trace"]["vlm_call"]["ignored_fact_count"] == 1


def test_live_loop_calls_vision_fact_extractor_for_s08_and_merges_facts(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s08_uses_vlm.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-s08-vlm"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000123",
                    capture_wall_ms=10000,
                    frame_seq=123,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000123.png"),
                )
            ]

    class RecordingVisionFactExtractor:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            self.calls.append(
                {
                    "vision": dict(vision),
                    "session_id": session_id,
                    "trigger_wall_ms": trigger_wall_ms,
                }
            )
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=["10000_000123"],
                        facts=[
                            VisionFact(
                                fact_id="fcs_page_visible",
                                state="seen",
                                source_frame_id="10000_000123",
                                expires_after_ms=2000,
                                evidence_note="FCS page visible.",
                            )
                        ],
                    ),
                },
            )()

        def close(self) -> None:
            return

    extractor = RecordingVisionFactExtractor()
    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-s08-vlm",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-s08-vlm",
        vision_mode="replay",
        vision_fact_extractor=extractor,
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
        stats = loop.stats.to_dict()
    finally:
        loop.close()

    assert response is not None
    assert len(extractor.calls) == 1
    request = model.calls[0]["request"]
    assert request.metadata["vision_fact_status"] == "available"
    assert request.metadata["vision_fact_active_step_ids"] == ["S08"]
    assert request.context["vision_fact_summary"]["seen_fact_ids"] == ["fcs_page_visible"]
    assert response.metadata["vision_fact_summary"]["seen_fact_ids"] == ["fcs_page_visible"]
    assert stats["vision_cycles"] == 1


def test_live_loop_passes_vision_session_id_to_vision_fact_extractor(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_with_distinct_vision_session.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-vision"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="1772872445010_000123",
                    capture_wall_ms=1772872445010,
                    frame_seq=123,
                    channel="panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "1772872445010_000123.png"),
                )
            ]

    class StaticVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            del vision, trigger_wall_ms
            assert session_id == "sess-vision"
            return type(
                "Result",
                (),
                {
                    "status": "vision_unavailable",
                    "error": None,
                    "metadata": {},
                    "observation": None,
                },
            )()

        def close(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-main",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-vision",
        vision_mode="live",
        vision_fact_extractor=StaticVisionFactExtractor(),
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=1772872445.0)
    finally:
        loop.close()

    assert response is not None


def test_extract_vision_fact_context_degrades_when_merge_raises(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_merge_failure.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "pack_id": "merge_failure_pack",
                "version": "v1",
                "steps": [{"id": "S01", "ui_targets": ["apu_switch"]}],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            del session_id
            return

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="1772872444950_000122",
                    capture_wall_ms=1772872444950,
                    frame_seq=122,
                    channel="panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "1772872444950_000122.png"),
                ),
                VisionObservation(
                    frame_id="1772872445010_000123",
                    capture_wall_ms=1772872445010,
                    frame_seq=123,
                    channel="panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "1772872445010_000123.png"),
                ),
            ]

    class StaticVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            assert session_id == "sess-merge-fail"
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=list(vision["frame_ids"]),
                        facts=[
                            VisionFact(
                                fact_id="supt_page_visible",
                                state="seen",
                                source_frame_id="1772872445010_000123",
                                expires_after_ms=2000,
                                evidence_note="SUPT page visible.",
                            )
                        ],
                    ),
                },
            )()

        def close(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-merge-fail",
        pack_path=pack_path,
        ui_map_path=Path(_default_pack_path()).parent / "ui_map.yaml",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-merge-fail",
        vision_mode="live",
        vision_fact_extractor=StaticVisionFactExtractor(),
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        vision_selection = loop._build_vision_selection(observation=obs, trigger_t_wall=1772872445.0)
        context = loop._extract_vision_fact_context(vision_selection=vision_selection)
    finally:
        loop.close()

    assert context["status"] == "extractor_failed"
    assert context["vision_facts"] == []
    assert context["vision_fact_summary"]["status"] == "extractor_failed"
    assert "vision fact id" in context["metadata"]["vision_fact_merge_error"]


def test_build_vision_fact_extractor_from_model_uses_pack_metadata_path(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    vision_facts_path = tmp_path / "configs" / "vision_facts_custom.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "pack_id": "custom_pack",
                "metadata": {"vision_facts_path": "configs/vision_facts_custom.yaml"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    vision_facts_path.parent.mkdir(parents=True, exist_ok=True)
    vision_facts_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "layout_id": "custom_layout",
                "facts": [
                    {
                        "fact_id": "fcs_page_visible",
                        "sticky": False,
                        "expires_after_ms": 4321,
                    }
                ],
                "step_bindings": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    model = OpenAICompatModel(client=FakeClient(), enable_multimodal=True)

    extractor = _build_vision_fact_extractor_from_model(
        model=model,
        lang="zh",
        pack_path=pack_path,
    )

    assert extractor is not None
    assert extractor._config["layout_id"] == "custom_layout"
    assert extractor._config["facts_by_id"]["fcs_page_visible"]["expires_after_ms"] == 4321


def test_live_loop_marks_vision_unavailable_without_sidecar(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_without_vision.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    executor = RecordingExecutor()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        session_id="sess-no-vision",
        vision_mode="replay",
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    vision = request.context["vision"]
    assert vision["status"] == "vision_unavailable"
    assert vision["frame_id"] is None
    assert vision["vision_used"] is False
    assert vision["frame_ids"] == []
    assert vision["sync_status"] is None
    assert vision["sync_miss_reason"] == "vision_port_unconfigured"
    assert response.metadata["vision_fallback_reason"] == "vision_unavailable"
    assert response.metadata["failure_code"] == "vision_unavailable"
    assert "vision_unavailable" in response.metadata["failure_codes"]


def test_live_loop_audit_fields_flow_into_request_response_and_overlay(monkeypatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_audit_fields.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])
    events: list[dict[str, Any]] = []

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-audit"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000123",
                    capture_wall_ms=10000,
                    frame_seq=123,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000123.png"),
                )
            ]

    class StaticVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            assert session_id == "sess-audit"
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=list(vision["frame_ids"]),
                        facts=[
                            VisionFact(
                                fact_id="supt_page_visible",
                                state="seen",
                                source_frame_id="10000_000123",
                                expires_after_ms=2000,
                                evidence_note="SUPT page visible.",
                            )
                        ],
                    ),
                },
            )()

        def close(self) -> None:
            return

    class StableVisionModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            self.calls.append({"observation": observation, "request": request})
            fused_step_id = request.context["deterministic_step_hint"]["inferred_step_id"]
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Turn on APU.",
                actions=[],
                explanations=["Turn on APU."],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": fused_step_id, "error_category": "OM"},
                        "next": {"step_id": fused_step_id},
                        "overlay": {
                            "targets": ["apu_switch"],
                            "evidence": [
                                {
                                    "target": "apu_switch",
                                    "type": "delta",
                                    "ref": "RECENT_UI_TARGETS.apu_switch",
                                    "quote": "Recent delta shows APU switch activity.",
                                }
                            ],
                        },
                        "explanations": ["Turn on APU."],
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    model = StableVisionModel()
    source = ReplayBiosReceiver(replay_path, speed=0.0)
    executor = _make_evented_overlay_executor(monkeypatch, events, session_id="sess-audit")
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=executor,
        session_id="sess-audit",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-audit",
        vision_mode="replay",
        vision_fact_extractor=StaticVisionFactExtractor(),
        event_sink=lambda event: events.append(event.to_dict()),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    tutor_request = next(event for event in events if event["kind"] == "tutor_request")
    tutor_response = next(event for event in events if event["kind"] == "tutor_response")
    overlay_requested = next(event for event in events if event["kind"] == "overlay_requested")

    request_meta = tutor_request["payload"]["metadata"]
    response_meta = tutor_response["payload"]["metadata"]
    overlay_payload = overlay_requested["payload"]

    assert request_meta["vision_used"] is True
    assert request_meta["frame_id"] == "10000_000123"
    assert request_meta["sync_delta_ms"] == 0
    assert request_meta["vision_fact_summary"]["status"] == "available"
    assert request_meta["vision_fallback_reason"] is None
    assert request_meta["layout_id"] == "fa18c_composite_panel_v2"
    assert response_meta["fused_step_id"] == request_meta["fused_step_id"]
    assert response_meta["fused_missing_conditions"] == request_meta["fused_missing_conditions"]
    assert overlay_payload["vision_used"] is True
    assert overlay_payload["frame_id"] == "10000_000123"
    assert overlay_payload["sync_delta_ms"] == 0
    assert overlay_payload["vision_fact_summary"]["status"] == "available"
    assert overlay_payload["fused_step_id"] == response_meta["fused_step_id"]
    assert overlay_payload["fused_missing_conditions"] == response_meta["fused_missing_conditions"]
    assert overlay_payload["vision_fallback_reason"] is None
    assert overlay_payload["layout_id"] == "fa18c_composite_panel_v2"


def test_live_loop_marks_vision_sync_miss_with_audit_metadata_and_stats(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_sync_miss.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class OutOfWindowVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-sync-miss"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="9500_000001",
                    capture_wall_ms=9500,
                    frame_seq=1,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "9500_000001.png"),
                )
            ]

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-sync-miss",
        vision_port=OutOfWindowVisionPort(),
        vision_session_id="sess-sync-miss",
        vision_mode="replay",
        vision_sync_window_ms=100,
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
        stats = loop.stats.to_dict()
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    assert request.metadata["vision_fact_status"] == "vision_unavailable"
    assert request.metadata["vision_fallback_reason"] == "vision_sync_miss"
    assert response.metadata["vision_fallback_reason"] == "vision_sync_miss"
    assert response.metadata["failure_code"] == "vision_sync_miss"
    assert "vision_sync_miss" in response.metadata["failure_codes"]
    assert stats["vision_cycles"] == 0
    assert stats["vision_sync_miss_count"] == 1
    assert stats["vision_text_fallback_count"] == 0


def test_live_loop_marks_vision_parse_fail_with_deterministic_metadata(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_parse_fail.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-parse-fail"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000123",
                    capture_wall_ms=10000,
                    frame_seq=123,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000123.png"),
                )
            ]

    class FailingVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            del vision, session_id, trigger_wall_ms
            return type(
                "Result",
                (),
                {
                    "status": "extractor_failed",
                    "error": "ValueError: invalid vision fact payload",
                    "metadata": {"error": "ValueError: invalid vision fact payload"},
                    "observation": None,
                },
            )()

        def close(self) -> None:
            return

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-parse-fail",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-parse-fail",
        vision_mode="replay",
        vision_fact_extractor=FailingVisionFactExtractor(),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    request = model.calls[0]["request"]
    assert request.metadata["vision_fallback_reason"] == "vision_parse_fail"
    assert response.metadata["vision_fallback_reason"] == "vision_parse_fail"
    assert response.metadata["failure_code"] == "vision_parse_fail"
    assert "vision_parse_fail" in response.metadata["failure_codes"]


def test_live_loop_tracks_vision_text_fallback_in_metadata_and_stats(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_text_fallback.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-text-fallback"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000123",
                    capture_wall_ms=10000,
                    frame_seq=123,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000123.png"),
                )
            ]

    class StaticVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=list(vision["frame_ids"]),
                        facts=[
                            VisionFact(
                                fact_id="supt_page_visible",
                                state="seen",
                                source_frame_id="10000_000123",
                                expires_after_ms=2000,
                                evidence_note="SUPT page visible.",
                            )
                        ],
                    ),
                },
            )()

        def close(self) -> None:
            return

    class TextFallbackModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Turn on APU.",
                actions=[],
                explanations=["Turn on APU."],
                metadata={
                    "provider": "openai_compat",
                    "multimodal_fallback_to_text": True,
                    "multimodal_failure_reason": "server rejected multimodal request",
                    "help_response": {
                        "diagnosis": {"step_id": "S02", "error_category": "OM"},
                        "next": {"step_id": request.context["deterministic_step_hint"]["inferred_step_id"]},
                        "overlay": {
                            "targets": ["apu_switch"],
                            "evidence": [
                                {
                                    "target": "apu_switch",
                                    "type": "delta",
                                    "ref": "RECENT_UI_TARGETS.apu_switch",
                                    "quote": "Recent delta shows APU switch activity.",
                                }
                            ],
                        },
                        "explanations": ["Turn on APU."],
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=TextFallbackModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-text-fallback",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-text-fallback",
        vision_mode="replay",
        vision_fact_extractor=StaticVisionFactExtractor(),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
        stats = loop.stats.to_dict()
    finally:
        loop.close()

    assert response is not None
    assert response.metadata["vision_fallback_reason"] == "vision_text_fallback"
    assert response.metadata["failure_code"] == "vision_text_fallback"
    assert "vision_text_fallback" in response.metadata["failure_codes"]
    assert stats["vision_cycles"] == 1
    assert stats["vision_text_fallback_count"] == 1


def test_live_loop_marks_vision_conflict_unresolved_when_model_disagrees_with_fused_step(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_conflict_unresolved.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-conflict"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000123",
                    capture_wall_ms=10000,
                    frame_seq=123,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000123.png"),
                )
            ]

    class StaticVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=list(vision["frame_ids"]),
                        facts=[
                            VisionFact(
                                fact_id="supt_page_visible",
                                state="seen",
                                source_frame_id="10000_000123",
                                expires_after_ms=2000,
                                evidence_note="SUPT page visible.",
                            )
                        ],
                    ),
                },
            )()

        def close(self) -> None:
            return

    class ConflictingModel:
        def __init__(self) -> None:
            self.fused_step_id: str | None = None

        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            fused_step_id = request.context["deterministic_step_hint"]["inferred_step_id"]
            self.fused_step_id = fused_step_id
            conflicting_step = "S01" if fused_step_id != "S01" else "S05"
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Check the highlighted switch.",
                actions=[],
                explanations=["Check the highlighted switch."],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": conflicting_step, "error_category": "OM"},
                        "next": {"step_id": conflicting_step},
                        "overlay": {
                            "targets": ["apu_switch"],
                            "evidence": [
                                {
                                    "target": "apu_switch",
                                    "type": "delta",
                                    "ref": "RECENT_UI_TARGETS.apu_switch",
                                    "quote": "Recent delta shows APU switch activity.",
                                }
                            ],
                        },
                        "explanations": ["Check the highlighted switch."],
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    model = ConflictingModel()
    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-conflict",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-conflict",
        vision_mode="replay",
        vision_fact_extractor=StaticVisionFactExtractor(),
    )
    loop._infer_preliminary_step_for_vision_facts = lambda obs: StepInferenceResult("S08", [])
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.metadata["fused_step_id"] == model.fused_step_id
    assert response.metadata["vision_fallback_reason"] == "vision_conflict_unresolved"
    assert response.metadata["failure_code"] == "vision_conflict_unresolved"
    assert "vision_conflict_unresolved" in response.metadata["failure_codes"]


def test_live_loop_uses_deterministic_fallback_when_visual_model_disagrees_with_s22_and_returns_no_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s22_visual_conflict.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-s22-visual-conflict"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000125",
                    capture_wall_ms=10000,
                    frame_seq=125,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000125.png"),
                )
            ]

    class StaticVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=list(vision["frame_ids"]),
                        facts=[
                            VisionFact(
                                fact_id="fcs_page_visible",
                                state="seen",
                                source_frame_id="10000_000125",
                                expires_after_ms=2000,
                                evidence_note="FCS page visible.",
                            ),
                            VisionFact(
                                fact_id="fcsmc_page_visible",
                                state="seen",
                                source_frame_id="10000_000125",
                                expires_after_ms=2000,
                                evidence_note="FCS-MC page visible.",
                            ),
                            VisionFact(
                                fact_id="fcsmc_final_go_result_visible",
                                state="seen",
                                source_frame_id="10000_000125",
                                expires_after_ms=600000,
                                evidence_note="Final GO visible.",
                            ),
                            VisionFact(
                                fact_id="hsi_page_visible",
                                state="seen",
                                source_frame_id="10000_000125",
                                expires_after_ms=2000,
                                evidence_note="HSI visible.",
                            ),
                        ],
                    ),
                },
            )()

        def close(self) -> None:
            return

    class WrongS18CompletionModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message=(
                    "FCS-MC final GO is visible, so S18 is complete. "
                    "Current visual evidence does not support more highlighting."
                ),
                actions=[],
                explanations=[
                    "FCS-MC final GO is visible, so S18 is complete. Current visual evidence does not support more highlighting."
                ],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S18", "error_category": "OM"},
                        "next": {"step_id": "S18"},
                        "overlay": {"targets": [], "evidence": []},
                        "explanations": [
                            "FCS-MC final GO is visible, so S18 is complete. Current visual evidence does not support more highlighting."
                        ],
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(inferred_step_id="S30", missing_conditions=()),
    )

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=WrongS18CompletionModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s22-visual-conflict",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-s22-visual-conflict",
        vision_mode="replay",
        vision_fact_extractor=StaticVisionFactExtractor(),
        lang="en",
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.actions
    assert response.actions[0]["target"] == "standby_altimeter_pressure_knob"
    assert response.metadata["fallback_overlay_used"] is True
    assert response.metadata["fallback_overlay_reason"] == "validator_repair"
    assert response.metadata["harness_validator_fallback_reason"] == "deterministic_step:S30"
    assert response.metadata["vision_fallback_reason"] is None
    assert response.metadata["final_public_response"]["actions"][0]["target"] == "standby_altimeter_pressure_knob"


def test_live_loop_rewrites_false_s08_completion_claim_while_preserving_navigation_target(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_false_s08_complete.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=1)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-s08-rewrite"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000124",
                    capture_wall_ms=10000,
                    frame_seq=124,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000124.png"),
                )
            ]

    class MenuStateVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=list(vision["frame_ids"]),
                        facts=[
                            VisionFact(
                                fact_id="tac_page_visible",
                                state="seen",
                                source_frame_id="10000_000124",
                                expires_after_ms=2000,
                                evidence_note="Left DDI TAC page visible.",
                            ),
                            VisionFact(
                                fact_id="supt_page_visible",
                                state="seen",
                                source_frame_id="10000_000124",
                                expires_after_ms=2000,
                                evidence_note="Left DDI SUPT page visible.",
                            ),
                            VisionFact(
                                fact_id="fcs_page_visible",
                                state="not_seen",
                                source_frame_id="10000_000124",
                                expires_after_ms=2000,
                                evidence_note="Left DDI is not yet on the FCS page.",
                            ),
                            VisionFact(
                                fact_id="bit_root_page_visible",
                                state="not_seen",
                                source_frame_id="10000_000124",
                                expires_after_ms=2000,
                                evidence_note="Right DDI is not on the BIT root page.",
                            ),
                        ],
                    ),
                },
            )()

        def close(self) -> None:
            return

    class FalseCompletionModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Displays are powered on with FCS page visible on Left DDI and BIT failures on Right DDI, indicating S08 is complete. Proceed to configure communications (S09).",
                actions=[],
                explanations=[
                    "Displays are powered on with FCS page visible on Left DDI and BIT failures on Right DDI, indicating S08 is complete. Proceed to configure communications (S09)."
                ],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S08", "error_category": "CO"},
                        "next": {"step_id": "S09"},
                        "overlay": {
                            "targets": ["left_mdi_pb15"],
                            "evidence": [
                                {
                                    "target": "left_mdi_pb15",
                                    "type": "rag",
                                    "ref": "RAG_SNIPPETS.fa18c_coldstart_quiz_8",
                                    "quote": "Select the FCS page on the left DDI.",
                                    "grounding_confidence": 0.9,
                                }
                            ],
                        },
                        "explanations": [
                            "Displays are powered on with FCS page visible on Left DDI and BIT failures on Right DDI, indicating S08 is complete. Proceed to configure communications (S09)."
                        ],
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=FalseCompletionModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s08-rewrite",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-s08-rewrite",
        vision_mode="replay",
        vision_fact_extractor=MenuStateVisionFactExtractor(),
        lang="en",
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.actions
    assert response.message != (
        "Displays are powered on with FCS page visible on Left DDI and BIT failures on Right DDI, indicating S08 is complete. Proceed to configure communications (S09)."
    )
    assert response.explanations == [response.message]
    assert response.metadata["completion_conflict_rewritten"] is True
    assert response.metadata["model_raw_help_response"]["next"]["step_id"] == "S09"
    assert response.metadata["model_raw_explanations"] == [
        "Displays are powered on with FCS page visible on Left DDI and BIT failures on Right DDI, indicating S08 is complete. Proceed to configure communications (S09)."
    ]
    assert response.metadata["final_public_response"]["message"] == response.message
    assert response.metadata["final_public_response"]["explanations"] == [response.message]


def test_completion_conflict_does_not_clear_s20_overlay_when_text_says_s19_complete(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_s20_s19_complete_text.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "deterministic_step_hint": {
                    "inferred_step_id": "S20",
                    "missing_conditions": ["vars.pitot_heat_on==true"],
                }
            },
        )
        response = TutorResponse(
            status="ok",
            message="S19 FCS BIT 已完成。当前处于 S20，需打开皮托管加热。",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "pitot_heater_switch",
                    "element_id": "pnt_409",
                }
            ],
            explanations=["S19 FCS BIT 已完成。当前处于 S20，需打开皮托管加热。"],
            metadata={
                "help_response": {
                    "diagnosis": {"step_id": "S20", "error_category": "OM"},
                    "next": {"step_id": "S20"},
                    "overlay": {"targets": ["pitot_heater_switch"], "evidence": []},
                    "explanations": ["S19 FCS BIT 已完成。当前处于 S20，需打开皮托管加热。"],
                },
                "next": {"step_id": "S20"},
                "diagnosis": {"step_id": "S20", "error_category": "OM"},
            },
        )

        rewritten = loop._rewrite_conflicting_step_completion_response(response, request)

        assert rewritten is False
        assert response.actions
        assert response.actions[0]["target"] == "pitot_heater_switch"
        assert response.metadata.get("completion_conflict_overlay_cleared") is not True
    finally:
        loop.close()


def test_live_loop_short_circuits_terminal_state_without_calling_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    replay_path = tmp_path / "bios_s25_terminal_s18_model.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "sess-s25-terminal-rewrite"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="10000_000160",
                    capture_wall_ms=10000,
                    frame_seq=160,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000160.png"),
                )
            ]

    class MenuStateVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "error": None,
                    "metadata": {},
                    "observation": VisionFactObservation(
                        session_id=session_id,
                        trigger_wall_ms=trigger_wall_ms,
                        frame_ids=list(vision["frame_ids"]),
                        facts=[
                            VisionFact(
                                fact_id="fcsmc_final_go_result_visible",
                                state="seen",
                                source_frame_id="10000_000160",
                                expires_after_ms=600000,
                                sticky=True,
                            ),
                            VisionFact(
                                fact_id="fcsmc_intermediate_result_visible",
                                state="not_seen",
                                source_frame_id="10000_000160",
                                expires_after_ms=2000,
                                sticky=False,
                            ),
                            VisionFact(
                                fact_id="fcsmc_in_test_visible",
                                state="not_seen",
                                source_frame_id="10000_000160",
                                expires_after_ms=2000,
                                sticky=False,
                            ),
                            VisionFact(
                                fact_id="fcsmc_page_visible",
                                state="seen",
                                source_frame_id="10000_000160",
                                expires_after_ms=2000,
                                sticky=False,
                            ),
                        ],
                    ),
                    "raw_text": "",
                    "parse_success": True,
                },
            )()

        def close(self) -> None:
            return

    model = RecordingModel()
    model.print_model_io = True

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=RecordingExecutor(),
        session_id="sess-s25-terminal-rewrite",
        vision_port=StaticVisionPort(),
        vision_session_id="sess-s25-terminal-rewrite",
        vision_mode="replay",
        vision_fact_extractor=MenuStateVisionFactExtractor(),
        lang="zh",
    )
    try:
        original_build_request = loop._build_request

        def patched_build_request(obs, *, vision_selection, vision_fact_context, **kwargs):
            request, prompt_meta, state_key = original_build_request(
                obs,
                vision_selection=vision_selection,
                vision_fact_context=vision_fact_context,
                **kwargs,
            )
            hint = dict(request.context.get("deterministic_step_hint", {}))
            hint["inferred_step_id"] = "S33"
            hint["missing_conditions"] = []
            hint["missing_conditions_count"] = 0
            hint["gate_blockers"] = []
            hint["gate_blocker_count"] = 0
            request.context["deterministic_step_hint"] = hint
            return request, prompt_meta, state_key

        loop._build_request = patched_build_request

        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert model.calls == []
    assert response.metadata["terminal_state_rewritten"] is True
    assert response.message == "当前冷启动流程已完成，无需继续操作。"
    assert response.explanations == ["当前冷启动流程已完成，无需继续操作。"]
    assert response.metadata["diagnosis"]["step_id"] == "S33"
    assert response.metadata["next"] == {"step_id": "S33"}
    assert response.actions == []
    assert response.metadata["fallback_overlay_used"] is False
    assert response.metadata["fallback_overlay_reason"] == "all_steps_complete"
    out = capsys.readouterr().out
    assert "[MODEL_IO][FINAL_PUBLIC_RESPONSE]" in out
    assert "当前冷启动流程已完成，无需继续操作。" in out


def test_rewrite_terminal_state_conflict_response_skips_when_gate_blockers_exist(tmp_path: Path) -> None:
    replay_path = tmp_path / "empty.jsonl"
    _write_replay(replay_path, [])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        lang="zh",
    )
    try:
        response = TutorResponse(
            status="ok",
            message="FCS-MC 页面可见，最终 GO 结果可见。",
            actions=[],
            explanations=["FCS-MC 页面可见，最终 GO 结果可见。"],
            metadata={
                "diagnosis": {"step_id": "S18", "error_category": "OM"},
                "next": {"step_id": "S18"},
                "help_response": {
                    "diagnosis": {"step_id": "S19", "error_category": "OM"},
                    "next": {"step_id": "S19"},
                },
            },
        )
        request = TutorRequest(
            request_id="terminal-gate-blocked",
            message="help",
            context={
                "deterministic_step_hint": {
                    "inferred_step_id": "S33",
                    "missing_conditions": [],
                    "gate_blockers": [{"ref": "GATES.S33.precondition", "reason": "blocked"}],
                }
            },
        )

        rewritten = loop._rewrite_terminal_state_conflict_response(response, request)
    finally:
        loop.close()

    assert rewritten is False
    assert response.message == "FCS-MC 页面可见，最终 GO 结果可见。"
    assert response.metadata["diagnosis"] == {"step_id": "S18", "error_category": "OM"}
    assert response.metadata["next"] == {"step_id": "S18"}


def test_rewrite_terminal_state_conflict_response_skips_short_circuit_response_without_model_payload(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "empty.jsonl"
    _write_replay(replay_path, [])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        lang="zh",
    )
    try:
        request = TutorRequest(
            request_id="terminal-short-circuit",
            message="help",
            context={
                "deterministic_step_hint": {
                    "inferred_step_id": "S33",
                    "missing_conditions": [],
                    "gate_blockers": [],
                }
            },
        )
        response = loop._build_terminal_state_response(request)

        rewritten = loop._rewrite_terminal_state_conflict_response(response, request)
    finally:
        loop.close()

    assert rewritten is False
    assert response.message == "当前冷启动流程已完成，无需继续操作。"
    assert response.explanations == ["当前冷启动流程已完成，无需继续操作。"]
    assert response.metadata["diagnosis"] == {"step_id": "S33"}
    assert response.metadata["next"] == {"step_id": "S33"}
    assert response.metadata["terminal_state_original_message"] == "当前冷启动流程已完成，无需继续操作。"
    assert response.metadata["terminal_state_original_explanations"] == []


def test_rewrite_terminal_state_conflict_response_clears_stale_actions(tmp_path: Path) -> None:
    replay_path = tmp_path / "empty.jsonl"
    _write_replay(replay_path, [])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        lang="zh",
    )
    try:
        response = TutorResponse(
            status="ok",
            message="请继续按 FCS BIT 开关。",
            actions=[
                {
                    "type": "overlay",
                    "intent": "highlight",
                    "target": "fcs_bit_switch",
                    "element_id": "pnt_999",
                }
            ],
            explanations=["请继续按 FCS BIT 开关。"],
            metadata={
                "diagnosis": {"step_id": "S18", "error_category": "OM"},
                "next": {"step_id": "S18"},
                "help_response": {
                    "diagnosis": {"step_id": "S19", "error_category": "OM"},
                    "next": {"step_id": "S19"},
                },
            },
        )
        request = TutorRequest(
            request_id="terminal-clear-actions",
            message="help",
            context={
                "deterministic_step_hint": {
                    "inferred_step_id": "S33",
                    "missing_conditions": [],
                    "gate_blockers": [],
                }
            },
        )

        rewritten = loop._rewrite_terminal_state_conflict_response(response, request)
    finally:
        loop.close()

    assert rewritten is True
    assert response.message == "当前冷启动流程已完成，无需继续操作。"
    assert response.actions == []
    assert response.metadata["terminal_state_original_actions"] == [
        {
            "type": "overlay",
            "intent": "highlight",
            "target": "fcs_bit_switch",
            "element_id": "pnt_999",
        }
    ]
    assert response.metadata["diagnosis"] == {"step_id": "S33", "error_category": "OM"}
    assert response.metadata["next"] == {"step_id": "S33"}


def test_live_loop_replaces_stale_s08_overlay_with_s09_action_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_path = tmp_path / "bios_s08_to_s09_overlay.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class WrongOverlayModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Step S08 is complete. Proceed to S09.",
                actions=[],
                explanations=["Step S08 is complete. Proceed to S09."],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S08", "error_category": "CO"},
                        "next": {"step_id": "S09"},
                        "overlay": {
                            "targets": ["left_mdi_pb15"],
                            "evidence": [
                                {
                                    "target": "left_mdi_pb15",
                                    "type": "gate",
                                    "ref": "GATES.S08.completion",
                                    "quote": "S08 completion allowed.",
                                    "grounding_confidence": 0.95,
                                }
                            ],
                        },
                        "explanations": ["Step S08 is complete. Proceed to S09."],
                        "confidence": 0.95,
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(inferred_step_id="S08", missing_conditions=()),
    )

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=WrongOverlayModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s08-to-s09-overlay",
        lang="en",
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.actions
    assert response.actions[0]["target"] == "ufc_comm1_channel_selector_pull"
    assert response.metadata["fallback_overlay_used"] is True
    assert response.metadata["fallback_overlay_reason"] == "validator_action_hint"
    assert response.metadata["harness_validator_fallback_reason"] == "deterministic_step:S08"
    assert response.metadata["response_mapping"]["rejected_targets_by_request_allowlist"] == ["left_mdi_pb15"]
    assert response.metadata["response_mapping"]["mapping_error"] == "overlay_target_not_in_request_allowlist"


def test_live_loop_overrides_s18_root_menu_overlay_with_action_hint_when_vision_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s18_action_hint_override.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class WrongS18OverlayModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Hold FCS BIT and continue BIT.",
                actions=[],
                explanations=[
                    "The FCS BIT switch is currently UP and the next action is to continue the BIT sequence."
                ],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S18", "error_category": "CO"},
                        "next": {"step_id": "S18"},
                        "overlay": {
                            "targets": ["fcs_bit_switch"],
                            "evidence": [
                                {
                                    "target": "fcs_bit_switch",
                                    "type": "gate",
                                    "ref": "GATES.S18.completion",
                                    "quote": "Continue the BIT sequence from the current switch state.",
                                    "grounding_confidence": 0.83,
                                }
                            ],
                        },
                        "explanations": [
                            "The FCS BIT switch is currently UP and the next action is to continue the BIT sequence."
                        ],
                        "confidence": 0.83,
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(inferred_step_id="S18", missing_conditions=()),
    )

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=WrongS18OverlayModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s18-action-hint-override",
        lang="en",
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.actions
    assert response.actions[0]["target"] == "right_mdi_pb5"
    assert response.metadata["fallback_overlay_used"] is True
    assert response.metadata["fallback_overlay_reason"] == "validator_action_hint"
    assert response.metadata["presentation_fallback_reason"] == "deterministic_step:S18"
    assert response.metadata["final_action_plan"]["source"] == "validator_action_hint"
    assert response.metadata.get("action_hint_overlay_override_used") is not True
    assert response.metadata["final_public_response"]["actions"][0]["target"] == "right_mdi_pb5"


def test_live_fixture_s18_bit_root_repairs_pb18_to_pb5() -> None:
    frame_ids = ["1779045062695_000060"]
    frame_id = frame_ids[0]
    context = {
        "deterministic_step_hint": {
            "action_hint": {"target": "right_mdi_pb5"},
            "inferred_step_id": "S18",
            "missing_conditions": ["vision_facts.fcsmc_page_visible==seen"],
            "overlay_step_id": "S18",
            "step_ui_targets": ["right_mdi_pb18", "right_mdi_pb5"],
        },
        "overlay_target_allowlist": ["right_mdi_pb18", "right_mdi_pb5"],
        "vision": {
            "observation_ref": "obs-s18-bit-root",
            "observation_seq": 9824,
            "observation_t_wall_ms": 1779045062600,
            "observation_t_wall_s": 1779045062.6004815,
            "sync_delta_ms": 92,
            "sync_miss_reason": "missing_pre_trigger_frame",
            "sync_status": "matched_future_fallback",
            "sync_window_ms": 250,
            "trigger_wall_ms": 1779045062603,
        },
        "vision_fact_summary": {
            "frame_ids": frame_ids,
            "fresh_fact_ids": ["bit_root_page_visible"],
            "not_seen_fact_ids": ["fcsmc_page_visible"],
            "seen_fact_ids": ["bit_root_page_visible"],
            "status": "available",
            "uncertain_fact_ids": [],
        },
        "vision_facts": [
            {
                "fact_id": "bit_root_page_visible",
                "source_frame_id": frame_id,
                "state": "seen",
            },
            {
                "fact_id": "fcsmc_page_visible",
                "source_frame_id": frame_id,
                "state": "not_seen",
            },
        ],
    }
    raw_help = {
        "diagnosis": {"step_id": "S18", "error_category": "CO"},
        "explanations": [
            "当前处于 S18 阶段，右 DDI 显示 BIT root 页面。请左键点击右 MDI PB18 进入 FCS-MC 页面。"
        ],
        "next": {"step_id": "S18"},
        "overlay": {
            "evidence": [
                {
                    "grounding_confidence": 0.95,
                    "quote": "[REDACTED_SOURCE_QUOTE]",
                    "ref": f"VISION_FACTS.bit_root_page_visible@{frame_id}",
                    "target": "right_mdi_pb18",
                    "type": "visual",
                }
            ],
            "targets": ["right_mdi_pb18"],
        },
    }
    request = TutorRequest(
        request_id="644fac65-eb37-4de1-a591-fd1de392c647",
        message="help",
        observation_ref="obs-s18-bit-root",
        context=context,
        metadata={},
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message="当前处于 S18 阶段，右 DDI 显示 BIT root 页面。请左键点击右 MDI PB18 进入 FCS-MC 页面。",
        actions=[],
        explanations=list(raw_help["explanations"]),
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": raw_help,
        },
    )
    loop = LiveDcsTutorLoop(
        source=_DelayedObservationSource(Observation()),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s18-fixture-repair",
        rag_top_k=0,
        lang="zh",
    )
    try:
        result = loop._validate_and_repair_live_help_response(
            response,
            request,
            prompt_meta={},
            state_key="fixture-s18",
            help_cycle_id=request.request_id,
            vision_selection=HelpCycleVisionSelection(
                status="available",
                observation_ref=context["vision"].get("observation_ref"),
                observation_seq=context["vision"].get("observation_seq"),
                observation_t_wall_s=context["vision"].get("observation_t_wall_s"),
                observation_t_wall_ms=context["vision"].get("observation_t_wall_ms"),
                trigger_wall_ms=context["vision"].get("trigger_wall_ms"),
                sync_window_ms=context["vision"].get("sync_window_ms"),
                vision_used=True,
                frame_id=frame_id,
                sync_status=context["vision"].get("sync_status"),
                sync_delta_ms=context["vision"].get("sync_delta_ms"),
                frame_stale=False,
                frame_ids=list(frame_ids),
                selected_frames=[],
                pre_trigger_frame=None,
                trigger_frame=None,
                sync_miss_reason=context["vision"].get("sync_miss_reason"),
            ),
            vision_fact_context={
                "status": "available",
                "vision_fact_summary": context["vision_fact_summary"],
                "vision_facts": context["vision_facts"],
            },
            vision_fact_active_step_ids=["S18"],
            terminal_state_short_circuited=False,
        )
    finally:
        loop.close()

    repaired = result.response
    assert [action["target"] for action in repaired.actions] == ["right_mdi_pb5"]
    assert repaired.message is not None
    assert "PB5" in repaired.message
    assert "PB18" not in repaired.message
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["rejected_model_target"] == "right_mdi_pb18"
    assert repaired.metadata["visual_hint_target"] == "right_mdi_pb5"
    assert repaired.metadata["final_action_plan"]["source"] == "visual_action_hint_repair"
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "right_mdi_pb5"


def _validate_compact_live_help_response(
    *,
    request: TutorRequest,
    response: TutorResponse,
    vision_status: str = "vision_not_required",
) -> TutorResponse:
    loop = LiveDcsTutorLoop(
        source=_DelayedObservationSource(Observation()),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id=f"sess-{request.request_id}",
        rag_top_k=0,
        lang="zh",
    )
    try:
        result = loop._validate_and_repair_live_help_response(
            response,
            request,
            prompt_meta={},
            state_key=f"state-{request.request_id}",
            help_cycle_id=request.request_id,
            vision_selection=HelpCycleVisionSelection(
                status=vision_status,
                observation_ref=None,
                observation_seq=None,
                observation_t_wall_s=None,
                observation_t_wall_ms=None,
                trigger_wall_ms=None,
                sync_window_ms=None,
                vision_used=False,
                frame_id=None,
                sync_status=None,
                sync_delta_ms=None,
                frame_stale=False,
                frame_ids=[],
                selected_frames=[],
                pre_trigger_frame=None,
                trigger_frame=None,
                sync_miss_reason=None,
            ),
            vision_fact_context={
                "status": vision_status,
                "vision_fact_summary": {"status": vision_status},
                "vision_facts": [],
            },
            vision_fact_active_step_ids=[],
            terminal_state_short_circuited=False,
        )
        return result.response
    finally:
        loop.close()


def _load_live_help_fixture(path: str) -> dict[str, Any]:
    return json.loads((Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8"))


def _fixture_request_and_model_response(fixture: dict[str, Any]) -> tuple[TutorRequest, TutorResponse]:
    request_payload = fixture["cycle"]["tutor_request"]
    context = dict(request_payload.get("context", {}))
    observations = fixture.get("context", {}).get("observations", [])
    if observations and isinstance(observations[0], dict):
        payload = observations[0].get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("vars"), dict):
            context.setdefault("vars", payload["vars"])
    request = TutorRequest(
        request_id=request_payload["request_id"],
        message=request_payload.get("message"),
        observation_ref=request_payload.get("observation_ref"),
        context=context,
        metadata=dict(request_payload.get("metadata", {})),
    )
    help_response = fixture["model_io"]["help_response"]
    explanations = [item for item in help_response.get("explanations", []) if isinstance(item, str)]
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message=explanations[0] if explanations else None,
        actions=[],
        explanations=explanations,
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": help_response,
        },
    )
    return request, response


def _fixture_request_and_raw_model_response(fixture: dict[str, Any]) -> tuple[TutorRequest, TutorResponse]:
    request, _ = _fixture_request_and_model_response(fixture)
    help_response = fixture["model_io"].get("model_raw_help_response") or fixture["model_io"]["help_response"]
    explanations = [item for item in help_response.get("explanations", []) if isinstance(item, str)]
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message=explanations[0] if explanations else None,
        actions=[],
        explanations=explanations,
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": help_response,
        },
    )
    return request, response


def _public_response_text(response: TutorResponse) -> str:
    final_public = response.metadata.get("final_public_response")
    if not isinstance(final_public, dict):
        return ""
    parts: list[str] = []
    message = final_public.get("message")
    if isinstance(message, str):
        parts.append(message)
    explanations = final_public.get("explanations")
    if isinstance(explanations, list):
        parts.extend(item for item in explanations if isinstance(item, str))
    return "\n".join(parts)


def _assert_live_fixture_expectations(fixture: dict[str, Any], response: TutorResponse) -> None:
    expectations = fixture.get("expectations")
    assert isinstance(expectations, dict)
    expected_step = expectations.get("expected_final_step_id")
    if isinstance(expected_step, str) and expected_step:
        assert response.metadata["diagnosis"]["step_id"] == expected_step
        assert response.metadata["next"]["step_id"] == expected_step
        assert response.metadata["final_public_response"]["next"]["step_id"] == expected_step
    expected_targets = expectations.get("expected_overlay_target_ids")
    if isinstance(expected_targets, list):
        assert response.metadata["final_overlay_targets"] == [
            item for item in expected_targets if isinstance(item, str)
        ]
    expected_source = expectations.get("final_response_source")
    if isinstance(expected_source, str) and expected_source:
        assert response.metadata["final_action_plan"]["source"] == expected_source
    if expectations.get("llm_decision_status") == "repaired":
        assert response.metadata["repair_applied"] is True
    expected_vlm_status = expectations.get("vlm_call_status")
    if isinstance(expected_vlm_status, str) and expected_vlm_status:
        assert response.metadata["vlm_call_status"] == expected_vlm_status


def test_live_help_fixture_311_replays_real_s09_comm1_complete_fixture() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/168fc73d-f98c-498e-a35c-fef91b06ee2e.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["diagnosis"]["step_id"] == "S10"
    assert repaired.metadata["next"]["step_id"] == "S10"
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["rejected_missing_conditions"] == ["vars.comm1_freq_134_000==true"]
    assert repaired.metadata["final_action_plan"]["source"] == "final_evidence_consistency_validator"
    assert repaired.metadata["final_public_response"]["next"]["step_id"] == "S10"
    assert "ufc_comm1_channel_selector_pull" not in repaired.metadata["final_overlay_targets"]
    assert "vars.comm1_freq_134_000==true" not in repaired.message
    assert "vars.comm1_freq_134_000==true" not in _public_response_text(repaired)


def test_live_help_fixture_311_replays_real_s10_left_engine_complete_fixture() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/c3c775ce-7a4d-4e7f-a841-3a022528138e.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["diagnosis"]["step_id"] != "S10"
    assert repaired.metadata["next"]["step_id"] != "S10"
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["rejected_missing_conditions"] == ["vars.engine_crank_left_complete==true"]
    assert repaired.metadata["final_action_plan"]["source"] in {
        "final_evidence_consistency_validator",
        "s10_left_engine_completion_guardrail",
    }
    assert "eng_crank_switch" not in repaired.metadata["final_overlay_targets"]
    assert "vars.engine_crank_left_complete==true" not in repaired.message
    assert "vars.engine_crank_left_complete==true" not in _public_response_text(repaired)


def test_live_help_fixture_311_keeps_s12_when_only_precondition_is_satisfied() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/c2a80099-8375-42a1-89eb-b09544b474e2.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)
    vars_map = request.context["vars"]
    assert vars_map["rpm_l_gte_60"] is True
    assert vars_map["ins_mode"] == 0
    assert vars_map["ins_mode_set"] is False
    assert vars_map["ins_mode_cv_or_gnd"] is False
    assert vars_map["ins_fast_align_complete"] is False

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["diagnosis"]["step_id"] == "S12"
    assert repaired.metadata["next"]["step_id"] == "S12"
    assert repaired.metadata["final_public_response"]["next"]["step_id"] == "S12"
    assert repaired.metadata["final_overlay_targets"] == ["ins_mode_knob"]
    assert [action["target"] for action in repaired.actions] == ["ins_mode_knob"]
    assert repaired.metadata.get("final_evidence_consistency_repair_applied") is not True
    assert repaired.metadata.get("rejected_missing_conditions", []) == []
    assert repaired.metadata["final_evidence_consistency_precondition_satisfied"] is True
    assert repaired.metadata["precondition_satisfied_conditions"] == ["vars.rpm_l_gte_60==true"]
    assert "radar_mode_knob" not in repaired.metadata["final_overlay_targets"]
    assert "completion_gate_already_satisfied:S12" not in repaired.metadata.get(
        "harness_validation_reasons",
        [],
    )


def test_final_evidence_split_uses_current_step_completion_predicates_only() -> None:
    loop = LiveDcsTutorLoop(
        source=_DelayedObservationSource(Observation()),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-issue-311-completion-split",
        rag_top_k=0,
        lang="zh",
    )
    try:
        completion, precondition = loop._split_completion_missing_conditions(
            "S12",
            [
                "vars.rpm_l_gte_60==true",
                "vars.ins_mode in [2,2]",
                "vars.ins_fast_align_complete==true",
                "vars.some_prior_step_gate==true",
            ],
        )
    finally:
        loop.close()

    assert completion == [
        "vars.ins_mode in [2,2]",
        "vars.ins_fast_align_complete==true",
    ]
    assert precondition == ["vars.rpm_l_gte_60==true"]


def test_live_help_fixture_315_repairs_unconfirmed_s08_tac_public_response() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/c2ddd0ad-eb91-416e-a3a8-be16816a9d49.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response, vision_status="available")

    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["message_category"] == "harness_validator_repair"
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "left_mdi_brightness_selector"
    public_text = _public_response_text(repaired)
    assert "TAC page" not in public_text
    assert "FCS page" not in public_text
    assert "not powered" in public_text or "未开启" in public_text
    final_public_json = json.dumps(repaired.metadata["final_public_response"], ensure_ascii=False)
    assert "VISION_FACTS.tac_page_visible" not in final_public_json


def test_live_help_fixture_314_keeps_s08_tac_bit_root_visual_recovery() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/3465ec45-e87c-48bd-8f1c-5f59d1abeafd.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response, vision_status="available")

    assert repaired.metadata["diagnosis"]["step_id"] == "S08"
    assert repaired.metadata["next"]["step_id"] == "S08"
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "left_mdi_pb18"
    assert repaired.metadata["final_overlay_targets"] == ["left_mdi_pb18"]
    assert "ufc_comm1_channel_selector_pull" not in repaired.metadata["final_overlay_targets"]
    public_text = _public_response_text(repaired)
    assert "S09" not in public_text
    assert "COMM1" not in public_text


def test_live_help_fixture_314_keeps_s08_supt_bit_root_visual_recovery() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/e6e6838b-b8ef-4e51-a4a8-1ea3d4771dad.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response, vision_status="available")

    assert repaired.metadata["diagnosis"]["step_id"] == "S08"
    assert repaired.metadata["next"]["step_id"] == "S08"
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "left_mdi_pb15"
    assert repaired.metadata["final_overlay_targets"] == ["left_mdi_pb15"]
    public_text = _public_response_text(repaired)
    assert "S09" not in public_text
    assert "COMM1" not in public_text
    assert "PB15" in public_text or "left_mdi_pb15" in public_text


def test_live_help_fixture_314_s09_requires_comm1_pull_before_numeric_sequence() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/ce8512ef-901a-4f08-a339-af10893b744f.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["final_overlay_targets"] == ["ufc_comm1_channel_selector_pull"]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "ufc_comm1_channel_selector_pull"
    assert "134.000" in _public_response_text(repaired)
    assert "1-3-4-0-0-0" not in _public_response_text(repaired)
    assert not {"ufc_key_1", "ufc_key_3", "ufc_key_4", "ufc_key_0"} & set(
        repaired.metadata["final_overlay_targets"]
    )


def test_live_help_fixture_314_s18_pb5_repair_rewrites_public_message() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/9608382d-ee62-486f-9d90-0c9d74997241.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)
    response.message = "当前 S18 尚未完成。请先操作 right_mdi_pb18，并确认该步骤条件已满足。"
    response.explanations = [response.message]

    repaired = _validate_compact_live_help_response(request=request, response=response, vision_status="available")

    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "right_mdi_pb5"
    public_text = _public_response_text(repaired)
    assert "right_mdi_pb5" in public_text or "PB5" in public_text
    assert "right_mdi_pb18" not in public_text
    assert "PB18" not in public_text


def test_live_help_fixture_314_s10_left_engine_uses_left_click_guidance() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/3de0e688-8bc4-4df1-b062-72f3134bc775.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["final_overlay_targets"] == ["eng_crank_switch"]
    public_text = _public_response_text(repaired)
    assert "LEFT" in public_text or "L 位置" in public_text
    assert "左键" in public_text or "left-click" in public_text
    assert "R 位置" not in public_text
    assert "右键" not in public_text
    assert "right-click" not in public_text


def test_live_help_fixture_314_s31_radar_altimeter_uses_mouse_wheel_guidance() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/5a7d5df7-2f08-4321-9a4a-6858131b4e6e.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["final_overlay_targets"] == ["radar_altimeter_bug_knob"]
    public_text = _public_response_text(repaired)
    assert "滚轮" in public_text or "mouse wheel" in public_text or "mouse-wheel" in public_text
    assert "200" in public_text and "40" in public_text


def test_live_help_fixture_314_s32_standby_attitude_uses_mouse_wheel_guidance() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/ffb5e69b-664b-47d8-9150-57d7cd75d233.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["final_overlay_targets"] == ["standby_attitude_cage_knob"]
    public_text = _public_response_text(repaired)
    assert "滚轮" in public_text or "mouse wheel" in public_text or "mouse-wheel" in public_text
    assert "备用姿态" in public_text or "standby attitude" in public_text.lower()


def test_live_help_fixture_314_s33_default_satisfied_uses_public_completion_message() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/7bea22f8-a4d1-4744-917b-f7ddc957f709.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["diagnosis"]["step_id"] == "S33"
    assert repaired.metadata["next"]["step_id"] == "S33"
    assert repaired.metadata["final_overlay_targets"] == []
    public_text = _public_response_text(repaired)
    assert "最新证据" not in public_text
    assert "evidence" not in public_text.lower()
    assert "AUTO" in public_text
    assert "完成" in public_text


def test_live_help_fixture_315_s28_raw_model_repair_advances_to_s29_guidance() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/8d30d919-108d-4068-8b58-a9467aecdb21.fixture.json"
    )
    request, response = _fixture_request_and_raw_model_response(fixture)
    request.context = dict(request.context)
    request.context["deterministic_step_hint"] = dict(request.context["deterministic_step_hint"])
    request.context["deterministic_step_hint"]["gate_blockers"] = [
        {
            "ref": "GATES.S28.precondition",
            "reason": "Old raw-step blocker must not be reused after repair.",
        }
    ]
    request.context["gates"] = {
        "S29.completion": {"status": "blocked", "reason": "BINGO fuel must be set on the IFEI."},
        "S29.precondition": {"status": "allowed"},
    }

    repaired = _validate_compact_live_help_response(request=request, response=response)

    _assert_live_fixture_expectations(fixture, repaired)
    assert repaired.metadata["diagnosis"]["step_id"] == "S29"
    assert repaired.metadata["next"]["step_id"] == "S29"
    assert repaired.metadata["final_action_plan"]["step_id"] == "S29"
    assert repaired.metadata["final_action_plan"]["overlay_step_id"] == "S29"
    assert repaired.metadata["final_overlay_targets"] == ["ifei_up_button"]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "ifei_up_button"
    assert repaired.actions[0]["evidence_refs"] == ["VARS.bingo_fuel_set"]
    assert repaired.metadata["help_response"]["overlay"]["evidence"][0]["ref"] == "VARS.bingo_fuel_set"
    assert repaired.metadata["harness_trace"]["model_decision"]["step_id"] == "S28"
    assert repaired.metadata["harness_trace"]["model_decision"]["overlay_targets"] == ["parking_brake_handle"]
    assert repaired.metadata["harness_trace"]["validator_result"]["rejected"] is True
    assert repaired.metadata["harness_trace"]["repair_result"]["path"] == "final_evidence_consistency_validator"
    public_text = _public_response_text(repaired)
    assert "BINGO" in public_text
    assert "S20" not in public_text
    assert "S22" not in public_text
    assert "parking_brake_handle" not in public_text
    assert "最新证据" not in public_text


def test_safe_fallback_overlay_syncs_message_after_completion_conflict_repair() -> None:
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "overlay_target_allowlist": ["right_mdi_pb18", "right_mdi_pb5"],
            "gates": [
                {"gate_id": "S18.completion", "status": "blocked"},
                {"gate_id": "S18.precondition", "status": "allowed"},
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S18",
                "overlay_step_id": "S18",
                "action_hint": {"target": "right_mdi_pb5"},
                "step_ui_targets": ["right_mdi_pb18", "right_mdi_pb5"],
            },
            "rag_topk": [],
        },
    )
    response = TutorResponse(
        status="ok",
        message="当前 S18 尚未完成。请先操作 right_mdi_pb18，并确认该步骤条件已满足。",
        explanations=["当前 S18 尚未完成。请先操作 right_mdi_pb18，并确认该步骤条件已满足。"],
        actions=[],
        metadata={
            "completion_conflict_rewritten": True,
            "rejected_model_targets": ["right_mdi_pb18"],
            "rejected_model_target": "right_mdi_pb18",
        },
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(Path("/dev/null")),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        used, reason = loop._apply_safe_fallback_overlay(response, request)

        assert used is True
        assert reason == "validator_action_hint"
        assert [action["target"] for action in response.actions] == ["right_mdi_pb5"]
        assert "right_mdi_pb5" in response.message or "PB5" in response.message
        assert "right_mdi_pb18" not in response.message
        assert "PB18" not in response.message
    finally:
        loop.close()


def test_live_help_fixture_306_1ab3_retracted_after_latch_advances_to_s22() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/1ab3eb59-6705-4a62-87d7-9fc15a2dcec0.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    _assert_live_fixture_expectations(fixture, repaired)
    assert repaired.metadata["diagnosis"]["step_id"] == "S22"
    assert repaired.metadata["next"]["step_id"] == "S22"
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["final_action_plan"]["source"] == "final_evidence_consistency_validator"
    assert repaired.metadata["final_public_response"]["next"]["step_id"] == "S22"
    assert repaired.metadata["final_overlay_targets"] == ["launch_bar_switch"]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "launch_bar_switch"
    public_text = _public_response_text(repaired)
    assert "S20 未完成" not in public_text
    assert "refuel_probe_switch" not in public_text


def test_live_help_fixture_306_2e04_extended_probe_advances_to_s21() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/2e04fdab-43ce-45d6-82e5-a9b0863d6c84.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    _assert_live_fixture_expectations(fixture, repaired)
    assert repaired.metadata["diagnosis"]["step_id"] == "S21"
    assert repaired.metadata["next"]["step_id"] == "S21"
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["final_action_plan"]["source"] == "final_evidence_consistency_validator"
    assert repaired.metadata["final_overlay_targets"] == ["refuel_probe_switch"]
    assert repaired.metadata["final_public_response"]["next"]["step_id"] == "S21"
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "refuel_probe_switch"
    public_text = _public_response_text(repaired)
    assert "S20 未完成" not in public_text
    assert "收起" in public_text


def test_live_help_fixture_306_20c_retracted_probe_latch_advances_to_s22() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/20c79783-092e-4128-ad47-5fa7d3774ed7.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    _assert_live_fixture_expectations(fixture, repaired)
    assert repaired.metadata["diagnosis"]["step_id"] == "S22"
    assert repaired.metadata["next"]["step_id"] == "S22"
    assert repaired.metadata["final_action_plan"]["source"] == "final_evidence_consistency_validator"
    assert repaired.metadata["final_overlay_targets"] == ["launch_bar_switch"]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "launch_bar_switch"


def test_live_help_fixture_306_6232_launch_bar_cycle_stops_at_s24_without_hook_latch() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/6232bd96-95f8-4905-9ae9-8f27e4515c3d.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    _assert_live_fixture_expectations(fixture, repaired)
    assert repaired.metadata["diagnosis"]["step_id"] == "S24"
    assert repaired.metadata["next"]["step_id"] == "S24"
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["final_action_plan"]["source"] == "final_evidence_consistency_validator"
    assert repaired.metadata["final_overlay_targets"] == ["arresting_hook_handle"]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "arresting_hook_handle"
    public_text = _public_response_text(repaired)
    assert "S23" not in public_text
    assert "S25" not in public_text


def test_live_help_fixture_306_da08_hook_up_advances_to_s26_after_polarity_fix() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/da08da4a-0909-4acb-a2e6-7027324fa989.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    _assert_live_fixture_expectations(fixture, repaired)
    assert repaired.metadata["diagnosis"]["step_id"] == "S26"
    assert repaired.metadata["next"]["step_id"] == "S26"
    assert repaired.metadata["final_overlay_targets"] == ["pitot_heater_switch"]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "pitot_heater_switch"
    public_text = _public_response_text(repaired)
    assert "放下阻钩" not in public_text
    assert "伸出阻钩" not in public_text


def test_live_help_fixture_306_7ea8_hook_down_advances_to_s25_without_trace_mismatch() -> None:
    fixture = _load_live_help_fixture(
        "artifacts/live_fixtures/7ea8bf09-dfe7-4e7b-a23f-6407af510e2a.fixture.json"
    )
    request, response = _fixture_request_and_model_response(fixture)

    repaired = _validate_compact_live_help_response(request=request, response=response)

    _assert_live_fixture_expectations(fixture, repaired)
    assert repaired.metadata["diagnosis"]["step_id"] == "S25"
    assert repaired.metadata["next"]["step_id"] == "S25"
    final_plan = repaired.metadata["final_action_plan"]
    assert final_plan["step_id"] == "S25"
    assert final_plan["overlay_step_id"] == "S25"
    assert repaired.metadata["final_overlay_targets"] == ["arresting_hook_handle"]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "arresting_hook_handle"
    public_text = _public_response_text(repaired)
    assert "抬起" in public_text or "收起" in public_text
    assert "放下阻钩" not in public_text


def test_live_help_fixture_294_s09_comm1_complete_advances_to_s10() -> None:
    request = TutorRequest(
        request_id="issue-294-s09-complete",
        message="help",
        context={
            "vars": {
                "comm1_freq_134_000": True,
                "comm1_freq_value": 13400,
                "ufc_scratchpad_number_display": " 134.000",
                "ufc_scratchpad_string_1_display": " 1",
                "ufc_scratchpad_string_2_display": "--",
            },
            "gates": {
                "S09.completion": {"status": "allowed"},
                "S10.completion": {
                    "status": "blocked",
                    "reason": "Left engine start must have begun.",
                    "reason_code": "s10_requires_engine_crank_left_complete",
                },
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "overlay_step_id": "S09",
                "missing_conditions": ["vars.comm1_freq_134_000==true"],
                "step_ui_targets": [
                    "ufc_comm1_channel_selector_pull",
                    "ufc_key_1",
                    "ufc_key_3",
                    "ufc_key_4",
                    "ufc_key_0",
                    "ufc_ent_button",
                ],
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["eng_crank_switch"],
            "rag_topk": [],
            "vision_fact_summary": {"status": "vision_not_required", "seen_fact_ids": [], "fresh_fact_ids": []},
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message="请继续设置 COMM1。",
        actions=[],
        explanations=["请继续设置 COMM1。"],
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S09", "error_category": "CO"},
                "next": {"step_id": "S09"},
                "overlay": {
                    "targets": ["ufc_comm1_channel_selector_pull"],
                    "evidence": [
                        {
                            "target": "ufc_comm1_channel_selector_pull",
                            "type": "var",
                            "ref": "VARS.comm1_freq_134_000",
                            "quote": "COMM1 is already tuned.",
                            "grounding_confidence": 0.9,
                        }
                    ],
                },
                "explanations": ["请继续设置 COMM1。"],
            },
        },
    )

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert [action["target"] for action in repaired.actions] == ["eng_crank_switch"]
    assert repaired.metadata["diagnosis"]["step_id"] == "S10"
    assert repaired.metadata["next"]["step_id"] == "S10"
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["rejected_missing_conditions"] == ["vars.comm1_freq_134_000==true"]
    assert repaired.metadata["s09_comm1_completion_guardrail_applied"] is True
    assert repaired.metadata["final_overlay_targets"] == ["eng_crank_switch"]
    assert repaired.metadata["final_action_plan"]["source"] == "final_evidence_consistency_validator"
    assert "vars.comm1_freq_134_000==true" not in repaired.message
    assert "134.000" in repaired.message
    assert "S10" in repaired.message


def test_live_help_fixture_310_s10_left_engine_complete_advances_past_crank() -> None:
    request = TutorRequest(
        request_id="issue-310-s10-left-complete",
        message="help",
        context={
            "vars": {
                "comm1_freq_134_000": True,
                "right_engine_nominal_start_params": True,
                "engine_crank_left": False,
                "engine_crank_left_complete": True,
                "rpm_l": 64,
                "rpm_l_gte_25": True,
                "rpm_l_gte_60": True,
                "left_engine_nominal_start_params": True,
                "left_engine_idle_ready": True,
                "throttle_l_not_off": True,
            },
            "gates": {
                "S10.completion": {
                    "status": "blocked",
                    "reason": "Left engine start must have begun.",
                    "reason_code": "s10_requires_engine_crank_left_complete",
                },
                "S12.completion": {
                    "status": "blocked",
                    "reason": "INS mode must be set to GND/CV.",
                    "reason_code": "s12_requires_ins_mode_set",
                },
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S10",
                "overlay_step_id": "S10",
                "missing_conditions": ["vars.engine_crank_left_complete==true"],
                "step_ui_targets": ["eng_crank_switch"],
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["eng_crank_switch"],
            "rag_topk": [],
            "vision_fact_summary": {"status": "vision_not_required", "seen_fact_ids": [], "fresh_fact_ids": []},
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message="当前处于 S10 步骤。请将 ENG CRANK 开关拨到 LEFT 位置以启动左发动机。",
        actions=[
            {
                "type": "highlight",
                "target": "eng_crank_switch",
                "intent": "guide",
                "evidence_refs": ["GATES.S10.completion"],
            }
        ],
        explanations=["当前处于 S10 步骤。请将 ENG CRANK 开关拨到 LEFT 位置以启动左发动机。"],
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S10", "error_category": "OM"},
                "next": {"step_id": "S10"},
                "overlay": {
                    "targets": ["eng_crank_switch"],
                    "evidence": [
                        {
                            "target": "eng_crank_switch",
                            "type": "gate",
                            "ref": "GATES.S10.completion",
                            "quote": "Left engine start must have begun.",
                            "grounding_confidence": 0.9,
                        }
                    ],
                },
                "explanations": ["当前处于 S10 步骤。请将 ENG CRANK 开关拨到 LEFT 位置以启动左发动机。"],
            },
        },
    )

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["diagnosis"]["step_id"] == "S12"
    assert repaired.metadata["next"]["step_id"] == "S12"
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["rejected_missing_conditions"] == ["vars.engine_crank_left_complete==true"]
    assert repaired.metadata["s10_left_engine_completion_guardrail_applied"] is True
    assert repaired.metadata["rejected_model_step_id"] == "S10"
    assert repaired.metadata["fallback_overlay_used"] is True
    assert repaired.metadata["final_action_plan"]["source"] == "s10_left_engine_completion_guardrail"
    assert repaired.metadata["final_overlay_targets"] == ["ins_mode_knob"]
    assert repaired.metadata["help_response"]["overlay"]["targets"] == ["ins_mode_knob"]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "ins_mode_knob"
    assert [action["target"] for action in repaired.actions] == ["ins_mode_knob"]
    assert "vars.engine_crank_left_complete==true" not in repaired.message
    assert "S10" not in repaired.message
    assert "S12" in repaired.message


def test_final_evidence_validator_rechecks_repaired_fallback_step() -> None:
    request = TutorRequest(
        request_id="issue-311-final-fallback-recheck",
        message="help",
        context={
            "vars": {
                "comm1_freq_134_000": True,
                "right_engine_nominal_start_params": True,
                "engine_crank_left": False,
                "engine_crank_left_complete": True,
                "rpm_l": 64,
                "rpm_l_gte_60": True,
                "left_engine_nominal_start_params": True,
                "throttle_l_not_off": True,
                "ins_mode": 2,
                "ins_mode_set": True,
                "ins_mode_cv_or_gnd": True,
                "ins_fast_align_complete": True,
                "radar_mode_opr": False,
            },
            "gates": {
                "S10.completion": {
                    "status": "blocked",
                    "reason": "Left engine start must have begun.",
                    "reason_code": "s10_requires_engine_crank_left_complete",
                },
                "S13.completion": {
                    "status": "blocked",
                    "step_id": "S13",
                    "gate_type": "completion",
                    "reason": "Radar knob must be set to OPR.",
                    "reason_code": "s13_requires_radar_mode_opr",
                },
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S10",
                "overlay_step_id": "S10",
                "missing_conditions": ["vars.engine_crank_left_complete==true"],
                "step_ui_targets": ["eng_crank_switch"],
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["eng_crank_switch", "ins_mode_knob", "radar_mode_knob"],
            "rag_topk": [],
            "vision_fact_summary": {"status": "vision_not_required", "seen_fact_ids": [], "fresh_fact_ids": []},
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message="当前处于 S12 步骤。请将 INS 旋钮转到 CV 或 GND。",
        actions=[
            {
                "type": "highlight",
                "target": "ins_mode_knob",
                "intent": "guide",
                "evidence_refs": ["GATES.S12.completion"],
            }
        ],
        explanations=["当前处于 S12 步骤。请将 INS 旋钮转到 CV 或 GND。"],
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "repair",
            "diagnosis": {"step_id": "S12", "error_category": "OM"},
            "next": {"step_id": "S12"},
            "harness_action_plan": {
                "step_id": "S12",
                "overlay_step_id": "S12",
                "targets": ["ins_mode_knob"],
                "text_only": False,
                "source": "deterministic_step:S12",
            },
            "help_response": {
                "diagnosis": {"step_id": "S12", "error_category": "OM"},
                "next": {"step_id": "S12"},
                "overlay": {
                    "targets": ["ins_mode_knob"],
                    "evidence": [
                        {
                            "target": "ins_mode_knob",
                            "type": "gate",
                            "ref": "GATES.S12.completion",
                            "quote": "INS alignment must be complete.",
                            "grounding_confidence": 0.9,
                        }
                    ],
                },
                "explanations": ["当前处于 S12 步骤。请将 INS 旋钮转到 CV 或 GND。"],
            },
        },
    )

    loop = LiveDcsTutorLoop(
        source=_DelayedObservationSource(Observation()),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-final-fallback-recheck",
        rag_top_k=0,
        lang="zh",
    )
    try:
        used, reason = loop._enforce_final_evidence_consistency_after_repairs(response, request)
    finally:
        loop.close()

    assert used is True
    assert reason == "final_evidence_consistency_validator"
    assert response.metadata["diagnosis"]["step_id"] == "S13"
    assert response.metadata["next"]["step_id"] == "S13"
    assert response.metadata["harness_action_plan"]["source"] == "final_evidence_consistency_validator"
    assert response.metadata["harness_action_plan"]["targets"] == ["radar_mode_knob"]
    assert "completion_gate_already_satisfied:S12" in response.metadata["final_evidence_consistency_reasons"]
    assert [action["target"] for action in response.actions] == ["radar_mode_knob"]
    assert "ins_mode_knob" not in [action["target"] for action in response.actions]


@pytest.mark.parametrize(
    ("next_step_id", "target", "expected_text"),
    [
        ("S28", "parking_brake_handle", "parking brake"),
        ("S29", "ifei_up_button", "bingo"),
        ("S30", "standby_altimeter_pressure_knob", "standby pressure altimeter"),
        ("S31", "radar_altimeter_bug_knob", "radar altimeter"),
        ("S32", "standby_attitude_cage_knob", "standby attitude"),
    ],
)
def test_final_evidence_consistency_repair_uses_repaired_action_guidance(
    monkeypatch: pytest.MonkeyPatch,
    next_step_id: str,
    target: str,
    expected_text: str,
) -> None:
    request = TutorRequest(
        request_id=f"issue-315-final-public-guidance-{next_step_id.lower()}",
        message="help",
        context={
            "vars": {},
            "gates": {
                f"{next_step_id}.completion": {"status": "blocked", "reason": f"{target} still needs action."},
                f"{next_step_id}.precondition": {"status": "allowed"},
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S26",
                "overlay_step_id": "S26",
                "missing_conditions": ["vars.pitot_heat_on==true"],
                "step_ui_targets": ["pitot_heat_switch"],
            },
            "overlay_target_allowlist": [target],
            "rag_topk": [],
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message=f"S26 latest evidence satisfied. Now entering {next_step_id}.",
        actions=[],
        explanations=[f"S26 latest evidence satisfied. Now entering {next_step_id}."],
        metadata={
            "provider": "fallback",
            "generation_mode": "fallback",
            "diagnosis": {"step_id": "S26", "error_category": "OM"},
            "next": {"step_id": "S26"},
            "help_response": {
                "diagnosis": {"step_id": "S26", "error_category": "OM"},
                "next": {"step_id": "S26"},
                "overlay": {"targets": [], "evidence": []},
                "explanations": [f"S26 latest evidence satisfied. Now entering {next_step_id}."],
            },
            "final_public_response": {
                "message": f"S26 latest evidence satisfied. Now entering {next_step_id}.",
                "explanations": [f"S26 latest evidence satisfied. Now entering {next_step_id}."],
                "diagnosis": {"step_id": "S26", "error_category": "OM"},
                "next": {"step_id": "S26"},
                "actions": [],
            },
        },
    )
    loop = LiveDcsTutorLoop(
        source=_DelayedObservationSource(Observation()),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-issue-315-final-public-guidance",
        rag_top_k=0,
        lang="en",
    )
    monkeypatch.setattr(
        loop,
        "_infer_after_completed_step",
        lambda *args, **kwargs: StepInferenceResult(next_step_id, (f"vars.{target}_complete==true",)),
    )
    try:
        used, reason = loop._rewrite_final_evidence_consistency_conflict_response(
            response,
            request,
            rejected_step_id="S26",
            rejected_missing_conditions=["vars.pitot_heat_on==true"],
        )
    finally:
        loop.close()

    assert used is True
    assert reason == "final_evidence_consistency_validator"
    assert response.metadata["next"]["step_id"] == next_step_id
    assert response.metadata["harness_action_plan"]["targets"] == [target]
    assert response.metadata["final_action_plan"]["step_id"] == next_step_id
    assert response.metadata["final_action_plan"]["targets"] == [target]
    assert response.metadata["final_action_plan_source"] == "final_evidence_consistency_validator"
    assert [action["target"] for action in response.actions] == [target]
    assert response.actions[0]["evidence_refs"] == [f"GATES.{next_step_id}.completion"]
    assert response.metadata["help_response"]["overlay"]["evidence"][0]["ref"] == response.actions[0]["evidence_refs"][0]
    assert "latest evidence satisfied" not in response.message
    assert expected_text in response.message.lower()
    assert response.metadata["help_response"]["explanations"] == [response.message]
    final_public = response.metadata["final_public_response"]
    assert final_public["message"] == response.message
    assert final_public["next"]["step_id"] == next_step_id
    assert final_public["actions"][0]["target"] == target
    assert "latest evidence satisfied" not in json.dumps(final_public)


def test_live_help_fixture_310_does_not_fall_back_to_s10_when_next_overlay_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_calls: list[tuple[str | None, str | None]] = []
    original_builder = LiveDcsTutorLoop._build_safe_fallback_overlay_help_obj

    def failing_s10_completion_fallback(
        self: LiveDcsTutorLoop,
        request: TutorRequest,
        *,
        override_inferred_step_id: str | None = None,
        override_overlay_step_id: str | None = None,
        ignore_request_allowlist: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        fallback_calls.append((override_inferred_step_id, override_overlay_step_id))
        if override_inferred_step_id == "S12":
            return None, "no_verifiable_evidence_ref"
        return original_builder(
            self,
            request,
            override_inferred_step_id=override_inferred_step_id,
            override_overlay_step_id=override_overlay_step_id,
            ignore_request_allowlist=ignore_request_allowlist,
        )

    monkeypatch.setattr(
        LiveDcsTutorLoop,
        "_build_safe_fallback_overlay_help_obj",
        failing_s10_completion_fallback,
    )

    request = TutorRequest(
        request_id="issue-310-s10-left-complete-fallback-fails",
        message="help",
        context={
            "vars": {
                "comm1_freq_134_000": True,
                "right_engine_nominal_start_params": True,
                "engine_crank_left": False,
                "engine_crank_left_complete": True,
                "rpm_l": 64,
                "rpm_l_gte_60": True,
                "left_engine_nominal_start_params": True,
                "throttle_l_not_off": True,
            },
            "gates": {
                "S10.completion": {
                    "status": "blocked",
                    "reason": "Left engine start must have begun.",
                    "reason_code": "s10_requires_engine_crank_left_complete",
                },
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S10",
                "overlay_step_id": "S10",
                "missing_conditions": ["vars.engine_crank_left_complete==true"],
                "step_ui_targets": ["eng_crank_switch"],
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["eng_crank_switch"],
            "rag_topk": [],
            "vision_fact_summary": {"status": "vision_not_required", "seen_fact_ids": [], "fresh_fact_ids": []},
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message="当前处于 S10 步骤。请将 ENG CRANK 开关拨到 LEFT 位置以启动左发动机。",
        actions=[
            {
                "type": "highlight",
                "target": "eng_crank_switch",
                "intent": "guide",
                "evidence_refs": ["GATES.S10.completion"],
            }
        ],
        explanations=["当前处于 S10 步骤。请将 ENG CRANK 开关拨到 LEFT 位置以启动左发动机。"],
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S10", "error_category": "OM"},
                "next": {"step_id": "S10"},
                "overlay": {
                    "targets": ["eng_crank_switch"],
                    "evidence": [
                        {
                            "target": "eng_crank_switch",
                            "type": "gate",
                            "ref": "GATES.S10.completion",
                            "quote": "Left engine start must have begun.",
                            "grounding_confidence": 0.9,
                        }
                    ],
                },
                "explanations": ["当前处于 S10 步骤。请将 ENG CRANK 开关拨到 LEFT 位置以启动左发动机。"],
            },
        },
    )

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["s10_left_engine_completion_guardrail_applied"] is True
    assert repaired.metadata["diagnosis"]["step_id"] == "S12"
    assert repaired.metadata["next"]["step_id"] == "S12"
    assert repaired.metadata["rejected_missing_conditions"] == ["vars.engine_crank_left_complete==true"]
    assert repaired.metadata["final_action_plan"]["source"] == "s10_left_engine_completion_guardrail"
    assert repaired.metadata["fallback_overlay_used"] is False
    assert repaired.metadata["s10_left_engine_completion_overlay_reason"] == "no_verifiable_evidence_ref"
    assert repaired.metadata["final_overlay_targets"] == []
    assert fallback_calls == [("S12", "S12")]
    assert "eng_crank_switch" not in [action["target"] for action in repaired.actions]
    assert "S10" not in repaired.message
    assert "S12" in repaired.message


def test_safe_fallback_overlay_rejects_missing_condition_satisfied_by_latest_telemetry() -> None:
    request = TutorRequest(
        request_id="issue-313-satisfied-missing-no-direct-fallback",
        message="help",
        context={
            "vars": {"comm1_freq_134_000": True},
            "gates": {"S09.completion": {"status": "allowed", "allowed": True}},
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "overlay_step_id": "S09",
                "missing_conditions": ["vars.comm1_freq_134_000==true"],
                "step_ui_targets": ["ufc_comm1_channel_selector_pull"],
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["ufc_comm1_channel_selector_pull"],
            "rag_topk": [],
        },
    )
    loop = LiveDcsTutorLoop(
        source=_DelayedObservationSource(Observation()),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-issue-313-satisfied-missing",
        rag_top_k=0,
        lang="zh",
    )
    try:
        fallback_help_obj, reason = loop._build_safe_fallback_overlay_help_obj(request)
    finally:
        loop.close()

    assert fallback_help_obj is None
    assert reason.startswith("evidence_conflict:")


def test_safe_fallback_overlay_rejects_already_satisfied_completion_gate() -> None:
    request = TutorRequest(
        request_id="issue-313-satisfied-gate-no-direct-fallback",
        message="help",
        context={
            "vars": {"comm1_freq_134_000": True},
            "gates": {"S09.completion": {"status": "allowed", "allowed": True}},
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "overlay_step_id": "S09",
                "missing_conditions": [],
                "step_ui_targets": ["ufc_comm1_channel_selector_pull"],
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["ufc_comm1_channel_selector_pull"],
            "rag_topk": [],
        },
    )
    loop = LiveDcsTutorLoop(
        source=_DelayedObservationSource(Observation()),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-issue-313-satisfied-gate",
        rag_top_k=0,
        lang="zh",
    )
    try:
        fallback_help_obj, reason = loop._build_safe_fallback_overlay_help_obj(request)
    finally:
        loop.close()

    assert fallback_help_obj is None
    assert reason.startswith("evidence_conflict:")
    assert "completion_gate_already_satisfied:S09" in reason


def test_safe_fallback_overlay_rechecks_override_step_against_latest_evidence() -> None:
    request = TutorRequest(
        request_id="issue-313-override-step-recheck",
        message="help",
        context={
            "vars": {
                "comm1_freq_134_000": True,
                "engine_crank_left_complete": True,
                "rpm_l": 64,
                "rpm_l_gte_60": True,
                "left_engine_nominal_start_params": True,
                "throttle_l_not_off": True,
            },
            "gates": {"S10.completion": {"status": "allowed", "allowed": True}},
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "overlay_step_id": "S09",
                "missing_conditions": ["vars.comm1_freq_134_000==true"],
                "step_ui_targets": ["ufc_comm1_channel_selector_pull"],
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["eng_crank_switch"],
            "rag_topk": [],
        },
    )
    loop = LiveDcsTutorLoop(
        source=_DelayedObservationSource(Observation()),
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-issue-313-override-recheck",
        rag_top_k=0,
        lang="zh",
    )
    try:
        fallback_help_obj, reason = loop._build_safe_fallback_overlay_help_obj(
            request,
            override_inferred_step_id="S10",
            override_overlay_step_id="S10",
            ignore_request_allowlist=True,
        )
    finally:
        loop.close()

    assert fallback_help_obj is None
    assert reason.startswith("evidence_conflict:")
    assert "completion_gate_already_satisfied:S10" in reason


def test_model_error_fallback_overlay_uses_emergency_presentation_source() -> None:
    request = TutorRequest(
        request_id="issue-313-model-error-presentation-fallback",
        message="help",
        context={
            "vars": {
                "comm1_freq_134_000": False,
                "ufc_comm1_pull_pressed": True,
                "ufc_scratchpad_number_display": "    .13",
            },
            "gates": {
                "S09.completion": {
                    "status": "blocked",
                    "reason": "COMM1 preset 1 must be programmed to 134.000 MHz.",
                    "reason_code": "s09_requires_comm1_freq_134_000",
                }
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "overlay_step_id": "S09",
                "missing_conditions": ["vars.comm1_freq_134_000==true"],
                "step_ui_targets": [
                    "ufc_comm1_channel_selector_pull",
                    "ufc_key_1",
                    "ufc_key_3",
                    "ufc_key_4",
                    "ufc_key_0",
                    "ufc_ent_button",
                ],
                "action_hint": {
                    "target": "ufc_key_4",
                    "reason": "COMM1 preset entry shows 13; press 4 next.",
                },
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": [
                "ufc_comm1_channel_selector_pull",
                "ufc_key_1",
                "ufc_key_3",
                "ufc_key_4",
                "ufc_key_0",
                "ufc_ent_button",
            ],
            "rag_topk": [],
            "vision_fact_summary": {"status": "vision_not_required", "seen_fact_ids": [], "fresh_fact_ids": []},
        },
    )
    response = TutorResponse(
        status="error",
        in_reply_to=request.request_id,
        message="降级提示：模型响应不可用。",
        actions=[],
        explanations=[],
        metadata={
            "provider": "fallback",
            "error_type": "ValueError",
            "error": "invalid model JSON",
        },
    )

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert repaired.metadata["fallback_overlay_used"] is True
    assert repaired.metadata["final_action_plan"]["source"] == "emergency_presentation_fallback"
    assert repaired.metadata["emergency_presentation_fallback_plan_source"] == "validator_action_hint"
    assert repaired.metadata["final_action_plan"]["targets"] == ["ufc_key_4"]
    assert repaired.metadata["harness_trace"]["final_action_plan"]["source"] == "emergency_presentation_fallback"
    assert repaired.metadata["harness_trace"]["final_action_plan"]["targets"] == ["ufc_key_4"]
    assert "validator_result" in repaired.metadata["harness_trace"]
    assert "repair_result" in repaired.metadata["harness_trace"]
    assert [action["target"] for action in repaired.actions] == ["ufc_key_4"]
    assert not str(repaired.metadata["fallback_overlay_reason"]).startswith("deterministic_step:")


def test_live_help_fixture_294_s09_uses_stage_action_hint_for_next_digit() -> None:
    request = TutorRequest(
        request_id="issue-294-s09-next-key",
        message="help",
        context={
            "vars": {
                "comm1_freq_134_000": False,
                "ufc_comm1_pull_pressed": True,
                "ufc_scratchpad_number_display": "    .13",
                "ufc_scratchpad_string_1_display": "1-",
                "ufc_scratchpad_string_2_display": "-",
            },
            "gates": {
                "S09.completion": {
                    "status": "blocked",
                    "reason": "COMM1 preset 1 must be programmed to 134.000 MHz.",
                    "reason_code": "s09_requires_comm1_freq_134_000",
                }
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S09",
                "overlay_step_id": "S09",
                "missing_conditions": ["vars.comm1_freq_134_000==true"],
                "step_ui_targets": [
                    "ufc_comm1_channel_selector_pull",
                    "ufc_key_1",
                    "ufc_key_3",
                    "ufc_key_4",
                    "ufc_key_0",
                    "ufc_ent_button",
                ],
                "action_hint": {
                    "target": "ufc_key_4",
                    "reason": "COMM1 preset entry shows 13; press 4 next.",
                },
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": [
                "ufc_comm1_channel_selector_pull",
                "ufc_key_1",
                "ufc_key_3",
                "ufc_key_4",
                "ufc_key_0",
                "ufc_ent_button",
            ],
            "rag_topk": [],
            "vision_fact_summary": {"status": "vision_not_required", "seen_fact_ids": [], "fresh_fact_ids": []},
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message="请拉出 COMM1。",
        actions=[],
        explanations=["请拉出 COMM1。"],
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S09", "error_category": "OM"},
                "next": {"step_id": "S09"},
                "overlay": {
                    "targets": ["ufc_comm1_channel_selector_pull"],
                    "evidence": [
                        {
                            "target": "ufc_comm1_channel_selector_pull",
                            "type": "var",
                            "ref": "VARS.comm1_freq_134_000",
                            "quote": "COMM1 is not tuned.",
                            "grounding_confidence": 0.9,
                        }
                    ],
                },
                "explanations": ["请拉出 COMM1。"],
            },
        },
    )

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert [action["target"] for action in repaired.actions] == ["ufc_key_4"]
    assert repaired.metadata["final_action_plan"]["source"] == "validator_action_hint"
    assert "action_hint_target_mismatch:ufc_comm1_channel_selector_pull" in repaired.metadata[
        "harness_validation_reasons"
    ]
    assert repaired.metadata["help_response"]["overlay"]["targets"] == ["ufc_key_4"]
    assert repaired.message == "COMM1 preset entry shows 13; press 4 next."


def test_live_help_fixture_299_s12_aligns_pb19_message_and_overlay() -> None:
    request = TutorRequest(
        request_id="2c936164-c567-4865-997c-ce65b827209d",
        message="help",
        context={
            "vars": {
                "ins_mode_cv_or_gnd": True,
                "ins_mode_set": True,
                "ins_fast_align_complete": False,
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S12",
                "overlay_step_id": "S12",
                "missing_conditions": ["vars.ins_fast_align_complete==true"],
                "step_ui_targets": ["ins_mode_knob", "ampcd_pb19"],
                "action_hint": {"target": "ampcd_pb19"},
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["ins_mode_knob", "ampcd_pb19"],
            "rag_topk": [{"snippet_id": "DCS FA-18C Early Access Guide EN_115"}],
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message="当前步骤 S12 未完成，需要设置 INS 模式。",
        actions=[],
        explanations=["当前步骤 S12 未完成，需要设置 INS 模式。"],
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S12", "error_category": "OM"},
                "next": {"step_id": "S12"},
                "overlay": {
                    "targets": ["ins_mode_knob"],
                    "evidence": [
                        {
                            "target": "ins_mode_knob",
                            "type": "rag",
                            "ref": "RAG_SNIPPETS.DCS FA-18C Early Access Guide EN_115",
                            "quote": "INS alignment guidance.",
                            "grounding_confidence": 0.9,
                        }
                    ],
                },
                "explanations": ["当前步骤 S12 未完成，需要设置 INS 模式。"],
            },
        },
    )

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert [action["target"] for action in repaired.actions] == ["ampcd_pb19"]
    assert "PB19" in repaired.message
    assert "设置 INS 模式" not in repaired.message
    assert repaired.metadata["validator_rejected"] is True
    assert repaired.metadata["repair_applied"] is True
    assert repaired.metadata["final_overlay_targets"] == ["ampcd_pb19"]
    assert repaired.metadata["final_action_plan"]["targets"] == ["ampcd_pb19"]
    assert repaired.metadata["final_action_plan"]["source"] == "validator_action_hint"
    assert "action_hint_target_mismatch:ins_mode_knob" in repaired.metadata["harness_validation_reasons"]
    assert repaired.metadata["model_raw_help_response"]["overlay"]["targets"] == ["ins_mode_knob"]
    assert repaired.metadata["help_response"]["overlay"]["targets"] == ["ampcd_pb19"]
    assert repaired.metadata["final_public_response"]["message"] == repaired.message
    assert repaired.metadata["final_public_response"]["explanations"] == [repaired.message]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "ampcd_pb19"


def test_live_help_fixture_299_s12_carrier_keeps_ins_knob_until_cv_mode() -> None:
    request = TutorRequest(
        request_id="issue-299-s12-carrier-gnd",
        message="help",
        context={
            "scenario_profile": "carrier",
            "vars": {
                "ins_mode": 2,
                "ins_mode_cv_or_gnd": True,
                "ins_mode_set": True,
                "ins_fast_align_complete": False,
            },
            "gates": {
                "S12.completion": {
                    "status": "blocked",
                    "reason_code": "s12_requires_ins_mode_cv",
                }
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S12",
                "overlay_step_id": "S12",
                "scenario_profile": "carrier",
                "missing_conditions": [
                    "vars.ins_mode==1",
                    "vars.ins_fast_align_complete==true",
                ],
                "step_ui_targets": ["ins_mode_knob", "ampcd_pb19"],
                "action_hint": {"target": "ampcd_pb19"},
                "observability_status": "observable",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["ins_mode_knob", "ampcd_pb19"],
            "rag_topk": [{"snippet_id": "DCS FA-18C Early Access Guide EN_115"}],
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message="INS 仍需设置到 CV。",
        actions=[],
        explanations=["INS 仍需设置到 CV。"],
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S12", "error_category": "OM"},
                "next": {"step_id": "S12"},
                "overlay": {
                    "targets": ["ins_mode_knob"],
                    "evidence": [
                        {
                            "target": "ins_mode_knob",
                            "type": "rag",
                            "ref": "RAG_SNIPPETS.DCS FA-18C Early Access Guide EN_115",
                            "quote": "Set INS mode to CV for carrier alignment.",
                            "grounding_confidence": 0.9,
                        }
                    ],
                },
                "explanations": ["INS 仍需设置到 CV。"],
            },
        },
    )

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert [action["target"] for action in repaired.actions] == ["ins_mode_knob"]
    assert "PB19" not in repaired.message
    assert repaired.metadata["final_overlay_targets"] == ["ins_mode_knob"]
    assert repaired.metadata["final_action_plan"]["targets"] == ["ins_mode_knob"]
    assert repaired.metadata["final_action_plan"]["source"] == "model"
    assert "action_hint_target_mismatch:ins_mode_knob" not in repaired.metadata["harness_validation_reasons"]


def test_live_help_fixture_299_s17_keeps_takeoff_trim_overlay() -> None:
    request = TutorRequest(
        request_id="19115196-2759-42d9-82b5-2822cb322285",
        message="help",
        context={
            "vars": {"takeoff_trim_set": False},
            "gates": {
                "S17.completion": {
                    "status": "blocked",
                    "reason_code": "s17_requires_takeoff_trim_pressed",
                }
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S17",
                "overlay_step_id": "S17",
                "missing_conditions": ["vars.takeoff_trim_set==true"],
                "step_ui_targets": ["takeoff_trim_button"],
                "action_hint": {"target": "takeoff_trim_button"},
                "observability_status": "partial",
                "requires_visual_confirmation": False,
            },
            "overlay_target_allowlist": ["takeoff_trim_button"],
        },
    )
    raw_message = "当前处于 S17 步骤，请按下 TAKEOFF TRIM 按钮以完成该步骤。"
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        message=raw_message,
        actions=[],
        explanations=[raw_message],
        metadata={
            "provider": "mock_qwen",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S17", "error_category": "OM"},
                "next": {"step_id": "S17"},
                "overlay": {
                    "targets": ["takeoff_trim_button"],
                    "evidence": [
                        {
                            "target": "takeoff_trim_button",
                            "type": "gate",
                            "ref": "GATES.S17.completion",
                            "quote": "Takeoff trim must be set.",
                            "grounding_confidence": 0.95,
                        }
                    ],
                },
                "explanations": [raw_message],
            },
        },
    )

    repaired = _validate_compact_live_help_response(request=request, response=response)

    assert [action["target"] for action in repaired.actions] == ["takeoff_trim_button"]
    assert repaired.metadata.get("completion_conflict_rewritten") is not True
    assert "takeoff_trim_button" in repaired.message
    assert repaired.explanations == [repaired.message]
    assert repaired.metadata["next"]["step_id"] == "S17"
    assert repaired.metadata["diagnosis"]["step_id"] == "S17"
    assert repaired.metadata["final_overlay_targets"] == ["takeoff_trim_button"]
    assert repaired.metadata["final_action_plan"]["targets"] == ["takeoff_trim_button"]
    assert repaired.metadata["final_public_response"]["message"] == repaired.message
    assert repaired.metadata["final_public_response"]["explanations"] == [repaired.message]
    assert repaired.metadata["final_public_response"]["actions"][0]["target"] == "takeoff_trim_button"


def test_completion_claim_parser_distinguishes_guidance_from_completion_claim() -> None:
    assert not _text_claims_step_complete(
        "当前处于 S17 步骤，请按下 TAKEOFF TRIM 按钮以完成该步骤。",
        "S17",
    )
    assert not _text_claims_step_complete(
        "当前处于 S17 步骤，请按下 TAKEOFF TRIM 按钮完成该步骤。",
        "S17",
    )
    assert _text_claims_step_complete(
        "S20 已经通过展开受油管来完成。当前进入下一步。",
        "S20",
    )
    assert _text_claims_step_complete(
        "当前处于 S17 步骤，已按下 TAKEOFF TRIM 按钮完成该步骤。",
        "S17",
    )


def test_map_response_actions_accepts_fake_llm_multi_target_help_response_when_enabled(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_multi_target_mapping.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-map-multi-target",
        lang="en",
        max_overlay_targets=2,
    )
    request = TutorRequest(
        intent="help",
        message="Need help with the FCS BIT sequence.",
        context={
            "vars": {"right_ddi_on": True, "fcs_bit_switch_up": False},
            "gates": {"S18.completion": {"status": "blocked"}},
            "overlay_target_allowlist": ["fcs_bit_switch", "right_mdi_pb5"],
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        metadata={
            "provider": "fake_llm",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S18", "error_category": "OM"},
                "next": {"step_id": "S18"},
                "overlay": {
                    "targets": ["fcs_bit_switch", "right_mdi_pb5"],
                    "evidence": [
                        {
                            "target": "fcs_bit_switch",
                            "type": "gate",
                            "ref": "GATES.S18.completion",
                            "quote": "Run the FCS BIT from the current blocked stage.",
                            "grounding_confidence": 0.92,
                        },
                        {
                            "target": "right_mdi_pb5",
                            "type": "gate",
                            "ref": "GATES.S18.completion",
                            "quote": "Run the FCS BIT from the current blocked stage.",
                            "grounding_confidence": 0.9,
                        },
                    ],
                },
                "explanations": ["Hold FCS BIT and press PB5 together."],
                "confidence": 0.91,
            },
        },
    )

    try:
        actions, mapping_meta = loop._map_response_actions(response, request)
    finally:
        loop.close()

    assert [action["target"] for action in actions] == ["fcs_bit_switch", "right_mdi_pb5"]
    assert [action["element_id"] for action in actions] == ["pnt_470", "pnt_83"]
    assert mapping_meta["allowed_evidence_ref_count"] >= 1


def test_map_response_actions_no_longer_backfills_legacy_s18_pb5_when_fcs_bit_is_highlighted_on_fcsmc_page(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_s18_backfill_pb5.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s18-backfill-pb5",
        lang="en",
        max_overlay_targets=2,
    )
    request = TutorRequest(
        intent="help",
        message="Need help with the FCS BIT sequence.",
        context={
            "vars": {"right_ddi_on": True, "fcs_bit_switch_up": False},
            "gates": {"S19.completion": {"status": "blocked"}},
            "overlay_target_allowlist": ["fcs_bit_switch", "right_mdi_pb5"],
            "deterministic_step_hint": {
                "inferred_step_id": "S19",
                "overlay_step_id": "S19",
                "requires_visual_confirmation": True,
                "action_hint": {"target": "fcs_bit_switch"},
            },
            "vision_fact_summary": {
                "status": "uncertain",
                "seen_fact_ids": ["fcsmc_page_visible"],
            },
            "vision_facts": [
                {
                    "fact_id": "fcsmc_page_visible",
                    "source_frame_id": "1773956832104_000021",
                }
            ],
        },
    )
    response = TutorResponse(
        status="ok",
        in_reply_to=request.request_id,
        metadata={
            "provider": "fake_llm",
            "generation_mode": "model",
            "help_response": {
                "diagnosis": {"step_id": "S19", "error_category": "OM"},
                "next": {"step_id": "S19"},
                "overlay": {
                    "targets": ["fcs_bit_switch"],
                    "evidence": [
                        {
                            "target": "fcs_bit_switch",
                            "type": "visual",
                            "ref": "VISION_FACTS.fcsmc_page_visible@1773956832104_000021",
                            "quote": "Right DDI explicitly displays the title 'FCS-MC' with MC1, MC2, FCSA, FCSB status lines.",
                            "grounding_confidence": 1.0,
                        }
                    ],
                },
                "explanations": ["Hold FCS BIT and press PB5 together."],
                "confidence": 0.91,
            },
        },
    )

    try:
        actions, mapping_meta = loop._map_response_actions(response, request)
    finally:
        loop.close()

    assert [action["target"] for action in actions] == ["fcs_bit_switch"]
    assert [action["element_id"] for action in actions] == ["pnt_470"]
    assert mapping_meta.get("s18_dual_overlay_backfill_applied") is not True


def test_live_loop_executes_fake_llm_multi_target_overlay_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_multi_target_live_loop.jsonl"
    _write_replay(
        replay_path,
        [
            {
                "schema_version": "v2",
                "seq": 1,
                "t_wall": 10.0,
                "aircraft": "FA-18C_hornet",
                "bios": {
                    "BATTERY_SW": 2,
                    "L_GEN_SW": 1,
                    "R_GEN_SW": 1,
                    "RIGHT_DDI_BRT_CTL": 0.5,
                    "FCS_BIT_SW": 1,
                    "RIGHT_DDI_PB_05": 1,
                },
                "delta": {
                    "FCS_BIT_SW": 1,
                    "RIGHT_DDI_PB_05": 1,
                },
            }
        ],
    )

    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(inferred_step_id="S19", missing_conditions=()),
    )
    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=MultiTargetHelpResponseModel(),
        action_executor=_make_multi_target_overlay_executor(
            monkeypatch,
            events,
            session_id="sess-live-multi-target",
        ),
        session_id="sess-live-multi-target",
        lang="en",
        max_overlay_targets=2,
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.metadata["prompt_build"]["max_overlay_targets"] == 2
    assert [action["target"] for action in response.actions] == ["fcs_bit_switch", "right_mdi_pb5"]
    assert [item["target"] for item in report["executed"]] == ["fcs_bit_switch", "right_mdi_pb5"]
    overlay_requested = [event for event in events if event.get("kind") == "overlay_requested"]
    assert [event["payload"]["target"] for event in overlay_requested] == ["pnt_470", "pnt_83"]


def test_live_loop_no_longer_backfills_legacy_s18_pb5_mid_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s18_dummy_single_target_json.jsonl"
    _write_replay(
        replay_path,
        [
            {
                "schema_version": "v2",
                "seq": 1,
                "t_wall": 10.0,
                "aircraft": "FA-18C_hornet",
                "bios": {
                    "BATTERY_SW": 2,
                    "L_GEN_SW": 1,
                    "R_GEN_SW": 1,
                    "RIGHT_DDI_BRT_CTL": 0.5,
                },
                "delta": {
                    "RIGHT_DDI_PB_05": 1,
                },
            }
        ],
    )

    class DummySingleTargetS19JsonModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="当前右 DDI 显示 FCS-MC 页面，表明已准备好进行 FCS BIT 测试。",
                actions=[],
                explanations=[
                    "当前右 DDI 显示 FCS-MC 页面，表明已准备好进行 FCS BIT 测试。下一步需要按住 FCS BIT 开关（向上），同时用左键点击右 DDI 上的 PB5 按钮以启动自检。",
                    "注意：只需在启动瞬间按住开关并按 PB5，看到测试开始（如出现 IN TEST 或 PBIT GO）后即可松开开关，无需持续按住直到测试完成。",
                ],
                metadata={
                    "provider": "dummy_llm",
                    "generation_mode": "model",
                    "help_response": {
                        "diagnosis": {"step_id": "S19", "error_category": "CO"},
                        "next": {"step_id": "S19"},
                        "overlay": {
                            "targets": ["fcs_bit_switch"],
                            "evidence": [
                                {
                                    "target": "fcs_bit_switch",
                                    "type": "visual",
                                    "ref": "VISION_FACTS.fcsmc_page_visible@1773957437530_000024",
                                    "quote": "Right DDI is on the FCS-MC sub-page, not the BIT root page.",
                                    "grounding_confidence": 1.0,
                                }
                            ],
                        },
                        "explanations": [
                            "当前右 DDI 显示 FCS-MC 页面，表明已准备好进行 FCS BIT 测试。下一步需要按住 FCS BIT 开关（向上），同时用左键点击右 DDI 上的 PB5 按钮以启动自检。",
                            "注意：只需在启动瞬间按住开关并按 PB5，看到测试开始（如出现 IN TEST 或 PBIT GO）后即可松开开关，无需持续按住直到测试完成。",
                        ],
                        "confidence": 0.9,
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(inferred_step_id="S19", missing_conditions=()),
    )
    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=DummySingleTargetS19JsonModel(),
        action_executor=_make_multi_target_overlay_executor_with_auto_clear(
            monkeypatch,
            events,
            session_id="sess-live-s18-dummy-json",
        ),
        session_id="sess-live-s18-dummy-json",
        lang="zh",
        max_overlay_targets=2,
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        loop._build_request = lambda *args, **kwargs: (
            TutorRequest(
                request_id=kwargs.get("request_id_override") or "dummy-s18-request",
                intent="help",
                message="help",
                observation_ref=obs.observation_id,
                context={
                    "vars": {"right_ddi_on": True, "fcs_bit_switch_up": False},
                    "gates": {"S19.completion": {"status": "blocked"}},
                    "overlay_target_allowlist": ["fcs_bit_switch", "right_mdi_pb5"],
                    "deterministic_step_hint": {
                        "inferred_step_id": "S19",
                        "overlay_step_id": "S19",
                        "requires_visual_confirmation": True,
                        "action_hint": {"target": "fcs_bit_switch"},
                        "step_ui_targets": ["fcs_bit_switch", "right_mdi_pb5"],
                    },
                    "vision": {
                        "vision_used": True,
                        "frame_ids": ["1773957437530_000024"],
                    },
                    "vision_fact_summary": {
                        "status": "uncertain",
                        "seen_fact_ids": ["fcsmc_page_visible"],
                        "frame_ids": ["1773957437530_000024"],
                    },
                    "vision_facts": [
                        {
                            "fact_id": "fcsmc_page_visible",
                            "state": "seen",
                            "source_frame_id": "1773957437530_000024",
                            "confidence": 1.0,
                            "evidence_note": "Right DDI is on the FCS-MC sub-page, not the BIT root page.",
                        }
                    ],
                },
                metadata={},
            ),
            {"max_overlay_targets": 2},
            "dummy-s18-state",
        )
        response, report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert [action["target"] for action in response.actions] == ["fcs_bit_switch"]
    assert response.metadata["response_mapping"].get("s18_dual_overlay_backfill_applied") is not True
    assert [item["target"] for item in report["executed"]] == ["fcs_bit_switch"]
    overlay_requested = [event for event in events if event.get("kind") == "overlay_requested"]
    assert [(event["payload"]["action"], event["payload"]["target"]) for event in overlay_requested] == [
        ("highlight", "pnt_470"),
    ]


def test_live_loop_clears_conflicting_overlay_before_fallback_rebuilds_current_step_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_conflicting_overlay_rewritten.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class ConflictingOverlayModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Set INS to GND.",
                actions=[],
                explanations=[
                    "INS is still OFF; set it to GND for S12.",
                    "Although visual evidence is unavailable, focus on the INS mode next.",
                ],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S12", "error_category": "CO"},
                        "next": {"step_id": "S12"},
                        "overlay": {
                            "targets": ["eng_crank_switch"],
                            "evidence": [
                                {
                                    "target": "eng_crank_switch",
                                    "type": "var",
                                    "ref": "VARS.ins_mode",
                                    "quote": "INS mode is OFF.",
                                    "grounding_confidence": 0.95,
                                }
                            ],
                        },
                        "explanations": [
                            "INS is still OFF; set it to GND for S12.",
                            "Although visual evidence is unavailable, focus on the INS mode next.",
                        ],
                        "confidence": 0.85,
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(
            inferred_step_id="S10",
            missing_conditions=("vars.engine_crank_left_complete==true",),
        ),
    )

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=ConflictingOverlayModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-conflicting-overlay-rewritten",
        lang="en",
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.metadata["completion_conflict_rewritten"] is True
    assert response.metadata["completion_conflict_overlay_cleared"] is True
    assert response.metadata["completion_conflict_original_actions"][0]["target"] == "eng_crank_switch"
    assert response.metadata["fallback_overlay_used"] is True
    assert response.actions
    assert response.actions[0]["target"] == "eng_crank_switch"
    assert response.message == "S10 is not complete yet. Please operate eng_crank_switch first and confirm that step is complete."
    assert "vars.engine_crank_left_complete" not in response.message
    assert response.metadata["final_public_response"]["actions"][0]["target"] == "eng_crank_switch"

def test_build_vision_selection_uses_observation_time_for_audit_anchor(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_observation_anchor.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-anchor",
        vision_mode="replay",
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        selection = loop._build_vision_selection(
            observation=loop._latest_enriched_obs,
            trigger_t_wall=10.25,
        )
    finally:
        loop.close()

    assert selection.observation_t_wall_s == 10.0
    assert selection.observation_t_wall_ms == 10000
    assert selection.trigger_wall_ms == 10250


def test_build_vision_selection_falls_back_when_observation_time_is_non_finite(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_observation_anchor_non_finite.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-anchor",
        vision_mode="replay",
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        assert loop._latest_enriched_obs is not None
        loop._latest_enriched_obs.payload["t_wall"] = math.nan
        selection = loop._build_vision_selection(
            observation=loop._latest_enriched_obs,
            trigger_t_wall=10.25,
        )
    finally:
        loop.close()

    assert selection.observation_t_wall_s == 10.25
    assert selection.observation_t_wall_ms == 10250
    assert selection.trigger_wall_ms == 10250


def test_build_vision_selection_falls_back_when_trigger_time_is_non_finite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "bios_trigger_anchor_non_finite.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=RecordingModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-anchor",
        vision_mode="replay",
    )
    monkeypatch.setattr("live_dcs.time.time", lambda: 42.5)
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        assert loop._latest_enriched_obs is not None
        loop._latest_enriched_obs.payload["t_wall"] = math.nan
        selection = loop._build_vision_selection(
            observation=loop._latest_enriched_obs,
            trigger_t_wall=math.nan,
        )
    finally:
        loop.close()

    assert selection.observation_t_wall_s == 42.5
    assert selection.observation_t_wall_ms == 42500
    assert selection.trigger_wall_ms == 42500


def test_fallback_overlay_for_s20_uses_only_refuel_probe_target(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_s20_remaining.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.5, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=5.0,
        lang="zh",
    )
    try:
        loop._step_interacted_targets = {"launch_bar_switch", "refuel_probe_switch"}

        s20_targets = loop.step_signal_profiles.get("S20", {}).get("ui_targets", [])
        assert s20_targets == ["refuel_probe_switch"]

        hint: dict[str, Any] = {
            "inferred_step_id": "S20",
            "overlay_step_id": "S20",
            "missing_conditions": [],
            "gate_blockers": [],
            "recent_ui_targets": [],
            "requires_visual_confirmation": False,
            "step_evidence_requirements": ["gate"],
            "observability": "observable",
        }
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "overlay_target_allowlist": list(loop.overlay_allowlist),
                "deterministic_step_hint": hint,
                "rag_topk": [],
                "gates": [
                    {"gate_id": "S20.completion", "status": "blocked"},
                    {"gate_id": "S20.precondition", "status": "allowed"},
                ],
            },
        )

        help_obj, reason = loop._build_safe_fallback_overlay_help_obj(request)

        assert help_obj is not None, f"expected overlay help_obj, got reason={reason}"
        actions = help_obj.get("overlay", {}).get("targets", [])
        assert actions == ["refuel_probe_switch"]
    finally:
        loop.close()


def test_step_interacted_targets_reset_on_step_change(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_step_reset.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        assert loop._step_interacted_targets == set()
        assert loop._last_inferred_step_id is None

        loop._step_interacted_targets.add("launch_bar_switch")
        loop._last_inferred_step_id = "S19"

        loop._step_interacted_targets = set()
        loop._last_inferred_step_id = "S08"
        assert loop._step_interacted_targets == set()
        assert loop._last_inferred_step_id == "S08"
    finally:
        loop.close()


def test_ingest_observation_clears_live_progress_state_on_power_loss(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_power_reset.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        loop._vision_fact_snapshot = {
            "fcsmc_final_go_result_visible": {
                "fact_id": "fcsmc_final_go_result_visible",
                "state": "seen",
                "source_frame_id": "1778572608745_000109",
                "expires_after_ms": 600000,
                "observed_at_wall_ms": 1778572608667,
                "expires_at_wall_ms": 1778573208667,
                "evidence_note": "",
                "sticky": True,
            }
        }
        loop._sticky_inference_step_id = "S19"
        loop._sticky_inference_missing_conditions = ("vars.pitot_heat_on==true",)
        loop._last_inferred_step_id = "S19"
        loop._step_interacted_targets = {"launch_bar_switch"}
        loop._refuel_probe_s20_latched_complete = True
        loop._refuel_probe_s21_latched_complete = True

        loop._ingest_observation(
            Observation(
                source="mock",
                payload={
                    "seq": 1,
                    "t_wall": 10.0,
                    "vars": {
                        "battery_on": False,
                        "power_available": False,
                    },
                },
            )
        )
    finally:
        loop.close()

    assert loop._vision_fact_snapshot == {}
    assert loop._sticky_inference_step_id is None
    assert loop._sticky_inference_missing_conditions == ()
    assert loop._last_inferred_step_id is None
    assert loop._step_interacted_targets == set()
    assert loop._refuel_probe_s20_latched_complete is False
    assert loop._refuel_probe_s21_latched_complete is False


def test_low_confidence_bootstrap_suppresses_early_battery_overlay(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_bootstrap_battery_false.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        request = TutorRequest(
            actor="learner",
            intent="help",
            message="help",
            context={
                "deterministic_step_hint": {
                    "inferred_step_id": "S01",
                    "missing_conditions": ["vars.battery_on==true"],
                },
                "state_harness": {
                    "telemetry_evidence": {
                        "source_status": "low_confidence_bootstrap",
                        "confidence": "low",
                        "observation_seq": 1,
                        "vars_source_missing_count": 101,
                    }
                },
                "vision_fact_summary": {
                    "status": "vision_not_required",
                    "frame_ids": ["1778943715191_000033"],
                    "seen_fact_ids": [],
                    "fresh_fact_ids": [],
                },
                "vision": {
                    "vision_used": True,
                    "frame_ids": ["1778943715191_000033"],
                },
            },
        )
        response = TutorResponse(
            status="ok",
            message="当前步骤 S01 未完成，电瓶开关未打开。",
            actions=[{"kind": "highlight", "target": "battery_switch"}],
            explanations=["当前步骤 S01 未完成，电瓶开关未打开。"],
            metadata={
                "help_response": {
                    "diagnosis": {"step_id": "S01", "error_category": "OM"},
                    "next": {"step_id": "S01"},
                    "overlay": {"targets": ["battery_switch"], "evidence": []},
                    "explanations": ["当前步骤 S01 未完成，电瓶开关未打开。"],
                }
            },
        )

        rewritten = loop._rewrite_low_confidence_bootstrap_response(response, request)

        assert rewritten is True
        assert response.actions == []
        assert "首帧遥测缺失较多" in response.message
        assert response.metadata["bootstrap_low_confidence_guardrail_applied"] is True
        assert loop._should_use_deterministic_overlay_fallback(response, request, None) is False
    finally:
        loop.close()


def test_build_request_remembers_launch_bar_interaction_before_next_help(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_launch_bar_memory.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(
            inferred_step_id="S20",
            missing_conditions=("vars.pitot_heat_on==true",),
        ),
    )
    monkeypatch.setattr(
        "live_dcs.enrich_bios_observation",
        lambda obs, *args, **kwargs: obs,
    )

    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path),
        model=FailingModel(),
        action_executor=RecordingExecutor(),
        cooldown_s=0,
        lang="zh",
    )
    try:
        loop._last_inferred_step_id = "S20"
        loop._ingest_observation(
            Observation(
                source="dcs_bios_raw",
                payload={
                    "schema_version": "v2",
                    "seq": 1,
                    "t_wall": 10.0,
                    "aircraft": "FA-18C_hornet",
                    "bios": {
                        "LAUNCH_BAR_SW": 1,
                    },
                    "delta": {
                        "LAUNCH_BAR_SW": 1,
                    },
                    "vars": {
                        "battery_on": True,
                        "power_available": True,
                    },
                },
            )
        )

        obs = Observation(
            source="mock",
            payload={
                "seq": 2,
                "t_wall": 30.0,
                "vars": {
                    "battery_on": True,
                    "power_available": True,
                    "probe_cycle_complete": True,
                    "pitot_heat_on": False,
                },
            },
        )
        vision_selection = loop._build_vision_selection(observation=obs, trigger_t_wall=30.0)
        vision_fact_context = loop._extract_vision_fact_context(vision_selection=vision_selection)
        request, _prompt_meta, _state_key = loop._build_request(
            obs,
            vision_selection=vision_selection,
            vision_fact_context=vision_fact_context,
        )
    finally:
        loop.close()

    hint = request.context["deterministic_step_hint"]
    assert hint["recent_ui_targets"] == []
    assert hint["step_interacted_targets"] == []
    assert hint["action_hint"]["target"] == "refuel_probe_switch"
