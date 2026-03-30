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
    _build_observation_source_from_args,
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
    _resolve_overlay_step_id,
    _resolve_step_overlay_allowlist,
    _sanitize_request_payload_for_event,
    _sanitize_response_payload_for_event,
    _sanitize_policy_error_for_user,
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
            "APU_CONTROL_SW": apu_switch,
            "APU_READY_LT": 0,
            "ENGINE_CRANK_SW": 1,
        },
        "delta": {"APU_CONTROL_SW": apu_switch},
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
                    "diagnosis": {"step_id": "S18", "error_category": "OM"},
                    "next": {"step_id": "S18"},
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
        lang="zh",
    )
    try:
        stats = loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert stats["frames"] == 1
    assert stats["help_cycles"] == 1
    assert stats["model_calls"] == 1
    assert len(model.calls) == 1
    request = model.calls[0]["request"]
    assert request is not None
    assert request.intent == "help"
    assert "candidate_steps" in request.context
    assert request.context["candidate_steps"] == [f"S{i:02d}" for i in range(1, 26)]
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
    gates = request.context["gates"]
    assert isinstance(gates, dict)
    assert "S03.completion" in gates
    assert gates["S03.completion"]["status"] in {"allowed", "blocked"}
    assert request.metadata["prompt_hash"]
    assert request.context["overlay_target_allowlist"] == ["apu_switch"]

    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    assert executor.calls[0][0]["type"] == "overlay"
    assert executor.calls[0][0]["target"] == "apu_switch"


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
        assert response_events[0].metadata["generation_mode"] == generation_mode
        for event in cycle_events:
            if event.kind.startswith("overlay_"):
                overlay_event_kinds.add(event.kind)
                assert event.payload["help_cycle_id"] == help_cycle_id
                assert event.metadata["generation_mode"] == generation_mode

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
    assert tutor_response_payloads[0]["metadata"]["failure_code"] == "vision_unavailable"
    response_mapping = tutor_response_payloads[0]["metadata"]["response_mapping"]
    assert response_mapping["rejected_targets_by_request_allowlist"] == ["battery_switch"]
    assert "overlay_target_not_in_request_allowlist" in response_mapping["mapping_errors"]
    assert tutor_response_payloads[0]["metadata"]["response_mapping_failure_codes"] == [ALLOWLIST_FAIL]
    assert tutor_response_payloads[0]["metadata"]["response_mapping_failure_code"] == ALLOWLIST_FAIL
    assert tutor_response_payloads[0]["metadata"]["response_mapping_failure_stage"] == "response_mapping"
    assert tutor_response_payloads[0]["metadata"]["fallback_overlay_used"] is True
    assert tutor_response_payloads[0]["metadata"]["fallback_overlay_reason"].startswith("deterministic_step:")


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
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

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
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    assert tutor_response_payload["metadata"]["failure_code"] == "vision_unavailable"
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
    _write_replay(replay_path, [_bios_frame(1, 11.0, apu_switch=0)])

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
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(executor.calls) == 0
    out = capsys.readouterr().out
    assert "dry_run_actions" in out
    assert "apu_switch" in out


def test_live_loop_dry_run_overlay_uses_executor_when_executor_is_dry_run(tmp_path: Path, capsys) -> None:
    replay_path = tmp_path / "bios_dry_run_exec.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 11.5, apu_switch=0)])

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
        loop.run(max_frames=1, auto_help_on_first_frame=True)
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

    for step_id in ("S11", "S12", "S13", "S14", "S16", "S20", "S21", "S23", "S24", "S25"):
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
    assert profiles["S18"]["ui_targets"] == ["fcs_bit_switch", "right_mdi_pb18", "right_mdi_pb5"]


def test_real_fa18c_pack_marks_non_display_partial_steps_as_non_visual() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pack_path = repo_root / "packs" / "fa18c_startup" / "pack.yaml"

    profiles = _load_step_signal_profiles(pack_path)

    for step_id in ("S17", "S19", "S22"):
        assert profiles[step_id]["observability"] == "partial"
        assert profiles[step_id]["observability_status"] == "partial"
        assert profiles[step_id]["requires_visual_confirmation"] is False

    for step_id in ("S02", "S07"):
        assert profiles[step_id]["observability"] == "observable"
        assert profiles[step_id]["observability_status"] == "observable"
        assert profiles[step_id]["requires_visual_confirmation"] is False

    assert profiles["S02"]["evidence_requirements"] == ["var", "delta", "gate"]
    assert profiles["S02"]["ui_targets"] == []
    assert profiles["S02"]["overlay_enabled"] is False
    assert profiles["S07"]["evidence_requirements"] == ["var", "delta", "gate"]
    assert profiles["S07"]["ui_targets"] == []
    assert profiles["S07"]["overlay_enabled"] is False
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
    _write_replay(replay_path, [_bios_frame(1, 13.0, apu_switch=0)])

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
        stats = loop.run(max_frames=1, auto_help_on_first_frame=True)
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
    _write_replay(replay_path, [_bios_frame(1, 19.0, apu_switch=0)])

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
        loop.run(max_frames=1, auto_help_on_first_frame=True)
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
    assert meta["fallback_overlay_reason"].startswith("deterministic_step:")


def test_live_loop_replaces_rejected_future_step_overlay_with_safe_current_step_overlay(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_wrong_future_overlay.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 19.0, apu_switch=0)])

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
        loop.run(max_frames=1, auto_help_on_first_frame=True)
    finally:
        loop.close()

    assert len(executor.calls) == 1
    assert executor.calls[0][0]["target"] == "apu_switch"

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    meta = tutor_response_payload["metadata"]
    assert meta["fallback_overlay_used"] is True
    assert meta["fallback_overlay_reason"] == "deterministic_step:S03"
    response_mapping = meta["response_mapping"]
    assert response_mapping["rejected_targets_by_request_allowlist"] == ["eng_crank_switch"]
    assert "overlay_target_not_in_request_allowlist" in response_mapping["mapping_errors"]
    assert tutor_response_payload["message"] == "S03 is not complete yet. Please satisfy: vars.apu_start_support_complete==true."
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
        assert response.message == "降级提示：你大概率卡在 S05，请先满足：vars.rpm_r>=25。"
        assert response.explanations == ["降级提示：你大概率卡在 S05，请先满足：vars.rpm_r>=25。"]
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
        assert response.message == "降级提示：你大概率卡在 S08，请先满足：vision_facts.fcs_page_visible==seen。"
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
                    "seen_fact_ids": ["bit_page_visible"],
                    "not_seen_fact_ids": [
                        "left_ddi_dark",
                        "right_ddi_dark",
                        "ampcd_dark",
                        "left_ddi_fcs_option_visible",
                        "fcs_page_visible",
                    ],
                    "uncertain_fact_ids": ["left_ddi_menu_root_visible"],
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
        assert fallback_help_obj["overlay"]["targets"] == ["left_mdi_pb18"]
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
                    "seen_fact_ids": ["bit_page_visible", "left_ddi_fcs_option_visible"],
                    "not_seen_fact_ids": [
                        "left_ddi_dark",
                        "right_ddi_dark",
                        "ampcd_dark",
                        "fcs_page_visible",
                    ],
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
                "seen_fact_ids": ["bit_page_visible", "left_ddi_fcs_option_visible"],
                "not_seen_fact_ids": ["left_ddi_dark", "fcs_page_visible"],
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
                "seen_fact_ids": ["bit_page_visible", "left_ddi_menu_root_visible"],
                "not_seen_fact_ids": ["left_ddi_dark", "fcs_page_visible"],
                "uncertain_fact_ids": ["left_ddi_fcs_option_visible", "left_ddi_fcs_page_button_visible"],
            }
        },
        allowed_targets=["left_mdi_pb18", "left_mdi_pb15", "left_mdi_brightness_selector"],
    )

    assert targets == ["left_mdi_pb18"]


def test_prefer_navigation_target_from_vision_context_returns_left_pb15_for_explicit_fcs_button_fact() -> None:
    targets = _prefer_navigation_target_from_vision_context(
        inferred_step_id="S08",
        missing_conditions=["vision_facts.fcs_page_visible==seen"],
        context={
            "vision_fact_summary": {
                "seen_fact_ids": ["bit_page_visible", "left_ddi_fcs_page_button_visible"],
                "not_seen_fact_ids": ["left_ddi_dark", "fcs_page_visible"],
                "uncertain_fact_ids": [],
            }
        },
        allowed_targets=["left_mdi_pb18", "left_mdi_pb15", "left_mdi_brightness_selector"],
    )

    assert targets == ["left_mdi_pb15"]


def test_visual_action_hint_overlay_override_rewrites_s08_tac_guidance_to_pb18(tmp_path: Path) -> None:
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
                    "seen_fact_ids": ["bit_page_visible", "left_ddi_menu_root_visible"],
                    "not_seen_fact_ids": ["left_ddi_dark", "fcs_page_visible"],
                    "uncertain_fact_ids": [
                        "left_ddi_fcs_option_visible",
                        "left_ddi_fcs_page_button_visible",
                    ],
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

        assert override_used is True
        assert override_reason == "deterministic_step:S08"
        assert response.actions
        assert response.actions[0]["target"] == "left_mdi_pb18"
        assert "Press PB18 first to switch to the SUPT page" in response.message
        assert response.explanations == [response.message]
        assert response.metadata["action_hint_overlay_override_kind"] == "visual_action_hint"
        assert response.metadata["visual_action_hint_override_used"] is True
        assert response.metadata["visual_action_hint_override_target"] == "left_mdi_pb18"
        assert response.metadata["action_hint_overlay_override_original_targets"] == ["left_mdi_pb15"]
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


def test_action_hint_overlay_override_rewrites_s18_fcsmc_step_to_fcs_bit_switch(tmp_path: Path) -> None:
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
                    {"gate_id": "S18.completion", "status": "allowed"},
                    {"gate_id": "S18.precondition", "status": "allowed"},
                ],
                "vision_fact_summary": {
                    "status": "uncertain",
                    "seen_fact_ids": ["fcs_page_visible", "right_ddi_fcsmc_page_visible"],
                    "not_seen_fact_ids": ["bit_page_visible", "bit_root_page_visible", "bit_page_failure_visible"],
                    "uncertain_fact_ids": ["fcs_bit_result_visible"],
                },
                "deterministic_step_hint": {
                    "inferred_step_id": "S18",
                    "overlay_step_id": "S18",
                    "missing_conditions": ["vision_facts.fcs_bit_result_visible==seen"],
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
        assert override_reason == "deterministic_step:S18"
        assert response.actions
        assert response.actions[0]["target"] == "fcs_bit_switch"
        assert "Hold the FCS BIT switch up" in response.message
        assert "PB5" in response.message
        assert response.metadata["action_hint_overlay_override_used"] is True
        assert response.metadata["action_hint_overlay_override_target"] == "fcs_bit_switch"
        assert response.metadata["action_hint_overlay_override_kind"] == "action_hint"
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
    allowed = ["fcs_bit_switch", "right_mdi_pb18", "right_mdi_pb5"]

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
        "target": "fcs_bit_switch",
        "reason": "Keep holding the FCS BIT switch up while you press Right DDI PB5 to start or continue the FCS-MC BIT.",
    }
    assert _build_procedural_action_hint(
        inferred_step_id="S18",
        vars_selected={"fcs_bit_switch_up": False},
        allowed_targets=allowed,
        vision_fact_summary={"seen_fact_ids": ["right_ddi_fcsmc_page_visible"]},
    ) == {
        "target": "fcs_bit_switch",
        "reason": "The right DDI is already on the FCS-MC page. Hold the FCS BIT switch up while pressing Right DDI PB5 to run the BIT.",
    }


def test_build_procedural_action_hint_for_s19_advances_after_probe_extends() -> None:
    allowed = ["refuel_probe_switch", "launch_bar_switch", "pitot_heater_switch"]

    assert _build_procedural_action_hint(
        inferred_step_id="S19",
        vars_selected={"probe_extended": False, "probe_cycle_complete": False},
        allowed_targets=allowed,
    ) == {
        "target": "refuel_probe_switch",
        "reason": "The refueling probe is not yet fully extended; move the probe switch to EXTEND first.",
    }
    assert _build_procedural_action_hint(
        inferred_step_id="S19",
        vars_selected={"probe_extended": False, "probe_cycle_complete": True},
        allowed_targets=allowed,
    ) == {
        "target": "launch_bar_switch",
        "reason": "The refueling probe has already been cycled in this startup session; continue the four-down checklist with the launch bar switch.",
    }


def test_action_hint_overlay_override_rewrites_s19_probe_backtrack_to_launch_bar() -> None:
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
            "next": {"step_id": "S20"},
            "diagnosis": {"step_id": "S19", "error_category": "OM"},
        },
    )
    request = TutorRequest(
        actor="learner",
        intent="help",
        message="help",
        context={
            "overlay_target_allowlist": ["refuel_probe_switch", "launch_bar_switch"],
            "gates": [
                {"gate_id": "S19.completion", "status": "allowed"},
                {"gate_id": "S19.precondition", "status": "allowed"},
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S19",
                "overlay_step_id": "S19",
                "requires_visual_confirmation": False,
                "step_evidence_requirements": ["gate", "rag", "delta"],
                "action_hint": {"target": "launch_bar_switch"},
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
        assert reason == "deterministic_step:S19"
        assert response.actions[0]["target"] == "launch_bar_switch"
        assert response.metadata["action_hint_overlay_override_target"] == "launch_bar_switch"
        assert "发射杆开关" in response.message
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
        missing_conditions=["vision_facts.bit_page_visible==seen"],
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
                    fact_id="fcs_reset_seen",
                    state="seen",
                    source_frame_id="1772872445010_000123",
                    confidence=0.91,
                    expires_after_ms=600000,
                    evidence_note="FCS reset evidence visible on the left DDI.",
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
                                fact_id="fcs_reset_seen",
                                state="seen",
                                source_frame_id="1772872445010_000123",
                                confidence=0.93,
                                expires_after_ms=600000,
                                evidence_note="FCS reset visible on the left DDI.",
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
    assert request.context["vision_fact_summary"]["seen_fact_ids"] == ["fcs_reset_seen"]
    assert request.metadata["vision_fact_status"] == "available"
    assert response.metadata["vision_fact_status"] == "available"
    assert response.metadata["vision_fact_summary"]["seen_fact_ids"] == ["fcs_reset_seen"]
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
        '{"facts":[{"fact_id":"fcs_reset_seen","state":"seen","source_frame_id":"1772872445010_000123",'
        '"confidence":0.93,"evidence_note":"FCS reset visible on the left DDI."}]}'
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
                                fact_id="fcs_reset_seen",
                                state="seen",
                                source_frame_id="1772872445010_000123",
                                confidence=0.93,
                                expires_after_ms=600000,
                                evidence_note="FCS reset visible on the left DDI.",
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
                                fact_id="fcs_reset_seen",
                                state="seen",
                                source_frame_id="1772872445010_000123",
                                confidence=0.9,
                                expires_after_ms=600000,
                                evidence_note="FCS reset visible.",
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
                                fact_id="fcs_reset_seen",
                                state="seen",
                                source_frame_id="10000_000123",
                                confidence=0.91,
                                expires_after_ms=600000,
                                evidence_note="FCS RESET visible.",
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
                        "confidence": 0.9,
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
                                fact_id="fcs_reset_seen",
                                state="seen",
                                source_frame_id="10000_000123",
                                confidence=0.9,
                                expires_after_ms=600000,
                                evidence_note="FCS RESET visible.",
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
                        "confidence": 0.88,
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
                                fact_id="fcs_reset_seen",
                                state="seen",
                                source_frame_id="10000_000123",
                                confidence=0.9,
                                expires_after_ms=600000,
                                evidence_note="FCS RESET visible.",
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
                        "confidence": 0.7,
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
                                fact_id="left_ddi_menu_root_visible",
                                state="seen",
                                source_frame_id="10000_000124",
                                confidence=0.99,
                                expires_after_ms=2000,
                                evidence_note="Left DDI root menu visible.",
                            ),
                            VisionFact(
                                fact_id="left_ddi_fcs_page_button_visible",
                                state="seen",
                                source_frame_id="10000_000124",
                                confidence=0.99,
                                expires_after_ms=2000,
                                evidence_note="Left DDI PB15 FCS button visible.",
                            ),
                            VisionFact(
                                fact_id="fcs_page_visible",
                                state="not_seen",
                                source_frame_id="10000_000124",
                                confidence=1.0,
                                expires_after_ms=2000,
                                evidence_note="Left DDI is not yet on the FCS page.",
                            ),
                            VisionFact(
                                fact_id="bit_page_visible",
                                state="not_seen",
                                source_frame_id="10000_000124",
                                confidence=1.0,
                                expires_after_ms=2000,
                                evidence_note="Right DDI is not on the top BIT page.",
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
                        "confidence": 0.95,
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
    assert response.metadata["fallback_overlay_reason"] == "deterministic_step:S08"
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
    assert response.metadata["fallback_overlay_reason"] == "deterministic_step:S18"
    assert response.metadata["action_hint_overlay_override_used"] is True
    assert response.metadata["action_hint_overlay_override_target"] == "right_mdi_pb5"
    assert response.metadata["action_hint_overlay_override_original_targets"] == ["fcs_bit_switch"]
    assert response.metadata["final_public_response"]["actions"][0]["target"] == "right_mdi_pb5"


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


def test_map_response_actions_backfills_s18_pb5_when_fcs_bit_is_highlighted_on_fcsmc_page(
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
            "gates": {"S18.completion": {"status": "blocked"}},
            "overlay_target_allowlist": ["fcs_bit_switch", "right_mdi_pb5"],
            "deterministic_step_hint": {
                "inferred_step_id": "S18",
                "overlay_step_id": "S18",
                "requires_visual_confirmation": True,
                "action_hint": {"target": "fcs_bit_switch"},
            },
            "vision_fact_summary": {
                "status": "uncertain",
                "seen_fact_ids": ["right_ddi_fcsmc_page_visible"],
            },
            "vision_facts": [
                {
                    "fact_id": "right_ddi_fcsmc_page_visible",
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
                "diagnosis": {"step_id": "S18", "error_category": "OM"},
                "next": {"step_id": "S18"},
                "overlay": {
                    "targets": ["fcs_bit_switch"],
                    "evidence": [
                        {
                            "target": "fcs_bit_switch",
                            "type": "visual",
                            "ref": "VISION_FACTS.right_ddi_fcsmc_page_visible@1773956832104_000021",
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

    assert [action["target"] for action in actions] == ["fcs_bit_switch", "right_mdi_pb5"]
    assert [action["element_id"] for action in actions] == ["pnt_470", "pnt_83"]
    assert mapping_meta["s18_dual_overlay_backfill_applied"] is True
    assert mapping_meta["s18_dual_overlay_backfill_reason"] == "s18_fcsmc_fcs_bit_implies_pb5"


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
        lambda *args, **kwargs: StepInferenceResult(inferred_step_id="S18", missing_conditions=()),
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


def test_live_loop_backfills_s18_pb5_without_clearing_first_target_mid_batch(
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

    class DummySingleTargetS18JsonModel:
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
                        "diagnosis": {"step_id": "S18", "error_category": "CO"},
                        "next": {"step_id": "S18"},
                        "overlay": {
                            "targets": ["fcs_bit_switch"],
                            "evidence": [
                                {
                                    "target": "fcs_bit_switch",
                                    "type": "visual",
                                    "ref": "VISION_FACTS.right_ddi_fcsmc_page_visible@1773957437530_000024",
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
        lambda *args, **kwargs: StepInferenceResult(inferred_step_id="S18", missing_conditions=()),
    )
    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=DummySingleTargetS18JsonModel(),
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
                    "gates": {"S18.completion": {"status": "blocked"}},
                    "overlay_target_allowlist": ["fcs_bit_switch", "right_mdi_pb5"],
                    "deterministic_step_hint": {
                        "inferred_step_id": "S18",
                        "overlay_step_id": "S18",
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
                        "seen_fact_ids": ["right_ddi_fcsmc_page_visible"],
                        "frame_ids": ["1773957437530_000024"],
                    },
                    "vision_facts": [
                        {
                            "fact_id": "right_ddi_fcsmc_page_visible",
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
    assert [action["target"] for action in response.actions] == ["fcs_bit_switch", "right_mdi_pb5"]
    assert response.metadata["response_mapping"]["s18_dual_overlay_backfill_applied"] is True
    assert [item["target"] for item in report["executed"]] == ["fcs_bit_switch", "right_mdi_pb5"]
    overlay_requested = [event for event in events if event.get("kind") == "overlay_requested"]
    assert [(event["payload"]["action"], event["payload"]["target"]) for event in overlay_requested] == [
        ("highlight", "pnt_470"),
        ("highlight", "pnt_83"),
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
    assert response.message == "S10 is not complete yet. Please operate eng_crank_switch first, then satisfy: vars.engine_crank_left_complete==true."
    assert response.metadata["final_public_response"]["actions"][0]["target"] == "eng_crank_switch"


def test_live_loop_advances_s18_to_s19_when_model_claims_final_go_under_vision_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s18_final_go_advances_s19.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class S18FinalGoModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Need more information/please confirm: right_mdi_pb5",
                actions=[],
                explanations=[
                    "Need more information/please confirm: right_mdi_pb5",
                    "The Right DDI shows the FCS-MC page with all systems reporting 'GO', indicating that the Built-In Test (BIT) has been successfully completed. The next step is to proceed with the startup sequence.",
                ],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S18", "error_category": "OM"},
                        "next": {"step_id": "S18"},
                        "overlay": {"targets": [], "evidence": []},
                        "explanations": [
                            "Need more information/please confirm: right_mdi_pb5",
                            "The Right DDI shows the FCS-MC page with all systems reporting 'GO', indicating that the Built-In Test (BIT) has been successfully completed. The next step is to proceed with the startup sequence.",
                        ],
                        "confidence": 0.95,
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
        model=S18FinalGoModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s18-final-go-advance",
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
    assert response.metadata["s18_visual_completion_rewritten"] is True
    assert response.metadata["next"]["step_id"] == "S19"
    assert response.metadata["diagnosis"]["step_id"] == "S19"
    assert response.actions
    assert response.actions[0]["target"] == "refuel_probe_switch"
    assert response.metadata["fallback_overlay_used"] is True
    assert response.metadata["fallback_overlay_reason"] == "deterministic_step:S19"
    assert response.metadata["final_public_response"]["actions"][0]["target"] == "refuel_probe_switch"
    assert response.metadata.get("action_hint_overlay_override_used") is not True


def test_live_loop_advances_s18_to_s19_when_model_reports_mc1_mc2_fcsa_fcsb_go_and_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s18_exit_page_after_final_go.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class S18ExitPageModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Need more information/please confirm: right_mdi_pb5",
                actions=[],
                explanations=[
                    "Need more information/please confirm: right_mdi_pb5",
                    "The Right DDI shows the FCS-MC page with all systems (MC1, MC2, FCSA, FCSB) reporting 'GO'. This indicates that the Built-In Test (BIT) has been successfully completed. The next step is to exit this maintenance page by pressing PB5 (EXIT) to return to the main BIT failures page or previous menu.",
                ],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S18", "error_category": "OM"},
                        "next": {"step_id": "S18"},
                        "overlay": {"targets": [], "evidence": []},
                        "explanations": [
                            "Need more information/please confirm: right_mdi_pb5",
                            "The Right DDI shows the FCS-MC page with all systems (MC1, MC2, FCSA, FCSB) reporting 'GO'. This indicates that the Built-In Test (BIT) has been successfully completed. The next step is to exit this maintenance page by pressing PB5 (EXIT) to return to the main BIT failures page or previous menu.",
                        ],
                        "confidence": 0.95,
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
        model=S18ExitPageModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s18-exit-page-advance",
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
    assert response.metadata["s18_visual_completion_rewritten"] is True
    assert response.metadata["next"]["step_id"] == "S19"
    assert response.actions
    assert response.actions[0]["target"] == "refuel_probe_switch"
    assert response.metadata.get("action_hint_overlay_override_used") is not True


def test_live_loop_advances_s18_to_s19_when_model_mentions_go_results_but_wrongly_repeats_pb5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s18_go_results_but_wrongly_repeats_pb5.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class S18GoResultsButWrongPb5Model:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Need more information/please confirm: right_mdi_pb5",
                actions=[],
                explanations=[
                    "Need more information/please confirm: right_mdi_pb5",
                    "The Right DDI shows the FCS-MC page with 'GO' results for MC1, MC2, FCSA, and FCSB. This indicates that the initial BIT root phase (pressing PB5 to enter FCS-MC) has been completed. The next step in the S18 procedure is to initiate the specific FCS BIT test by holding the FCS BIT switch and pressing PB5 again.",
                ],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S18", "error_category": "OM"},
                        "next": {"step_id": "S18"},
                        "overlay": {"targets": [], "evidence": []},
                        "explanations": [
                            "Need more information/please confirm: right_mdi_pb5",
                            "The Right DDI shows the FCS-MC page with 'GO' results for MC1, MC2, FCSA, and FCSB. This indicates that the initial BIT root phase (pressing PB5 to enter FCS-MC) has been completed. The next step in the S18 procedure is to initiate the specific FCS BIT test by holding the FCS BIT switch and pressing PB5 again.",
                        ],
                        "confidence": 0.95,
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
        model=S18GoResultsButWrongPb5Model(),
        action_executor=RecordingExecutor(),
        session_id="sess-s18-go-results-wrong-pb5",
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
    assert response.metadata.get("s18_visual_completion_rewritten") is not True
    assert response.metadata.get("fallback_overlay_reason") != "deterministic_step:S19"
    assert not response.actions or response.actions[0]["target"] != "refuel_probe_switch"


def test_live_loop_does_not_advance_s18_from_structured_fact_when_evidence_lacks_full_final_go_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "bios_s18_partial_structured_go.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    class PartialStructuredGoModel:
        def explain_error(self, observation: Observation, request=None) -> TutorResponse:
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message="Need more information/please confirm: right_mdi_pb5",
                actions=[],
                explanations=[
                    "Need more information/please confirm: right_mdi_pb5",
                ],
                metadata={
                    "provider": "mock_qwen",
                    "help_response": {
                        "diagnosis": {"step_id": "S18", "error_category": "CO"},
                        "next": {"step_id": "S18"},
                        "overlay": {"targets": [], "evidence": []},
                        "explanations": [
                            "Need more information/please confirm: right_mdi_pb5",
                        ],
                        "confidence": 0.95,
                    },
                },
            )

        def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
            return self.explain_error(observation, request)

    class PartialStructuredGoVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):
            return VisionFactExtractionResult(
                status="available",
                observation=VisionFactObservation(
                    trigger_wall_ms=trigger_wall_ms,
                    session_id=session_id,
                    frame_ids=["1772872445010_000123"],
                    facts=[
                        VisionFact(
                            fact_id="fcs_bit_result_visible",
                            state="seen",
                            source_frame_id="1772872445010_000123",
                            confidence=0.95,
                            expires_after_ms=600000,
                            evidence_note="Right DDI FCS-MC page shows FCSA GO and FCSB GO.",
                            sticky=True,
                            observed_at_wall_ms=trigger_wall_ms,
                        )
                    ],
                    summary="seen=fcs_bit_result_visible",
                    metadata={},
                ),
                metadata={"vision_fact_summary": {"status": "available", "seen_fact_ids": ["fcs_bit_result_visible"]}},
            )

    monkeypatch.setattr(
        "live_dcs.infer_step_id",
        lambda *args, **kwargs: StepInferenceResult(inferred_step_id="S18", missing_conditions=()),
    )

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    loop = LiveDcsTutorLoop(
        source=source,
        model=PartialStructuredGoModel(),
        action_executor=RecordingExecutor(),
        session_id="sess-s18-partial-structured-go",
        lang="en",
        vision_fact_extractor=PartialStructuredGoVisionFactExtractor(),
    )
    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert response.metadata.get("s18_visual_completion_rewritten") is not True
    assert response.metadata["vision_fact_summary"]["seen_fact_ids"] == []
    assert response.metadata["vision_fact_summary"]["uncertain_fact_ids"] == ["fcs_bit_result_visible"]
    assert response.metadata.get("fallback_overlay_reason") != "deterministic_step:S19"


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
