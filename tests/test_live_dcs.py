from __future__ import annotations

import builtins
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import time

import pytest
import yaml

from adapters.action_executor import OverlayActionExecutor
from adapters.dcs.overlay.sender import DcsOverlaySender
from adapters.source_chunk_refs import build_source_chunk_ref
from core.help_failure import ALLOWLIST_FAIL, EVIDENCE_FAIL
from core.types import Observation, TutorRequest, TutorResponse
from live_dcs import (
    CompositeHelpTrigger,
    LiveDcsTutorLoop,
    ReplayBiosReceiver,
    StdinHelpTrigger,
    UdpHelpTrigger,
    build_arg_parser,
    _is_help_trigger_payload,
    _load_overlay_allowlist,
    _load_step_signal_profiles,
    _normalize_cached_response_metadata,
    _sanitize_policy_error_for_user,
)
from tools.index_docs import build_index
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


class FailingModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        raise RuntimeError("model unavailable")


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


class QueryOnlyKnowledge:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        self.calls.append({"text": text, "k": k})
        return [
            {
                "doc_id": "manual",
                "section": "S03",
                "page_or_heading": "S03",
                "snippet": "APU switch to ON and wait for APU READY.",
                "snippet_id": "manual_s03_1",
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
                "section": Path("S03"),
                "page_or_heading": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "snippet": {"text": "APU switch to ON"},
                "snippet_id": Path("manual_s03_1"),
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
                "section": "S03",
                "page_or_heading": "S03",
                "snippet": "APU switch ON.",
                "snippet_id": "meta_s03_1",
                "score": 1.0,
            }
        ]
        meta = {
            "cache_hit": True,
            "grounding_missing": False,
            "grounding_reason": None,
            "snippet_ids": ["meta_s03_1"],
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
                "snippet": {"text": "APU switch ON"},
                "snippet_id": Path("meta_s03_1"),
                "score": float("inf"),
                "extra": {"nested": True},
            }
        ]
        meta = {
            "cache_hit": "yes",
            "grounding_missing": 0,
            "grounding_reason": {"unexpected": "mapping"},
            "snippet_ids": [Path("meta_s03_1"), {"nested": True}],
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
    hint = request.context["deterministic_step_hint"]
    assert isinstance(hint, dict)
    assert hint.get("inferred_step_id")
    assert isinstance(hint.get("requires_visual_confirmation"), bool)
    gates = request.context["gates"]
    assert isinstance(gates, dict)
    assert "S03.completion" in gates
    assert gates["S03.completion"]["status"] in {"allowed", "blocked"}
    assert request.metadata["prompt_hash"]

    assert len(executor.calls) == 1
    assert len(executor.calls[0]) == 1
    assert executor.calls[0][0]["type"] == "overlay"
    assert executor.calls[0][0]["target"] == "apu_switch"
    assert executor.calls[0][0]["element_id"] == _apu_element_id_from_ui_map()

    kinds = [event.kind for event in events]
    assert "observation" in kinds
    assert "tutor_request" in kinds
    assert "tutor_response" in kinds
    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    assert isinstance(tutor_response_payload["metadata"].get("requires_visual_confirmation"), bool)


def test_live_loop_records_grounding_snippet_ids_when_index_available(tmp_path: Path) -> None:
    replay_path = tmp_path / "bios_grounding.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0, apu_switch=0)])

    doc = tmp_path / "grounding.md"
    doc.write_text(
        "# S03\nF/A-18C Cold Start MVP subset checklist.\nAPU switch ON and wait for APU READY.\n",
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
    assert req_meta["grounding_snippet_ids"]
    rag_topk = tutor_request_payload["context"]["rag_topk"]
    assert rag_topk
    assert rag_topk[0]["snippet_id"]
    assert rag_topk[0]["doc_id"] == "grounding"

    tutor_response_payload = next(event.payload for event in events if event.kind == "tutor_response")
    prompt_build = tutor_response_payload["metadata"]["prompt_build"]
    assert prompt_build["grounding_missing"] is False
    assert prompt_build["rag_snippet_ids"] == req_meta["grounding_snippet_ids"]


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
    assert isinstance(tutor_request_payload["context"]["grounding_query"], str)
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
    assert req_meta["source_chunk_refs"] == ["fa18c_startup_master/fa18c_startup_master_1:1-28"]


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
    assert req_meta["source_chunk_refs"] == ["fa18c_startup_master/fa18c_startup_master_1:1-28"]


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
    assert req_meta["source_chunk_refs"] == ["fa18c_startup_master/fa18c_startup_master_1:1-28"]


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
    assert req_meta["grounding_snippet_ids"] == ["manual_s03_1"]
    assert req_meta["grounding_index_path"] is None
    assert tutor_request_payload["context"]["grounding_reason"] is None
    assert isinstance(tutor_request_payload["context"]["grounding_query"], str)
    assert tutor_request_payload["context"]["rag_topk"][0]["snippet_id"] == "manual_s03_1"


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
    assert set(first.keys()) <= {"doc_id", "section", "page_or_heading", "snippet", "snippet_id", "score"}
    assert isinstance(first["doc_id"], str)
    assert isinstance(first["section"], str)
    assert isinstance(first["page_or_heading"], str)
    assert isinstance(first["snippet"], str)
    assert first["snippet_id"] == "snippet_0"
    assert req_meta["grounding_snippet_ids"] == ["snippet_0"]
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
    assert req_meta["grounding_index_path"] == "meta://store"
    assert req_meta["grounding_snippet_ids"] == ["meta_s03_1"]


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
    assert set(first.keys()) <= {"doc_id", "section", "page_or_heading", "snippet", "snippet_id", "score"}
    assert first["snippet_id"] == "snippet_0"
    assert isinstance(first["doc_id"], str)
    assert isinstance(first["section"], str)
    assert isinstance(first["page_or_heading"], str)
    assert isinstance(first["snippet"], str)
    assert isinstance(req_meta["grounding_index_path"], str)
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
    assert executor.calls[0] == []
    tutor_response_payloads = [event.payload for event in events if event.kind == "tutor_response"]
    assert len(tutor_response_payloads) == 1
    assert tutor_response_payloads[0]["metadata"]["failure_code"] == ALLOWLIST_FAIL
    response_mapping = tutor_response_payloads[0]["metadata"]["response_mapping"]
    assert response_mapping["rejected_targets_by_request_allowlist"] == ["battery_switch"]
    assert "overlay_target_not_in_request_allowlist" in response_mapping["mapping_errors"]
    rejected_payloads = [event.payload for event in events if event.kind == "overlay_rejected"]
    assert len(rejected_payloads) == 1
    assert rejected_payloads[0]["failure_code"] == ALLOWLIST_FAIL


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
                        "diagnosis": {"step_id": "S02", "error_category": "OM"},
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
    assert tutor_response_payload["metadata"]["failure_code"] == EVIDENCE_FAIL
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


def test_udp_help_trigger_receives_help_datagram() -> None:
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
        "  - id: S02\n"
        "    observability: unknown\n"
        "    evidence_requirements: [visual, rag]\n",
        encoding="utf-8",
    )

    profiles = _load_step_signal_profiles(pack)
    assert profiles["S01"]["observability"] == "observable"
    assert profiles["S01"]["observability_status"] == "observable"
    assert profiles["S01"]["evidence_requirements"] == ["var", "gate"]
    assert profiles["S01"]["requires_visual_confirmation"] is False
    assert profiles["S02"]["observability"] == "unobservable"
    assert profiles["S02"]["observability_status"] == "unobservable"
    assert profiles["S02"]["evidence_requirements"] == ["visual", "rag"]
    assert profiles["S02"]["requires_visual_confirmation"] is True


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
    assert gate_blockers, "expected at least one inferred gate blocker"
    inferred_step_id = hint.get("inferred_step_id")
    assert isinstance(inferred_step_id, str) and inferred_step_id
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


def test_live_dcs_cli_scenario_profile_defaults_airfield_and_accepts_carrier() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.scenario_profile == "airfield"

    args = parser.parse_args(["--scenario-profile", "carrier"])
    assert args.scenario_profile == "carrier"


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
