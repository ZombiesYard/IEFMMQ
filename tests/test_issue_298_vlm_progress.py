from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import live_dcs as live_dcs_module
import pytest
from adapters.step_inference import StepInferenceResult
from core.types import Observation, TutorRequest, TutorResponse
from core.types_v2 import VisionObservation
from live_dcs import LiveDcsTutorLoop, ReplayBiosReceiver


class _NoopModel:
    def explain_error(self, obs, request):  # pragma: no cover
        raise AssertionError("model is not used by these progress-state tests")


class _NoopExecutor:
    def execute_actions(self, actions):  # pragma: no cover
        return {"executed": [], "failed": [], "dry_run": []}


class _RecordingModel:
    def __init__(self) -> None:
        self.requests: list[TutorRequest] = []

    def explain_error(self, obs, request):
        self.requests.append(request)
        step_id = request.context["deterministic_step_hint"]["inferred_step_id"]
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id,
            message=f"Proceed with {step_id}.",
            actions=[],
            metadata={},
        )


def _write_replay(path: Path, frames: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(frame) for frame in frames), encoding="utf-8")


def _bios_frame(seq: int, t_wall: float) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "seq": seq,
        "t_wall": t_wall,
        "aircraft": "FA-18C_hornet",
        "bios": {"BATTERY_SW": 2, "L_GEN_SW": 1, "R_GEN_SW": 1},
        "delta": {},
    }


def _issue_298_a7a6bf18_payload() -> dict[str, Any]:
    return {
        "seq": 11105,
        "t_wall": 1779220674.6965468,
        "vars": {
            "battery_on": True,
            "power_available": True,
            "left_ddi_on": True,
            "fcs_reset_complete": True,
            "flap_auto": False,
        },
    }


def _loop(tmp_path: Path, session_id: str) -> LiveDcsTutorLoop:
    replay_path = tmp_path / f"{session_id}.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0)])
    return LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=_NoopModel(),
        action_executor=_NoopExecutor(),
        session_id=session_id,
    )


def _request(step_id: str, missing_conditions: list[str]) -> TutorRequest:
    return TutorRequest(
        request_id=f"cycle-{step_id.lower()}",
        actor="learner",
        intent="help",
        message="help",
        observation_ref=f"obs-{step_id.lower()}",
        context={
            "vars": {},
            "gates": {},
            "recent_actions": {"recent_buttons": []},
            "deterministic_step_hint": {
                "inferred_step_id": step_id,
                "missing_conditions": list(missing_conditions),
            },
            "vision_facts": [],
        },
        metadata={},
    )


def _trusted_response(step_id: str) -> TutorResponse:
    return TutorResponse(
        status="ok",
        in_reply_to=f"cycle-{step_id.lower()}",
        message=f"Proceed with {step_id}.",
        actions=[],
        metadata={
            "harness_action_plan": {
                "step_id": step_id,
                "overlay_step_id": step_id,
                "targets": [],
                "text_only": True,
                "source": "final_evidence_consistency_validator",
            }
        },
    )


def test_issue_298_keeps_s08_active_when_page_navigation_unconfirmed(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-s08-page-unconfirmed")
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)

        active = loop._active_step_ids_for_vision_facts(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
        )
    finally:
        loop.close()

    assert active == ["S08"]


def test_issue_298_remembers_final_response_progress_for_next_vision_gate(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-final-progress")
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)

        updated = loop._remember_final_response_progress(
            _trusted_response("S09"),
            _request("S08", ["vision_facts.fcs_page_visible==seen"]),
        )
        active = loop._active_step_ids_for_vision_facts(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
        )
    finally:
        loop.close()

    assert updated is True
    assert loop._last_inferred_step_id == "S09"
    assert loop._sticky_inference_step_id == "S09"
    assert active == ["S09"]


def test_issue_298_preliminary_gate_uses_stabilized_inference_before_vlm(tmp_path: Path, monkeypatch) -> None:
    replay_path = tmp_path / "issue-298-preliminary-stabilized.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0)])
    loop = LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=_NoopModel(),
        action_executor=_NoopExecutor(),
        session_id="issue-298-preliminary-stabilized",
    )
    calls: dict[str, Any] = {}

    def fake_infer_step_id(*args, **kwargs) -> StepInferenceResult:
        calls["vision_facts"] = kwargs.get("vision_facts")
        return StepInferenceResult("S19", ("vision_facts.fcsmc_final_go_result_visible==seen",))

    def fake_stabilize(
        inference: StepInferenceResult,
        vars_selected: dict[str, Any],
        *,
        recent_ui_targets: list[str],
    ) -> StepInferenceResult:
        calls["stabilized_from"] = inference.inferred_step_id
        calls["stabilized_recent_targets"] = list(recent_ui_targets)
        return StepInferenceResult("S20", ("vars.ext_refuel_probe_value>=60000",))

    monkeypatch.setattr(live_dcs_module, "infer_step_id", fake_infer_step_id)
    monkeypatch.setattr(loop, "_stabilize_live_inference", fake_stabilize)

    try:
        obs = loop.source.get_observation()
        assert obs is not None
        inference = loop._infer_preliminary_step_for_vision_facts(obs)
    finally:
        loop.close()

    assert calls["vision_facts"] is None
    assert calls["stabilized_from"] == "S19"
    assert inference.inferred_step_id == "S20"
    assert inference.missing_conditions == ("vars.ext_refuel_probe_value>=60000",)


@pytest.mark.parametrize(
    ("raw_step_id", "raw_missing", "gate_step_id", "request_step_id", "request_missing"),
    [
        (
            "S08",
            ("vision_facts.fcs_page_visible==seen",),
            "S09",
            "S09",
            ("vars.comm1_freq_134_000==true",),
        ),
        (
            "S15",
            ("vision_facts.fcs_page_visible==seen",),
            "S16",
            "S16",
            ("vars.flap_auto==true",),
        ),
        (
            "S19",
            ("vars.ext_refuel_probe_value>=60000",),
            "S19",
            "S20",
            ("vars.ext_refuel_probe_value>=60000",),
        ),
    ],
)
def test_issue_298_run_help_cycle_skips_extractor_for_stabilized_nonvisual_step(
    tmp_path: Path,
    monkeypatch,
    raw_step_id: str,
    raw_missing: tuple[str, ...],
    gate_step_id: str,
    request_step_id: str,
    request_missing: tuple[str, ...],
) -> None:
    replay_path = tmp_path / f"issue-298-stabilized-{request_step_id.lower()}-cycle.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0)])

    class StaticVisionPort:
        def start(self, session_id: str) -> None:
            assert session_id == f"issue-298-stabilized-{request_step_id.lower()}-cycle"

        def stop(self) -> None:
            return

        def poll(self) -> list[VisionObservation]:
            return [
                VisionObservation(
                    frame_id="10000_000001",
                    capture_wall_ms=10000,
                    frame_seq=1,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "10000_000001.png"),
                )
            ]

    class FailingVisionFactExtractor:
        def extract(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError(f"VLM extractor should not be called for stabilized {request_step_id}")

        def close(self) -> None:
            return

    def fake_infer_step_id(*args, **kwargs) -> StepInferenceResult:
        return StepInferenceResult(raw_step_id, raw_missing)

    stabilize_calls = 0

    def fake_stabilize(
        inference: StepInferenceResult,
        vars_selected: dict[str, Any],
        *,
        recent_ui_targets: list[str],
    ) -> StepInferenceResult:
        nonlocal stabilize_calls
        stabilize_calls += 1
        if stabilize_calls == 1:
            return StepInferenceResult(gate_step_id, raw_missing if gate_step_id == raw_step_id else request_missing)
        return StepInferenceResult(request_step_id, request_missing)

    source = ReplayBiosReceiver(replay_path, speed=0.0)
    model = _RecordingModel()
    loop = LiveDcsTutorLoop(
        source=source,
        model=model,
        action_executor=_NoopExecutor(),
        session_id=f"issue-298-stabilized-{request_step_id.lower()}-cycle",
        vision_port=StaticVisionPort(),
        vision_session_id=f"issue-298-stabilized-{request_step_id.lower()}-cycle",
        vision_mode="replay",
        vision_fact_extractor=FailingVisionFactExtractor(),
    )
    monkeypatch.setattr(live_dcs_module, "infer_step_id", fake_infer_step_id)
    monkeypatch.setattr(loop, "_stabilize_live_inference", fake_stabilize)
    stale_visual_step_id = "S19" if request_step_id == "S20" else raw_step_id
    loop._last_inferred_step_id = stale_visual_step_id
    loop._sticky_inference_step_id = stale_visual_step_id
    loop._sticky_inference_missing_conditions = raw_missing
    loop._vision_fact_snapshot = {
        "fcs_page_visible": {
            "fact_id": "fcs_page_visible",
            "state": "seen",
            "sticky": False,
            "observed_at_wall_ms": 9000,
            "expires_at_wall_ms": 600000,
        },
        "fcsmc_final_go_result_visible": {
            "fact_id": "fcsmc_final_go_result_visible",
            "state": "not_seen",
            "sticky": True,
            "observed_at_wall_ms": 9000,
            "expires_at_wall_ms": 600000,
        },
    }

    try:
        obs = source.get_observation()
        assert obs is not None
        loop._ingest_observation(obs)
        response, _report = loop.run_help_cycle(trigger_t_wall=10.0)
    finally:
        loop.close()

    assert response is not None
    assert model.requests[0].metadata["vision_fact_status"] == "vision_not_required"
    assert model.requests[0].metadata["vision_fact_active_step_ids"] == [request_step_id]
    assert model.requests[0].metadata["vision_fact_extractor_used"] is False
    assert model.requests[0].context["vision_facts"] == []
    assert not any(
        item.get("source") in {"sticky_state", "visual_anchor"}
        for item in model.requests[0].context.get("candidate_steps", [])
        if isinstance(item, dict)
    )
    assert response.metadata["vlm_call_status"] == "not_required"
    assert response.metadata["harness_trace"]["vlm_call"]["extractor_called"] is False


def test_issue_298_a7a6bf18_s16_flap_ignores_sticky_s15_visual_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _issue_298_a7a6bf18_payload()

    class IdleSource:
        def get_observation(self):  # pragma: no cover
            return None

        def close(self) -> None:
            return

    class StaticVisionPort:
        def __init__(self) -> None:
            self._polled = False

        def start(self, session_id: str) -> None:
            assert session_id == "issue-298-a7a6bf18-s16"

        def stop(self) -> None:
            return

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return [
                VisionObservation(
                    frame_id="1779220674784_000406",
                    capture_wall_ms=1779220674784,
                    frame_seq=406,
                    channel="composite_panel",
                    layout_id="fa18c_composite_panel_v2",
                    image_uri=str(tmp_path / "1779220674784_000406.png"),
                )
            ]

    class FailingIfCalledVisionFactExtractor:
        def extract(self, vision, *, session_id: str | None, trigger_wall_ms: int):  # pragma: no cover
            del vision, session_id, trigger_wall_ms
            raise AssertionError("VLM extractor should not be called for issue #298 S16/flap")

        def close(self) -> None:
            return

    request_id = UUID("a7a6bf18-2fe7-42b6-abe9-c9a257117647")
    monkeypatch.setattr(live_dcs_module, "uuid4", lambda: request_id)

    model = _RecordingModel()
    loop = LiveDcsTutorLoop(
        source=IdleSource(),
        model=model,
        action_executor=_NoopExecutor(),
        session_id="issue-298-a7a6bf18-s16",
        vision_port=StaticVisionPort(),
        vision_session_id="issue-298-a7a6bf18-s16",
        vision_mode="replay",
        vision_fact_extractor=FailingIfCalledVisionFactExtractor(),
    )
    loop._last_inferred_step_id = "S15"
    loop._sticky_inference_step_id = "S15"
    loop._sticky_inference_missing_conditions = ()
    loop._vision_fact_snapshot = {
        "fcs_page_visible": {
            "fact_id": "fcs_page_visible",
            "state": "seen",
            "source_frame_id": "old-s15-frame",
            "sticky": False,
            "observed_at_wall_ms": 1779220670000,
            "expires_at_wall_ms": 1779221270000,
        },
        "fcsmc_final_go_result_visible": {
            "fact_id": "fcsmc_final_go_result_visible",
            "state": "not_seen",
            "source_frame_id": "old-s18-frame",
            "sticky": True,
            "observed_at_wall_ms": 1779220670000,
            "expires_at_wall_ms": 1779221270000,
        },
    }

    try:
        obs = Observation(source="synthetic_issue_298", payload=frame)
        loop._latest_raw_obs = obs
        loop._latest_enriched_obs = obs
        loop._accumulated_vars.update(frame["vars"])
        loop.telemetry_window_ring.add_delta(frame["vars"], t_wall=frame["t_wall"], seq=frame["seq"])
        response, _report = loop.run_help_cycle(trigger_t_wall=1779220674.7)
        stats = loop.stats.to_dict()
    finally:
        loop.close()

    assert response is not None
    assert model.requests[0].request_id == str(request_id)
    assert model.requests[0].metadata["vision_fact_status"] == "vision_not_required"
    assert model.requests[0].metadata["vision_fact_active_step_ids"] == ["S16"]
    assert model.requests[0].metadata["vision_fact_extractor_used"] is False
    assert model.requests[0].context["vision_facts"] == []
    assert response.metadata["vision_fact_extractor_used"] is False
    assert response.metadata["vlm_call_status"] == "not_required"
    assert response.metadata["harness_trace"]["vlm_call"]["extractor_called"] is False
    assert response.metadata["harness_trace"]["vlm_call"]["ignored_fact_count"] == 2
    assert stats["vision_cycles"] == 0


def test_issue_298_keeps_sticky_s15_visual_gate_until_fcs_reset_completes(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-s15-still-needs-reset")
    try:
        loop._sticky_inference_step_id = "S15"
        loop._sticky_inference_missing_conditions = ()

        inference = loop._stabilize_live_inference(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
            {
                "battery_on": True,
                "power_available": True,
                "left_ddi_on": True,
                "fcs_reset_complete": False,
                "flap_auto": False,
            },
            recent_ui_targets=[],
        )
    finally:
        loop.close()

    assert inference.inferred_step_id == "S15"


def test_issue_298_keeps_sticky_s15_when_visual_hold_is_unresolved(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-s15-visual-hold")
    try:
        loop._sticky_inference_step_id = "S15"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_x_marks_visible==seen",)

        inference = loop._stabilize_live_inference(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
            {
                "battery_on": True,
                "power_available": True,
                "left_ddi_on": True,
                "fcs_reset_complete": True,
                "flap_auto": False,
            },
            recent_ui_targets=[],
        )
    finally:
        loop.close()

    assert inference.inferred_step_id == "S15"
    assert inference.missing_conditions == ("vision_facts.fcs_page_x_marks_visible==seen",)


def test_issue_298_does_not_remember_spoofed_final_plan_only_progress(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-spoofed-final-plan-only")
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
        response = TutorResponse(
            status="ok",
            in_reply_to="cycle-spoofed-final-plan",
            message="Provider metadata claims a trusted source.",
            actions=[],
            metadata={
                "final_action_plan": {
                    "step_id": "S09",
                    "targets": [],
                    "text_only": True,
                    "source": "final_evidence_consistency_validator",
                }
            },
        )

        updated = loop._remember_final_response_progress(
            response,
            _request("S08", ["vision_facts.fcs_page_visible==seen"]),
        )
        active = loop._active_step_ids_for_vision_facts(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
        )
    finally:
        loop.close()

    assert updated is False
    assert loop._last_inferred_step_id == "S08"
    assert loop._sticky_inference_step_id == "S08"
    assert active == ["S08"]


def test_issue_298_does_not_remember_untrusted_next_as_progress(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-untrusted-next")
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
        response = TutorResponse(
            status="ok",
            in_reply_to="cycle-untrusted-next",
            message="Model says a later step.",
            actions=[],
            metadata={
                "next": {"step_id": "S20"},
                "final_action_plan": {
                    "step_id": "S20",
                    "targets": [],
                    "text_only": True,
                    "source": "model",
                },
            },
        )

        updated = loop._remember_final_response_progress(
            response,
            _request("S08", ["vision_facts.fcs_page_visible==seen"]),
        )
        active = loop._active_step_ids_for_vision_facts(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
        )
    finally:
        loop.close()

    assert updated is False
    assert loop._last_inferred_step_id == "S08"
    assert loop._sticky_inference_step_id == "S08"
    assert active == ["S08"]


def test_issue_298_remembers_visual_final_progress_as_visual_hold(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-visual-final-progress")
    try:
        loop._last_inferred_step_id = "S18"
        loop._sticky_inference_step_id = "S18"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcsmc_page_visible==seen",)

        updated = loop._remember_final_response_progress(
            _trusted_response("S19"),
            _request("S19", ["vision_facts.fcsmc_final_go_result_visible==seen"]),
        )
        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S17", ()))
    finally:
        loop.close()

    assert updated is True
    assert loop._sticky_inference_step_id == "S19"
    assert loop._sticky_inference_missing_conditions == ("vision_facts.fcsmc_final_go_result_visible==seen",)
    assert active == ["S19"]
