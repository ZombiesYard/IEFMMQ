from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.help_orchestrator import (
    HelpCycleDecisionResult,
    HelpCycleOrchestrator,
    HelpCycleRequestBundle,
)
from core.types import Observation, TutorRequest, TutorResponse


class FakeTelemetryPort:
    def __init__(self) -> None:
        self.observation = Observation(
            observation_id="obs-1",
            source="fake-telemetry",
            payload={"vars": {"battery_on": True}},
        )

    def current_observation(self) -> Observation | None:
        return self.observation


class FakeVisionPort:
    def __init__(self, active_step_ids: Sequence[str]) -> None:
        self.active_step_ids_seen: list[tuple[str, ...]] = []
        self._active_step_ids = tuple(active_step_ids)

    def active_step_ids(self, observation: Observation) -> Sequence[str]:
        assert observation.observation_id == "obs-1"
        return self._active_step_ids

    def extract(
        self,
        observation: Observation,
        *,
        help_cycle_id: str,
        active_step_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        assert help_cycle_id
        self.active_step_ids_seen.append(tuple(active_step_ids))
        return {
            "status": "available",
            "vision_facts": [{"fact_id": "tac_page_visible", "state": "seen"}],
            "vision_fact_summary": {
                "status": "available",
                "seen_fact_ids": ["tac_page_visible"],
                "fresh_fact_ids": ["tac_page_visible"],
            },
        }


class FakeCandidatePort:
    def __init__(self) -> None:
        self.vision_context_seen: Mapping[str, Any] | None = None

    def build_request(
        self,
        observation: Observation,
        *,
        vision_context: Mapping[str, Any],
        help_cycle_id: str,
    ) -> HelpCycleRequestBundle:
        self.vision_context_seen = vision_context
        request = TutorRequest(
            request_id=help_cycle_id,
            actor="learner",
            intent="help",
            message="help",
            observation_ref=observation.observation_id,
            context={
                "candidate_steps": [{"step_id": "S08", "source": "visual_anchor"}],
                "vision_fact_summary": dict(vision_context["vision_fact_summary"]),
                "deterministic_step_hint": {"inferred_step_id": "S08", "overlay_step_id": "S08"},
            },
            metadata={"prompt_hash": "abc"},
        )
        return HelpCycleRequestBundle(
            request=request,
            prompt_metadata={"prompt_tokens_est": 12},
            state_key="state-1",
        )


class FakeLlmPort:
    def __init__(self) -> None:
        self.requests: list[TutorRequest] = []

    def adjudicate(self, observation: Observation, request: TutorRequest) -> TutorResponse:
        self.requests.append(request)
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id,
            message="Press PB.",
            actions=[{"type": "overlay", "target": "left_mdi_pb18"}],
            metadata={"provider": "fake-llm", "help_response": {"next": {"step_id": "S08"}}},
        )


class FakeValidatorPort:
    def __init__(self) -> None:
        self.responses_seen: list[TutorResponse] = []

    def validate_and_repair(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> HelpCycleDecisionResult:
        self.responses_seen.append(response)
        response.metadata["validator_rejected"] = False
        response.metadata["final_action_plan_source"] = "model"
        return HelpCycleDecisionResult(
            response=response,
            fallback_overlay_used=False,
            fallback_overlay_reason="not_needed",
        )


class FakeOverlayPort:
    def __init__(self) -> None:
        self.actions_seen: list[Mapping[str, Any]] = []

    def execute(self, actions: Sequence[Mapping[str, Any] | Any]) -> Mapping[str, Any]:
        self.actions_seen = [dict(action) for action in actions if isinstance(action, Mapping)]
        return {"executed": list(self.actions_seen)}


class FakeAuditSink:
    def __init__(self) -> None:
        self.records: list[tuple[str, Mapping[str, Any]]] = []

    def record(self, stage: str, payload: Mapping[str, Any]) -> None:
        self.records.append((stage, payload))


def test_help_cycle_orchestrator_runs_with_fake_ports_only() -> None:
    telemetry = FakeTelemetryPort()
    vision = FakeVisionPort(active_step_ids=["S08"])
    candidates = FakeCandidatePort()
    llm = FakeLlmPort()
    validator = FakeValidatorPort()
    overlay = FakeOverlayPort()
    audit = FakeAuditSink()

    result = HelpCycleOrchestrator(
        telemetry=telemetry,
        vision=vision,
        candidates=candidates,
        llm=llm,
        validator=validator,
        actions=overlay,
        audit=audit,
        id_factory=lambda: "cycle-1",
    ).run()

    assert result is not None
    assert result.request.request_id == "cycle-1"
    assert result.response.metadata["final_action_plan_source"] == "model"
    assert result.action_report["executed"] == [{"type": "overlay", "target": "left_mdi_pb18"}]
    assert llm.requests == [result.request]
    assert validator.responses_seen == [result.response]
    assert candidates.vision_context_seen is not None
    assert [stage for stage, _ in audit.records] == [
        "telemetry",
        "vision",
        "request",
        "llm_adjudication",
        "decision_validation",
        "action_planning",
    ]


def test_help_cycle_orchestrator_passes_active_steps_to_vision_port() -> None:
    vision = FakeVisionPort(active_step_ids=["S08", "S09"])

    result = HelpCycleOrchestrator(
        telemetry=FakeTelemetryPort(),
        vision=vision,
        candidates=FakeCandidatePort(),
        llm=FakeLlmPort(),
        validator=FakeValidatorPort(),
        actions=FakeOverlayPort(),
        audit=FakeAuditSink(),
        id_factory=lambda: "cycle-visual",
    ).run()

    assert result is not None
    assert vision.active_step_ids_seen == [("S08", "S09")]
