"""
Application-level help-cycle orchestration ports.

The concrete live DCS loop wires these ports to adapters. Tests can wire fake
ports, which keeps harness orchestration verifiable without DCS, model servers,
or overlay transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from core.types import Observation, TutorRequest, TutorResponse


@dataclass(frozen=True)
class HelpCycleRequestBundle:
    request: TutorRequest
    prompt_metadata: Mapping[str, Any] = field(default_factory=dict)
    state_key: str | None = None


@dataclass(frozen=True)
class HelpCycleDecisionResult:
    response: TutorResponse
    fallback_overlay_used: bool = False
    fallback_overlay_reason: str = "not_needed"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HelpCycleRunResult:
    observation: Observation
    request: TutorRequest
    response: TutorResponse
    action_report: Mapping[str, Any]
    vision_context: Mapping[str, Any]
    active_step_ids: tuple[str, ...]
    prompt_metadata: Mapping[str, Any]
    state_key: str | None
    help_cycle_id: str


@dataclass(frozen=True)
class PreparedHelpCycle:
    observation: Observation
    request: TutorRequest
    prompt_metadata: Mapping[str, Any] = field(default_factory=dict)
    state_key: str | None = None
    vision_context: Mapping[str, Any] = field(default_factory=dict)
    active_step_ids: Sequence[str] = ()
    help_cycle_id: str | None = None


class TelemetryEvidenceInputPort(Protocol):
    def current_observation(self) -> Observation | None:
        ...


class VisionFactExtractionPort(Protocol):
    def active_step_ids(self, observation: Observation) -> Sequence[str]:
        ...

    def extract(
        self,
        observation: Observation,
        *,
        help_cycle_id: str,
        active_step_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        ...


class CandidateGenerationPort(Protocol):
    def build_request(
        self,
        observation: Observation,
        *,
        vision_context: Mapping[str, Any],
        help_cycle_id: str,
    ) -> HelpCycleRequestBundle:
        ...


class LlmAdjudicationPort(Protocol):
    def adjudicate(self, observation: Observation, request: TutorRequest) -> TutorResponse:
        ...


class DecisionValidationRepairPort(Protocol):
    def validate_and_repair(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> HelpCycleDecisionResult:
        ...


class ActionPlanningPort(Protocol):
    def execute(self, actions: Sequence[Mapping[str, Any] | Any]) -> Mapping[str, Any]:
        ...


class TraceAuditSinkPort(Protocol):
    def record(self, stage: str, payload: Mapping[str, Any]) -> None:
        ...


class NoopTraceAuditSink:
    def record(self, stage: str, payload: Mapping[str, Any]) -> None:
        return None


def _string_tuple(raw: Sequence[str] | None) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple, set)):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _request_id(request: TutorRequest) -> str | None:
    request_id = getattr(request, "request_id", None)
    return request_id if isinstance(request_id, str) and request_id else None


class HelpCycleOrchestrator:
    def __init__(
        self,
        *,
        telemetry: TelemetryEvidenceInputPort | None = None,
        vision: VisionFactExtractionPort | None = None,
        candidates: CandidateGenerationPort | None = None,
        llm: LlmAdjudicationPort,
        validator: DecisionValidationRepairPort,
        actions: ActionPlanningPort,
        audit: TraceAuditSinkPort | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.telemetry = telemetry
        self.vision = vision
        self.candidates = candidates
        self.llm = llm
        self.validator = validator
        self.actions = actions
        self.audit = audit if audit is not None else NoopTraceAuditSink()
        self.id_factory = id_factory if id_factory is not None else lambda: str(uuid4())

    def run(self) -> HelpCycleRunResult | None:
        if self.telemetry is None or self.vision is None or self.candidates is None:
            raise ValueError("telemetry, vision, and candidates ports are required for run()")

        observation = self.telemetry.current_observation()
        if observation is None:
            return None

        help_cycle_id = self.id_factory()
        self.audit.record("telemetry", {"observation_id": observation.observation_id})

        active_step_ids = _string_tuple(self.vision.active_step_ids(observation))
        vision_context = self.vision.extract(
            observation,
            help_cycle_id=help_cycle_id,
            active_step_ids=active_step_ids,
        )
        self.audit.record(
            "vision",
            {
                "help_cycle_id": help_cycle_id,
                "active_step_ids": list(active_step_ids),
                "status": vision_context.get("status"),
            },
        )

        bundle = self.candidates.build_request(
            observation,
            vision_context=vision_context,
            help_cycle_id=help_cycle_id,
        )
        self.audit.record(
            "request",
            {
                "help_cycle_id": help_cycle_id,
                "request_id": _request_id(bundle.request),
                "state_key": bundle.state_key,
            },
        )
        return self.run_prepared(
            PreparedHelpCycle(
                observation=observation,
                request=bundle.request,
                prompt_metadata=bundle.prompt_metadata,
                state_key=bundle.state_key,
                vision_context=vision_context,
                active_step_ids=active_step_ids,
                help_cycle_id=help_cycle_id,
            )
        )

    def run_prepared(self, prepared: PreparedHelpCycle) -> HelpCycleRunResult:
        help_cycle_id = prepared.help_cycle_id or _request_id(prepared.request) or self.id_factory()
        response = self.llm.adjudicate(prepared.observation, prepared.request)
        self.audit.record(
            "llm_adjudication",
            {
                "help_cycle_id": help_cycle_id,
                "request_id": _request_id(prepared.request),
                "status": response.status,
            },
        )

        decision = self.validator.validate_and_repair(response, prepared.request)
        decision.response.metadata = dict(decision.response.metadata)
        decision.response.metadata["fallback_overlay_used"] = bool(decision.fallback_overlay_used)
        decision.response.metadata["fallback_overlay_reason"] = decision.fallback_overlay_reason
        for key, value in decision.metadata.items():
            if isinstance(key, str) and key:
                decision.response.metadata[key] = value
        self.audit.record(
            "decision_validation",
            {
                "help_cycle_id": help_cycle_id,
                "fallback_overlay_used": decision.fallback_overlay_used,
                "fallback_overlay_reason": decision.fallback_overlay_reason,
            },
        )

        action_report = self.actions.execute(decision.response.actions)
        self.audit.record(
            "action_planning",
            {
                "help_cycle_id": help_cycle_id,
                "action_count": len(decision.response.actions),
            },
        )

        return HelpCycleRunResult(
            observation=prepared.observation,
            request=prepared.request,
            response=decision.response,
            action_report=action_report,
            vision_context=dict(prepared.vision_context),
            active_step_ids=_string_tuple(prepared.active_step_ids),
            prompt_metadata=dict(prepared.prompt_metadata),
            state_key=prepared.state_key,
            help_cycle_id=help_cycle_id,
        )


__all__ = [
    "ActionPlanningPort",
    "CandidateGenerationPort",
    "DecisionValidationRepairPort",
    "HelpCycleDecisionResult",
    "HelpCycleOrchestrator",
    "HelpCycleRequestBundle",
    "HelpCycleRunResult",
    "LlmAdjudicationPort",
    "PreparedHelpCycle",
    "TelemetryEvidenceInputPort",
    "TraceAuditSinkPort",
    "VisionFactExtractionPort",
]
