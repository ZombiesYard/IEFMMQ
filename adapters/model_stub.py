"""
Deterministic model stub for controlled experiments.

Modes:
 - A: rule card (short template-driven guidance)
 - B: llm stub (static phrasing)
 - C: rag+stub (adds a fake retrieval note)
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import TutorResponse, Observation, TutorRequest
from ports.model_port import ModelPort


@dataclass
class ModelStub(ModelPort):
    mode: str = "A"

    def plan_next_step(self, observation: Observation, request: TutorRequest | None = None) -> TutorResponse:
        message, card = self._card_for(observation, request)
        return TutorResponse(message=message, actions=[], metadata={"card": card})

    def explain_error(self, observation: Observation, request: TutorRequest | None = None) -> TutorResponse:
        message, card = self._card_for(observation, request, explain=True)
        return TutorResponse(message=message, actions=[], metadata={"card": card})

    def _card_for(self, observation: Observation, request: TutorRequest | None, explain: bool = False):
        hint = observation.procedure_hint or "unknown"
        base = f"Next step: {hint}" if not explain else f"Error noted at {hint}"

        if self.mode == "A":
            card = f"[RULE] {base}"
            message = f"{base}. Follow checklist."
        elif self.mode == "B":
            card = f"[LLM-STUB] {base}"
            message = f"{base}. (stubbed llm guidance)"
        elif self.mode == "C":
            card = f"[RAG+STUB] {base} | ref: fa18c_startup_master"
            message = f"{base}. Retrieved canonical snippet."
        else:
            card = f"[UNKNOWN] {base}"
            message = base
        return message, card

