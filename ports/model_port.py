"""
Model port defines interfaces for tutor reasoning (LLM/VLM/RAG).
"""

from __future__ import annotations

from typing import Protocol, List, Dict, Any

from core.types import TutorResponse, TutorRequest, Observation


class ModelPort(Protocol):
    def plan_next_step(self, observation: Observation, request: TutorRequest | None = None) -> TutorResponse:
        ...

    def explain_error(self, observation: Observation, request: TutorRequest | None = None) -> TutorResponse:
        ...

