"""
Procedure engine: simple step state machine.

States: pending -> active -> done/blocked, with ability to rewind a completed
step back to active. Only one step may be active at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    BLOCKED = "blocked"


@dataclass
class StepState:
    step_id: str
    status: StepStatus = StepStatus.PENDING
    activated_at: Optional[datetime] = None
    prompt_count: int = 0


class ProcedureEngine:
    def __init__(self, steps: Sequence[dict]):
        if not steps:
            raise ValueError("steps must not be empty")
        self._order = [s["id"] for s in steps]
        self._steps: Dict[str, StepState] = {sid: StepState(step_id=sid) for sid in self._order}
        self.events: List[dict] = []

    # --- helpers
    def _emit(self, event_type: str, step_id: str, reason: Optional[str] = None) -> None:
        evt = {"type": event_type, "step_id": step_id, "timestamp": _ts()}
        if reason:
            evt["reason"] = reason
        self.events.append(evt)

    def _active_id(self) -> Optional[str]:
        for sid, st in self._steps.items():
            if st.status == StepStatus.ACTIVE:
                return sid
        return None

    def _last_done(self) -> Optional[str]:
        for sid in reversed(self._order):
            if self._steps[sid].status == StepStatus.DONE:
                return sid
        return None

    # --- public API
    def activate_next(self) -> StepState:
        active = self._active_id()
        if active:
            return self._steps[active]
        for sid in self._order:
            st = self._steps[sid]
            if st.status == StepStatus.PENDING:
                st.status = StepStatus.ACTIVE
                st.activated_at = datetime.now(timezone.utc)
                self._emit("step_activated", sid)
                return st
        raise RuntimeError("No pending steps to activate")

    def complete_active(self) -> StepState:
        active = self._active_id()
        if not active:
            raise RuntimeError("No active step to complete")
        st = self._steps[active]
        st.status = StepStatus.DONE
        self._emit("step_completed", st.step_id)
        return st

    def block_active(self, reason: str) -> StepState:
        active = self._active_id()
        if not active:
            raise RuntimeError("No active step to block")
        st = self._steps[active]
        st.status = StepStatus.BLOCKED
        self._emit("step_blocked", st.step_id, reason=reason)
        return st

    def rewind(self) -> StepState:
        active = self._active_id()
        if active:
            # put current active back to pending before rewinding
            self._steps[active].status = StepStatus.PENDING
        last_done = self._last_done()
        if not last_done:
            raise RuntimeError("No completed step to rewind to")
        st = self._steps[last_done]
        st.status = StepStatus.ACTIVE
        st.activated_at = datetime.now(timezone.utc)
        self._emit("step_activated", st.step_id)
        return st

    def request_prompt(self, step_id: Optional[str] = None) -> int:
        """Increment prompt counter for the active (or provided) step."""
        sid = step_id or self._active_id()
        if not sid:
            raise RuntimeError("No active step for prompt request")
        st = self._steps[sid]
        st.prompt_count += 1
        return st.prompt_count

    def check_timeout(self, now: datetime, timeout_seconds: float) -> Optional[StepState]:
        active = self._active_id()
        if not active:
            return None
        st = self._steps[active]
        if not st.activated_at:
            return None
        elapsed = (now - st.activated_at).total_seconds()
        if elapsed > timeout_seconds:
            st.status = StepStatus.BLOCKED
            self._emit("step_blocked", st.step_id, reason="timeout")
            return st
        return None

    # inspection helpers
    def status(self, step_id: str) -> StepStatus:
        return self._steps[step_id].status

    def active_step(self) -> Optional[str]:
        return self._active_id()

