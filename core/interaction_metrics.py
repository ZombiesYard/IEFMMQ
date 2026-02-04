"""
Interaction telemetry metrics derived from event logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class InteractionMetrics:
    stall_time_total_s: float = 0.0
    stall_time_max_s: float = 0.0
    help_requests: int = 0
    llm_triggers: int = 0
    ui_clicks: int = 0
    ui_click_failures: int = 0
    highlight_requests: int = 0
    highlight_acks: int = 0
    clear_requests: int = 0
    clear_acks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stall_time_total_s": self.stall_time_total_s,
            "stall_time_max_s": self.stall_time_max_s,
            "help_requests": self.help_requests,
            "llm_triggers": self.llm_triggers,
            "ui_clicks": self.ui_clicks,
            "ui_click_failures": self.ui_click_failures,
            "highlight_requests": self.highlight_requests,
            "highlight_acks": self.highlight_acks,
            "clear_requests": self.clear_requests,
            "clear_acks": self.clear_acks,
        }


def _event_kind(ev: Dict[str, Any]) -> str:
    return ev.get("kind") or ev.get("type") or ""


def compute_interaction_metrics(events: Iterable[Dict[str, Any]]) -> InteractionMetrics:
    metrics = InteractionMetrics()
    step_start: Dict[str, datetime] = {}
    stalls: List[float] = []

    for ev in events:
        kind = _event_kind(ev)
        payload = ev.get("payload") or {}

        if kind == "step_activated":
            sid = payload.get("step_id")
            ts = _parse_time(payload.get("timestamp")) or _parse_time(ev.get("timestamp"))
            if sid and ts:
                step_start[sid] = ts
        elif kind in ("step_completed", "step_blocked"):
            sid = payload.get("step_id")
            ts = _parse_time(payload.get("timestamp")) or _parse_time(ev.get("timestamp"))
            if sid and ts and sid in step_start:
                delta = (ts - step_start[sid]).total_seconds()
                if delta >= 0:
                    stalls.append(delta)
                step_start.pop(sid, None)

        if kind == "tutor_request":
            intent = (payload.get("intent") or "").lower()
            if "help" in intent or "hint" in intent:
                metrics.help_requests += 1

        if kind == "tutor_response":
            meta = payload.get("metadata") or {}
            if meta.get("model") or meta.get("llm"):
                metrics.llm_triggers += 1

        if kind == "ui.user_clicked":
            metrics.ui_clicks += 1
            if payload.get("success") is False:
                metrics.ui_click_failures += 1
        elif kind == "ui.highlight_requested":
            metrics.highlight_requests += 1
        elif kind == "ui.highlight_ack":
            metrics.highlight_acks += 1
        elif kind == "ui.clear_requested":
            metrics.clear_requests += 1
        elif kind == "ui.clear_ack":
            metrics.clear_acks += 1

    if stalls:
        metrics.stall_time_total_s = sum(stalls)
        metrics.stall_time_max_s = max(stalls)
    return metrics


__all__ = ["InteractionMetrics", "compute_interaction_metrics"]
