from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from adapters.dcs.overlay.sender import DcsOverlaySender
from core.overlay import OverlayPlanner
from core.types import Event


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_ui_map_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "ui_map.yaml"


def _load_pack_ui_targets(pack_path: Path) -> set[str] | None:
    data = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"pack must be a mapping: {pack_path}")
    ui_targets = data.get("ui_targets")
    if ui_targets is None:
        return None
    if not isinstance(ui_targets, list):
        raise ValueError(f"pack.ui_targets must be a list: {pack_path}")
    out: set[str] = set()
    for item in ui_targets:
        if not isinstance(item, str) or not item:
            raise ValueError(f"pack.ui_targets must contain non-empty strings: {pack_path}")
        out.add(item)
    return out


@dataclass
class ActionExecutionReport:
    executed: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    dry_run: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": list(self.executed),
            "rejected": list(self.rejected),
            "dropped": list(self.dropped),
            "dry_run": list(self.dry_run),
        }


class OverlayActionExecutor:
    """
    Execute tutor actions with overlay-only safety enforcement.

    Notes:
    - Only `type="overlay"` actions are executable.
    - Target allowlist is enforced from `ui_map` (optionally narrowed by `pack.ui_targets`).
    - `pulse_enabled=True` uses a blocking sleep; `execute_actions` blocks the calling thread
      for `ttl_s` per executed target before issuing `clear`.
    """

    def __init__(
        self,
        *,
        sender: DcsOverlaySender | None = None,
        ui_map_path: str | Path | None = None,
        pack_path: str | Path | None = None,
        max_targets: int = 1,
        ttl_s: float = 2.0,
        pulse_enabled: bool = False,
        dry_run: bool = False,
        expect_ack: bool = True,
        session_id: str | None = None,
        event_sink: Callable[[Event], None] | None = None,
    ) -> None:
        resolved_ui_map = Path(ui_map_path) if ui_map_path else _default_ui_map_path()
        self._planner = OverlayPlanner(str(resolved_ui_map))
        base_allowlist = set(self._planner.elements.keys())
        if pack_path is not None:
            pack_allowlist = _load_pack_ui_targets(Path(pack_path))
            if pack_allowlist is not None:
                base_allowlist = base_allowlist.intersection(pack_allowlist)
        self._allowlist = base_allowlist

        self.max_targets = max(0, max_targets)
        self.ttl_s = max(0.0, float(ttl_s))
        self.pulse_enabled = pulse_enabled
        self.dry_run = dry_run
        self.expect_ack = expect_ack
        self.session_id = session_id
        self.event_sink = event_sink

        if sender is None:
            self._sender = DcsOverlaySender(
                session_id=session_id,
                event_sink=event_sink,
            )
            self._owns_sender = True
        else:
            self._sender = sender
            self._owns_sender = False

    def close(self) -> None:
        if self._owns_sender:
            self._sender.close()

    def __enter__(self) -> "OverlayActionExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if not self.event_sink:
            return
        self.event_sink(Event(kind=kind, payload=dict(payload), t_wall=time.time(), session_id=self.session_id))

    def _sender_emits_to_executor_sink(self) -> bool:
        sender_sink = getattr(self._sender, "event_sink", None)
        return bool(self.event_sink) and sender_sink is self.event_sink

    def _reject(self, report: ActionExecutionReport, *, reason: str, action_idx: int, action: Any) -> None:
        detail = {
            "reason": reason,
            "action_index": action_idx,
            "action": action if isinstance(action, Mapping) else {"raw": str(action)},
        }
        report.rejected.append(detail)
        self._emit("overlay_failed", detail)

    def execute_actions(self, actions: Sequence[Mapping[str, Any] | Any]) -> ActionExecutionReport:
        report = ActionExecutionReport()
        executed_count = 0

        for idx, action in enumerate(actions):
            if not isinstance(action, Mapping):
                self._reject(report, reason="invalid_action_payload", action_idx=idx, action=action)
                continue

            action_type = action.get("type")
            if action_type != "overlay":
                self._reject(report, reason="rejected_non_overlay_action", action_idx=idx, action=action)
                continue

            target = action.get("target")
            if not isinstance(target, str) or not target:
                self._reject(report, reason="invalid_overlay_target", action_idx=idx, action=action)
                continue

            if target not in self._allowlist:
                self._reject(report, reason="overlay_target_not_in_allowlist", action_idx=idx, action=action)
                continue

            if executed_count >= self.max_targets:
                drop_detail = {
                    "reason": "max_targets_exceeded",
                    "action_index": idx,
                    "target": target,
                }
                report.dropped.append(drop_detail)
                continue

            try:
                intent = self._planner.plan(target, intent="highlight")
            except (KeyError, ValueError):
                self._reject(report, reason="overlay_target_unmappable", action_idx=idx, action=action)
                continue
            if self.dry_run:
                preview = {
                    "target": target,
                    "element_id": intent.element_id,
                    "intent": "highlight",
                    "ttl_s": self.ttl_s,
                    "pulse_enabled": self.pulse_enabled,
                }
                report.dry_run.append(preview)
                self._emit("overlay_dry_run", preview)
                executed_count += 1
                continue

            ack = self._sender.send_intent(intent, expect_ack=self.expect_ack)
            if not self._sender_emits_to_executor_sink():
                requested = {
                    "action": "highlight",
                    "target": intent.element_id,
                    "target_name": target,
                }
                self._emit("overlay_requested", requested)
                if isinstance(ack, Mapping):
                    ack_payload = dict(ack)
                    ack_payload["intent"] = "highlight"
                    ack_payload["target"] = intent.element_id
                    kind = "overlay_applied" if ack.get("status") == "ok" else "overlay_failed"
                    self._emit(kind, ack_payload)
            applied = {
                "target": target,
                "element_id": intent.element_id,
                "intent": "highlight",
                "ack": ack,
            }
            report.executed.append(applied)
            executed_count += 1

            if self.pulse_enabled and self.ttl_s > 0:
                time.sleep(self.ttl_s)
                clear_intent = self._planner.plan(target, intent="clear")
                self._sender.send_intent(clear_intent, expect_ack=False)

        return report


def execute_overlay_actions(
    actions: Sequence[Mapping[str, Any] | Any],
    *,
    sender: DcsOverlaySender | None = None,
    ui_map_path: str | Path | None = None,
    pack_path: str | Path | None = None,
    max_targets: int = 1,
    ttl_s: float = 2.0,
    pulse_enabled: bool = False,
    dry_run: bool = False,
    expect_ack: bool = True,
    session_id: str | None = None,
    event_sink: Callable[[Event], None] | None = None,
) -> dict[str, Any]:
    with OverlayActionExecutor(
        sender=sender,
        ui_map_path=ui_map_path,
        pack_path=pack_path,
        max_targets=max_targets,
        ttl_s=ttl_s,
        pulse_enabled=pulse_enabled,
        dry_run=dry_run,
        expect_ack=expect_ack,
        session_id=session_id,
        event_sink=event_sink,
    ) as executor:
        return executor.execute_actions(actions).to_dict()


__all__ = ["ActionExecutionReport", "OverlayActionExecutor", "execute_overlay_actions"]
