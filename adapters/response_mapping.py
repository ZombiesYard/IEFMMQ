"""
Map LLM HelpResponse payloads into v1 TutorResponse with safe overlay actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.overlay import OverlayPlanner
from core.types import TutorRequest, TutorResponse


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_ui_map_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "ui_map.yaml"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def map_help_response_to_tutor_response(
    help_obj: Mapping[str, Any] | None,
    *,
    request: TutorRequest | None = None,
    status: str = "ok",
    max_overlay_targets: int = 1,
    ui_map_path: str | Path | None = None,
    overlay_intent: str = "highlight",
) -> TutorResponse:
    metadata: dict[str, Any] = {}
    if isinstance(help_obj, Mapping):
        metadata["help_response"] = dict(help_obj)

    if max_overlay_targets < 0:
        max_overlay_targets = 0

    explanations_raw = help_obj.get("explanations") if isinstance(help_obj, Mapping) else None
    explanations = [item for item in explanations_raw if isinstance(item, str) and item] if isinstance(explanations_raw, list) else []
    message = explanations[0] if explanations else None

    raw_targets: list[str] = []
    if isinstance(help_obj, Mapping):
        overlay = help_obj.get("overlay")
        if isinstance(overlay, Mapping):
            targets = overlay.get("targets")
            if isinstance(targets, list):
                raw_targets = [item for item in targets if isinstance(item, str) and item]

    deduped_targets = _dedupe_preserve_order(raw_targets)
    selected_targets = deduped_targets[:max_overlay_targets] if max_overlay_targets > 0 else []
    dropped_for_limit = deduped_targets[max_overlay_targets:] if max_overlay_targets > 0 else deduped_targets
    if dropped_for_limit:
        metadata["dropped_targets"] = dropped_for_limit

    rejected_targets: list[str] = []
    actions: list[dict[str, Any]] = []
    planner: OverlayPlanner | None = None
    planner_error: str | None = None
    try:
        resolved_ui_map = Path(ui_map_path) if ui_map_path else _default_ui_map_path()
        planner = OverlayPlanner(str(resolved_ui_map))
    except Exception as exc:  # pragma: no cover - defensive path
        planner_error = f"{type(exc).__name__}: {exc}"

    if planner is None:
        rejected_targets.extend(selected_targets)
        if planner_error:
            metadata["overlay_mapping_error"] = planner_error
    else:
        for target in selected_targets:
            try:
                actions.append(planner.plan(target, intent=overlay_intent).to_action())
            except Exception:
                rejected_targets.append(target)

    if rejected_targets:
        metadata["rejected_targets"] = rejected_targets

    effective_status = status
    if status not in {"ok", "pending", "error"}:
        effective_status = "error"
        metadata["mapping_error"] = f"invalid_status:{status}"

    return TutorResponse(
        status=effective_status,
        in_reply_to=request.request_id if request else None,
        message=message,
        actions=actions,
        explanations=explanations,
        metadata=metadata,
    )


__all__ = ["map_help_response_to_tutor_response"]
