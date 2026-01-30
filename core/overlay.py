"""
Overlay planner: map abstract cockpit targets to simulator-specific element ids.

Overlay intents:
- highlight
- clear
- pulse

Actions are emitted as dicts so they can be serialized directly in TutorResponse.actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import yaml


@dataclass(frozen=True)
class OverlayIntent:
    intent: str  # highlight | clear | pulse
    target: str  # abstract target name, e.g., battery_switch
    element_id: str  # simulator-specific id, e.g., pnt_331
    style: Dict[str, object] | None = None

    def to_action(self) -> Dict[str, object]:
        action = {
            "type": "overlay",
            "intent": self.intent,
            "target": self.target,
            "element_id": self.element_id,
        }
        if self.style:
            action["style"] = self.style
        return action


class OverlayPlanner:
    def __init__(self, ui_map_path: str):
        with open(ui_map_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("UI map must be a YAML mapping")
        if "version" in data and data["version"] != "v1":
            raise ValueError(f"Unsupported UI map version: {data['version']!r}")
        elements = data.get("cockpit_elements", {})
        if not isinstance(elements, dict):
            raise ValueError("UI map 'cockpit_elements' must be a mapping")
        default_style = data.get("default_overlay", {})
        if default_style is not None and not isinstance(default_style, dict):
            raise ValueError("UI map 'default_overlay' must be a mapping if provided")
        self.elements = elements
        self.default_style = default_style

    def plan(self, target: str, intent: str = "highlight") -> OverlayIntent:
        if intent not in {"highlight", "clear", "pulse"}:
            raise ValueError(f"Unknown overlay intent: {intent!r}")
        entry = self.elements.get(target)
        if not entry or "dcs_id" not in entry:
            raise KeyError(f"Unknown target '{target}' in UI map")
        style = dict(self.default_style) if self.default_style else None
        return OverlayIntent(intent=intent, target=target, element_id=entry["dcs_id"], style=style)

