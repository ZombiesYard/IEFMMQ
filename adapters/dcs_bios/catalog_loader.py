from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class DcsBiosControl:
    identifier: str
    category: str
    description: str
    control_type: str
    value_type: str
    outputs: list[dict]
    inputs: list[dict]

    def to_dict(self, aircraft: str) -> Dict[str, Any]:
        return {
            "id": f"{aircraft}/{self.identifier}",
            "identifier": self.identifier,
            "category": self.category,
            "description": self.description,
            "control_type": self.control_type,
            "value_type": self.value_type,
            "outputs": self.outputs,
            "inputs": self.inputs,
        }


def _flatten_controls(data: dict) -> Iterable[tuple[str, dict, dict]]:
    for category, controls in data.items():
        if not isinstance(controls, dict):
            continue
        for key, control in controls.items():
            if not isinstance(control, dict):
                continue
            yield category, key, control


def _extract_outputs(control: dict) -> list[dict]:
    outputs = []
    for out in control.get("outputs", []) or []:
        if not isinstance(out, dict):
            continue
        outputs.append(
            {
                "address": out.get("address"),
                "mask": out.get("mask"),
                "shift_by": out.get("shift_by"),
                "max_value": out.get("max_value"),
                "suffix": out.get("suffix"),
                "type": out.get("type"),
                "description": out.get("description"),
            }
        )
    return outputs


def _extract_inputs(control: dict) -> list[dict]:
    inputs = []
    for inp in control.get("inputs", []) or []:
        if not isinstance(inp, dict):
            continue
        inputs.append(
            {
                "interface": inp.get("interface"),
                "description": inp.get("description"),
                "max_value": inp.get("max_value"),
                "suggested_step": inp.get("suggested_step"),
                "argument": inp.get("argument"),
            }
        )
    return inputs


def _value_type(outputs: list[dict]) -> str:
    for out in outputs:
        if out.get("type") == "string":
            return "string"
    return "integer"


def load_control_reference(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_catalog(aircraft: str, data: dict) -> List[DcsBiosControl]:
    controls: list[DcsBiosControl] = []
    for category, key, control in _flatten_controls(data):
        identifier = str(control.get("identifier") or key)
        outputs = _extract_outputs(control)
        inputs = _extract_inputs(control)
        controls.append(
            DcsBiosControl(
                identifier=identifier,
                category=str(control.get("category") or category),
                description=str(control.get("description") or ""),
                control_type=str(control.get("control_type") or ""),
                value_type=_value_type(outputs),
                outputs=outputs,
                inputs=inputs,
            )
        )
    return controls


def write_catalog_json(aircraft: str, controls: List[DcsBiosControl], output: Path) -> None:
    payload = {
        "aircraft": aircraft,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "controls": [c.to_dict(aircraft) for c in controls],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_controls_lua(aircraft: str, controls: List[DcsBiosControl], output: Path) -> None:
    lines = ["return {", f"  aircraft = {json.dumps(aircraft)},", "  controls = {"]
    for c in controls:
        lines.append(
            f"    {{ id = {json.dumps(f'{aircraft}/{c.identifier}')}, kind = {json.dumps(c.value_type)} }},"
        )
    lines.append("  }")
    lines.append("}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DCS-BIOS control catalog.")
    parser.add_argument("--aircraft", required=True, help="Aircraft name (e.g., FA-18C_hornet)")
    parser.add_argument("--input", required=True, help="Path to DCS-BIOS control reference JSON")
    parser.add_argument("--output", required=True, help="Output catalog JSON path")
    parser.add_argument("--controls-lua", help="Optional Lua control list for hub script")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    data = load_control_reference(Path(args.input))
    controls = build_catalog(args.aircraft, data)
    write_catalog_json(args.aircraft, controls, Path(args.output))
    if args.controls_lua:
        write_controls_lua(args.aircraft, controls, Path(args.controls_lua))
    print(f"[CATALOG] wrote {len(controls)} controls to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
