"""
Stable variable resolver for telemetry frames.

Maps raw telemetry (bios/lo/etc) to stable vars used by packs and gating.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from core.types_v2 import TelemetryFrame


class VarResolverError(ValueError):
    pass


def _safe_eval(expr: str, ctx: Mapping[str, Any]) -> Any:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise VarResolverError(f"Invalid expression: {expr}") from exc

    def resolve_attr(value: Any, attr: str) -> Any:
        if isinstance(value, dict):
            return value.get(attr)
        return None

    def eval_node(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return ctx.get(n.id)
        if isinstance(n, ast.Attribute):
            base = eval_node(n.value)
            return resolve_attr(base, n.attr)
        if isinstance(n, ast.UnaryOp):
            operand = eval_node(n.operand)
            if isinstance(n.op, ast.Not):
                return not bool(operand)
            if isinstance(n.op, ast.USub):
                return -operand if operand is not None else None
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                for v in n.values:
                    if not bool(eval_node(v)):
                        return False
                return True
            if isinstance(n.op, ast.Or):
                for v in n.values:
                    if bool(eval_node(v)):
                        return True
                return False
        if isinstance(n, ast.Compare):
            left = eval_node(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = eval_node(comp)
                if left is None or right is None:
                    return False
                if isinstance(op, ast.Eq) and not (left == right):
                    return False
                if isinstance(op, ast.NotEq) and not (left != right):
                    return False
                if isinstance(op, ast.Gt) and not (left > right):
                    return False
                if isinstance(op, ast.GtE) and not (left >= right):
                    return False
                if isinstance(op, ast.Lt) and not (left < right):
                    return False
                if isinstance(op, ast.LtE) and not (left <= right):
                    return False
                left = right
            return True
        if isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)
            if left is None or right is None:
                return None
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
        raise VarResolverError(f"Unsupported expression: {ast.dump(n, include_attributes=False)}")

    return eval_node(node)


@dataclass
class VarResolver:
    rules: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VarResolver":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        rules = data.get("vars") if isinstance(data, dict) else None
        if not isinstance(rules, dict):
            raise VarResolverError("telemetry_map.yaml missing top-level 'vars' mapping")
        return cls(rules=dict(rules))

    def resolve(self, frame: TelemetryFrame | Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(frame, TelemetryFrame):
            data = frame.to_dict()
        else:
            data = dict(frame)

        context = {
            "bios": data.get("bios") or {},
            "lo": data.get("lo") or {},
            "cockpit_args": data.get("cockpit_args") or {},
            "vars": dict(data.get("vars") or {}),
        }
        resolved = dict(context["vars"])

        for key, expr in self.rules.items():
            if expr is None:
                resolved[key] = None
                context["vars"] = resolved
                continue
            if isinstance(expr, str):
                expr_text = expr.strip()
                if expr_text.startswith("derived(") and expr_text.endswith(")"):
                    expr_text = expr_text[len("derived(") : -1].strip()
                value = _safe_eval(expr_text, context)
            else:
                value = expr
            resolved[key] = value
            context["vars"] = resolved
        return resolved

    def apply(self, frame: TelemetryFrame | Mapping[str, Any]) -> TelemetryFrame | Dict[str, Any]:
        resolved = self.resolve(frame)
        if isinstance(frame, TelemetryFrame):
            frame.vars = resolved
            return frame
        data = dict(frame)
        data["vars"] = resolved
        return data


__all__ = ["VarResolver", "VarResolverError"]
