"""
Stable variable resolver for telemetry frames.

Maps raw telemetry (bios/lo/etc) to stable vars used by packs and gating.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from core.types_v2 import TelemetryFrame


class VarResolverError(ValueError):
    pass


_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")
_REF_ROOTS = {"bios", "lo", "cockpit_args", "vars"}
_MISSING = object()


def _to_number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _NUMERIC_RE.match(text):
            if "." in text:
                return float(text)
            return int(text)
    return None


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _attribute_chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    return parts


class _SourceRefCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.refs: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _attribute_chain(node)
        if chain and chain[0] in _REF_ROOTS:
            self.refs.add(".".join(chain))
            return
        self.generic_visit(node)


def _extract_source_refs(expr: str) -> set[str]:
    try:
        root = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise VarResolverError(f"Invalid expression: {expr}") from exc
    collector = _SourceRefCollector()
    collector.visit(root)
    return collector.refs


def _read_ref(ctx: Mapping[str, Any], ref: str) -> Any:
    parts = ref.split(".")
    current: Any = ctx.get(parts[0], _MISSING)
    if current is _MISSING:
        return _MISSING
    for part in parts[1:]:
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
            if current is _MISSING:
                return _MISSING
            continue
        return _MISSING
    return current


def _find_missing_refs(expr: str, ctx: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for ref in sorted(_extract_source_refs(expr)):
        value = _read_ref(ctx, ref)
        if value is _MISSING or _is_missing_value(value):
            missing.append(ref)
    return missing


def _safe_eval(expr: str, ctx: Mapping[str, Any]) -> Any:
    """
    Evaluate a restricted expression with consistent None handling:
    - Arithmetic: if any operand is None, return None.
    - Boolean ops: use truthiness (None -> False).
    - Comparisons: if either side is None, return False.
    """
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
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id != "num":
                raise VarResolverError(
                    "Unsupported function call. Only num(value) is supported in expressions."
                )
            if len(n.args) != 1 or n.keywords:
                raise VarResolverError("num(...) expects exactly one positional argument.")
            return _to_number(eval_node(n.args[0]))
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
            raise VarResolverError(
                f"Unsupported boolean operator:{type(n.op).__name__} in expression: {expr}"
            )
        if isinstance(n, ast.Compare):
            left = eval_node(n.left)
            for op, comp in zip(n.ops, n.comparators):
                # Check for unsupported operators before evaluating comparator
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                    raise VarResolverError(
                        f"Unsupported comparison operator: {op.__class__.__name__}. "
                        f"Only ==, !=, >, >=, <, <= are supported."
                    )
                right = eval_node(comp)
                if left is None or right is None:
                    return False
                if isinstance(op, ast.Eq):
                    if not (left == right):
                        return False
                elif isinstance(op, ast.NotEq):
                    if not (left != right):
                        return False
                elif isinstance(op, ast.Gt):
                    try:
                        ok = left > right
                    except TypeError:
                        return False
                    if not ok:
                        return False
                elif isinstance(op, ast.GtE):
                    try:
                        ok = left >= right
                    except TypeError:
                        return False
                    if not ok:
                        return False
                elif isinstance(op, ast.Lt):
                    try:
                        ok = left < right
                    except TypeError:
                        return False
                    if not ok:
                        return False
                elif isinstance(op, ast.LtE):
                    try:
                        ok = left <= right
                    except TypeError:
                        return False
                    if not ok:
                        return False
                left = right
            return True
        if isinstance(n, ast.BinOp):
            left = _to_number(eval_node(n.left))
            right = _to_number(eval_node(n.right))
            if left is None or right is None:
                return None
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                if right == 0:
                    raise VarResolverError(f"Division by zero in expression: {expr}")
                return left / right
        raise VarResolverError(f"Unsupported expression: {ast.dump(n, include_attributes=False)}")

    try:
        return eval_node(node)
    except VarResolverError:
        raise
    except Exception as exc:
        raise VarResolverError(f"Failed to evaluate expression: {expr}") from exc


def evaluate_expression_with_missing_refs(expr: str, ctx: Mapping[str, Any]) -> tuple[Any, list[str]]:
    missing_refs = _find_missing_refs(expr, ctx)
    value = _safe_eval(expr, ctx)
    return value, missing_refs


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
        """Resolve vars in rule order; forward references are not supported."""
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
        source_missing_vars: set[str] = set()

        for key, expr in self.rules.items():
            if expr is None:
                resolved[key] = None
                context["vars"] = resolved
                if key != "vars_source_missing":
                    source_missing_vars.add(key)
                continue
            missing_refs: list[str] = []
            if isinstance(expr, str):
                expr_text = expr.strip()
                if expr_text.startswith("derived(") and expr_text.endswith(")"):
                    expr_text = expr_text[len("derived(") : -1].strip()
                
                try:
                    missing_refs = _find_missing_refs(expr_text, context)
                    value = _safe_eval(expr_text, context)
                except VarResolverError as exc:
                    raise VarResolverError(f"Failed to resolve var '{key}': {exc}") from exc
            else:
                value = expr
            resolved[key] = value
            context["vars"] = resolved
            if key != "vars_source_missing" and (value is None or missing_refs):
                source_missing_vars.add(key)
        resolved["vars_source_missing"] = sorted(source_missing_vars)
        return resolved

    def apply(self, frame: TelemetryFrame | Mapping[str, Any]) -> TelemetryFrame | Dict[str, Any]:
        """Apply resolved vars; mutates TelemetryFrame in place, Mapping returns a new dict."""
        resolved = self.resolve(frame)
        if isinstance(frame, TelemetryFrame):
            frame.vars = resolved
            return frame
        data = dict(frame)
        data["vars"] = resolved
        return data


__all__ = ["VarResolver", "VarResolverError"]
