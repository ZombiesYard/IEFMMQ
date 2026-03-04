#!/usr/bin/env python3
"""Minimal CI guardrails for SimTutor.

This script enforces repository-level safety and architecture invariants that
are easy to validate statically in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = [
    REPO_ROOT / "core",
    REPO_ROOT / "ports",
    REPO_ROOT / "adapters",
    REPO_ROOT / "simtutor",
    REPO_ROOT / "live_dcs.py",
]


@dataclass(frozen=True)
class GuardViolation:
    rule: str
    path: Path
    detail: str


def _iter_py_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.py") if p.is_file())


def _iter_production_py_files() -> list[Path]:
    files: list[Path] = []
    for path in PRODUCTION_PATHS:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(_iter_py_files(path))
    return sorted(set(files))


def _check_forbidden_automation() -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    pattern = re.compile(r"\bperformClickableAction\b")
    for path in _iter_production_py_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            violations.append(
                GuardViolation(
                    rule="no_click_automation",
                    path=path,
                    detail="Forbidden API `performClickableAction` detected.",
                )
            )
    return violations


def _check_core_purity() -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    forbidden_markers = [
        ("dcs_marker", re.compile(r"\bDCS\b")),
        ("udp_socket_usage", re.compile(r"\bsocket\.")),
    ]
    for path in _iter_py_files(REPO_ROOT / "core"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for rule_suffix, marker in forbidden_markers:
            if marker.search(text):
                violations.append(
                    GuardViolation(
                        rule=f"core_purity_{rule_suffix}",
                        path=path,
                        detail=f"Core layer contains forbidden marker: {marker.pattern}",
                    )
                )
    return violations


def _check_core_imports() -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    adapter_import = re.compile(r"^\s*(from|import)\s+adapters(\.|$)", re.MULTILINE)
    for path in _iter_py_files(REPO_ROOT / "core"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if adapter_import.search(text):
            violations.append(
                GuardViolation(
                    rule="core_import_boundary",
                    path=path,
                    detail="Core layer must not import from adapters.",
                )
            )
    return violations


def _check_schema_v1_presence() -> list[GuardViolation]:
    """Basic sanity: required v1 schema files should exist."""
    violations: list[GuardViolation] = []
    required = [
        REPO_ROOT / "simtutor/schemas/v1/observation.schema.json",
        REPO_ROOT / "simtutor/schemas/v1/tutor_request.schema.json",
        REPO_ROOT / "simtutor/schemas/v1/tutor_response.schema.json",
        REPO_ROOT / "simtutor/schemas/v1/event.schema.json",
    ]
    for path in required:
        if not path.exists():
            violations.append(
                GuardViolation(
                    rule="schema_v1_presence",
                    path=path,
                    detail="Required v1 schema file is missing.",
                )
            )
    return violations


def _normalize_guard_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return path
    try:
        _ = resolved.relative_to(REPO_ROOT)
        return resolved
    except ValueError:
        return REPO_ROOT / "Doc" / "Evaluation" / path.name


def _display_violation_path(path: Path) -> str:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _check_eval_doc_drift() -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    try:
        try:
            from tools.regenerate_eval_docs import (
                EvalDocDriftError,
                EvalDocRegenerationError,
                regenerate_eval_docs,
            )
        except ModuleNotFoundError:
            from regenerate_eval_docs import (  # type: ignore
                EvalDocDriftError,
                EvalDocRegenerationError,
                regenerate_eval_docs,
            )
    except Exception as exc:
        violations.append(
            GuardViolation(
                rule="eval_docs_regeneration_error",
                path=REPO_ROOT / "tools" / "regenerate_eval_docs.py",
                detail=f"Doc regeneration import failed: {exc}",
            )
        )
        return violations

    try:
        regenerate_eval_docs(
            index_path=REPO_ROOT / "Doc" / "Evaluation" / "index.json",
            policy_path=REPO_ROOT / "knowledge_source_policy.yaml",
            repo_root=REPO_ROOT,
            check=True,
            strict_policy=True,
        )
    except EvalDocDriftError as exc:
        for path in exc.drift_paths:
            violations.append(
                GuardViolation(
                    rule="eval_docs_drift",
                    path=_normalize_guard_path(path),
                    detail="Generated Doc/Evaluation markdown is out of date. Run: python -m tools.regenerate_eval_docs",
                )
            )
    except EvalDocRegenerationError as exc:
        violations.append(
            GuardViolation(
                rule="eval_docs_regeneration_error",
                path=REPO_ROOT / "knowledge_source_policy.yaml",
                detail=f"Doc regeneration pipeline error: {exc}",
            )
        )
    return violations


def run_guards() -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    violations.extend(_check_forbidden_automation())
    violations.extend(_check_core_purity())
    violations.extend(_check_core_imports())
    violations.extend(_check_schema_v1_presence())
    violations.extend(_check_eval_doc_drift())
    return violations


def main() -> int:
    violations = run_guards()
    if not violations:
        print("ci_guard: OK")
        return 0

    print("ci_guard: FAILED")
    for idx, v in enumerate(violations, start=1):
        print(f"{idx}. [{v.rule}] {_display_violation_path(v.path)}: {v.detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
