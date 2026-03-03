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


def _check_private_issue_contract_not_public() -> list[GuardViolation]:
    """Enforce that private issue templates are not committed to the public repo."""
    violations: list[GuardViolation] = []
    forbidden = REPO_ROOT / ".github/ISSUE_TEMPLATE/llm-task.yml"
    if forbidden.exists():
        violations.append(
            GuardViolation(
                rule="private_contract_leak",
                path=forbidden,
                detail="Private LLM issue template must not be committed.",
            )
        )
    return violations


def run_guards() -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    violations.extend(_check_forbidden_automation())
    violations.extend(_check_core_purity())
    violations.extend(_check_core_imports())
    violations.extend(_check_schema_v1_presence())
    violations.extend(_check_private_issue_contract_not_public())
    return violations


def main() -> int:
    violations = run_guards()
    if not violations:
        print("ci_guard: OK")
        return 0

    print("ci_guard: FAILED")
    for idx, v in enumerate(violations, start=1):
        rel = v.path.relative_to(REPO_ROOT) if v.path.exists() else v.path
        print(f"{idx}. [{v.rule}] {rel}: {v.detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
