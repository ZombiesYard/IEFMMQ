from __future__ import annotations

from pathlib import Path

import tools.ci_guard as ci_guard


def test_private_contract_leak_check_detects_public_template(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".github/ISSUE_TEMPLATE").mkdir(parents=True)
    forbidden = tmp_path / ".github/ISSUE_TEMPLATE/llm-task.yml"
    forbidden.write_text("name: x", encoding="utf-8")

    monkeypatch.setattr(ci_guard, "REPO_ROOT", tmp_path)
    violations = ci_guard._check_private_issue_contract_not_public()
    assert len(violations) == 1
    assert violations[0].rule == "private_contract_leak"
    assert violations[0].path == forbidden


def test_private_contract_leak_check_passes_when_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ci_guard, "REPO_ROOT", tmp_path)
    violations = ci_guard._check_private_issue_contract_not_public()
    assert violations == []

