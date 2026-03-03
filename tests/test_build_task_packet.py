from __future__ import annotations

from pathlib import Path

import pytest

from tools.build_task_packet import (
    IssueData,
    build_task_packet,
    fetch_issue_via_gh,
    validate_issue_body,
)


def _issue_body_complete() -> str:
    return """### Goal
Implement one issue.

### Scope
- Must change: A
- May change: B
- Must not change: C

### Acceptance
- [ ] cond-1

### Test Plan
- run unit tests

### Out of Scope
- no refactor
"""


def test_validate_issue_body_missing_required_sections() -> None:
    body = """### Goal
Only goal present.
"""
    missing = validate_issue_body(body)
    assert "Scope" in missing
    assert "Acceptance" in missing
    assert "Test Plan" in missing
    assert "Out of Scope" in missing


def test_build_task_packet_generates_output(tmp_path: Path) -> None:
    private_dir = tmp_path / ".simtutor-private"
    private_dir.mkdir()
    (private_dir / "issue_contract.md").write_text("# contract", encoding="utf-8")
    (private_dir / "issue_prompt_wrapper.md").write_text(
        "ISSUE {{ISSUE_NUMBER}} {{ISSUE_TITLE}}\n{{ISSUE_BODY}}\n{{REQUIRED_SECTIONS}}",
        encoding="utf-8",
    )
    issue = IssueData(
        number=42,
        title="Packet Build",
        body=_issue_body_complete(),
        url="https://example.com/issue/42",
    )
    output = tmp_path / "task_packet_42.md"
    result = build_task_packet(issue=issue, private_dir=private_dir, output_path=output)
    assert result == output
    text = output.read_text(encoding="utf-8")
    assert "# contract" in text
    assert "ISSUE 42 Packet Build" in text
    assert "### Goal" in text
    assert "- Goal" in text
    assert "## Output Contract" in text


def test_build_task_packet_raises_on_missing_sections(tmp_path: Path) -> None:
    private_dir = tmp_path / ".simtutor-private"
    private_dir.mkdir()
    (private_dir / "issue_contract.md").write_text("# contract", encoding="utf-8")
    (private_dir / "issue_prompt_wrapper.md").write_text("{{ISSUE_BODY}}", encoding="utf-8")
    issue = IssueData(number=9, title="x", body="### Goal\nx", url="u")
    with pytest.raises(ValueError):
        build_task_packet(issue=issue, private_dir=private_dir, output_path=tmp_path / "out.md")


def test_fetch_issue_via_gh_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completed:
        returncode = 1
        stdout = ""
        stderr = "bad request"

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return _Completed()

    monkeypatch.setattr("tools.build_task_packet.subprocess.run", _fake_run)
    with pytest.raises(RuntimeError, match="Failed to fetch issue via gh"):
        fetch_issue_via_gh(1, repo="owner/repo")

