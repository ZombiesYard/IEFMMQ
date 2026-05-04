from __future__ import annotations

import stat
from pathlib import Path

from tools.copilot_review_autofix import (
    build_codex_exec_command,
    default_autofix_bundle_output_path,
    default_last_message_output_path,
    get_current_branch,
    has_staged_changes,
    render_autofix_prompt,
    resolve_codex_binary,
    run_git,
    write_text,
)


def test_render_autofix_prompt_contains_execution_rules() -> None:
    rendered = render_autofix_prompt(
        "## PR 概览\n- PR: #205",
        repo="ZombiesYard/IEFMMQ",
        pr_number=205,
    )

    assert "请直接在当前工作区修改代码" in rendered
    assert "不要自动执行 git push" in rendered
    assert "接受了哪些 review items" in rendered
    assert "## PR 概览" in rendered


def test_build_codex_exec_command_includes_selected_options(tmp_path: Path) -> None:
    prompt_path = tmp_path / "bundle.md"
    last_message_path = tmp_path / "last.md"

    cmd = build_codex_exec_command(
        codex_bin="/usr/local/bin/codex",
        prompt_path=prompt_path,
        last_message_path=last_message_path,
        model="gpt-5.5",
        profile="default",
        sandbox="workspace-write",
        approval_policy="never",
    )

    assert cmd[:2] == ["/usr/local/bin/codex", "exec"]
    assert "-m" in cmd and "gpt-5.5" in cmd
    assert "-p" in cmd and "default" in cmd
    assert "-s" in cmd and "workspace-write" in cmd
    assert "-a" in cmd and "never" in cmd
    assert "-o" in cmd and str(last_message_path) in cmd
    assert cmd[-1] == "-"


def test_write_text_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "bundle.md"

    returned = write_text(str(path), "hello")

    assert returned == path
    assert path.read_text(encoding="utf-8") == "hello"


def test_run_git_raises_clear_error_on_failure(monkeypatch) -> None:
    class Result:
        returncode = 1
        stdout = ""
        stderr = "fatal: bad revision"

    monkeypatch.setattr("tools.copilot_review_autofix.subprocess.run", lambda *args, **kwargs: Result())

    try:
        run_git("status")
    except RuntimeError as exc:
        assert "fatal: bad revision" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("run_git should have raised RuntimeError")


def test_get_current_branch_uses_git_rev_parse(monkeypatch) -> None:
    monkeypatch.setattr("tools.copilot_review_autofix.run_git", lambda *args: "feature/test")

    assert get_current_branch() == "feature/test"


def test_has_staged_changes_checks_git_diff_exit_code(monkeypatch) -> None:
    class Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    monkeypatch.setattr("tools.copilot_review_autofix.subprocess.run", lambda *args, **kwargs: Result(1))
    assert has_staged_changes() is True

    monkeypatch.setattr("tools.copilot_review_autofix.subprocess.run", lambda *args, **kwargs: Result(0))
    assert has_staged_changes() is False


def test_resolve_codex_binary_prefers_explicit_existing_path(tmp_path: Path) -> None:
    codex_path = tmp_path / "codex"
    codex_path.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_path.chmod(codex_path.stat().st_mode | stat.S_IXUSR)

    resolved = resolve_codex_binary(str(codex_path))

    assert resolved == str(codex_path)


def test_resolve_codex_binary_rejects_non_executable_explicit_path(tmp_path: Path) -> None:
    codex_path = tmp_path / "codex"
    codex_path.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    try:
        resolve_codex_binary(str(codex_path))
    except RuntimeError as exc:
        assert "is not executable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("resolve_codex_binary should reject non-executable files")


def test_resolve_codex_binary_uses_which(monkeypatch) -> None:
    monkeypatch.setattr("tools.copilot_review_autofix.shutil.which", lambda candidate: "/usr/bin/codex" if candidate == "codex" else None)

    resolved = resolve_codex_binary("")

    assert resolved == "/usr/bin/codex"


def test_default_last_message_output_path_includes_pr_number() -> None:
    assert default_last_message_output_path(206) == ".tmp/copilot_autofix_last_message.md"


def test_default_autofix_bundle_output_path_includes_pr_number() -> None:
    assert default_autofix_bundle_output_path(206) == ".tmp/copilot_review_bundle.md"
