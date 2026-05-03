from __future__ import annotations

from tools.copilot_review_digest import PullRequestReviewDigest, ReviewComment, ReviewThread
from tools.copilot_review_loop import (
    FailedRunLog,
    PullRequestSnapshot,
    fetch_failed_log_text,
    fetch_checks_output,
    render_bundle_codex_zh,
)


def test_fetch_checks_output_returns_stdout_on_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.copilot_review_loop._run_gh_allow_failure",
        lambda *args: (0, "build\tpass\t32s\nlint\tpass\t9s\n", ""),
    )

    rendered = fetch_checks_output("ZombiesYard/IEFMMQ", 205)

    assert "build" in rendered
    assert "lint" in rendered


def test_fetch_checks_output_falls_back_to_error_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.copilot_review_loop._run_gh_allow_failure",
        lambda *args: (1, "", "no checks reported on the 'feature/test' branch"),
    )

    rendered = fetch_checks_output("ZombiesYard/IEFMMQ", 205)

    assert rendered.startswith("[checks unavailable]")
    assert "no checks reported" in rendered


def test_render_bundle_codex_zh_contains_checks_and_review_items() -> None:
    snapshot = PullRequestSnapshot(
        repo="ZombiesYard/IEFMMQ",
        number=205,
        title="Example PR",
        url="https://github.com/ZombiesYard/IEFMMQ/pull/205",
        head_ref="feature/test",
        base_ref="main",
        is_draft=False,
        review_decision="CHANGES_REQUESTED",
        head_sha="abc123",
        checks_output="build\tpass\t32s",
    )
    digest = PullRequestReviewDigest(
        repo="ZombiesYard/IEFMMQ",
        number=205,
        title="Example PR",
        url="https://github.com/ZombiesYard/IEFMMQ/pull/205",
        head_ref="feature/test",
        base_ref="main",
        threads=(
            ReviewThread(
                path="core/example.py",
                line=17,
                original_line=17,
                is_resolved=False,
                comments=(
                    ReviewComment(
                        author_login="github-copilot[bot]",
                        body="Please add a regression test here.",
                        url="https://example.test/comment/7",
                        created_at="2026-05-03T12:00:00Z",
                    ),
                ),
            ),
        ),
    )

    rendered = render_bundle_codex_zh(snapshot, digest)

    assert "## PR 概览" in rendered
    assert "Review decision: CHANGES_REQUESTED" in rendered
    assert "## CI / Checks" in rendered
    assert "build\tpass\t32s" in rendered
    assert "Location: core/example.py:17" in rendered
    assert "Please add a regression test here." in rendered


def test_render_bundle_codex_zh_includes_failed_run_logs() -> None:
    snapshot = PullRequestSnapshot(
        repo="ZombiesYard/IEFMMQ",
        number=205,
        title="Example PR",
        url="https://github.com/ZombiesYard/IEFMMQ/pull/205",
        head_ref="feature/test",
        base_ref="main",
        is_draft=False,
        review_decision="",
        head_sha="abc123",
        checks_output="guard\tfail\t1m12s",
    )
    digest = PullRequestReviewDigest(
        repo="ZombiesYard/IEFMMQ",
        number=205,
        title="Example PR",
        url="https://github.com/ZombiesYard/IEFMMQ/pull/205",
        head_ref="feature/test",
        base_ref="main",
        threads=(),
    )
    failed_runs = (
        FailedRunLog(
            database_id=123,
            name="CI",
            workflow_name="guard-and-test",
            conclusion="failure",
            url="https://github.com/ZombiesYard/IEFMMQ/actions/runs/123",
            log_excerpt="pytest failed in tests/test_example.py::test_case",
        ),
    )

    rendered = render_bundle_codex_zh(snapshot, digest, failed_runs)

    assert "## Failed Actions Logs" in rendered
    assert "guard-and-test" in rendered
    assert "pytest failed in tests/test_example.py::test_case" in rendered


def test_fetch_failed_log_text_truncates_long_logs(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.copilot_review_loop._run_gh_allow_failure",
        lambda *args: (0, "A" * 50, ""),
    )

    rendered = fetch_failed_log_text("ZombiesYard/IEFMMQ", 123, max_chars=20)

    assert rendered.endswith("...[truncated]...")
