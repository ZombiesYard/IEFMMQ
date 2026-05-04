from __future__ import annotations

from tools.copilot_review_digest import (
    PullRequestReviewDigest,
    ReviewComment,
    ReviewThread,
    _parse_repo_slug,
    _parse_github_datetime,
    is_copilot_login,
    normalize_threads,
    render_codex_prompt_zh,
    resolve_pr_number,
)


def test_parse_repo_slug_supports_https_and_ssh() -> None:
    assert _parse_repo_slug("https://github.com/ZombiesYard/IEFMMQ.git") == "ZombiesYard/IEFMMQ"
    assert _parse_repo_slug("git@github.com:ZombiesYard/IEFMMQ.git") == "ZombiesYard/IEFMMQ"


def test_is_copilot_login_matches_common_variants() -> None:
    assert is_copilot_login("copilot")
    assert is_copilot_login("github-copilot[bot]")
    assert is_copilot_login("Copilot-Pull-Request-Reviewer[bot]")
    assert not is_copilot_login("octocat")


def test_normalize_threads_keeps_only_copilot_comments_and_filters_resolved() -> None:
    raw_threads = [
        {
            "isResolved": False,
            "path": "adapters/example.py",
            "line": 42,
            "originalLine": 42,
            "comments": {
                "nodes": [
                    {
                        "body": "Please add a test for this branch.",
                        "url": "https://example.test/comment/1",
                        "createdAt": "2026-05-03T12:00:00Z",
                        "author": {"login": "github-copilot[bot]"},
                    },
                    {
                        "body": "Human reply",
                        "url": "https://example.test/comment/2",
                        "createdAt": "2026-05-03T12:01:00Z",
                        "author": {"login": "zombiesyard"},
                    },
                ]
            },
        },
        {
            "isResolved": True,
            "path": "adapters/example.py",
            "line": 99,
            "originalLine": 99,
            "comments": {
                "nodes": [
                    {
                        "body": "Resolved Copilot note",
                        "url": "https://example.test/comment/3",
                        "createdAt": "2026-05-03T12:02:00Z",
                        "author": {"login": "github-copilot[bot]"},
                    }
                ]
            },
        },
    ]

    threads = normalize_threads(raw_threads, include_resolved=False)

    assert len(threads) == 1
    assert threads[0].path == "adapters/example.py"
    assert threads[0].line == 42
    assert len(threads[0].comments) == 1
    assert threads[0].comments[0].author_login == "github-copilot[bot]"
    assert threads[0].comments[0].body == "Please add a test for this branch."


def test_render_codex_prompt_zh_contains_review_location_and_comment() -> None:
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
                        body="This branch looks untested.",
                        url="https://example.test/comment/9",
                        created_at="2026-05-03T12:00:00Z",
                    ),
                ),
            ),
        ),
    )

    rendered = render_codex_prompt_zh(digest)

    assert "PR: #205" in rendered
    assert "Location: core/example.py:17" in rendered
    assert "This branch looks untested." in rendered
    assert "你准备拒绝的 review items 及理由" in rendered


def test_normalize_threads_can_filter_to_latest_commit_window() -> None:
    raw_threads = [
        {
            "isResolved": False,
            "path": "adapters/example.py",
            "line": 42,
            "originalLine": 42,
            "comments": {
                "nodes": [
                    {
                        "body": "Old Copilot note.",
                        "url": "https://example.test/comment/1",
                        "createdAt": "2026-05-03T12:00:00Z",
                        "author": {"login": "github-copilot[bot]"},
                    },
                    {
                        "body": "New Copilot note.",
                        "url": "https://example.test/comment/2",
                        "createdAt": "2026-05-03T12:10:00Z",
                        "author": {"login": "github-copilot[bot]"},
                    },
                ]
            },
        },
    ]

    threads = normalize_threads(
        raw_threads,
        include_resolved=False,
        created_after=_parse_github_datetime("2026-05-03T12:05:00Z"),
    )

    assert len(threads) == 1
    assert len(threads[0].comments) == 1
    assert threads[0].comments[0].body == "New Copilot note."


def test_resolve_pr_number_uses_current_branch_when_no_explicit_pr(monkeypatch) -> None:
    monkeypatch.setattr("tools.copilot_review_digest._run_git", lambda *args: "feature/test")
    monkeypatch.setattr(
        "tools.copilot_review_digest._run_gh",
        lambda *args: '{"number": 205}' if args[:3] == ("pr", "view", "feature/test") else "{}",
    )

    resolved = resolve_pr_number("ZombiesYard/IEFMMQ", "")

    assert resolved == 205
