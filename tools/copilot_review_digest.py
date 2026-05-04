"""
Fetch GitHub Copilot PR review comments and render them as a Codex-friendly digest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ReviewComment:
    author_login: str
    body: str
    url: str
    created_at: str


@dataclass(frozen=True)
class ReviewThread:
    path: str
    line: int | None
    original_line: int | None
    is_resolved: bool
    comments: tuple[ReviewComment, ...]


@dataclass(frozen=True)
class PullRequestReviewDigest:
    repo: str
    number: int
    title: str
    url: str
    head_ref: str
    base_ref: str
    threads: tuple[ReviewThread, ...]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch GitHub Copilot PR review comments and render a digest for Codex."
    )
    parser.add_argument("--repo", default="", help="Repository slug like owner/repo. Defaults to origin remote.")
    parser.add_argument("--pr", default="", help="PR number or PR URL. Defaults to the PR for the current branch.")
    parser.add_argument(
        "--format",
        choices=("codex-zh", "markdown", "json"),
        default="codex-zh",
        help="Output format.",
    )
    parser.add_argument(
        "--include-resolved",
        action="store_true",
        help="Include threads already resolved on GitHub.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--since-latest-commit",
        action="store_true",
        help="Only include Copilot comments created after the PR's latest head commit.",
    )
    return parser


def _run_gh(*args: str) -> str:
    result = subprocess.run(
        ["gh", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "gh command failed"
        raise RuntimeError(f"gh {' '.join(args)} failed: {message}")
    return result.stdout


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _parse_repo_slug(remote_url: str) -> str:
    text = remote_url.strip()
    https_match = re.match(r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", text)
    if https_match:
        return f"{https_match.group(1)}/{https_match.group(2)}"
    ssh_match = re.match(r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", text)
    if ssh_match:
        return f"{ssh_match.group(1)}/{ssh_match.group(2)}"
    raise ValueError(f"unsupported GitHub remote URL: {remote_url!r}")


def resolve_repo_slug(explicit_repo: str) -> str:
    if explicit_repo.strip():
        return explicit_repo.strip()
    origin = _run_git("remote", "get-url", "origin")
    return _parse_repo_slug(origin)


def _pr_number_from_arg(pr_ref: str) -> int | None:
    text = pr_ref.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = re.search(r"/pull/(\d+)", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"unsupported PR reference: {pr_ref!r}")


def resolve_pr_number(repo: str, explicit_pr: str) -> int:
    number = _pr_number_from_arg(explicit_pr)
    if number is not None:
        return number
    current_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    payload = json.loads(_run_gh("pr", "view", current_branch, "-R", repo, "--json", "number"))
    pr_number = payload.get("number")
    if isinstance(pr_number, int) and pr_number > 0:
        return pr_number
    raise RuntimeError("failed to resolve PR number for current branch")


def fetch_pr_metadata(repo: str, pr_number: int) -> dict[str, Any]:
    payload = json.loads(
        _run_gh(
            "pr",
            "view",
            str(pr_number),
            "-R",
            repo,
            "--json",
            "number,title,url,headRefName,baseRefName",
        )
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("gh pr view did not return an object")
    return dict(payload)


def fetch_pr_latest_commit_at(repo: str, pr_number: int) -> str:
    owner, name = repo.split("/", 1)
    query = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            committedDate
            pushedDate
          }
        }
      }
    }
  }
}
""".strip()
    payload = json.loads(
        _run_gh(
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pr_number}",
        )
    )
    nodes = (
        payload.get("data", {})
        .get("repository", {})
        .get("pullRequest", {})
        .get("commits", {})
        .get("nodes", [])
    )
    if not isinstance(nodes, list) or not nodes:
        return ""
    commit = (nodes[0] or {}).get("commit", {})
    if not isinstance(commit, Mapping):
        return ""
    pushed_date = str(commit.get("pushedDate") or "").strip()
    committed_date = str(commit.get("committedDate") or "").strip()
    return pushed_date or committed_date


def fetch_review_threads(repo: str, pr_number: int) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    query = """
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $after) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          isResolved
          path
          line
          originalLine
          comments(first: 30) {
            nodes {
              body
              url
              createdAt
              author {
                login
              }
            }
          }
        }
      }
    }
  }
}
""".strip()
    threads: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pr_number}",
        ]
        if after:
            args.extend(["-F", f"after={after}"])
        payload = json.loads(_run_gh(*args))
        page = (
            payload.get("data", {})
            .get("repository", {})
            .get("pullRequest", {})
            .get("reviewThreads", {})
        )
        nodes = page.get("nodes", [])
        if isinstance(nodes, list):
            threads.extend(item for item in nodes if isinstance(item, Mapping))
        page_info = page.get("pageInfo", {})
        has_next = bool(page_info.get("hasNextPage"))
        end_cursor = page_info.get("endCursor")
        if not has_next or not isinstance(end_cursor, str) or not end_cursor:
            break
        after = end_cursor
    return threads


def is_copilot_login(login: str) -> bool:
    return "copilot" in login.strip().lower()


def normalize_threads(
    raw_threads: Sequence[Mapping[str, Any]],
    *,
    include_resolved: bool,
    created_after: datetime | None = None,
) -> tuple[ReviewThread, ...]:
    normalized: list[ReviewThread] = []
    for item in raw_threads:
        is_resolved = bool(item.get("isResolved"))
        if is_resolved and not include_resolved:
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            path = "<unknown>"
        line = item.get("line")
        original_line = item.get("originalLine")
        raw_comments = item.get("comments", {}).get("nodes", [])
        comments: list[ReviewComment] = []
        if isinstance(raw_comments, list):
            for raw in raw_comments:
                if not isinstance(raw, Mapping):
                    continue
                author = raw.get("author") or {}
                login = str(author.get("login") or "").strip()
                if not login or not is_copilot_login(login):
                    continue
                created_at = str(raw.get("createdAt") or "").strip()
                if created_after is not None:
                    created_at_dt = _parse_github_datetime(created_at)
                    if created_at_dt is None or created_at_dt < created_after:
                        continue
                body = _normalize_body(str(raw.get("body") or ""))
                comments.append(
                    ReviewComment(
                        author_login=login,
                        body=body,
                        url=str(raw.get("url") or "").strip(),
                        created_at=created_at,
                    )
                )
        if not comments:
            continue
        normalized.append(
            ReviewThread(
                path=path,
                line=line if isinstance(line, int) else None,
                original_line=original_line if isinstance(original_line, int) else None,
                is_resolved=is_resolved,
                comments=tuple(comments),
            )
        )
    return tuple(normalized)


def _normalize_body(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def _parse_github_datetime(text: str) -> datetime | None:
    value = text.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def build_digest(
    repo: str,
    pr_number: int,
    *,
    include_resolved: bool,
    since_latest_commit: bool = False,
) -> PullRequestReviewDigest:
    metadata = fetch_pr_metadata(repo, pr_number)
    raw_threads = fetch_review_threads(repo, pr_number)
    created_after = None
    if since_latest_commit:
        latest_commit_at = fetch_pr_latest_commit_at(repo, pr_number)
        created_after = _parse_github_datetime(latest_commit_at)
    threads = normalize_threads(
        raw_threads,
        include_resolved=include_resolved,
        created_after=created_after,
    )
    return PullRequestReviewDigest(
        repo=repo,
        number=pr_number,
        title=str(metadata.get("title") or ""),
        url=str(metadata.get("url") or ""),
        head_ref=str(metadata.get("headRefName") or ""),
        base_ref=str(metadata.get("baseRefName") or ""),
        threads=threads,
    )


def render_markdown(digest: PullRequestReviewDigest) -> str:
    lines = [
        f"# Copilot Review Digest for PR #{digest.number}",
        "",
        f"- Repo: `{digest.repo}`",
        f"- Title: {digest.title}",
        f"- URL: {digest.url}",
        f"- Branch: `{digest.head_ref}` -> `{digest.base_ref}`",
        f"- Copilot threads: {len(digest.threads)}",
        "",
    ]
    if not digest.threads:
        lines.extend(["No Copilot review comments found.", ""])
        return "\n".join(lines)
    for idx, thread in enumerate(digest.threads, start=1):
        location = _thread_location(thread)
        lines.extend([f"## {idx}. `{location}`", ""])
        lines.append(f"- Resolved: {'yes' if thread.is_resolved else 'no'}")
        lines.append("")
        for comment in thread.comments:
            lines.extend(
                [
                    f"### Comment by `{comment.author_login}`",
                    "",
                    comment.body or "_empty comment_",
                    "",
                    f"- URL: {comment.url}",
                    f"- Created: {comment.created_at}",
                    "",
                ]
            )
    return "\n".join(lines)


def render_codex_prompt_zh(digest: PullRequestReviewDigest) -> str:
    lines = [
        "请根据下面的 GitHub Copilot review comments 检查并修改当前分支代码。",
        "你可以接受或拒绝某条 review 意见，但如果拒绝，必须明确说明理由。",
        "请优先修复真实问题，保持改动最小，并在完成后运行相关测试。",
        "",
        f"PR: #{digest.number}",
        f"Repo: {digest.repo}",
        f"Title: {digest.title}",
        f"URL: {digest.url}",
        f"Branch: {digest.head_ref} -> {digest.base_ref}",
        f"Copilot unresolved/resolved threads included: {len(digest.threads)}",
        "",
    ]
    if not digest.threads:
        lines.extend(
            [
                "当前没有抓到 Copilot review comments。",
                "如果你预期应该有 comment，请确认：",
                "1. Copilot review 已完成",
                "2. 当前 PR 编号正确",
                "3. comment 确实来自 Copilot reviewer",
            ]
        )
        return "\n".join(lines)

    for idx, thread in enumerate(digest.threads, start=1):
        location = _thread_location(thread)
        lines.extend(
            [
                f"## Review Item {idx}",
                f"- Location: {location}",
                f"- Resolved on GitHub: {'yes' if thread.is_resolved else 'no'}",
                "",
            ]
        )
        for comment_idx, comment in enumerate(thread.comments, start=1):
            lines.extend(
                [
                    f"### Copilot Comment {idx}.{comment_idx}",
                    comment.body or "(empty comment)",
                    "",
                    f"- URL: {comment.url}",
                    f"- Created: {comment.created_at}",
                    "",
                ]
            )

    lines.extend(
        [
            "请输出：",
            "1. 你准备接受的 review items",
            "2. 你准备拒绝的 review items 及理由",
            "3. 实际代码修改",
            "4. 运行过的测试及结果",
        ]
    )
    return "\n".join(lines)


def render_json(digest: PullRequestReviewDigest) -> str:
    payload = {
        "repo": digest.repo,
        "number": digest.number,
        "title": digest.title,
        "url": digest.url,
        "head_ref": digest.head_ref,
        "base_ref": digest.base_ref,
        "threads": [
            {
                "path": thread.path,
                "line": thread.line,
                "original_line": thread.original_line,
                "is_resolved": thread.is_resolved,
                "comments": [
                    {
                        "author_login": comment.author_login,
                        "body": comment.body,
                        "url": comment.url,
                        "created_at": comment.created_at,
                    }
                    for comment in thread.comments
                ],
            }
            for thread in digest.threads
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _thread_location(thread: ReviewThread) -> str:
    line = thread.line if thread.line is not None else thread.original_line
    return f"{thread.path}:{line}" if isinstance(line, int) else thread.path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        repo = resolve_repo_slug(args.repo)
        pr_number = resolve_pr_number(repo, args.pr)
        digest = build_digest(
            repo,
            pr_number,
            include_resolved=bool(args.include_resolved),
            since_latest_commit=bool(args.since_latest_commit),
        )
        if args.format == "markdown":
            rendered = render_markdown(digest)
        elif args.format == "json":
            rendered = render_json(digest)
        else:
            rendered = render_codex_prompt_zh(digest)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            print(rendered)
        return 0
    except Exception as exc:  # pragma: no cover - exercised via CLI behavior
        parser.exit(
            1,
            f"error: {exc}\n"
            "tip: if the current branch has no PR, rerun with --pr <number> or --pr <url>\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
