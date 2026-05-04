"""
Bundle GitHub PR review context for Codex and optionally re-request Copilot review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from tools.copilot_review_digest import (
    PullRequestReviewDigest,
    build_digest,
    resolve_pr_number,
    resolve_repo_slug,
)


DEFAULT_BUNDLE_OUTPUT_TEMPLATE = ".tmp/copilot_review_bundle_pr{pr}.md"


@dataclass(frozen=True)
class PullRequestSnapshot:
    repo: str
    number: int
    title: str
    url: str
    head_ref: str
    base_ref: str
    is_draft: bool
    review_decision: str
    head_sha: str
    checks_output: str


@dataclass(frozen=True)
class FailedRunLog:
    database_id: int
    name: str
    workflow_name: str
    conclusion: str
    url: str
    log_excerpt: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bundle Copilot review context for Codex and optionally re-request Copilot review."
    )
    parser.add_argument("--repo", default="", help="Repository slug like owner/repo. Defaults to origin remote.")
    parser.add_argument("--pr", default="", help="PR number or URL. Defaults to the PR for the current branch.")
    parser.add_argument(
        "--include-resolved",
        action="store_true",
        help="Include resolved Copilot review threads in the digest.",
    )
    parser.add_argument(
        "--since-latest-commit",
        action="store_true",
        help="Only include Copilot comments created after the PR's latest head commit.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path. Defaults to .tmp/copilot_review_bundle_pr<PR>.md.",
    )
    parser.add_argument(
        "--request-copilot-review",
        action="store_true",
        help="After generating the bundle, re-request Copilot review for the PR.",
    )
    parser.add_argument(
        "--include-failed-run-logs",
        action="store_true",
        help="Include excerpts from failed GitHub Actions runs for the PR head commit.",
    )
    parser.add_argument(
        "--max-failed-runs",
        type=int,
        default=2,
        help="Maximum number of failed workflow runs to include when --include-failed-run-logs is set.",
    )
    parser.add_argument(
        "--max-log-chars",
        type=int,
        default=12000,
        help="Maximum number of characters per failed workflow log excerpt.",
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


def _run_gh_allow_failure(*args: str) -> tuple[int, str, str]:
    result = subprocess.run(
        ["gh", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def fetch_snapshot(repo: str, pr_number: int) -> PullRequestSnapshot:
    payload = json.loads(
        _run_gh(
            "pr",
            "view",
            str(pr_number),
            "-R",
            repo,
            "--json",
            "number,title,url,headRefName,baseRefName,isDraft,reviewDecision,headRefOid",
        )
    )
    checks_output = fetch_checks_output(repo, pr_number)
    return PullRequestSnapshot(
        repo=repo,
        number=int(payload.get("number") or pr_number),
        title=str(payload.get("title") or ""),
        url=str(payload.get("url") or ""),
        head_ref=str(payload.get("headRefName") or ""),
        base_ref=str(payload.get("baseRefName") or ""),
        is_draft=bool(payload.get("isDraft")),
        review_decision=str(payload.get("reviewDecision") or ""),
        head_sha=str(payload.get("headRefOid") or ""),
        checks_output=checks_output,
    )


def fetch_checks_output(repo: str, pr_number: int) -> str:
    code, stdout, stderr = _run_gh_allow_failure("pr", "checks", str(pr_number), "-R", repo)
    text = stdout.strip()
    if code == 0 and text:
        return text
    error_text = stderr.strip() or text or "unavailable"
    return f"[checks unavailable] {error_text}"


def fetch_failed_run_logs(
    repo: str,
    head_sha: str,
    *,
    max_runs: int,
    max_log_chars: int,
) -> tuple[FailedRunLog, ...]:
    if not head_sha.strip():
        return ()
    payload = json.loads(
        _run_gh(
            "api",
            f"repos/{repo}/actions/runs?head_sha={head_sha}&per_page=20",
        )
    )
    workflow_runs = payload.get("workflow_runs", [])
    if not isinstance(workflow_runs, list):
        return ()

    failed_runs: list[FailedRunLog] = []
    for run in workflow_runs:
        if not isinstance(run, Mapping):
            continue
        conclusion = str(run.get("conclusion") or "").strip().lower()
        if conclusion in {"success", "neutral", "skipped"}:
            continue
        database_id = run.get("id")
        if not isinstance(database_id, int):
            continue
        log_text = fetch_failed_log_text(repo, database_id, max_chars=max_log_chars)
        failed_runs.append(
            FailedRunLog(
                database_id=database_id,
                name=str(run.get("name") or ""),
                workflow_name=str(run.get("display_title") or run.get("name") or ""),
                conclusion=conclusion or "unknown",
                url=str(run.get("html_url") or "").strip(),
                log_excerpt=log_text,
            )
        )
        if len(failed_runs) >= max_runs:
            break
    return tuple(failed_runs)


def fetch_failed_log_text(repo: str, run_id: int, *, max_chars: int) -> str:
    code, stdout, stderr = _run_gh_allow_failure("run", "view", str(run_id), "-R", repo, "--log-failed")
    text = stdout.strip()
    if code != 0:
        error_text = stderr.strip() or text or "failed to fetch log"
        return f"[failed to fetch log] {error_text}"[:max_chars]
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]..."


def render_bundle_codex_zh(
    snapshot: PullRequestSnapshot,
    digest: PullRequestReviewDigest,
    failed_run_logs: Sequence[FailedRunLog] = (),
) -> str:
    lines = [
        "请处理这个 PR 当前的 Copilot review 与检查状态。",
        "你可以接受或拒绝某条 review 意见，但如果拒绝，必须明确说明理由。",
        "请优先修复真实问题，保持改动最小，并在完成后运行相关测试。",
        "",
        "## PR 概览",
        f"- PR: #{snapshot.number}",
        f"- Repo: {snapshot.repo}",
        f"- Title: {snapshot.title}",
        f"- URL: {snapshot.url}",
        f"- Branch: {snapshot.head_ref} -> {snapshot.base_ref}",
        f"- Draft: {'yes' if snapshot.is_draft else 'no'}",
        f"- Review decision: {snapshot.review_decision or 'none'}",
        "",
        "## CI / Checks",
        "```text",
        snapshot.checks_output or "[checks unavailable]",
        "```",
        "",
    ]
    if failed_run_logs:
        lines.append("## Failed Actions Logs")
        for idx, run in enumerate(failed_run_logs, start=1):
            lines.extend(
                [
                    f"### Failed Run {idx}",
                    f"- Workflow: {run.workflow_name or run.name}",
                    f"- Conclusion: {run.conclusion}",
                    f"- URL: {run.url}",
                    "```text",
                    run.log_excerpt or "[empty log]",
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Copilot Review Digest",
            _render_digest_section(digest),
            "",
            "请输出：",
            "1. 你准备接受的 review items",
            "2. 你准备拒绝的 review items 及理由",
            "3. 实际代码修改",
            "4. 运行过的测试及结果",
            "5. 如果还有未解决风险，请明确指出",
        ]
    )
    return "\n".join(lines)


def _render_digest_section(digest: PullRequestReviewDigest) -> str:
    if not digest.threads:
        return "\n".join(
            [
                "当前没有抓到 Copilot review comments。",
                "如果你预期应该有 comment，请确认 Copilot review 已完成，并且当前 PR 编号正确。",
            ]
        )

    lines: list[str] = []
    for idx, thread in enumerate(digest.threads, start=1):
        lines.extend(
            [
                f"### Review Item {idx}",
                f"- Location: {_thread_location(thread)}",
                f"- Resolved on GitHub: {'yes' if thread.is_resolved else 'no'}",
                "",
            ]
        )
        for comment_idx, comment in enumerate(thread.comments, start=1):
            lines.extend(
                [
                    f"#### Copilot Comment {idx}.{comment_idx}",
                    comment.body or "(empty comment)",
                    "",
                    f"- URL: {comment.url}",
                    f"- Created: {comment.created_at}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip()


def _thread_location(thread: object) -> str:
    line = getattr(thread, "line", None)
    original_line = getattr(thread, "original_line", None)
    path = str(getattr(thread, "path", "<unknown>"))
    chosen = line if isinstance(line, int) else original_line
    return f"{path}:{chosen}" if isinstance(chosen, int) else path


def request_copilot_review(repo: str, pr_number: int) -> str:
    return _run_gh("pr", "edit", str(pr_number), "-R", repo, "--add-reviewer", "@copilot").strip()


def write_output(text: str, output_path: str) -> None:
    if not output_path:
        print(text)
        return
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def default_bundle_output_path(pr_number: int) -> str:
    return DEFAULT_BUNDLE_OUTPUT_TEMPLATE.format(pr=pr_number)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        repo = resolve_repo_slug(args.repo)
        pr_number = resolve_pr_number(repo, args.pr)
        snapshot = fetch_snapshot(repo, pr_number)
        digest = build_digest(
            repo,
            pr_number,
            include_resolved=args.include_resolved,
            since_latest_commit=args.since_latest_commit,
        )
        failed_run_logs: tuple[FailedRunLog, ...] = ()
        if args.include_failed_run_logs:
            failed_run_logs = fetch_failed_run_logs(
                repo,
                snapshot.head_sha,
                max_runs=max(1, args.max_failed_runs),
                max_log_chars=max(500, args.max_log_chars),
            )
        rendered = render_bundle_codex_zh(snapshot, digest, failed_run_logs)

        if args.request_copilot_review:
            result = request_copilot_review(repo, pr_number)
            rendered = "\n".join(
                [
                    rendered,
                    "",
                    "## Copilot Review Request",
                    result or "Copilot review request submitted.",
                ]
            )

        output_path = args.output or default_bundle_output_path(pr_number)
        write_output(rendered, output_path)
        print(f"Review bundle written to: {output_path}")
        print(f"PR #{snapshot.number}: {snapshot.title}")
        print(f"Copilot threads included: {len(digest.threads)}")
        if args.since_latest_commit:
            print("Filtered to comments since latest head commit: yes")
        if args.include_failed_run_logs:
            print(f"Failed run logs included: {len(failed_run_logs)}")
        return 0
    except Exception as exc:  # pragma: no cover - exercised via CLI behavior
        parser.exit(
            1,
            f"error: {exc}\n"
            "tip: if the current branch has no PR, rerun with --pr <number> or --pr <url>\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
