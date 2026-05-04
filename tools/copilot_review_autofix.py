"""
Generate a Codex autofix prompt from Copilot review context and run `codex exec`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

from tools.copilot_review_digest import build_digest, resolve_pr_number, resolve_repo_slug
from tools.copilot_review_loop import (
    DEFAULT_BUNDLE_OUTPUT_TEMPLATE,
    FailedRunLog,
    default_bundle_output_path,
    fetch_failed_run_logs,
    fetch_snapshot,
    request_copilot_review,
    render_bundle_codex_zh,
)


DEFAULT_LAST_MESSAGE_OUTPUT_TEMPLATE = ".tmp/copilot_autofix_last_message_pr{pr}.md"
DEFAULT_CODEX_CANDIDATES = ("codex",)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Codex autofix prompt from Copilot review context and run codex exec."
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
    parser.add_argument(
        "--bundle-output",
        default="",
        help=f"Where to write the generated Codex bundle prompt. Defaults to {DEFAULT_BUNDLE_OUTPUT_TEMPLATE}.",
    )
    parser.add_argument(
        "--last-message-output",
        default="",
        help="Where to write the last Codex message. Defaults to .tmp/copilot_autofix_last_message_pr<PR>.md.",
    )
    parser.add_argument(
        "--codex-model",
        default="",
        help="Optional model override for `codex exec`.",
    )
    parser.add_argument(
        "--codex-bin",
        default="",
        help="Optional explicit path to the `codex` executable. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--codex-profile",
        default="",
        help="Optional profile override for `codex exec`.",
    )
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
        help="Sandbox mode passed to `codex exec`.",
    )
    parser.add_argument(
        "--codex-approval-policy",
        choices=("untrusted", "on-request", "never"),
        default="never",
        help="Approval policy passed to `codex exec`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate the prompt bundle and print the codex command, without running Codex.",
    )
    parser.add_argument(
        "--post-command",
        action="append",
        default=[],
        help="Optional shell command to run after Codex finishes, before any git/PR automation. Repeatable.",
    )
    parser.add_argument(
        "--git-commit",
        action="store_true",
        help="Stage all changes with `git add -A` and create a commit after Codex/post commands succeed.",
    )
    parser.add_argument(
        "--commit-message",
        default="chore: apply Codex autofix for Copilot review",
        help="Commit message used with --git-commit.",
    )
    parser.add_argument(
        "--git-push",
        action="store_true",
        help="Push the current branch after a successful autofix commit.",
    )
    parser.add_argument(
        "--request-copilot-review",
        action="store_true",
        help="Re-request Copilot review after a successful push.",
    )
    return parser


def render_autofix_prompt(
    bundle_text: str,
    *,
    repo: str,
    pr_number: int,
) -> str:
    lines = [
        f"请处理 {repo} 的 PR #{pr_number}。",
        "下面已经整理好了 PR 状态、Copilot review comments，以及可选的失败 CI 日志。",
        "请直接在当前工作区修改代码，不要只给建议。",
        "要求：",
        "1. 先判断哪些 review 意见应该接受，哪些应该拒绝。",
        "2. 如果拒绝某条 review 意见，必须在最终总结里明确说明理由。",
        "3. 优先修复真实问题，保持改动最小。",
        "4. 运行与你改动相关的测试，并在最终总结里写出结果。",
        "5. 不要自动执行 git push、gh pr edit、gh merge 等远程操作。",
        "",
        "下面是整理后的 PR 修复包：",
        "",
        bundle_text.rstrip(),
        "",
        "完成后请在最终总结中至少包含：",
        "- 接受了哪些 review items",
        "- 拒绝了哪些 review items 及理由",
        "- 修改了哪些文件",
        "- 运行了哪些测试，结果如何",
        "- 是否还有剩余风险",
    ]
    return "\n".join(lines)


def write_text(path_text: str, content: str) -> Path:
    path = Path(path_text)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def default_last_message_output_path(pr_number: int) -> str:
    return DEFAULT_LAST_MESSAGE_OUTPUT_TEMPLATE.format(pr=pr_number)


def build_codex_exec_command(
    *,
    codex_bin: str,
    prompt_path: Path,
    last_message_path: Path,
    model: str,
    profile: str,
    sandbox: str,
    approval_policy: str,
) -> list[str]:
    cmd = [
        codex_bin,
        "exec",
        "-C",
        str(Path.cwd()),
        "-s",
        sandbox,
        "-a",
        approval_policy,
        "-o",
        str(last_message_path),
        "-",
    ]
    if model:
        cmd[2:2] = ["-m", model]
    if profile:
        insert_at = 2 if not model else 4
        cmd[insert_at:insert_at] = ["-p", profile]
    return cmd


def _is_executable_file(path_text: str) -> bool:
    path = Path(path_text)
    return path.is_file() and os.access(path, os.X_OK)


def resolve_codex_binary(explicit_path: str) -> str:
    if explicit_path.strip():
        candidate = explicit_path.strip()
        if _is_executable_file(candidate):
            return candidate
        if Path(candidate).is_file():
            raise RuntimeError(f"configured codex binary is not executable: {candidate}")
        raise RuntimeError(f"configured codex binary does not exist: {candidate}")

    for candidate in DEFAULT_CODEX_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        "could not locate `codex` in PATH; rerun with --codex-bin /path/to/codex"
    )


def run_codex_exec(command: Sequence[str], prompt_text: str) -> int:
    result = subprocess.run(
        list(command),
        input=prompt_text,
        text=True,
    )
    return int(result.returncode)


def run_shell_command(command: str) -> None:
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"post command failed with exit code {result.returncode}: {command}")


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result.stdout.strip()


def get_current_branch() -> str:
    return run_git("rev-parse", "--abbrev-ref", "HEAD")


def has_staged_changes() -> bool:
    result = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
    return result.returncode == 1


def stage_all_and_commit(commit_message: str) -> bool:
    run_git("add", "-A")
    if not has_staged_changes():
        return False
    run_git("commit", "-m", commit_message)
    return True


def push_current_branch(branch_name: str) -> None:
    run_git("push", "origin", branch_name)


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
        bundle_text = render_bundle_codex_zh(snapshot, digest, failed_run_logs)
        prompt_text = render_autofix_prompt(bundle_text, repo=repo, pr_number=pr_number)

        bundle_output = args.bundle_output or default_bundle_output_path(pr_number)
        last_message_output = args.last_message_output or default_last_message_output_path(pr_number)
        bundle_path = write_text(bundle_output, prompt_text)
        last_message_path = Path(last_message_output)
        if last_message_path.parent != Path("."):
            last_message_path.parent.mkdir(parents=True, exist_ok=True)

        command = build_codex_exec_command(
            codex_bin=resolve_codex_binary(args.codex_bin),
            prompt_path=bundle_path,
            last_message_path=last_message_path,
            model=args.codex_model,
            profile=args.codex_profile,
            sandbox=args.codex_sandbox,
            approval_policy=args.codex_approval_policy,
        )

        print(f"Prompt bundle written to: {bundle_path}")
        print(f"Codex last message path: {last_message_path}")
        print("Codex command:")
        print(" ".join(command))

        if args.dry_run:
            if args.post_command:
                print("Post commands:")
                for command_text in args.post_command:
                    print(f"- {command_text}")
            if args.git_commit:
                print(f"Will commit with message: {args.commit_message}")
            if args.git_push:
                print(f"Will push current branch: {get_current_branch()}")
            if args.request_copilot_review:
                print("Will re-request Copilot review after push.")
            print("Dry run enabled; Codex was not executed.")
            return 0

        exit_code = run_codex_exec(command, prompt_text)
        if exit_code != 0:
            raise RuntimeError(f"codex exec failed with exit code {exit_code}")
        print("Codex autofix completed.")

        if args.post_command:
            print("Running post commands...")
            for command_text in args.post_command:
                print(f"> {command_text}")
                run_shell_command(command_text)

        committed = False
        branch_name = get_current_branch()
        if args.git_commit:
            committed = stage_all_and_commit(args.commit_message)
            if committed:
                print(f"Created commit on branch {branch_name}: {args.commit_message}")
            else:
                print("No staged changes detected after `git add -A`; skipping commit.")

        if args.git_push:
            if not committed:
                raise RuntimeError("refusing to push automatically because no commit was created in this run")
            push_current_branch(branch_name)
            print(f"Pushed branch to origin: {branch_name}")

        if args.request_copilot_review:
            if not args.git_push:
                raise RuntimeError("requesting Copilot review automatically requires --git-push in this tool")
            request_result = request_copilot_review(repo, pr_number)
            print(request_result or "Copilot review request submitted.")
        return 0
    except Exception as exc:  # pragma: no cover - exercised via CLI behavior
        parser.exit(
            1,
            f"error: {exc}\n"
            "tip: if the current branch has no PR, rerun with --pr <number> or --pr <url>\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
