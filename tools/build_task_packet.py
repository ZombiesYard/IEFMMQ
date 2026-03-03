#!/usr/bin/env python3
"""Build a private task packet from a GitHub issue for Codex execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import subprocess
import sys
import textwrap
import re


DEFAULT_PRIVATE_DIR = Path.home() / ".simtutor-private"
REQUIRED_SECTIONS = ("Goal", "Scope", "Acceptance", "Test Plan", "Out of Scope")

DEFAULT_CONTRACT_TEMPLATE = textwrap.dedent(
    """\
    # Private Issue Contract

    ## ROLE
    You are principal software architect + senior Python engineer.

    ## LANGUAGE
    Reply in Chinese.

    ## HARD CONSTRAINTS
    - One issue only; do not expand scope.
    - Keep architecture boundaries and safety guardrails.
    - Keep v1 schema compatibility unless issue explicitly requires otherwise.
    """
)

DEFAULT_WRAPPER_TEMPLATE = textwrap.dedent(
    """\
    ## Task Input
    - Issue Number: {{ISSUE_NUMBER}}
    - Issue Title: {{ISSUE_TITLE}}
    - Issue URL: {{ISSUE_URL}}

    ## Required Sections
    {{REQUIRED_SECTIONS}}

    ## Issue Body
    {{ISSUE_BODY}}
    """
)

DEFAULT_ISSUE_FIELDS_TEMPLATE = textwrap.dedent(
    """\
    ### Goal
    [One-paragraph objective]

    ### Scope
    - Must change:
    - May change:
    - Must not change:

    ### Acceptance
    - [ ] acceptance criterion 1
    - [ ] acceptance criterion 2

    ### Test Plan
    - commands:
    - scenarios:

    ### Out of Scope
    - item 1
    """
)

PACKET_OUTPUT_REQUIREMENTS = textwrap.dedent(
    """\
    ## Output Contract
    The implementation output must include:

    1. 变更摘要
    2. 文件清单
    3. 关键设计决策
    4. 测试结果（命令+结果）
    5. 风险与降级策略
    """
)


@dataclass(frozen=True)
class IssueData:
    number: int
    title: str
    body: str
    url: str


def _section_present(body: str, section_name: str) -> bool:
    heading = re.compile(
        rf"(?im)^\s*#{{1,6}}\s*{re.escape(section_name)}\s*:?\s*$"
    )
    label = re.compile(rf"(?im)^\s*{re.escape(section_name)}\s*:\s*.+$")
    bold_label = re.compile(rf"(?im)^\s*\*\*{re.escape(section_name)}\*\*\s*:\s*.*$")
    return bool(heading.search(body) or label.search(body) or bold_label.search(body))


def validate_issue_body(body: str) -> list[str]:
    missing: list[str] = []
    for section in REQUIRED_SECTIONS:
        if not _section_present(body, section):
            missing.append(section)
    return missing


def parse_issue_json(raw: str) -> IssueData:
    payload = json.loads(raw)
    return IssueData(
        number=int(payload["number"]),
        title=str(payload.get("title", "")).strip(),
        body=str(payload.get("body", "")),
        url=str(payload.get("url", "")).strip(),
    )


def fetch_issue_via_gh(issue_number: int, repo: str | None = None) -> IssueData:
    cmd = [
        "gh",
        "issue",
        "view",
        str(issue_number),
        "--json",
        "number,title,body,url,labels,assignees",
    ]
    if repo:
        cmd.extend(["--repo", repo])
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        err = completed.stderr.strip() or completed.stdout.strip() or "unknown gh error"
        raise RuntimeError(f"Failed to fetch issue via gh: {err}")
    return parse_issue_json(completed.stdout)


def _read_required_file(path: Path, help_hint: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {help_hint}")
    return path.read_text(encoding="utf-8")


def render_packet(issue: IssueData, contract_text: str, wrapper_text: str) -> str:
    required_sections_md = "\n".join(f"- {s}" for s in REQUIRED_SECTIONS)
    issue_json = json.dumps(
        {
            "number": issue.number,
            "title": issue.title,
            "url": issue.url,
        },
        ensure_ascii=False,
        indent=2,
    )
    rendered_wrapper = (
        wrapper_text.replace("{{ISSUE_NUMBER}}", str(issue.number))
        .replace("{{ISSUE_TITLE}}", issue.title)
        .replace("{{ISSUE_URL}}", issue.url)
        .replace("{{ISSUE_BODY}}", issue.body)
        .replace("{{ISSUE_JSON}}", issue_json)
        .replace("{{REQUIRED_SECTIONS}}", required_sections_md)
    )
    return (
        f"{contract_text.rstrip()}\n\n---\n\n"
        f"{rendered_wrapper.rstrip()}\n\n---\n\n"
        f"{PACKET_OUTPUT_REQUIREMENTS.rstrip()}\n"
    )


def bootstrap_private_dir(private_dir: Path, overwrite: bool = False) -> None:
    private_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "issue_contract.md": DEFAULT_CONTRACT_TEMPLATE,
        "issue_prompt_wrapper.md": DEFAULT_WRAPPER_TEMPLATE,
        "issue_fields_template.md": DEFAULT_ISSUE_FIELDS_TEMPLATE,
    }
    for name, content in files.items():
        path = private_dir / name
        if path.exists() and not overwrite:
            continue
        path.write_text(content, encoding="utf-8")


def build_task_packet(
    issue: IssueData,
    private_dir: Path,
    output_path: Path,
    allow_missing_sections: bool = False,
) -> Path:
    missing = validate_issue_body(issue.body)
    if missing and not allow_missing_sections:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Issue body missing required sections: {missing_list}. "
            "Fix the issue body before generating task packet."
        )

    contract_text = _read_required_file(
        private_dir / "issue_contract.md",
        "Run: python -m tools.build_task_packet --init-private-dir",
    )
    wrapper_text = _read_required_file(
        private_dir / "issue_prompt_wrapper.md",
        "Run: python -m tools.build_task_packet --init-private-dir",
    )
    packet = render_packet(issue=issue, contract_text=contract_text, wrapper_text=wrapper_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packet, encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--issue-number", type=int, help="GitHub issue number to fetch via gh.")
    source.add_argument("--issue-json", type=str, help="Path to issue JSON payload.")
    p.add_argument("--repo", type=str, default=None, help="owner/repo for gh issue view.")
    p.add_argument(
        "--private-dir",
        type=str,
        default=str(DEFAULT_PRIVATE_DIR),
        help="Private directory that stores local contracts and wrappers.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output task packet path. Defaults to /tmp/task_packet_<issue>.md",
    )
    p.add_argument(
        "--allow-missing-sections",
        action="store_true",
        help="Generate packet even when required issue sections are missing.",
    )
    p.add_argument(
        "--init-private-dir",
        action="store_true",
        help="Create default private templates in --private-dir and exit if no issue source is provided.",
    )
    p.add_argument(
        "--overwrite-private-templates",
        action="store_true",
        help="Overwrite existing templates when using --init-private-dir.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    private_dir = Path(args.private_dir).expanduser().resolve()
    if args.init_private_dir:
        bootstrap_private_dir(private_dir, overwrite=bool(args.overwrite_private_templates))
        print(f"Initialized private templates at: {private_dir}")
        if not args.issue_number and not args.issue_json:
            return 0

    if not args.issue_number and not args.issue_json:
        parser.error("one of --issue-number/--issue-json is required unless only --init-private-dir is used")

    if args.issue_json:
        issue = parse_issue_json(Path(args.issue_json).read_text(encoding="utf-8"))
    else:
        issue = fetch_issue_via_gh(issue_number=int(args.issue_number), repo=args.repo)

    output = Path(args.output).expanduser() if args.output else Path(f"/tmp/task_packet_{issue.number}.md")
    output = output.resolve()
    try:
        result = build_task_packet(
            issue=issue,
            private_dir=private_dir,
            output_path=output,
            allow_missing_sections=bool(args.allow_missing_sections),
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Task packet generated: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
