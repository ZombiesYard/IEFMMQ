"""
Aggregate HelpResponse failure codes from JSONL event logs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_help_failures(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    primary_counts: Counter[str] = Counter()
    all_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    total_tutor_responses = 0

    for event in events:
        kind = event.get("kind") or event.get("type")
        if kind != "tutor_response":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue

        total_tutor_responses += 1
        status = payload.get("status")
        if isinstance(status, str) and status:
            status_counts[status] += 1

        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            continue

        provider = metadata.get("provider")
        if isinstance(provider, str) and provider:
            provider_counts[provider] += 1

        failure_code = metadata.get("failure_code")
        if isinstance(failure_code, str) and failure_code:
            primary_counts[failure_code] += 1

        failure_codes = metadata.get("failure_codes")
        if isinstance(failure_codes, list):
            for code in failure_codes:
                if isinstance(code, str) and code:
                    all_counts[code] += 1

    return {
        "total_tutor_responses": total_tutor_responses,
        "primary_failure_code_counts": dict(sorted(primary_counts.items())),
        "all_failure_code_counts": dict(sorted(all_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "provider_counts": dict(sorted(provider_counts.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate HelpResponse failure codes from JSONL logs.")
    parser.add_argument("log", help="Path to JSONL event log")
    args = parser.parse_args(argv)

    summary = summarize_help_failures(load_jsonl(args.log))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
