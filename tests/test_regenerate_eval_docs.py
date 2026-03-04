from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.regenerate_eval_docs import (
    EvalDocDriftError,
    _compute_version_stamp,
    build_regenerated_docs,
    regenerate_eval_docs,
)


def _write_sample_index_and_policy(tmp_path: Path) -> tuple[Path, Path]:
    index_path = tmp_path / "Doc" / "Evaluation" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "documents": [
            {
                "doc_id": "sample_doc",
                "source_path": "Doc\\Evaluation\\sample_doc.md",
                "chunks": [
                    {
                        "chunk_id": "sample_doc_0",
                        "heading": "Sample Title",
                        "text": "Intro line 1\nIntro line 2",
                    },
                    {
                        "chunk_id": "sample_doc_1",
                        "heading": "Section A",
                        "text": "Alpha\nBeta",
                    },
                ],
            }
        ]
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    policy_path = tmp_path / "knowledge_source_policy.yaml"
    policy_path.write_text(
        "policy_id: sample_policy\n"
        "version: v1\n"
        "index_path: Doc/Evaluation/index.json\n"
        "allow:\n"
        "  - doc_id: sample_doc\n"
        "    chunk_id: sample_doc_0\n"
        "    line_range: [1, 2]\n"
        "  - doc_id: sample_doc\n"
        "    chunk_id: sample_doc_1\n"
        "    line_range: [1, 2]\n",
        encoding="utf-8",
    )
    return index_path, policy_path


def test_regenerate_eval_docs_snapshot(tmp_path: Path) -> None:
    index_path, policy_path = _write_sample_index_and_policy(tmp_path)
    rendered = build_regenerated_docs(
        index_path=index_path,
        policy_path=policy_path,
        repo_root=tmp_path,
        version_stamp_override="snapshot-v1",
    )

    assert len(rendered) == 1
    snapshot_path = Path(__file__).resolve().parent / "fixtures" / "eval_doc_snapshot_sample.md"
    expected = snapshot_path.read_text(encoding="utf-8")
    assert rendered[0].content == expected


def test_regenerate_eval_docs_check_mode_detects_drift(tmp_path: Path) -> None:
    index_path, policy_path = _write_sample_index_and_policy(tmp_path)
    changed = regenerate_eval_docs(
        index_path=index_path,
        policy_path=policy_path,
        repo_root=tmp_path,
        check=False,
    )
    assert changed == [(tmp_path / "Doc" / "Evaluation" / "sample_doc.md").resolve()]

    generated_path = changed[0]
    generated_path.write_text(generated_path.read_text(encoding="utf-8") + "\nmanual edit\n", encoding="utf-8")

    with pytest.raises(EvalDocDriftError) as exc_info:
        regenerate_eval_docs(
            index_path=index_path,
            policy_path=policy_path,
            repo_root=tmp_path,
            check=True,
        )
    assert exc_info.value.drift_paths == [generated_path]


def test_compute_version_stamp_is_stable_across_line_endings(tmp_path: Path) -> None:
    index_lf = tmp_path / "index_lf.json"
    index_crlf = tmp_path / "index_crlf.json"
    policy_lf = tmp_path / "policy_lf.yaml"
    policy_crlf = tmp_path / "policy_crlf.yaml"

    index_payload = '{\n  "documents": []\n}\n'
    policy_payload = "policy_id: p\nversion: v1\nindex_path: Doc/Evaluation/index.json\nallow: []\n"

    index_lf.write_text(index_payload, encoding="utf-8")
    index_crlf.write_text(index_payload.replace("\n", "\r\n"), encoding="utf-8")
    policy_lf.write_text(policy_payload, encoding="utf-8")
    policy_crlf.write_text(policy_payload.replace("\n", "\r\n"), encoding="utf-8")

    stamp_lf = _compute_version_stamp(index_lf, policy_lf)
    stamp_crlf = _compute_version_stamp(index_crlf, policy_crlf)
    assert stamp_lf == stamp_crlf
