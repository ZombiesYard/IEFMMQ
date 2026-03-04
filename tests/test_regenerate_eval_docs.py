from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.regenerate_eval_docs as regen
from tools.regenerate_eval_docs import (
    EvalDocDriftError,
    EvalDocRegenerationError,
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


def test_build_regenerated_docs_rejects_absolute_source_path(tmp_path: Path) -> None:
    index_path = tmp_path / "Doc" / "Evaluation" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "documents": [
            {
                "doc_id": "sample_doc",
                "source_path": "/tmp/evil.md",
                "chunks": [{"chunk_id": "sample_doc_0", "heading": "H", "text": "L1"}],
            }
        ]
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    policy_path = tmp_path / "knowledge_source_policy.yaml"
    policy_path.write_text(
        "policy_id: p\n"
        "version: v1\n"
        "index_path: Doc/Evaluation/index.json\n"
        "allow:\n"
        "  - doc_id: sample_doc\n"
        "    chunk_id: sample_doc_0\n"
        "    line_range: [1, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalDocRegenerationError, match="must be relative to repo root"):
        build_regenerated_docs(index_path=index_path, policy_path=policy_path, repo_root=tmp_path)


def test_build_regenerated_docs_rejects_path_traversal(tmp_path: Path) -> None:
    index_path = tmp_path / "Doc" / "Evaluation" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "documents": [
            {
                "doc_id": "sample_doc",
                "source_path": "../outside.md",
                "chunks": [{"chunk_id": "sample_doc_0", "heading": "H", "text": "L1"}],
            }
        ]
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    policy_path = tmp_path / "knowledge_source_policy.yaml"
    policy_path.write_text(
        "policy_id: p\n"
        "version: v1\n"
        "index_path: Doc/Evaluation/index.json\n"
        "allow:\n"
        "  - doc_id: sample_doc\n"
        "    chunk_id: sample_doc_0\n"
        "    line_range: [1, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalDocRegenerationError, match="escapes repo root"):
        build_regenerated_docs(index_path=index_path, policy_path=policy_path, repo_root=tmp_path)


def test_regenerate_eval_docs_reports_non_utf8_existing_file(tmp_path: Path) -> None:
    index_path, policy_path = _write_sample_index_and_policy(tmp_path)
    changed = regenerate_eval_docs(
        index_path=index_path,
        policy_path=policy_path,
        repo_root=tmp_path,
        check=False,
    )
    generated_path = changed[0]
    generated_path.write_bytes(b"\xff\xfe\xfd")

    with pytest.raises(EvalDocRegenerationError, match="not valid UTF-8"):
        regenerate_eval_docs(
            index_path=index_path,
            policy_path=policy_path,
            repo_root=tmp_path,
            check=True,
        )


def test_build_regenerated_docs_parses_inputs_once_for_version_stamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_path, policy_path = _write_sample_index_and_policy(tmp_path)
    call_count = {"json_loads": 0, "yaml_safe_load": 0}
    real_json_loads = regen.json.loads
    real_yaml_safe_load = regen.yaml.safe_load

    def _counting_json_loads(*args, **kwargs):
        call_count["json_loads"] += 1
        return real_json_loads(*args, **kwargs)

    def _counting_yaml_safe_load(*args, **kwargs):
        call_count["yaml_safe_load"] += 1
        return real_yaml_safe_load(*args, **kwargs)

    monkeypatch.setattr(regen.json, "loads", _counting_json_loads)
    monkeypatch.setattr(regen.yaml, "safe_load", _counting_yaml_safe_load)

    rendered = build_regenerated_docs(
        index_path=index_path,
        policy_path=policy_path,
        repo_root=tmp_path,
    )
    assert len(rendered) == 1
    assert call_count["json_loads"] == 1
    assert call_count["yaml_safe_load"] == 1


def test_render_preserves_selected_leading_blank_lines(tmp_path: Path) -> None:
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
                        "text": "\nAlpha",
                    }
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
        "    line_range: [1, 2]\n",
        encoding="utf-8",
    )
    rendered = build_regenerated_docs(
        index_path=index_path,
        policy_path=policy_path,
        repo_root=tmp_path,
        version_stamp_override="snapshot-v1",
    )
    assert len(rendered) == 1
    assert "# Sample Title\n\n\nAlpha" in rendered[0].content
