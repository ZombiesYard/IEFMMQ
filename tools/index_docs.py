"""
Index markdown/pdf documents into a simple JSON structure for offline retrieval.

Note: this module is a placeholder for future LLM integration and is not
referenced by the runtime code paths today.

Usage:
  python -m tools.index_docs --output index.json
  # defaults to indexing all docs under Doc/Evaluation
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import PyPDF2


def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def chunk_markdown(path: Path) -> List[Dict]:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    chunks = []
    current_heading = None
    buf: List[str] = []
    for line in lines:
        if line.startswith("#"):
            if buf:
                chunks.append({"heading": current_heading, "text": "\n".join(buf)})
                buf = []
            current_heading = line.lstrip("# ").strip()
        else:
            buf.append(line)
    if buf:
        chunks.append({"heading": current_heading, "text": "\n".join(buf)})
    return chunks


def chunk_pdf(path: Path) -> List[Dict]:
    reader = PyPDF2.PdfReader(str(path))
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        chunks.append({"page": i + 1, "text": text})
    return chunks


def build_index(input_paths: List[str], output: str):
    documents = []
    for p in input_paths:
        path = Path(p)
        if path.is_dir():
            files = sorted(path.glob("**/*"))
        else:
            files = [path]
        for file in files:
            if file.suffix.lower() in {".md", ".markdown"}:
                chunks = chunk_markdown(file)
            elif file.suffix.lower() == ".pdf":
                chunks = chunk_pdf(file)
            else:
                continue
            doc_entry = {
                "doc_id": file.stem,
                "source_path": str(file),
                "chunks": [],
            }
            for idx, ch in enumerate(chunks):
                doc_entry["chunks"].append(
                    {
                        "chunk_id": f"{file.stem}_{idx}",
                        "heading": ch.get("heading"),
                        "page": ch.get("page"),
                        "text": ch["text"],
                    }
                )
            documents.append(doc_entry)
    out = {"documents": documents}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="Index markdown/pdf documents")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Files or directories to index (default: Doc/Evaluation)",
    )
    parser.add_argument("--output", required=True, help="Output index JSON path")
    args = parser.parse_args()

    inputs = args.input or ["Doc/Evaluation"]
    build_index(inputs, args.output)
    print(f"Wrote index to {args.output} from {inputs}")


if __name__ == "__main__":
    main()
