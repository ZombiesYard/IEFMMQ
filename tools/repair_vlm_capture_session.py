from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            rows.append(payload)
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temp_path.replace(path)


def _normalize_frame_id(text: str) -> str:
    value = str(text).strip().replace("\\", "/")
    name = Path(value).name
    if name.endswith(".png"):
        name = name[:-4]
    if name.endswith("_vlm"):
        name = name[:-4]
    return name


def _parse_seq_range(text: str) -> tuple[int, int]:
    raw = str(text).strip()
    if ":" in raw:
        start_text, end_text = raw.split(":", 1)
    elif "-" in raw:
        start_text, end_text = raw.split("-", 1)
    else:
        raise ValueError(f"expected seq range like 153:176, got {text!r}")
    start = int(start_text)
    end = int(end_text)
    if start > end:
        raise ValueError(f"invalid descending seq range: {text!r}")
    return start, end


def _move_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def _backup_session(session_dir: Path, backup_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / f"{session_dir.name}_{timestamp}"
    suffix = 1
    while backup_dir.exists():
        backup_dir = backup_root / f"{session_dir.name}_{timestamp}_{suffix}"
        suffix += 1
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(session_dir, backup_dir)
    return backup_dir


def _plan_progress_row(
    *,
    plan_row: dict[str, str],
    old_row: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = dict(plan_row)
    for field in ("seq", "total"):
        try:
            out[field] = int(out[field])
        except (TypeError, ValueError):
            pass
    category_id = str(plan_row.get("category_id", "")).strip()
    out.update(
        {
            "captured_frame_id": old_row.get("captured_frame_id") or "",
            "capture_reason": f"manual_plan:{category_id}" if category_id else old_row.get("capture_reason", ""),
            "artifact_image_path": old_row.get("artifact_image_path") or "",
            "raw_image_path": old_row.get("raw_image_path") or "",
            "captured_at_wall_ms": old_row.get("captured_at_wall_ms") or "",
            "status": old_row.get("status") or "captured",
            "operator_note": old_row.get("operator_note") or "",
        }
    )
    return out


def repair_session(
    *,
    session_dir: Path,
    remove_frame_ids: set[str],
    remove_frame_seq_ranges: list[tuple[int, int]],
    backup_root: Path,
    apply: bool,
    expected_final_count: int | None,
) -> dict[str, Any]:
    session_dir = session_dir.expanduser().resolve()
    backup_root = backup_root.expanduser().resolve()
    frames_path = session_dir / "frames.jsonl"
    capture_index_path = session_dir / "capture_index.jsonl"
    plan_path = session_dir / "capture_plan.csv"
    progress_csv_path = session_dir / "capture_plan_progress.csv"
    progress_jsonl_path = session_dir / "capture_plan_progress.jsonl"

    if not session_dir.exists():
        raise FileNotFoundError(f"session dir does not exist: {session_dir}")
    for required in [frames_path, capture_index_path, plan_path, progress_csv_path, progress_jsonl_path]:
        if not required.exists():
            raise FileNotFoundError(f"required file is missing: {required}")

    frames = _load_jsonl(frames_path)
    capture_index = _load_jsonl(capture_index_path)
    progress = _load_jsonl(progress_jsonl_path)
    plan_fieldnames, plan_rows = _load_csv(plan_path)
    progress_fieldnames, _progress_csv_rows = _load_csv(progress_csv_path)

    remove_ids = {_normalize_frame_id(frame_id) for frame_id in remove_frame_ids if str(frame_id).strip()}
    for start, end in remove_frame_seq_ranges:
        for frame in frames:
            frame_seq = int(frame.get("frame_seq", -1))
            if start <= frame_seq <= end:
                remove_ids.add(str(frame.get("frame_id", "")))
    remove_ids.discard("")

    existing_ids = {str(frame.get("frame_id", "")) for frame in frames}
    missing_ids = sorted(remove_ids - existing_ids)
    if missing_ids:
        raise ValueError(f"requested frame ids are not present in frames.jsonl: {missing_ids}")

    remaining_progress = [
        row for row in progress if _normalize_frame_id(str(row.get("captured_frame_id", ""))) not in remove_ids
    ]
    if len(remaining_progress) > len(plan_rows):
        raise ValueError("more remaining progress rows than capture plan rows")
    if expected_final_count is not None and len(remaining_progress) != expected_final_count:
        raise ValueError(
            f"expected {expected_final_count} remaining progress rows, got {len(remaining_progress)}"
        )

    rebuilt_progress: list[dict[str, Any]] = []
    frame_to_reason: dict[str, str] = {}
    for index, old_row in enumerate(remaining_progress):
        plan_row = dict(plan_rows[index])
        plan_row["seq"] = str(index + 1)
        rebuilt = _plan_progress_row(plan_row=plan_row, old_row=old_row)
        rebuilt_progress.append(rebuilt)
        frame_id = _normalize_frame_id(str(rebuilt.get("captured_frame_id", "")))
        if frame_id:
            frame_to_reason[frame_id] = str(rebuilt.get("capture_reason", ""))

    remaining_frames = [
        frame for frame in frames if str(frame.get("frame_id", "")) not in remove_ids
    ]
    remaining_capture_index: list[dict[str, Any]] = []
    for row in capture_index:
        frame_id = str(row.get("frame_id", ""))
        if frame_id in remove_ids:
            continue
        if frame_id in frame_to_reason:
            row = dict(row)
            row["capture_reason"] = frame_to_reason[frame_id]
        remaining_capture_index.append(row)

    summary = {
        "session_dir": str(session_dir),
        "apply": apply,
        "original_frames": len(frames),
        "original_progress_rows": len(progress),
        "remove_count": len(remove_ids),
        "remove_frame_ids": sorted(remove_ids),
        "remaining_frames": len(remaining_frames),
        "remaining_progress_rows": len(rebuilt_progress),
        "next_manual_plan_start_seq": len(rebuilt_progress) + 1,
    }
    if not apply:
        return summary

    backup_dir = _backup_session(session_dir, backup_root)
    quarantine_dir = session_dir / ".repair_quarantine" / backup_dir.name
    moved_files: list[str] = []
    for frame_id in sorted(remove_ids):
        raw_src = session_dir / "raw" / f"{frame_id}.png"
        artifact_src = session_dir / "artifacts" / f"{frame_id}_vlm.png"
        if _move_if_exists(raw_src, quarantine_dir / "raw" / raw_src.name):
            moved_files.append(str(raw_src))
        if _move_if_exists(artifact_src, quarantine_dir / "artifacts" / artifact_src.name):
            moved_files.append(str(artifact_src))

    _write_jsonl_atomic(frames_path, remaining_frames)
    _write_jsonl_atomic(capture_index_path, remaining_capture_index)
    _write_jsonl_atomic(progress_jsonl_path, rebuilt_progress)
    _write_csv_atomic(progress_csv_path, progress_fieldnames, rebuilt_progress)
    summary["backup_dir"] = str(backup_dir)
    summary["quarantine_dir"] = str(quarantine_dir)
    summary["moved_files"] = moved_files

    summary_path = session_dir / "capture_repair_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair a manual VLM capture session after bad extra frames.")
    parser.add_argument("--session-dir", required=True, help="Capture session directory to repair.")
    parser.add_argument(
        "--remove-frame-id",
        action="append",
        default=[],
        help="Frame id, raw PNG name, or *_vlm artifact stem to remove. May be repeated.",
    )
    parser.add_argument(
        "--remove-frame-seq-range",
        action="append",
        default=[],
        help="Inclusive frame_seq range to remove, for example 153:176.",
    )
    parser.add_argument(
        "--backup-root",
        default=".tmp/capture_repair_backups",
        help="Where to store a full copy of the session before applying changes.",
    )
    parser.add_argument("--expected-final-count", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="Apply the repair. Without this, only print a dry run.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        summary = repair_session(
            session_dir=Path(args.session_dir),
            remove_frame_ids=set(args.remove_frame_id),
            remove_frame_seq_ranges=[_parse_seq_range(item) for item in args.remove_frame_seq_range],
            backup_root=Path(args.backup_root),
            apply=bool(args.apply),
            expected_final_count=args.expected_final_count,
        )
    except Exception as exc:
        print(f"[REPAIR_VLM_CAPTURE] failed: {type(exc).__name__}: {exc}")
        return 1

    print("[REPAIR_VLM_CAPTURE] " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
