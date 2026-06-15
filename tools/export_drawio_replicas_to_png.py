#!/usr/bin/env python3
"""Export thesis draw.io replica pages as PNG assets.

The important constraint is that draw.io must do the rendering. This script
only calls the draw.io CLI; it does not parse or redraw diagram geometry.
"""

from __future__ import annotations

import argparse
import binascii
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib


PAGE_NAMES = [
    "fig_system_architecture",
    "fig_evidence_authority",
    "fig_help_cycle",
    "fig_vlm_pipeline",
    "fig_ontology_evolution",
    "fig_evaluation_dataflow",
    "fig_pilot_protocol",
    "fig_harness_validation",
    "fig_deployment_topology",
    "fig_experiment_export",
]

DEFAULT_SOURCE = "paper/figures/drawio/cropped_replicas_all.drawio"

DEFAULT_OUTPUT_ROOTS = ["paper/figures", "paper_zh/figures"]

DRAWIO_CANDIDATES = [
    "/mnt/k/drawIO/draw.io/draw.io.exe",
    "/mnt/c/Program Files/draw.io/draw.io.exe",
    "/mnt/c/Program Files (x86)/draw.io/draw.io.exe",
    "drawio",
    "draw.io",
]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngError(RuntimeError):
    pass


def resolve_drawio(explicit: str | None) -> str:
    candidates = [explicit] if explicit else []
    candidates.extend(DRAWIO_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.expanduser(candidate)
        if os.path.isabs(expanded) and Path(expanded).exists():
            return expanded
        located = shutil.which(expanded)
        if located:
            return located
    raise FileNotFoundError(
        "draw.io CLI not found. Set DRAWIO_CLI or pass --drawio."
    )


def wsl_windows_path(path: Path) -> str:
    if shutil.which("wslpath"):
        return subprocess.check_output(
            ["wslpath", "-w", str(path)], text=True
        ).strip()
    return str(path)


def read_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if not data.startswith(PNG_SIGNATURE):
        raise PngError("not a PNG file")
    chunks: list[tuple[bytes, bytes]] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 8 > len(data):
            raise PngError("truncated PNG chunk header")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        crc_end = offset + 8 + length + 4
        if crc_end > len(data):
            raise PngError("truncated PNG chunk data")
        chunks.append((chunk_type, chunk_data))
        offset = crc_end
        if chunk_type == b"IEND":
            break
    return chunks


def paeth_predictor(left: int, up: int, upper_left: int) -> int:
    predictor = left + up - upper_left
    left_distance = abs(predictor - left)
    up_distance = abs(predictor - up)
    upper_left_distance = abs(predictor - upper_left)
    if left_distance <= up_distance and left_distance <= upper_left_distance:
        return left
    if up_distance <= upper_left_distance:
        return up
    return upper_left


def unfilter_png_rows(
    filtered: bytes, width: int, height: int, bytes_per_pixel: int
) -> list[bytes]:
    row_bytes = width * bytes_per_pixel
    expected = (row_bytes + 1) * height
    if len(filtered) != expected:
        raise PngError(
            f"unexpected decompressed size {len(filtered)}; expected {expected}"
        )

    rows: list[bytes] = []
    previous = bytearray(row_bytes)
    offset = 0
    for _ in range(height):
        filter_type = filtered[offset]
        source = filtered[offset + 1 : offset + 1 + row_bytes]
        offset += row_bytes + 1
        row = bytearray(row_bytes)

        for i, source_byte in enumerate(source):
            left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
            up = previous[i]
            upper_left = previous[i - bytes_per_pixel] if i >= bytes_per_pixel else 0

            if filter_type == 0:
                value = source_byte
            elif filter_type == 1:
                value = source_byte + left
            elif filter_type == 2:
                value = source_byte + up
            elif filter_type == 3:
                value = source_byte + ((left + up) // 2)
            elif filter_type == 4:
                value = source_byte + paeth_predictor(left, up, upper_left)
            else:
                raise PngError(f"unsupported PNG filter type {filter_type}")
            row[i] = value & 0xFF

        rows.append(bytes(row))
        previous = row
    return rows


def chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_png(
    path: Path,
    rows: list[bytes],
    width: int,
    bit_depth: int,
    color_type: int,
    compression: int,
    filter_method: int,
    interlace: int,
) -> None:
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        len(rows),
        bit_depth,
        color_type,
        compression,
        filter_method,
        interlace,
    )
    payload = bytearray()
    for row in rows:
        payload.append(0)
        payload.extend(row)
    data = (
        PNG_SIGNATURE
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(payload), level=9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def dark_pixel_counts(rows: list[bytes], width: int, channels: int) -> list[int]:
    counts: list[int] = []
    for row in rows:
        count = 0
        for offset in range(0, width * channels, channels):
            red = row[offset]
            green = row[offset + 1]
            blue = row[offset + 2]
            alpha = row[offset + 3] if channels == 4 else 255
            if alpha > 10 and (red < 215 or green < 215 or blue < 215):
                count += 1
        counts.append(count)
    return counts


def crop_bottom_note(rows: list[bytes], width: int, channels: int) -> tuple[list[bytes], int]:
    counts = dark_pixel_counts(rows, width, channels)
    threshold = max(10, width // 250)
    first_candidate_row = int(len(rows) * 0.62)
    active_rows = [
        index
        for index in range(first_candidate_row, len(rows))
        if counts[index] > threshold
    ]
    if not active_rows:
        return rows, len(rows)

    groups: list[tuple[int, int]] = []
    group_start = active_rows[0]
    group_end = active_rows[0]
    for row_index in active_rows[1:]:
        if row_index - group_end <= 42:
            group_end = row_index
        else:
            groups.append((group_start, group_end))
            group_start = row_index
            group_end = row_index
    groups.append((group_start, group_end))

    cluster_top, cluster_end = groups[-1]
    cluster_height = cluster_end - cluster_top + 1
    if cluster_end < int(len(rows) * 0.83) or cluster_top < int(len(rows) * 0.62):
        return rows, len(rows)
    if cluster_height < 6:
        return rows, len(rows)

    crop_at = max(1, cluster_top - 18)
    return rows[:crop_at], crop_at


def crop_png_bottom_note(source: Path, destination: Path) -> tuple[int, int, int]:
    chunks = read_chunks(source.read_bytes())
    ihdr_chunks = [data for chunk_type, data in chunks if chunk_type == b"IHDR"]
    if len(ihdr_chunks) != 1:
        raise PngError("PNG must contain one IHDR chunk")
    width, height, bit_depth, color_type, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", ihdr_chunks[0])
    )
    if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        raise PngError("only 8-bit non-interlaced PNG files are supported")
    if color_type == 6:
        channels = 4
    elif color_type == 2:
        channels = 3
    else:
        raise PngError(f"unsupported PNG color type {color_type}")

    compressed = b"".join(data for chunk_type, data in chunks if chunk_type == b"IDAT")
    rows = unfilter_png_rows(zlib.decompress(compressed), width, height, channels)
    cropped_rows, cropped_height = crop_bottom_note(rows, width, channels)
    write_png(
        destination,
        cropped_rows,
        width,
        bit_depth,
        color_type,
        compression,
        filter_method,
        interlace,
    )
    return width, height, cropped_height


def export_page(
    drawio: str,
    source: Path,
    page_index: int,
    destination: Path,
    tmp_dir: Path,
    crop_bottom: bool,
) -> tuple[int, int, int]:
    raw = tmp_dir / f"{PAGE_NAMES[page_index]}_raw.png"
    cmd = [
        drawio,
        "--export",
        "--format",
        "png",
        "--page-index",
        str(page_index),
        "--output",
        wsl_windows_path(raw),
        wsl_windows_path(source),
    ]
    subprocess.run(cmd, check=True)
    if not raw.exists():
        raise FileNotFoundError(f"draw.io did not create {raw}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if crop_bottom:
        return crop_png_bottom_note(raw, destination)
    shutil.copyfile(raw, destination)
    width, height, _bit_depth = png_dimensions(raw)
    return width, height, height


def png_dimensions(source: Path) -> tuple[int, int, int]:
    chunks = read_chunks(source.read_bytes())
    ihdr_chunks = [data for chunk_type, data in chunks if chunk_type == b"IHDR"]
    if len(ihdr_chunks) != 1:
        raise PngError("PNG must contain one IHDR chunk")
    width, height, bit_depth, _color_type, _compression, _filter_method, _interlace = (
        struct.unpack(">IIBBBBB", ihdr_chunks[0])
    )
    return width, height, bit_depth


def copy_drawio_source(source: Path) -> None:
    for destination in [
        Path("paper/figures/drawio/cropped_replicas_all.drawio"),
        Path("paper_zh/figures/drawio/cropped_replicas_all.drawio"),
    ]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--drawio", default=os.environ.get("DRAWIO_CLI"))
    parser.add_argument(
        "--output-root",
        action="append",
        default=[],
        help="Figure directory to receive fig_*.png files. Can be repeated.",
    )
    parser.add_argument(
        "--no-copy-source",
        action="store_true",
        help="Do not copy the source .drawio into paper figure source folders.",
    )
    parser.add_argument(
        "--crop-bottom-note",
        action="store_true",
        help="Crop the bottom note row after draw.io export. Disabled by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    drawio = resolve_drawio(args.drawio)
    output_roots = [Path(root) for root in (args.output_root or DEFAULT_OUTPUT_ROOTS)]

    with tempfile.TemporaryDirectory(prefix="iefmmq_drawio_png_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for page_index, page_name in enumerate(PAGE_NAMES):
            first_target = tmp_dir / f"{page_name}.png"
            width, original_height, cropped_height = export_page(
                drawio,
                source,
                page_index,
                first_target,
                tmp_dir,
                args.crop_bottom_note,
            )
            for output_root in output_roots:
                target = output_root / f"{page_name}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(first_target, target)
            print(
                f"{page_name}: {width}x{original_height} -> "
                f"{width}x{cropped_height}"
            )

    if not args.no_copy_source:
        copy_drawio_source(source)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, PngError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
