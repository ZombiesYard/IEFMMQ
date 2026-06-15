#!/usr/bin/env python3
"""Export simple uncompressed draw.io thesis diagrams to vector PDFs.

This is intentionally small and dependency-free. It supports the subset used by
the imagegen-cropped thesis replica diagrams: text cells, rectangle cells, and
orthogonal connector edges with optional waypoints. Bottom note cells are
skipped by default so the exported paper figures do not include the extra
explanatory footer text.
"""

from __future__ import annotations

import argparse
import html
import math
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


PDF_SCALE = 0.72
INK = "#111827"
MUTED = "#334155"
LINE = "#374151"


@dataclass(frozen=True)
class Geometry:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def port(self, px: float, py: float) -> tuple[float, float]:
        return self.x + self.w * px, self.y + self.h * py


def parse_style(style: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in style.split(";"):
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = value
        else:
            result[part] = "1"
    return result


def color(raw: str | None, default: str = INK) -> str:
    if not raw or raw == "none":
        return default
    return raw


def rgb(raw: str) -> tuple[float, float, float]:
    raw = raw.strip().lstrip("#")
    if len(raw) != 6:
        raw = "111827"
    return tuple(int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def clean_label(raw: str | None) -> str:
    if not raw:
        return ""
    text = raw.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"</?(div|p)[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def pdf_text(raw: str) -> str:
    text = raw.encode("ascii", "replace").decode("ascii")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def is_note_cell(cell: ET.Element) -> bool:
    cell_id = cell.attrib.get("id", "")
    if "_note_" in cell_id:
        return True
    value = clean_label(cell.attrib.get("value")).lower()
    note_starts = (
        "models propose",
        "higher authority",
        "the llm is invoked",
        "training and benchmark evidence",
        "the ontology narrows",
        "a vlm benchmark",
        "the pilot describes",
        "the model proposes",
        "the prototype is",
        "exports keep",
    )
    return value.startswith(note_starts)


class PdfCanvas:
    def __init__(self, width: float, height: float, scale: float = PDF_SCALE) -> None:
        self.scale = scale
        self.width = width * scale
        self.height = height * scale
        self.ops: list[str] = []

    def sx(self, x: float) -> float:
        return x * self.scale

    def sy(self, y: float) -> float:
        return self.height - y * self.scale

    def rect(self, geom: Geometry, fill: str, stroke: str, width: float = 1.0, dashed: bool = False) -> None:
        fr, fg, fb = rgb(fill)
        sr, sg, sb = rgb(stroke)
        dash = "[6 5] 0 d" if dashed else "[] 0 d"
        self.ops.append(
            "q "
            f"{fr:.3f} {fg:.3f} {fb:.3f} rg "
            f"{sr:.3f} {sg:.3f} {sb:.3f} RG "
            f"{max(0.2, width * self.scale):.2f} w {dash} "
            f"{self.sx(geom.x):.2f} {self.sy(geom.bottom):.2f} {self.sx(geom.w):.2f} {self.sx(geom.h):.2f} re B Q"
        )

    def line(self, points: list[tuple[float, float]], stroke: str = LINE, width: float = 2.0, dashed: bool = False) -> None:
        if len(points) < 2:
            return
        sr, sg, sb = rgb(stroke)
        dash = "[8 6] 0 d" if dashed else "[] 0 d"
        start, *rest = points
        parts = [
            "q",
            f"{sr:.3f} {sg:.3f} {sb:.3f} RG",
            f"{max(0.3, width * self.scale):.2f} w",
            dash,
            f"{self.sx(start[0]):.2f} {self.sy(start[1]):.2f} m",
        ]
        parts.extend(f"{self.sx(x):.2f} {self.sy(y):.2f} l" for x, y in rest)
        parts.append("S Q")
        self.ops.append(" ".join(parts))

    def arrow_head(self, start: tuple[float, float], end: tuple[float, float], fill: str = LINE) -> None:
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        head = 10.0
        left = (end[0] - head * math.cos(angle - math.pi / 6), end[1] - head * math.sin(angle - math.pi / 6))
        right = (end[0] - head * math.cos(angle + math.pi / 6), end[1] - head * math.sin(angle + math.pi / 6))
        r, g, b = rgb(fill)
        self.ops.append(
            "q "
            f"{r:.3f} {g:.3f} {b:.3f} rg {r:.3f} {g:.3f} {b:.3f} RG "
            f"{self.sx(end[0]):.2f} {self.sy(end[1]):.2f} m "
            f"{self.sx(left[0]):.2f} {self.sy(left[1]):.2f} l "
            f"{self.sx(right[0]):.2f} {self.sy(right[1]):.2f} l h f Q"
        )

    def draw_text(
        self,
        geom: Geometry,
        text: str,
        *,
        size: float,
        font: str,
        color_raw: str,
        align: str,
        valign: str = "middle",
    ) -> None:
        if not text:
            return
        fit = size
        max_width = max(8.0, geom.w - 18.0)
        explicit: list[str] = []
        for line in text.splitlines() or [""]:
            width_chars = max(8, int(max_width / max(1.0, fit * 0.48)))
            explicit.extend(textwrap.wrap(line, width=width_chars, break_long_words=False) or [""])
        while fit > 8:
            line_h = fit * 1.18
            if len(explicit) * line_h <= geom.h - 8:
                break
            fit -= 1
            explicit = []
            for line in text.splitlines() or [""]:
                width_chars = max(8, int(max_width / max(1.0, fit * 0.48)))
                explicit.extend(textwrap.wrap(line, width=width_chars, break_long_words=False) or [""])

        line_h = fit * 1.18
        total_h = len(explicit) * line_h
        if valign == "top":
            y = geom.y + fit + 8
        else:
            y = geom.y + (geom.h - total_h) / 2 + fit
        r, g, b = rgb(color_raw)
        for line in explicit:
            approx_w = len(line) * fit * 0.48
            if align == "left":
                x = geom.x + 8
            elif align == "right":
                x = geom.right - approx_w - 8
            else:
                x = geom.x + (geom.w - approx_w) / 2
            self.ops.append(
                "BT "
                f"/{font} {fit * self.scale:.2f} Tf "
                f"{r:.3f} {g:.3f} {b:.3f} rg "
                f"{self.sx(x):.2f} {self.sy(y):.2f} Td "
                f"({pdf_text(line)}) Tj ET"
            )
            y += line_h

    def save(self, path: Path) -> None:
        stream = "\n".join(self.ops).encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
                "/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> /Contents 7 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>",
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
        offsets: list[int] = []
        pos = len(chunks[0])
        for idx, obj in enumerate(objects, start=1):
            offsets.append(pos)
            chunk = f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
            chunks.append(chunk)
            pos += len(chunk)
        xref_pos = pos
        xref = [b"xref\n", f"0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
        xref.extend(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets)
        trailer = (
            b"trailer\n"
            + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
        )
        path.write_bytes(b"".join(chunks + xref + [trailer]))


def geometry(cell: ET.Element) -> Geometry | None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return None
    return Geometry(
        float(geom.attrib.get("x", "0")),
        float(geom.attrib.get("y", "0")),
        float(geom.attrib.get("width", "0")),
        float(geom.attrib.get("height", "0")),
    )


def waypoints(cell: ET.Element) -> list[tuple[float, float]]:
    geom = cell.find("mxGeometry")
    if geom is None:
        return []
    points: list[tuple[float, float]] = []
    for point in geom.findall("./Array[@as='points']/mxPoint"):
        points.append((float(point.attrib.get("x", "0")), float(point.attrib.get("y", "0"))))
    return points


def edge_endpoint(cell: ET.Element, styles: dict[str, str], geoms: dict[str, Geometry], *, source: bool) -> tuple[float, float]:
    ref = cell.attrib.get("source" if source else "target", "")
    geom = geoms.get(ref)
    if geom is None:
        own = geometry(cell)
        if own is None:
            return (0.0, 0.0)
        return (own.x, own.y)
    px_key = "exitX" if source else "entryX"
    py_key = "exitY" if source else "entryY"
    if px_key in styles and py_key in styles:
        return geom.port(float(styles[px_key]), float(styles[py_key]))
    return geom.port(1.0 if source else 0.0, 0.5)


def export_page(diagram: ET.Element, out_dir: Path, *, keep_notes: bool) -> Path:
    name = diagram.attrib["name"]
    model = diagram.find("mxGraphModel")
    if model is None:
        raise ValueError(f"{name}: missing mxGraphModel")
    page_w = float(model.attrib.get("pageWidth", "842"))
    page_h = float(model.attrib.get("pageHeight", "595"))
    cells = model.findall(".//mxCell")
    kept = [cell for cell in cells if keep_notes or not is_note_cell(cell)]
    geoms = {cell.attrib["id"]: geom for cell in kept if (geom := geometry(cell)) is not None and cell.attrib.get("vertex") == "1"}
    max_bottom = max((geom.bottom for geom in geoms.values()), default=page_h)
    for cell in kept:
        if cell.attrib.get("edge") == "1":
            for _, y in waypoints(cell):
                max_bottom = max(max_bottom, y)
    canvas = PdfCanvas(page_w, min(page_h, max_bottom + 24))

    # Draw filled cells first.
    for cell in kept:
        if cell.attrib.get("vertex") != "1":
            continue
        geom = geometry(cell)
        if geom is None:
            continue
        styles = parse_style(cell.attrib.get("style", ""))
        if "text" in styles and styles.get("strokeColor") == "none" and styles.get("fillColor") == "none":
            continue
        fill = color(styles.get("fillColor"), "#ffffff")
        stroke = color(styles.get("strokeColor"), "#111111")
        width = float(styles.get("strokeWidth", "1") or "1")
        dashed = styles.get("dashed") == "1"
        canvas.rect(geom, fill, stroke, width=width, dashed=dashed)

    # Then connectors.
    for cell in kept:
        if cell.attrib.get("edge") != "1":
            continue
        styles = parse_style(cell.attrib.get("style", ""))
        pts = [edge_endpoint(cell, styles, geoms, source=True)]
        pts.extend(waypoints(cell))
        pts.append(edge_endpoint(cell, styles, geoms, source=False))
        stroke = color(styles.get("strokeColor"), LINE)
        width = float(styles.get("strokeWidth", "2") or "2")
        dashed = styles.get("dashed") == "1"
        canvas.line(pts, stroke=stroke, width=width, dashed=dashed)
        if styles.get("endArrow", "classic") != "none":
            canvas.arrow_head(pts[-2], pts[-1], fill=stroke)

    # Text last for legibility.
    for cell in kept:
        if cell.attrib.get("vertex") != "1":
            continue
        text = clean_label(cell.attrib.get("value"))
        if not text:
            continue
        geom = geometry(cell)
        if geom is None:
            continue
        styles = parse_style(cell.attrib.get("style", ""))
        font_style = int(styles.get("fontStyle", "0") or "0")
        font = "F2" if font_style & 1 else "F3" if font_style & 2 else "F1"
        size = float(styles.get("fontSize", "24") or "24")
        align = styles.get("align", "center")
        valign = styles.get("verticalAlign", "middle")
        font_color = color(styles.get("fontColor"), INK if size >= 24 else MUTED)
        canvas.draw_text(geom, text, size=size, font=font, color_raw=font_color, align=align, valign=valign)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.pdf"
    canvas.save(out_path)
    return out_path


def export_all(input_path: Path, output_dirs: list[Path], *, keep_notes: bool) -> list[Path]:
    root = ET.parse(input_path).getroot()
    diagrams = root.findall("diagram")
    if not diagrams:
        raise ValueError(f"No diagrams found in {input_path}")
    written: list[Path] = []
    for out_dir in output_dirs:
        for diagram in diagrams:
            written.append(export_page(diagram, out_dir, keep_notes=keep_notes))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Uncompressed multi-page .drawio file")
    parser.add_argument("--output-dir", type=Path, action="append", required=True, help="Directory for exported PDFs")
    parser.add_argument("--keep-notes", action="store_true", help="Keep bottom note cells instead of cropping them away")
    args = parser.parse_args()
    written = export_all(args.input, args.output_dir, keep_notes=args.keep_notes)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
