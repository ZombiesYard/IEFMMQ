#!/usr/bin/env python3
"""Generate thesis-specific draw.io source and PDF figures.

The script intentionally uses only the Python standard library. The generated
PDFs are simple vector drawings that can be included by LaTeX with graphicx.
"""

from __future__ import annotations

import math
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


PAGE_W = 842
PAGE_H = 595
PDF_SCALE = 0.72

ROOT = Path(__file__).resolve().parent
DRAWIO_OUT = ROOT / "drawio" / "thesis_figures.drawio"

COLORS = {
    "sim": ("#e5e7eb", "#6b7280"),
    "adapter": ("#fce7f3", "#be185d"),
    "core": ("#e0f2fe", "#0369a1"),
    "data": ("#fef3c7", "#b45309"),
    "model": ("#dbeafe", "#1d4ed8"),
    "vlm": ("#dcfce7", "#15803d"),
    "evidence": ("#ede9fe", "#6d28d9"),
    "eval": ("#f3e8ff", "#7e22ce"),
    "output": ("#ecfccb", "#4d7c0f"),
    "manual": ("#fee2e2", "#b91c1c"),
}


@dataclass(frozen=True)
class Box:
    key: str
    x: float
    y: float
    w: float
    h: float
    text: str
    kind: str = "core"


@dataclass(frozen=True)
class Arrow:
    src: str
    dst: str
    label: str = ""
    dashed: bool = False


@dataclass(frozen=True)
class Figure:
    name: str
    filename: str
    title: str
    boxes: tuple[Box, ...]
    arrows: tuple[Arrow, ...]
    notes: tuple[str, ...] = ()


def hex_to_rgb(raw: str) -> tuple[float, float, float]:
    raw = raw.lstrip("#")
    return tuple(int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_lines(text: str, width: float, size: float) -> list[str]:
    result: list[str] = []
    max_chars = max(8, int(width / (size * 0.52)))
    for raw in text.split("\n"):
        if not raw:
            result.append("")
        else:
            result.extend(textwrap.wrap(raw, width=max_chars, break_long_words=False) or [""])
    return result


class PdfCanvas:
    def __init__(self, width: int = PAGE_W, height: int = PAGE_H):
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def _y(self, y: float) -> float:
        return self.height - y

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str) -> None:
        fr, fg, fb = hex_to_rgb(fill)
        sr, sg, sb = hex_to_rgb(stroke)
        self.ops.append(
            "q "
            f"{fr:.3f} {fg:.3f} {fb:.3f} rg "
            f"{sr:.3f} {sg:.3f} {sb:.3f} RG "
            "1.2 w "
            f"{x:.1f} {self._y(y + h):.1f} {w:.1f} {h:.1f} re B Q"
        )

    def line(self, points: list[tuple[float, float]], color: str = "#374151", dashed: bool = False) -> None:
        if len(points) < 2:
            return
        r, g, b = hex_to_rgb(color)
        dash = "[5 4] 0 d " if dashed else "[] 0 d "
        first = points[0]
        rest = points[1:]
        parts = [
            "q",
            f"{r:.3f} {g:.3f} {b:.3f} RG",
            "1.2 w",
            dash,
            f"{first[0]:.1f} {self._y(first[1]):.1f} m",
        ]
        parts.extend(f"{x:.1f} {self._y(y):.1f} l" for x, y in rest)
        parts.append("S Q")
        self.ops.append(" ".join(parts))

    def arrow(self, start: tuple[float, float], end: tuple[float, float], label: str = "", dashed: bool = False) -> None:
        self.line([start, end], dashed=dashed)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        head = 8.0
        left = (
            end[0] - head * math.cos(angle - math.pi / 6),
            end[1] - head * math.sin(angle - math.pi / 6),
        )
        right = (
            end[0] - head * math.cos(angle + math.pi / 6),
            end[1] - head * math.sin(angle + math.pi / 6),
        )
        r, g, b = hex_to_rgb("#374151")
        self.ops.append(
            "q "
            f"{r:.3f} {g:.3f} {b:.3f} rg {r:.3f} {g:.3f} {b:.3f} RG "
            f"{end[0]:.1f} {self._y(end[1]):.1f} m "
            f"{left[0]:.1f} {self._y(left[1]):.1f} l "
            f"{right[0]:.1f} {self._y(right[1]):.1f} l h f Q"
        )
        if label:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2 - 4
            self.text(mx - min(80, len(label) * 2.7), my, label, size=8.2, color="#374151")

    def text(self, x: float, y: float, text: str, size: float = 9, color: str = "#111827") -> None:
        r, g, b = hex_to_rgb(color)
        self.ops.append(
            "BT "
            f"/F1 {size:.1f} Tf "
            f"{r:.3f} {g:.3f} {b:.3f} rg "
            f"{x:.1f} {self._y(y):.1f} Td "
            f"({escape_pdf_text(text)}) Tj ET"
        )

    def centered_text(self, x: float, y: float, w: float, h: float, text: str, size: float = 9) -> None:
        fit_size = size
        while fit_size > 6.6:
            lines = wrap_lines(text, w - 10, fit_size)
            line_h = fit_size * 1.18
            if len(lines) * line_h <= h - 5:
                break
            fit_size -= 0.3
        lines = wrap_lines(text, w - 10, fit_size)
        line_h = fit_size * 1.18
        total = len(lines) * line_h
        cur_y = y + (h - total) / 2 + fit_size
        for line in lines:
            approx_w = len(line) * fit_size * 0.52
            self.text(x + (w - approx_w) / 2, cur_y, line, size=fit_size)
            cur_y += line_h

    def box(self, box: Box) -> None:
        fill, stroke = COLORS[box.kind]
        self.rect(box.x, box.y, box.w, box.h, fill, stroke)
        self.centered_text(box.x, box.y, box.w, box.h, box.text, size=9.8)

    def save(self, path: Path) -> None:
        content = "\n".join(self.ops).encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
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


def edge(src: Box, dst: Box) -> tuple[tuple[float, float], tuple[float, float]]:
    sx, sy = src.x + src.w / 2, src.y + src.h / 2
    dx, dy = dst.x + dst.w / 2, dst.y + dst.h / 2
    if abs(dx - sx) > abs(dy - sy):
        if dx >= sx:
            return (src.x + src.w, sy), (dst.x, dy)
        return (src.x, sy), (dst.x + dst.w, dy)
    if dy >= sy:
        return (sx, src.y + src.h), (dx, dst.y)
    return (sx, src.y), (dx, dst.y + dst.h)


def render_pdf(fig: Figure, path: Path) -> None:
    def scaled_box(box: Box) -> Box:
        return Box(
            box.key,
            box.x * PDF_SCALE,
            box.y * PDF_SCALE,
            box.w * PDF_SCALE,
            box.h * PDF_SCALE,
            box.text,
            box.kind,
        )

    canvas = PdfCanvas(int(PAGE_W * PDF_SCALE), int(PAGE_H * PDF_SCALE))
    canvas.text(18, 24, fig.title, size=14.5)
    boxes = tuple(scaled_box(box) for box in fig.boxes)
    by_key = {b.key: b for b in boxes}
    for box in boxes:
        canvas.box(box)
    for arrow in fig.arrows:
        start, end = edge(by_key[arrow.src], by_key[arrow.dst])
        canvas.arrow(start, end, arrow.label, dashed=arrow.dashed)
    y = 548 * PDF_SCALE
    for note in fig.notes:
        canvas.text(18, y, note, size=7.8, color="#4b5563")
        y += 12
    canvas.save(path)


class DrawioWriter:
    def __init__(self):
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": "2026-06-09T00:00:00.000Z",
                "agent": "SimTutor thesis figure generator",
                "version": "24.0.0",
                "type": "device",
            },
        )

    def add_figure(self, fig: Figure) -> None:
        root = ET.Element(
            "mxGraphModel",
            {
                "dx": "0",
                "dy": "0",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(PAGE_W),
                "pageHeight": str(PAGE_H),
                "math": "0",
                "shadow": "0",
            },
        )
        r = ET.SubElement(root, "root")
        ET.SubElement(r, "mxCell", {"id": "0"})
        ET.SubElement(r, "mxCell", {"id": "1", "parent": "0"})
        by_key: dict[str, str] = {}
        next_id = 2
        title_id = str(next_id)
        next_id += 1
        title = ET.SubElement(
            r,
            "mxCell",
            {
                "id": title_id,
                "value": fig.title,
                "style": "text;html=1;strokeColor=none;fillColor=none;fontSize=18;fontStyle=1;align=left;",
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(title, "mxGeometry", {"x": "24", "y": "14", "width": "760", "height": "28", "as": "geometry"})
        for box in fig.boxes:
            cid = str(next_id)
            next_id += 1
            by_key[box.key] = cid
            fill, stroke = COLORS[box.kind]
            cell = ET.SubElement(
                r,
                "mxCell",
                {
                    "id": cid,
                    "value": box.text.replace("\n", "<br>"),
                    "style": (
                        "rounded=1;whiteSpace=wrap;html=1;fontSize=10;fontFamily=Helvetica;"
                        f"fillColor={fill};strokeColor={stroke};align=center;verticalAlign=middle;"
                    ),
                    "vertex": "1",
                    "parent": "1",
                },
            )
            ET.SubElement(
                cell,
                "mxGeometry",
                {"x": str(box.x), "y": str(box.y), "width": str(box.w), "height": str(box.h), "as": "geometry"},
            )
        for arrow in fig.arrows:
            cid = str(next_id)
            next_id += 1
            cell = ET.SubElement(
                r,
                "mxCell",
                {
                    "id": cid,
                    "value": arrow.label,
                    "style": (
                        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
                        "html=1;endArrow=block;endFill=1;strokeColor=#374151;fontSize=8;"
                        + ("dashed=1;" if arrow.dashed else "")
                    ),
                    "edge": "1",
                    "parent": "1",
                    "source": by_key[arrow.src],
                    "target": by_key[arrow.dst],
                },
            )
            ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": fig.filename, "name": fig.name})
        diagram.append(root)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(ET, "indent"):
            ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)


def figures() -> list[Figure]:
    return [
        Figure(
            "2.1 Evidence Sources",
            "fig_bg_evidence_sources.pdf",
            "F/A-18C cold-start evidence sources",
            (
                Box("task", 30, 70, 140, 58, "33-step cold-start\nordered P1-P6 procedure", "sim"),
                Box("tel", 210, 60, 170, 82, "Telemetry-visible steps\nDCS-BIOS variables\nS01, S03-S13, S16,\nS20-S33", "evidence"),
                Box("vis", 430, 60, 170, 82, "Vision-priority steps\ncomposite panel + VLM\nS08, S15, S18, S19", "vlm"),
                Box("man", 650, 60, 150, 82, "Manual / out-of-layout\nexpert confirmation\nS14, S17", "manual"),
                Box("bios", 210, 190, 170, 64, "DCS-BIOS stream\nswitches, gauges,\nwarning lights", "data"),
                Box("panel", 430, 190, 170, 64, "Composite panel image\nleft DDI, AMPCD,\nright DDI", "data"),
                Box("manual", 650, 190, 150, 64, "Operator judgement\noutside captured panel", "data"),
                Box("packet", 305, 325, 220, 64, "Evidence packet\ntelemetry + visual facts\n+ manual state", "core"),
                Box("gates", 565, 325, 180, 64, "Procedure gates\npreconditions and\ncompletion rules", "core"),
                Box("tutor", 390, 450, 210, 64, "Grounded tutor output\nstep guidance + overlay\nwith evidence refs", "output"),
            ),
            (
                Arrow("task", "tel"),
                Arrow("task", "vis"),
                Arrow("task", "man"),
                Arrow("tel", "bios"),
                Arrow("vis", "panel"),
                Arrow("man", "manual"),
                Arrow("bios", "packet"),
                Arrow("panel", "packet"),
                Arrow("manual", "packet"),
                Arrow("packet", "gates"),
                Arrow("gates", "tutor"),
                Arrow("packet", "tutor"),
            ),
            ("The task cannot be grounded by language alone because completion evidence comes from mixed sources.",),
        ),
        Figure(
            "3.1 System Architecture",
            "fig_system_architecture.pdf",
            "Evidence-grounded tutoring architecture",
            (
                Box("dcs", 30, 85, 130, 58, "DCS F/A-18C\nQuest 3 VR session", "sim"),
                Box("bios", 30, 200, 130, 56, "DCS-BIOS\ntelemetry stream", "sim"),
                Box("vision", 30, 315, 130, 56, "Vision sidecar\ncomposite capture", "sim"),
                Box("ports", 205, 80, 150, 80, "Adapter ports\ntelemetry, vision,\nmodel, knowledge", "adapter"),
                Box("pack", 205, 205, 150, 80, "Procedure pack\nsteps, gates,\nUI targets, facts", "data"),
                Box("runtime", 400, 80, 170, 80, "SimTutor runtime\nhelp-cycle orchestration", "core"),
                Box("evidence", 400, 205, 170, 80, "Evidence packet\nstate, gates, facts,\nrecent actions", "core"),
                Box("harness", 400, 330, 170, 80, "Harness validation\nschema, allowlist,\nrepair or fallback", "core"),
                Box("llm", 625, 65, 160, 62, "Text LLM\nstep guidance", "model"),
                Box("vlm", 625, 155, 160, 62, "VLM adapter\nvisual facts only", "vlm"),
                Box("rag", 625, 245, 160, 62, "Retrieved manuals\nand checklist snippets", "data"),
                Box("out", 625, 370, 160, 62, "Cockpit overlay\nJSONL log\nexperiment export", "output"),
            ),
            (
                Arrow("dcs", "bios"),
                Arrow("dcs", "vision"),
                Arrow("bios", "ports"),
                Arrow("vision", "ports"),
                Arrow("ports", "runtime"),
                Arrow("pack", "runtime"),
                Arrow("runtime", "evidence"),
                Arrow("evidence", "harness"),
                Arrow("evidence", "llm"),
                Arrow("evidence", "vlm"),
                Arrow("rag", "llm"),
                Arrow("llm", "harness"),
                Arrow("vlm", "evidence"),
                Arrow("harness", "out"),
            ),
        ),
        Figure(
            "3.2 Live Help Cycle",
            "fig_help_cycle.pdf",
            "Live evidence-grounded help cycle",
            (
                Box("trig", 25, 80, 105, 55, "Help trigger\nX1 request", "sim"),
                Box("snap", 155, 80, 105, 55, "Telemetry\nsnapshot", "adapter"),
                Box("vision", 285, 80, 105, 55, "Vision capture\nif required", "adapter"),
                Box("infer", 415, 80, 105, 55, "Step and gate\ninference", "core"),
                Box("packet", 545, 80, 105, 55, "Evidence\npacket", "core"),
                Box("rag", 675, 80, 105, 55, "Knowledge\nretrieval", "data"),
                Box("prompt", 675, 250, 105, 55, "Prompt and\nschema", "core"),
                Box("llm", 545, 250, 105, 55, "Text LLM\nresponse", "model"),
                Box("map", 415, 250, 105, 55, "Response\nmapping", "adapter"),
                Box("harness", 285, 250, 105, 55, "Harness\nvalidation", "core"),
                Box("overlay", 155, 250, 105, 55, "Overlay and\ntext output", "output"),
                Box("log", 25, 250, 105, 55, "Event log\nand export", "data"),
            ),
            (
                Arrow("trig", "snap"),
                Arrow("snap", "vision"),
                Arrow("vision", "infer"),
                Arrow("infer", "packet"),
                Arrow("packet", "rag"),
                Arrow("rag", "prompt"),
                Arrow("prompt", "llm"),
                Arrow("llm", "map"),
                Arrow("map", "harness"),
                Arrow("harness", "overlay"),
                Arrow("overlay", "log"),
                Arrow("harness", "log", "trace", dashed=True),
            ),
            ("The LLM never acts directly on the simulator; mapped responses pass through harness validation first.",),
        ),
        Figure(
            "3.3 Harness Validation",
            "fig_harness_validation.pdf",
            "Harness and evidence validation",
            (
                Box("packet", 50, 95, 170, 70, "EvidencePacket\ntelemetry, gates,\nvision facts, actions", "core"),
                Box("candidates", 50, 245, 170, 70, "StepCandidate list\nranked alternatives\nwith evidence refs", "core"),
                Box("spec", 50, 395, 170, 70, "StepHarnessSpec\nallowed targets,\nrequired facts", "data"),
                Box("decision", 330, 120, 170, 70, "HarnessDecision\nLLM structured output", "model"),
                Box("schema", 330, 270, 170, 70, "Validation gates\nJSON Schema,\nallowlist, evidence", "core"),
                Box("planner", 330, 420, 170, 70, "State-action planner\nrepair or text-only\nfallback", "core"),
                Box("plan", 610, 175, 170, 70, "HarnessActionPlan\naccepted targets,\nguidance, trace", "output"),
                Box("reject", 610, 350, 170, 70, "Safe rejection\nno unsafe overlay\nwhen evidence fails", "manual"),
            ),
            (
                Arrow("packet", "decision"),
                Arrow("candidates", "decision"),
                Arrow("spec", "schema"),
                Arrow("decision", "schema"),
                Arrow("schema", "planner"),
                Arrow("planner", "plan"),
                Arrow("schema", "reject", "failed checks", dashed=True),
                Arrow("reject", "plan", "text-only", dashed=True),
            ),
        ),
        Figure(
            "4.1 VLM Pipeline",
            "fig_vlm_pipeline.pdf",
            "Small-data VLM adaptation pipeline",
            (
                Box("capture", 25, 95, 105, 58, "DCS screenshot\ncapture", "sim"),
                Box("pre", 150, 95, 105, 58, "AI pre-labels\nper visual fact", "model"),
                Box("review", 275, 95, 105, 58, "Human review\nLabel Studio", "manual"),
                Box("jsonl", 400, 95, 105, 58, "Reviewed JSONL\nground truth", "data"),
                Box("sft", 525, 95, 105, 58, "Bilingual SFT\nEN + ZH rows", "data"),
                Box("lora", 650, 95, 105, 58, "LoRA\nfine-tuning", "vlm"),
                Box("bench", 525, 280, 130, 62, "Holdout benchmark\nbase vs LoRA", "eval"),
                Box("errors", 340, 280, 130, 62, "Error analysis\ncritical false positives", "eval"),
                Box("ontology", 155, 280, 130, 62, "Ontology revision\n8 facts to 13 facts", "core"),
                Box("runtime", 650, 430, 130, 62, "Runtime VLM\nvisual fact extractor", "output"),
            ),
            (
                Arrow("capture", "pre"),
                Arrow("pre", "review"),
                Arrow("review", "jsonl"),
                Arrow("jsonl", "sft"),
                Arrow("sft", "lora"),
                Arrow("lora", "bench"),
                Arrow("bench", "errors"),
                Arrow("errors", "ontology"),
                Arrow("ontology", "capture", "next capture", dashed=True),
                Arrow("lora", "runtime"),
            ),
            ("Each benchmark prediction remains traceable to a reviewed label and original cockpit image.",),
        ),
        Figure(
            "4.2 Ontology Evolution",
            "fig_ontology_evolution.pdf",
            "Visual fact ontology evolution",
            (
                Box("v1", 40, 75, 210, 70, "8-fact v1\nmixed abstraction levels\ncompound completion targets", "manual"),
                Box("insgo", 60, 190, 170, 55, "ins_go\nprocedural completion\njudgement", "manual"),
                Box("fcsbit", 60, 285, 170, 55, "fcs_bit_result\nsingle broad FCS-MC\ncompletion fact", "manual"),
                Box("v2", 590, 75, 210, 70, "13-fact v2\nlocally verifiable\nvisual atoms", "vlm"),
                Box("ins1", 575, 185, 230, 45, "ins_grnd_alignment_text_visible", "vlm"),
                Box("ins2", 575, 240, 230, 45, "ins_ok_text_visible", "vlm"),
                Box("fcs1", 575, 315, 230, 42, "fcsmc_page_visible", "vlm"),
                Box("fcs2", 575, 365, 230, 42, "fcsmc_intermediate_result_visible", "vlm"),
                Box("fcs3", 575, 415, 230, 42, "fcsmc_in_test_visible", "vlm"),
                Box("fcs4", 575, 465, 230, 42, "fcsmc_final_go_result_visible", "vlm"),
                Box("principle", 300, 255, 215, 80, "Design principle\nVLM extracts observable evidence;\nprocedure engine performs\nprocedural interpretation", "core"),
                Box("result", 300, 410, 215, 65, "Observed effect\ncritical false positives\nreduced by 64-89%", "eval"),
            ),
            (
                Arrow("v1", "principle"),
                Arrow("principle", "v2"),
                Arrow("insgo", "ins1"),
                Arrow("insgo", "ins2"),
                Arrow("fcsbit", "fcs1"),
                Arrow("fcsbit", "fcs2"),
                Arrow("fcsbit", "fcs3"),
                Arrow("fcsbit", "fcs4"),
                Arrow("principle", "result"),
            ),
        ),
        Figure(
            "6.1 Evaluation Data Flow",
            "fig_evaluation_dataflow.pdf",
            "Evaluation data flow across research questions",
            (
                Box("rq2", 45, 85, 180, 70, "RQ2 VLM evaluation\nholdout screenshots\nbase vs LoRA", "eval"),
                Box("rq1", 330, 85, 180, 70, "RQ1 technical readiness\nreplay suite\nharness fixtures", "eval"),
                Box("rq3", 615, 85, 180, 70, "RQ3 formative pilot\nVR trials\nwith/without tutor", "eval"),
                Box("bench", 45, 235, 180, 70, "Benchmark artifacts\nfact accuracy, F1,\ncritical false positives", "data"),
                Box("runtime", 330, 235, 180, 70, "Runtime artifacts\nJSONL logs,\ntrace metrics", "data"),
                Box("pilot", 615, 235, 180, 70, "Pilot exports\nstep coding,\nhelp cycles, quality notes", "data"),
                Box("tables", 180, 405, 210, 70, "Evaluation tables\ntechnical and pilot\nsummaries", "output"),
                Box("limits", 455, 405, 210, 70, "Interpretation limits\nno inferential claims,\nno learning transfer claims", "manual"),
            ),
            (
                Arrow("rq2", "bench"),
                Arrow("rq1", "runtime"),
                Arrow("rq3", "pilot"),
                Arrow("bench", "tables"),
                Arrow("runtime", "tables"),
                Arrow("pilot", "tables"),
                Arrow("tables", "limits"),
            ),
        ),
        Figure(
            "A.1 Deployment Topology",
            "fig_deployment_topology.pdf",
            "Deployment topology",
            (
                Box("dcs", 55, 90, 165, 60, "Windows simulator host\nDCS World F/A-18C\nQuest 3 via Oculus Link", "sim"),
                Box("bios", 55, 205, 165, 55, "DCS-BIOS\nUDP telemetry", "sim"),
                Box("sidecar", 55, 315, 165, 55, "Vision sidecar\nVR mirror capture", "adapter"),
                Box("live", 310, 115, 185, 70, "SimTutor live-dcs\nhelp trigger, orchestration,\noverlay dispatch", "core"),
                Box("config", 310, 255, 185, 60, "Saved Games DCS\nSimTutorConfig.lua\noverlay command channel", "data"),
                Box("logs", 310, 395, 185, 60, "Runtime logs\nJSONL events and\ntrial exports", "data"),
                Box("gpu", 595, 90, 190, 70, "Remote GPU server\nvLLM endpoint\nSSH tunnel", "model"),
                Box("text", 595, 220, 190, 60, "Text help model\nOpenAI-compatible API", "model"),
                Box("vision", 595, 340, 190, 60, "Vision model\nQwen/Gemma LoRA\nvisual fact extraction", "vlm"),
            ),
            (
                Arrow("dcs", "bios"),
                Arrow("dcs", "sidecar"),
                Arrow("bios", "live"),
                Arrow("sidecar", "live"),
                Arrow("live", "config"),
                Arrow("config", "dcs", "overlay", dashed=True),
                Arrow("live", "logs"),
                Arrow("live", "gpu"),
                Arrow("gpu", "text"),
                Arrow("gpu", "vision"),
                Arrow("text", "live", dashed=True),
                Arrow("vision", "live", dashed=True),
            ),
        ),
        Figure(
            "A.2 Experiment Export",
            "fig_experiment_export.pdf",
            "Detailed experiment export pipeline",
            (
                Box("log", 35, 90, 150, 60, "Raw JSONL event log\none file per trial", "data"),
                Box("meta", 35, 210, 150, 60, "Session metadata\nparticipant, condition,\nstudy id, model", "data"),
                Box("pack", 35, 330, 150, 60, "Pack and taxonomy\nsteps, UI map,\nerror codes", "data"),
                Box("export", 285, 170, 190, 80, "experiment-export\nfreezes one trial into\nstudy-ready artifacts", "core"),
                Box("quality", 285, 330, 190, 60, "Quality gate\ncompleteness and\nmetadata checks", "eval"),
                Box("trial", 575, 85, 210, 55, "trial_summary.csv", "output"),
                Box("steps", 575, 155, 210, 55, "step_coding.csv", "output"),
                Box("help", 575, 225, 210, 55, "help_cycles.csv", "output"),
                Box("timeline", 575, 295, 210, 55, "action_timeline.csv", "output"),
                Box("analyze", 575, 395, 210, 65, "experiment-analyze\ncondition summaries,\nstep accuracy, help quality", "core"),
            ),
            (
                Arrow("log", "export"),
                Arrow("meta", "export"),
                Arrow("pack", "export"),
                Arrow("quality", "export"),
                Arrow("export", "trial"),
                Arrow("export", "steps"),
                Arrow("export", "help"),
                Arrow("export", "timeline"),
                Arrow("trial", "analyze"),
                Arrow("steps", "analyze"),
                Arrow("help", "analyze"),
                Arrow("timeline", "analyze"),
            ),
        ),
    ]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    writer = DrawioWriter()
    figs = figures()
    for fig in figs:
        render_pdf(fig, ROOT / fig.filename)
        writer.add_figure(fig)
    writer.save(DRAWIO_OUT)
    print(f"Wrote {DRAWIO_OUT}")
    for fig in figs:
        print(f"Wrote {ROOT / fig.filename}")


if __name__ == "__main__":
    main()
