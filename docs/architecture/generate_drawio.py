#!/usr/bin/env python3
"""Generate diagrams.drawio with all 6 SimTutor architecture diagrams on separate tabs."""
from __future__ import annotations
import io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

COLORS = {
    "core":    ("#e0f2fe", "#7dd3fc"),
    "adapter": ("#fce7f3", "#f9a8d4"),
    "data":    ("#fef3c7", "#fcd34d"),
    "ext":     ("#ffffff", "#999999"),
    "hw":      ("#e5e7eb", "#9ca3af"),
    "vlm":     ("#dcfce7", "#86efac"),
    "llm":     ("#dbeafe", "#93c5fd"),
    "port":    ("#ede9fe", "#c4b5fd"),
}

@dataclass
class Box:
    x: float; y: float; w: float; h: float; label: str; kind: str
    sub: str = ""

@dataclass
class Arrow:
    x1: float; y1: float; x2: float; y2: float; label: str = ""
    route: Optional[list[tuple[float, float]]] = None  # intermediate waypoints

class DrawioGen:
    def __init__(self):
        self.next_id = 2
        self.cells: list[ET.Element] = []

    def new_id(self) -> str:
        sid = str(self.next_id)
        self.next_id += 1
        return sid

    def box(self, b: Box) -> str:
        fid = self.new_id()
        fill, stroke = COLORS.get(b.kind, COLORS["core"])
        dashed = 1 if b.kind == "ext" else 0
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"dashed={dashed};fontSize=10;fontFamily=Helvetica;align=center;"
            f"verticalAlign=middle;"
        )
        lines = b.label.count("<br>") + 1
        if lines > 1:
            # use table-like html rendering for multi-line
            html_label = b.label.replace("<br>", "<br/>")
            if b.sub:
                html_label += f'<br><font style="font-size:7px;color:#6b7280;">{b.sub}</font>'
        else:
            html_label = b.label
            if b.sub:
                html_label += f'<br><font style="font-size:7px;color:#6b7280;">{b.sub}</font>'

        style += "overflow=hidden;"
        cell = ET.Element("mxCell", {
            "id": fid, "value": html_label,
            "style": style, "vertex": "1", "parent": "1",
        })
        cell.append(ET.Element("mxGeometry", {
            "x": str(b.x), "y": str(b.y),
            "width": str(b.w), "height": str(b.h),
            "as": "geometry",
        }))
        self.cells.append(cell)
        return fid

    def zone(self, x: float, y: float, w: float, h: float, label: str) -> str:
        fid = self.new_id()
        style = (
            "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#d1d5db;"
            "strokeWidth=1;dashed=1;dashPattern=4 3;fontSize=9;fontColor=#6b7280;"
            "align=left;verticalAlign=top;fontFamily=Helvetica;"
        )
        cell = ET.Element("mxCell", {
            "id": fid, "value": label,
            "style": style, "vertex": "1", "parent": "1",
        })
        cell.append(ET.Element("mxGeometry", {
            "x": str(x), "y": str(y),
            "width": str(w), "height": str(h),
            "as": "geometry",
        }))
        self.cells.append(cell)
        return fid

    def arrow(self, src_id: str, tgt_id: str, label: str = "",
              exit_x: float = 1.0, exit_y: float = 0.5,
              entry_x: float = 0.0, entry_y: float = 0.5,
              color: str = "#6b7280", dash: bool = False,
              waypoints: Optional[list[tuple[float, float]]] = None) -> str:
        fid = self.new_id()
        dash_str = ";dashed=1" if dash else ""
        style = (
            f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
            f"jettySize=auto;html=1;exitX={exit_x};exitY={exit_y};"
            f"entryX={entry_x};entryY={entry_y};"
            f"strokeColor={color};strokeWidth=1.5;fontSize=8;fontColor={color};"
            f"endArrow=block;endFill=1;{dash_str}"
        )
        cell = ET.Element("mxCell", {
            "id": fid, "value": label,
            "style": style, "edge": "1", "parent": "1",
            "source": src_id, "target": tgt_id,
        })
        geo = ET.Element("mxGeometry", {"relative": "1", "as": "geometry"})
        if waypoints:
            arr = ET.Element("Array", {"as": "points"})
            for wx, wy in waypoints:
                arr.append(ET.Element("mxPoint", {"x": str(wx), "y": str(wy)}))
            geo.append(arr)
        cell.append(geo)
        self.cells.append(cell)
        return fid

    def down_arrow(self, src_id: str, tgt_id: str, label: str = "",
                   exit_x: float = 0.5, exit_y: float = 1.0,
                   entry_x: float = 0.5, entry_y: float = 0.0,
                   color: str = "#2563eb", dash: bool = False) -> str:
        return self.arrow(src_id, tgt_id, label,
                          exit_x=exit_x, exit_y=exit_y,
                          entry_x=entry_x, entry_y=entry_y,
                          color=color, dash=dash)

    def build_diagram(self, name: str, diagram_id: str, page_w: int, page_h: int) -> ET.Element:
        self.next_id = 2
        self.cells.clear()
        root = ET.Element("mxGraphModel", {
            "dx": "0", "dy": "0", "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1",
            "arrows": "1", "fold": "1", "page": "1",
            "pageScale": "1", "pageWidth": str(page_w),
            "pageHeight": str(page_h), "math": "0", "shadow": "0",
        })
        r = ET.Element("root")
        r.append(ET.Element("mxCell", {"id": "0"}))
        r.append(ET.Element("mxCell", {"id": "1", "parent": "0"}))
        # diagram-specific render logic — pass `self` so it can call box/arrow
        return root, r

    def finalize(self, root: ET.Element, r: ET.Element) -> ET.Element:
        for c in self.cells:
            r.append(c)
        root.append(r)
        return root

    def make_diagram(self, name: str, diagram_id: str,
                     page_w: int, page_h: int,
                     build_fn) -> ET.Element:
        self.next_id = 2
        self.cells.clear()
        root = ET.Element("mxGraphModel", {
            "dx": "0", "dy": "0", "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1",
            "arrows": "1", "fold": "1", "page": "1",
            "pageScale": "1", "pageWidth": str(page_w),
            "pageHeight": str(page_h), "math": "0", "shadow": "0",
        })
        r = ET.Element("root")
        r.append(ET.Element("mxCell", {"id": "0"}))
        r.append(ET.Element("mxCell", {"id": "1", "parent": "0"}))
        build_fn()
        for c in self.cells:
            r.append(c)
        root.append(r)

        diag = ET.Element("diagram", {"id": diagram_id, "name": name})
        diag.append(root)
        return diag


def rect_center(b: Box) -> tuple[float, float]:
    return (b.x + b.w / 2, b.y + b.h / 2)


def build_all() -> ET.Element:
    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net",
        "modified": "2026-05-26T00:00:00.000Z",
        "agent": "SimTutor",
        "version": "24.0.0",
        "type": "device",
    })

    gen = DrawioGen()

    # ==================== DIAGRAM 1: SYSTEM OVERVIEW ====================
    def d1():
        gen.zone(8, 8, 185, 280, "Hardware / Sim")
        gen.zone(210, 8, 380, 280, "SimTutor Process (live_dcs.py)")
        gen.zone(610, 8, 390, 280, "Remote &amp; Output")

        q3   = gen.box(Box(22, 40, 155, 38, "Quest 3 HMD<br>(fixed platform)", "hw"))
        dcs  = gen.box(Box(22, 92, 155, 38, "DCS F/A-18C<br>cold-start mission", "hw"))
        bios = gen.box(Box(22, 144, 155, 38, "DCS-BIOS<br>UDP 239.255.50.10:5010", "hw"))
        pack = gen.box(Box(22, 200, 155, 38, "Pack Config<br>packs/fa18c_startup/", "data"))
        vfc  = gen.box(Box(22, 248, 155, 38, "Vision Facts<br>13 facts, sticky", "data"))

        l1 = gen.box(Box(230, 40, 175, 34, "LiveDcsTutorLoop<br>live_dcs.py", "adapter"))
        l2 = gen.box(Box(230, 84, 175, 34, "build_help_prompt_result()<br>adapters/prompting.py", "adapter"))
        l3 = gen.box(Box(230, 128, 175, 34, "EvidencePacket<br>core/evidence_packet.py", "core"))
        l4 = gen.box(Box(230, 172, 175, 34, "plan_harness_action()<br>core/harness_validation.py", "core"))
        l5 = gen.box(Box(230, 216, 175, 34, "OverlayActionExecutor<br>adapters/action_executor.py", "adapter"))
        l6 = gen.box(Box(230, 256, 175, 34, "experiment-export<br>core/experiment_export.py", "core"))

        r1 = gen.box(Box(625, 40, 180, 38, "OpenAICompatModel<br>Qwen3-8B — text LLM", "llm"))
        r2 = gen.box(Box(625, 92, 180, 38, "VisionFactExtractor<br>Qwen3.5-27B — VLM", "vlm"))
        r3 = gen.box(Box(825, 40, 160, 44, "vLLM Server<br>Remote GPU", "ext"))
        r4 = gen.box(Box(825, 98, 160, 44, "simtutor-base<br>simtutor-vision (LoRA)", "ext"))
        r5 = gen.box(Box(825, 160, 160, 38, "JSONL Event Log<br>logs/*.jsonl", "data"))
        r6 = gen.box(Box(825, 212, 160, 38, "CSV Artifacts<br>trial_summary, step_coding...", "data"))
        r7 = gen.box(Box(825, 260, 160, 38, "Analysis Output<br>study_summary, figures", "data"))

        ovc = gen.box(Box(445, 256, 155, 34, "SimTutorConfig.lua<br>DCS overlay config", "hw"))

        gen.down_arrow(q3, dcs)
        gen.down_arrow(dcs, bios)
        gen.arrow(bios, l1, "telemetry")
        gen.arrow(dcs, l1, "VR mirror")
        gen.down_arrow(l1, l2)
        gen.down_arrow(l2, l3)
        gen.down_arrow(l3, l4)
        gen.arrow(l3, r1, "evidence for prompt", waypoints=[(415, 145), (415, 59), (620, 59)])
        gen.arrow(l3, r2, "VLM facts", waypoints=[(415, 145), (550, 145), (550, 111), (620, 111)])
        gen.arrow(l4, l5)
        gen.arrow(l5, ovc, "write config")
        gen.arrow(r1, r3, "SSH tunnel")
        gen.arrow(r2, r4, "SSH tunnel")
        gen.down_arrow(l5, l6, exit_x=0.66)
        gen.arrow(l6, r5, "freeze trial")
        gen.down_arrow(r5, r6)
        gen.down_arrow(r6, r7)
        gen.arrow(pack, l3, "steps/gates")

    diag1 = gen.make_diagram("1. System Overview", "d1", 1020, 380, d1)

    # ==================== DIAGRAM 2: LIVE HELP CYCLE ====================
    def d2():
        steps = [
            ("1. Participant presses X1", "adapters/windows_global_help_trigger.py", "adapter"),
            ("2. Telemetry snapshot", "adapters/dcs_bios/receiver.py", "adapter"),
            ("3. Enrich derived vars", "adapters/telemetry_pipeline.py", "adapter"),
            ("4. Trigger vision capture", "adapters/vision_capture_trigger.py", "adapter"),
            ("5. Deterministic inference", "adapters/step_inference.py", "adapter"),
            ("6. Evaluate pack gates", "adapters/pack_gates.py", "adapter"),
            ("7. Build EvidencePacket", "core/evidence_packet.py", "core"),
            ("8. VLM extraction (if visual step)", "adapters/vision_fact_extractor.py", "vlm"),
            ("9. RAG knowledge retrieval", "adapters/knowledge_local.py", "adapter"),
            ("10. Build LLM prompt", "adapters/prompting.py", "adapter"),
            ("11. Call text LLM", "adapters/openai_compat_model.py", "llm"),
            ("12. Map response", "adapters/response_mapping.py", "adapter"),
            ("13. Harness validate &amp; plan", "core/harness_validation.py", "core"),
            ("14. Dispatch overlay", "adapters/action_executor.py", "adapter"),
            ("15. Log to JSONL", "core/event_store.py", "core"),
        ]
        boxW, boxX, boxH, gap = 400, 100, 34, 38
        tagX, tagW = 30, 55
        prev_id = None
        for i, (title, file, kind) in enumerate(steps):
            y = 30 + i * gap
            b = Box(boxX, y, boxW, boxH, f"{title}<br>{file}", kind)
            fid = gen.box(b)
            # layer badge
            layer = {"core": "CORE", "adapter": "ADAPT", "vlm": "VLM", "llm": "LLM"}.get(kind, kind.upper())
            gen.box(Box(tagX, y + 6, tagW, 14, layer, kind))
            if prev_id:
                gen.down_arrow(prev_id, fid)
            prev_id = fid

    diag2 = gen.make_diagram("2. Live Help Cycle", "d2", 620, 640, d2)

    # ==================== DIAGRAM 3: VLM / LLM SEPARATION ====================
    def d3():
        req  = gen.box(Box(310, 8, 200, 30, "Help Request Received (X1)", "core"))
        gate = gen.box(Box(270, 100, 200, 38, "requires_visual_confirmation?<br>from pack step config", "core"))
        ve   = gen.box(Box(20, 60, 230, 34, "VisionFactExtractor<br>adapters/vision_fact_extractor.py", "vlm"))
        vf   = gen.box(Box(20, 108, 230, 42, "Composite Panel Screenshots<br>pre_trigger_frame + trigger_frame", "data"))
        vo   = gen.box(Box(20, 164, 230, 42, "13 VisionFact Objects<br>seen / not_seen / uncertain", "data"))
        vs   = gen.box(Box(20, 220, 230, 34, "Vision Fact Summary<br>→ EvidencePacket.vision_evidence", "data"))
        le   = gen.box(Box(500, 60, 240, 34, "OpenAICompatModel<br>adapters/openai_compat_model.py", "llm"))
        lp   = gen.box(Box(500, 108, 240, 42, "Text Prompt<br>Evidence + Candidates + RAG + Specs", "data"))
        lo   = gen.box(Box(500, 164, 240, 42, "HarnessDecision JSON<br>step, diagnosis, targets, evidence_refs", "data"))
        lr   = gen.box(Box(500, 220, 240, 34, "TutorResponse<br>→ overlay + text guidance to cockpit", "data"))
        vm   = gen.box(Box(80, 280, 200, 32, "simtutor-vision (LoRA)<br>Qwen3.5-27B on remote GPU", "ext"))
        lm   = gen.box(Box(530, 280, 200, 32, "simtutor-base (text)<br>Qwen3-8B on remote GPU", "ext"))
        call = gen.box(Box(310, 220, 200, 46, "Visual-Priority Steps Only:<br>S08, S15, S18, S19", "vlm"))

        gen.arrow(req, gate, "check step config", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        # VLM branch
        gen.arrow(gate, ve, "Yes (visual step)", exit_x=0.0, entry_x=1.0,
                  waypoints=[(260, 119), (260, 77)])
        gen.down_arrow(ve, vf)
        gen.down_arrow(vf, vo)
        gen.down_arrow(vo, vs)
        gen.arrow(ve, vm, "HTTP/SSH tunnel", exit_x=0.0, entry_x=0.0,
                  waypoints=[(10, 77), (10, 296)])
        # LLM branch
        gen.arrow(gate, le, "No, or after VLM →", exit_x=1.0, entry_x=0.0)
        gen.down_arrow(le, lp)
        gen.down_arrow(lp, lo)
        gen.down_arrow(lo, lr)
        gen.arrow(le, lm, "HTTP/SSH tunnel", exit_x=1.0, entry_x=1.0,
                  waypoints=[(750, 77), (750, 296)])
        # Cross connection
        gen.arrow(vs, lp, "facts injected into evidence", exit_x=1.0, entry_x=0.0,
                  waypoints=[(250, 237), (250, 129)])
        # Text only shortcut
        gen.arrow(gate, le, "text only (skip VLM)", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0,
                  waypoints=[(370, 138), (370, 50), (620, 50)],
                  color="#9ca3af", dash=True)

    diag3 = gen.make_diagram("3. VLM / LLM Separation", "d3", 820, 340, d3)

    # ==================== DIAGRAM 4: HARNESS ENGINEERING OBJECTS ====================
    def d4():
        ev1 = gen.box(Box(8,  10, 170, 34, "TelemetryEvidence<br>source_status, confidence", "data"))
        ev2 = gen.box(Box(190, 10, 170, 34, "TelemetryWindowDigest<br>windowed var analysis", "data"))
        ev3 = gen.box(Box(372, 10, 170, 34, "VisionEvidence<br>13 facts, anchors", "data"))
        ev4 = gen.box(Box(554, 10, 170, 34, "GateEvidence<br>blocked/satisfied gates", "data"))
        ev5 = gen.box(Box(736, 10, 170, 34, "RecentActionEvidence<br>target_ids, deltas", "data"))

        inf = gen.box(Box(8,  65, 170, 30, "infer_step_id()<br>adapters/step_inference.py", "adapter"))
        det = gen.box(Box(195, 65, 160, 30, "DeterministicCandidate<br>step_id, missing_conditions", "data"))
        bep = gen.box(Box(480, 65, 200, 30, "build_evidence_packet()<br>core/evidence_packet.py", "core"))

        ep  = gen.box(Box(340, 115, 210, 34, "EvidencePacket<br>+ conflicts tuple (4 types)", "data"))
        bsc = gen.box(Box(280, 170, 200, 30, "build_step_candidates()<br>max 8, ordered by confidence", "core"))
        sc  = gen.box(Box(530, 170, 180, 30, "StepCandidate[]<br>source, confidence, evidence_refs", "data"))

        shs = gen.box(Box(20, 230, 180, 30, "StepHarnessSpec[]<br>core/step_harness.py", "core"))
        ldc = gen.box(Box(280, 230, 200, 30, "HarnessDecision (LLM output)<br>JSON Schema contract", "llm"))
        sap = gen.box(Box(20, 290, 180, 30, "State-Action Planner<br>S08/S09/S12/S18/S19 rules", "core"))
        pha = gen.box(Box(310, 290, 200, 30, "plan_harness_action()<br>core/harness_validation.py", "core"))
        vld = gen.box(Box(560, 285, 200, 30, "validate_final_evidence_consistency()", "core"))
        hap = gen.box(Box(340, 345, 210, 30, "HarnessActionPlan<br>targets, guidance, text_only, repaired", "data"))
        ecr = gen.box(Box(580, 345, 180, 30, "EvidenceConsistencyResult<br>accepted, rejected, repair", "data"))
        out = gen.box(Box(380, 390, 170, 22, "→ Public TutorResponse<br>safe for cockpit display", "data"))

        # evidence → build_evidence_packet
        for ev_id in [ev1, ev2, ev3, ev4, ev5]:
            gen.arrow(ev_id, bep, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0,
                      waypoints=None)  # simplified direct

        gen.down_arrow(bep, ep)
        gen.down_arrow(inf, det)
        gen.arrow(det, ep)
        gen.down_arrow(ep, bsc, exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        gen.arrow(bsc, sc)
        gen.arrow(ep, ldc, "evidence feeds prompt", exit_x=1.0, entry_x=0.0,
                  waypoints=[(550, 132), (550, 245)])
        gen.arrow(sc, ldc, "candidates", exit_x=0.5, exit_y=1.0, entry_x=0.3, entry_y=0.0)
        gen.arrow(shs, ldc, "specs")
        gen.down_arrow(ldc, pha)
        gen.arrow(sap, pha)
        gen.down_arrow(pha, hap)
        gen.arrow(pha, vld, "", exit_x=1.0, entry_x=0.0)
        gen.down_arrow(vld, ecr)
        gen.arrow(hap, out, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        gen.arrow(ecr, out, "", exit_x=0.0, entry_x=1.0,
                  waypoints=[(570, 360), (560, 360), (560, 401)])

    diag4 = gen.make_diagram("4. Harness Engineering Objects", "d4", 960, 420, d4)

    # ==================== DIAGRAM 5: EXPERIMENT EXPORT AND ANALYSIS ====================
    def d5():
        jlog = gen.box(Box(15, 15, 165, 38, "Raw JSONL Event Log<br>one .jsonl per trial", "data"))
        cli  = gen.box(Box(15, 72, 165, 42, "CLI Metadata<br>--participant-id --condition<br>--study-id --model-name...", "data"))
        exp  = gen.box(Box(230, 40, 200, 36, "experiment-export<br>core/experiment_export.py", "core"))
        exp2 = gen.box(Box(230, 112, 200, 30, "Repeat per participant/trial", "core"))
        ana  = gen.box(Box(230, 170, 200, 36, "experiment-analyze<br>All trials combined", "core"))
        pki  = gen.box(Box(15, 132, 165, 42, "Pack &amp; Taxonomy<br>pack.yaml, taxonomy.yaml<br>ui_map.yaml, bios_to_ui.yaml", "data"))
        qg   = gen.box(Box(15, 200, 165, 34, "quality_gate.passed?<br>must be true", "core"))

        oy, oh, op_gap = 8, 26, 2
        ocol_x = 500
        output_names = [
            "trial_summary.csv", "step_coding.csv", "help_cycles.csv",
            "action_timeline.csv", "quality_gate.json", "session.json",
            "raw_events.jsonl (copy)",
        ]
        oids = []
        for i, name in enumerate(output_names):
            oids.append(gen.box(Box(ocol_x, oy + i * (oh + op_gap), 170, oh, name, "data")))

        ay = 170
        acol_x = 690
        agg_names = [
            "study_summary.csv", "condition_summary.csv",
            "step_accuracy_by_condition.csv", "help_quality_summary.csv",
            "fig_*.png (optional)",
        ]
        aids = []
        for i, name in enumerate(agg_names):
            aids.append(gen.box(Box(acol_x, ay + i * (oh + op_gap), 175, oh, name, "data")))

        gen.arrow(jlog, exp)
        gen.arrow(cli, exp, waypoints=[(180, 93), (220, 93), (220, 58)])
        gen.arrow(pki, exp, "step coding rules", waypoints=[(180, 153), (220, 153), (220, 68)])
        for oid in oids:
            gen.arrow(exp, oid)
        gen.down_arrow(exp, exp2)
        gen.down_arrow(exp2, ana)
        gen.arrow(qg, exp2, "yes →")
        for aid in aids:
            gen.arrow(ana, aid)

    diag5 = gen.make_diagram("5. Experiment Export and Analysis", "d5", 880, 330, d5)

    # ==================== DIAGRAM 6: DEPLOYMENT TOPOLOGY ====================
    def d6():
        gen.zone(8, 8, 370, 335, "Windows Simulator Host")
        gen.zone(450, 8, 352, 240, "Remote GPU Server (cloud-247)")

        dcs  = gen.box(Box(22, 42, 155, 32, "DCS World F/A-18C", "hw"))
        bios = gen.box(Box(22, 88, 155, 32, "DCS-BIOS (UDP:5010)", "hw"))
        q3   = gen.box(Box(22, 134, 155, 32, "Quest 3 (Oculus Link)", "hw"))
        vs   = gen.box(Box(200, 42, 160, 32, "Vision Sidecar<br>capture_vision_sidecar.py", "vlm"))
        ld   = gen.box(Box(200, 88, 160, 32, "SimTutor live-dcs", "adapter"))
        gui  = gen.box(Box(200, 134, 160, 32, "GUI Launcher<br>simtutor_launcher.py", "adapter"))
        sg   = gen.box(Box(22, 188, 200, 32, "Saved Games\\DCS\\<br>SimTutorConfig.lua", "hw"))
        logf = gen.box(Box(22, 238, 200, 32, "logs/*.jsonl<br>Runtime event log", "data"))
        art  = gen.box(Box(22, 288, 200, 32, "artifacts/experiments/<br>CSV per-trial exports", "data"))
        ex2  = gen.box(Box(250, 188, 110, 32, "Experiment Export", "core"))

        vllm  = gen.box(Box(470, 42, 310, 32, "vLLM Server (localhost:8000)", "ext"))
        sbase = gen.box(Box(470, 88, 310, 32, "simtutor-base — Qwen3-8B (text LLM)", "llm"))
        svis  = gen.box(Box(470, 134, 310, 32, "simtutor-vision — Qwen3.5-27B + LoRA", "vlm"))
        ssh   = gen.box(Box(470, 180, 310, 32, "SSH Tunnel Endpoint", "ext"))

        gen.down_arrow(dcs, bios)
        gen.down_arrow(bios, q3)
        gen.arrow(dcs, vs, "VR mirror")
        gen.arrow(bios, ld, "telemetry")
        gen.arrow(vs, ld, "frame trigger")
        gen.arrow(ld, sg, "write config",
                  waypoints=[(360, 104), (370, 104), (370, 204), (222, 204)])
        gen.down_arrow(sg, logf)
        gen.down_arrow(logf, art)
        gen.arrow(logf, ex2)
        gen.arrow(ld, sbase, "SSH: text LLM call",
                  waypoints=[(360, 104), (410, 104), (410, 104)],
                  dash=True)
        gen.arrow(vs, svis, "SSH: VLM call",
                  waypoints=[(360, 58), (360, 58)],
                  dash=True)
        gen.down_arrow(sbase, svis)
        gen.down_arrow(svis, ssh)

    diag6 = gen.make_diagram("6. Deployment Topology", "d6", 820, 370, d6)

    for d in [diag1, diag2, diag3, diag4, diag5, diag6]:
        mxfile.append(d)

    return mxfile


def main():
    mxfile = build_all()

    # Write clean XML without minidom (which can add whitespace text nodes)
    output_path = "docs/architecture/diagrams.drawio"
    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        _write_element(f, mxfile, indent=0)
    size = os.path.getsize(output_path)
    print(f"Wrote {output_path} ({size} bytes, 6 diagram tabs)")


def _write_element(f, elem, indent):
    import io
    tag = elem.tag
    # Remove namespace prefix if present
    if '}' in tag:
        tag = tag.split('}', 1)[1]

    # Gather attribs
    attrs = ' '.join(f'{k}="{_esc_attr(v)}"' for k, v in sorted(elem.items()))
    start = f'{"  " * indent}<{tag} {attrs}'.rstrip() + '>'

    # Get text and tail
    text = elem.text or ''
    tail = elem.tail or ''
    children = list(elem)

    if not children and not text:
        # Self-closing
        f.write(f'{"  " * indent}<{tag} {attrs}/>\n'.replace(' >', '>'))
    elif not children:
        # Leaf with text
        f.write(f'{start}{_esc_text(text)}</{tag}>\n')
    else:
        # Has children
        f.write(f'{start}\n')
        if text.strip():
            f.write(f'{"  " * (indent+1)}{_esc_text(text)}\n')
        for child in children:
            _write_element(f, child, indent + 1)
        f.write(f'{"  " * indent}</{tag}>\n')


def _esc_attr(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _esc_text(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


if __name__ == "__main__":
    main()
