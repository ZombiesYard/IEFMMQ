#!/usr/bin/env python3
"""Generate diagrams_single.drawio — all 6 diagrams on one large page, stacked vertically."""
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
PAGE_W = 1050
# offsets for each diagram
OFFSETS = [0, 420, 1100, 1480, 1940, 2310]
TITLES = [
    "1. System Overview",
    "2. Live Help Cycle (15 stages)",
    "3. VLM / LLM Separation",
    "4. Harness Engineering Objects",
    "5. Experiment Export and Analysis Pipeline",
    "6. Deployment Topology",
]

@dataclass
class Box:
    x: float; y: float; w: float; h: float; label: str; kind: str
    sub: str = ""

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
            f"verticalAlign=middle;overflow=hidden;"
        )
        label = b.label
        if b.sub:
            label += f'<br><font style="font-size:7px;color:#6b7280;">{b.sub}</font>'
        cell = ET.Element("mxCell", {
            "id": fid, "value": label,
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

    def title_box(self, x: float, y: float, text: str) -> str:
        fid = self.new_id()
        style = (
            "rounded=1;whiteSpace=wrap;html=1;fillColor=#1e40af;strokeColor=#1e3a8a;"
            "fontSize=13;fontColor=#ffffff;fontFamily=Helvetica;align=left;verticalAlign=middle;"
        )
        cell = ET.Element("mxCell", {
            "id": fid, "value": f'<b>{text}</b>',
            "style": style, "vertex": "1", "parent": "1",
        })
        cell.append(ET.Element("mxGeometry", {
            "x": str(x), "y": str(y), "width": str(PAGE_W-40), "height": "28",
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

    def clear(self):
        self.next_id = 2
        self.cells.clear()

    def cells_list(self) -> list[ET.Element]:
        return list(self.cells)


def build_all() -> ET.Element:
    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net",
        "modified": "2026-05-26T00:00:00.000Z",
        "agent": "SimTutor",
        "version": "24.0.0",
        "type": "device",
    })

    gen = DrawioGen()

    # We'll build into one big diagram
    root = ET.Element("mxGraphModel", {
        "dx": "0", "dy": "0", "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1",
        "arrows": "1", "fold": "1", "page": "1",
        "pageScale": "1", "pageWidth": str(PAGE_W),
        "pageHeight": "2700", "math": "0", "shadow": "0",
    })
    r = ET.Element("root")
    r.append(ET.Element("mxCell", {"id": "0"}))
    r.append(ET.Element("mxCell", {"id": "1", "parent": "0"}))

    # ---- DIAGRAM 1: SYSTEM OVERVIEW (y=0..380) ----
    def off(b: Box, oy: float) -> Box:
        return Box(b.x, b.y + oy, b.w, b.h, b.label, b.kind, b.sub)

    oy = OFFSETS[0]
    gen.clear()
    gen.title_box(20, oy + 2, TITLES[0])

    gen.zone(off(Box(8, 34, 185, 280, "", ""), oy).x, off(Box(8, 34, 185, 280, "", ""), oy).y,
             185, 280, "Hardware / Sim")
    gen.zone(off(Box(210, 34, 380, 280, "", ""), oy).x, off(Box(210, 34, 380, 280, "", ""), oy).y,
             380, 280, "SimTutor Process (live_dcs.py)")
    gen.zone(off(Box(610, 34, 390, 280, "", ""), oy).x, off(Box(610, 34, 390, 280, "", ""), oy).y,
             390, 280, "Remote &amp; Output")

    q3   = gen.box(off(Box(22, 60, 155, 38,  "Quest 3 HMD<br>(fixed platform)", "hw"), oy))
    dcs  = gen.box(off(Box(22, 112, 155, 38, "DCS F/A-18C<br>cold-start mission", "hw"), oy))
    bios = gen.box(off(Box(22, 164, 155, 38, "DCS-BIOS<br>UDP 239.255.50.10:5010", "hw"), oy))
    pack = gen.box(off(Box(22, 220, 155, 38, "Pack Config<br>packs/fa18c_startup/", "data"), oy))
    vfc  = gen.box(off(Box(22, 268, 155, 38, "Vision Facts<br>13 facts, sticky", "data"), oy))

    l1 = gen.box(off(Box(230, 60, 175, 34, "LiveDcsTutorLoop<br>live_dcs.py", "adapter"), oy))
    l2 = gen.box(off(Box(230, 104, 175, 34, "build_help_prompt_result()<br>adapters/prompting.py", "adapter"), oy))
    l3 = gen.box(off(Box(230, 148, 175, 34, "EvidencePacket<br>core/evidence_packet.py", "core"), oy))
    l4 = gen.box(off(Box(230, 192, 175, 34, "plan_harness_action()<br>core/harness_validation.py", "core"), oy))
    l5 = gen.box(off(Box(230, 236, 175, 34, "OverlayActionExecutor<br>adapters/action_executor.py", "adapter"), oy))
    l6 = gen.box(off(Box(230, 276, 175, 34, "experiment-export<br>core/experiment_export.py", "core"), oy))

    r1 = gen.box(off(Box(625, 60, 180, 38, "OpenAICompatModel<br>Qwen3-8B — text LLM", "llm"), oy))
    r2 = gen.box(off(Box(625, 112, 180, 38, "VisionFactExtractor<br>Qwen3.5-27B — VLM", "vlm"), oy))
    r3 = gen.box(off(Box(825, 60, 160, 44, "vLLM Server<br>Remote GPU", "ext"), oy))
    r4 = gen.box(off(Box(825, 118, 160, 44, "simtutor-base<br>simtutor-vision (LoRA)", "ext"), oy))
    r5 = gen.box(off(Box(825, 180, 160, 38, "JSONL Event Log<br>logs/*.jsonl", "data"), oy))
    r6 = gen.box(off(Box(825, 232, 160, 38, "CSV Artifacts<br>trial_summary, step_coding...", "data"), oy))
    r7 = gen.box(off(Box(825, 280, 160, 38, "Analysis Output<br>study_summary, figures", "data"), oy))
    ovc = gen.box(off(Box(445, 276, 155, 34, "SimTutorConfig.lua<br>DCS overlay config", "hw"), oy))

    gen.down_arrow(q3, dcs); gen.down_arrow(dcs, bios)
    gen.arrow(bios, l1, "telemetry"); gen.arrow(dcs, l1, "VR mirror")
    gen.down_arrow(l1, l2); gen.down_arrow(l2, l3); gen.down_arrow(l3, l4)
    gen.arrow(l3, r1, "evidence for prompt", waypoints=[(415, oy+165), (415, oy+79)])
    gen.arrow(l3, r2, "VLM facts", waypoints=[(415, oy+165), (550, oy+131)])
    gen.arrow(l4, l5); gen.arrow(l5, ovc, "write config")
    gen.arrow(r1, r3, "SSH tunnel"); gen.arrow(r2, r4, "SSH tunnel")
    gen.down_arrow(l5, l6, exit_x=0.66)
    gen.arrow(l6, r5, "freeze trial")
    gen.down_arrow(r5, r6); gen.down_arrow(r6, r7)
    gen.arrow(pack, l3, "steps/gates")

    # ---- DIAGRAM 2: LIVE HELP CYCLE (y=420..1060) ----
    oy = OFFSETS[1]
    gen.title_box(20, oy + 2, TITLES[1])

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
        y = oy + 40 + i * gap
        b = Box(boxX, y, boxW, boxH, f"{title}<br>{file}", kind)
        fid = gen.box(b)
        layer = {"core": "CORE", "adapter": "ADAPT", "vlm": "VLM", "llm": "LLM"}.get(kind, kind.upper())
        gen.box(Box(tagX, y + 6, tagW, 14, layer, kind))
        if prev_id:
            gen.down_arrow(prev_id, fid)
        prev_id = fid

    # ---- DIAGRAM 3: VLM / LLM SEPARATION (y=1100..1440) ----
    oy = OFFSETS[2]
    gen.title_box(20, oy + 2, TITLES[2])

    req  = gen.box(off(Box(310, 40, 200, 30, "Help Request Received (X1)", "core"), oy))
    gate = gen.box(off(Box(270, 130, 200, 38, "requires_visual_confirmation?<br>from pack step config", "core"), oy))
    ve   = gen.box(off(Box(20, 90, 230, 34, "VisionFactExtractor<br>adapters/vision_fact_extractor.py", "vlm"), oy))
    vf   = gen.box(off(Box(20, 138, 230, 42, "Composite Panel Screenshots<br>pre_trigger_frame + trigger_frame", "data"), oy))
    vo   = gen.box(off(Box(20, 194, 230, 42, "13 VisionFact Objects<br>seen / not_seen / uncertain", "data"), oy))
    vs   = gen.box(off(Box(20, 250, 230, 34, "Vision Fact Summary<br>→ EvidencePacket.vision_evidence", "data"), oy))
    le   = gen.box(off(Box(500, 90, 240, 34, "OpenAICompatModel<br>adapters/openai_compat_model.py", "llm"), oy))
    lp   = gen.box(off(Box(500, 138, 240, 42, "Text Prompt<br>Evidence + Candidates + RAG + Specs", "data"), oy))
    lo   = gen.box(off(Box(500, 194, 240, 42, "HarnessDecision JSON<br>step, diagnosis, targets, evidence_refs", "data"), oy))
    lr   = gen.box(off(Box(500, 250, 240, 34, "TutorResponse<br>→ overlay + text guidance to cockpit", "data"), oy))
    vm   = gen.box(off(Box(80, 310, 200, 32, "simtutor-vision (LoRA)<br>Qwen3.5-27B on remote GPU", "ext"), oy))
    lm   = gen.box(off(Box(530, 310, 200, 32, "simtutor-base (text)<br>Qwen3-8B on remote GPU", "ext"), oy))
    call = gen.box(off(Box(310, 250, 200, 46, "Visual-Priority Steps Only:<br>S08, S15, S18, S19", "vlm"), oy))

    gen.arrow(req, gate, "check step config", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
    gen.arrow(gate, ve, "Yes (visual step)", exit_x=0.0, entry_x=1.0,
              waypoints=[(260, oy+149), (260, oy+107)])
    gen.down_arrow(ve, vf); gen.down_arrow(vf, vo); gen.down_arrow(vo, vs)
    gen.arrow(ve, vm, "HTTP/SSH", exit_x=0.0, entry_x=0.0,
              waypoints=[(10, oy+107), (10, oy+326)])
    gen.arrow(gate, le, "No, or after VLM", exit_x=1.0, entry_x=0.0)
    gen.down_arrow(le, lp); gen.down_arrow(lp, lo); gen.down_arrow(lo, lr)
    gen.arrow(le, lm, "HTTP/SSH", exit_x=1.0, entry_x=1.0,
              waypoints=[(750, oy+107), (750, oy+326)])
    gen.arrow(vs, lp, "facts injected", exit_x=1.0, entry_x=0.0,
              waypoints=[(250, oy+267), (250, oy+159)])
    gen.arrow(gate, le, "text only", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0,
              waypoints=[(370, oy+168), (370, oy+80), (620, oy+80)],
              color="#9ca3af", dash=True)

    # ---- DIAGRAM 4: HARNESS ENGINEERING (y=1480..1900) ----
    oy = OFFSETS[3]
    gen.title_box(20, oy + 2, TITLES[3])

    ev1 = gen.box(off(Box(8,  40, 170, 34, "TelemetryEvidence<br>source_status, confidence", "data"), oy))
    ev2 = gen.box(off(Box(190, 40, 170, 34, "TelemetryWindowDigest<br>windowed var analysis", "data"), oy))
    ev3 = gen.box(off(Box(372, 40, 170, 34, "VisionEvidence<br>13 facts, anchors", "data"), oy))
    ev4 = gen.box(off(Box(554, 40, 170, 34, "GateEvidence<br>blocked/satisfied gates", "data"), oy))
    ev5 = gen.box(off(Box(736, 40, 170, 34, "RecentActionEvidence<br>target_ids, deltas", "data"), oy))
    inf = gen.box(off(Box(8,  95, 170, 30, "infer_step_id()<br>adapters/step_inference.py", "adapter"), oy))
    det = gen.box(off(Box(195, 95, 160, 30, "DeterministicCandidate<br>step_id, missing_conditions", "data"), oy))
    bep = gen.box(off(Box(480, 95, 200, 30, "build_evidence_packet()<br>core/evidence_packet.py", "core"), oy))
    ep  = gen.box(off(Box(340, 145, 210, 34, "EvidencePacket<br>+ conflicts tuple (4 types)", "data"), oy))
    bsc = gen.box(off(Box(280, 200, 200, 30, "build_step_candidates()<br>max 8, ordered by confidence", "core"), oy))
    sc  = gen.box(off(Box(530, 200, 180, 30, "StepCandidate[]<br>source, confidence, evidence_refs", "data"), oy))
    shs = gen.box(off(Box(20, 260, 180, 30, "StepHarnessSpec[]<br>core/step_harness.py", "core"), oy))
    ldc = gen.box(off(Box(280, 260, 200, 30, "HarnessDecision (LLM output)<br>JSON Schema contract", "llm"), oy))
    sap = gen.box(off(Box(20, 320, 180, 30, "State-Action Planner<br>S08/S09/S12/S18/S19 rules", "core"), oy))
    pha = gen.box(off(Box(310, 320, 200, 30, "plan_harness_action()<br>core/harness_validation.py", "core"), oy))
    vld = gen.box(off(Box(560, 315, 200, 30, "validate_final_evidence_consistency()", "core"), oy))
    hap = gen.box(off(Box(340, 375, 210, 30, "HarnessActionPlan<br>targets, guidance, text_only, repaired", "data"), oy))
    ecr = gen.box(off(Box(580, 375, 180, 30, "EvidenceConsistencyResult<br>accepted, rejected, repair", "data"), oy))
    out = gen.box(off(Box(380, 420, 170, 22, "→ Public TutorResponse<br>safe for cockpit display", "data"), oy))

    for ev_id in [ev1, ev2, ev3, ev4, ev5]:
        gen.arrow(ev_id, bep, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
    gen.down_arrow(bep, ep)
    gen.down_arrow(inf, det); gen.arrow(det, ep)
    gen.down_arrow(ep, bsc); gen.arrow(bsc, sc)
    gen.arrow(ep, ldc, "evidence feeds prompt", exit_x=1.0, entry_x=0.0,
              waypoints=[(550, oy+162), (550, oy+275)])
    gen.arrow(sc, ldc, "candidates", exit_x=0.5, exit_y=1.0, entry_x=0.3, entry_y=0.0)
    gen.arrow(shs, ldc, "specs")
    gen.down_arrow(ldc, pha); gen.arrow(sap, pha)
    gen.down_arrow(pha, hap)
    gen.arrow(pha, vld, "", exit_x=1.0, entry_x=0.0)
    gen.down_arrow(vld, ecr)
    gen.arrow(hap, out, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
    gen.arrow(ecr, out, "", exit_x=0.0, entry_x=1.0,
              waypoints=[(570, oy+390), (560, oy+390)])

    # ---- DIAGRAM 5: EXPERIMENT EXPORT (y=1940..2270) ----
    oy = OFFSETS[4]
    gen.title_box(20, oy + 2, TITLES[4])

    jlog = gen.box(off(Box(15, 50, 165, 38, "Raw JSONL Event Log<br>one .jsonl per trial", "data"), oy))
    cli  = gen.box(off(Box(15, 107, 165, 42, "CLI Metadata<br>--participant-id --condition<br>--study-id --model-name...", "data"), oy))
    exp  = gen.box(off(Box(230, 75, 200, 36, "experiment-export<br>core/experiment_export.py", "core"), oy))
    exp2 = gen.box(off(Box(230, 147, 200, 30, "Repeat per participant/trial", "core"), oy))
    ana  = gen.box(off(Box(230, 205, 200, 36, "experiment-analyze<br>All trials combined", "core"), oy))
    pki  = gen.box(off(Box(15, 167, 165, 42, "Pack &amp; Taxonomy<br>pack.yaml, taxonomy.yaml<br>ui_map.yaml, bios_to_ui.yaml", "data"), oy))
    qg   = gen.box(off(Box(15, 235, 165, 34, "quality_gate.passed?<br>must be true", "core"), oy))

    out_names = ["trial_summary.csv", "step_coding.csv", "help_cycles.csv",
                 "action_timeline.csv", "quality_gate.json", "session.json", "raw_events.jsonl (copy)"]
    oids = []
    for i, name in enumerate(out_names):
        oids.append(gen.box(off(Box(500, 43 + i * 28, 180, 26, name, "data"), oy)))

    agg_names = ["study_summary.csv", "condition_summary.csv",
                 "step_accuracy_by_condition.csv", "help_quality_summary.csv", "fig_*.png (optional)"]
    aids = []
    for i, name in enumerate(agg_names):
        aids.append(gen.box(off(Box(690, 205 + i * 28, 195, 26, name, "data"), oy)))

    gen.arrow(jlog, exp)
    gen.arrow(cli, exp, waypoints=[(180, oy+128), (220, oy+128), (220, oy+93)])
    gen.arrow(pki, exp, "step coding rules", waypoints=[(180, oy+188), (220, oy+188), (220, oy+104)])
    for oid in oids:
        gen.arrow(exp, oid)
    gen.down_arrow(exp, exp2); gen.down_arrow(exp2, ana)
    gen.arrow(qg, exp2, "yes →")
    for aid in aids:
        gen.arrow(ana, aid)

    # ---- DIAGRAM 6: DEPLOYMENT TOPOLOGY (y=2310..2650) ----
    oy = OFFSETS[5]
    gen.title_box(20, oy + 2, TITLES[5])

    gen.zone(off(Box(8, 34, 370, 335, "", ""), oy).x, off(Box(8, 34, 370, 335, "", ""), oy).y,
             370, 335, "Windows Simulator Host")
    gen.zone(off(Box(450, 34, 352, 240, "", ""), oy).x, off(Box(450, 34, 352, 240, "", ""), oy).y,
             352, 240, "Remote GPU Server (cloud-247)")

    dcs2  = gen.box(off(Box(22, 68, 155, 32, "DCS World F/A-18C", "hw"), oy))
    bios2 = gen.box(off(Box(22, 114, 155, 32, "DCS-BIOS (UDP:5010)", "hw"), oy))
    q3_2  = gen.box(off(Box(22, 160, 155, 32, "Quest 3 (Oculus Link)", "hw"), oy))
    vs2   = gen.box(off(Box(200, 68, 160, 32, "Vision Sidecar<br>capture_vision_sidecar.py", "vlm"), oy))
    ld2   = gen.box(off(Box(200, 114, 160, 32, "SimTutor live-dcs", "adapter"), oy))
    gui2  = gen.box(off(Box(200, 160, 160, 32, "GUI Launcher<br>simtutor_launcher.py", "adapter"), oy))
    sg2   = gen.box(off(Box(22, 214, 200, 32, "Saved Games\\DCS\\<br>SimTutorConfig.lua", "hw"), oy))
    logf2 = gen.box(off(Box(22, 264, 200, 32, "logs/*.jsonl<br>Runtime event log", "data"), oy))
    art2  = gen.box(off(Box(22, 314, 200, 32, "artifacts/experiments/<br>CSV per-trial exports", "data"), oy))
    ex2_2 = gen.box(off(Box(250, 214, 110, 32, "Experiment Export", "core"), oy))

    vllm2  = gen.box(off(Box(470, 68, 310, 32, "vLLM Server (localhost:8000)", "ext"), oy))
    sbase2 = gen.box(off(Box(470, 114, 310, 32, "simtutor-base — Qwen3-8B (text LLM)", "llm"), oy))
    svis2  = gen.box(off(Box(470, 160, 310, 32, "simtutor-vision — Qwen3.5-27B + LoRA", "vlm"), oy))
    ssh2   = gen.box(off(Box(470, 206, 310, 32, "SSH Tunnel Endpoint", "ext"), oy))

    gen.down_arrow(dcs2, bios2); gen.down_arrow(bios2, q3_2)
    gen.arrow(dcs2, vs2, "VR mirror"); gen.arrow(bios2, ld2, "telemetry")
    gen.arrow(vs2, ld2, "frame trigger")
    gen.arrow(ld2, sg2, "write config",
              waypoints=[(360, oy+130), (370, oy+130), (370, oy+230), (222, oy+230)])
    gen.down_arrow(sg2, logf2); gen.down_arrow(logf2, art2)
    gen.arrow(logf2, ex2_2)
    gen.arrow(ld2, sbase2, "SSH: text LLM call", dash=True,
              waypoints=[(360, oy+130), (410, oy+130)])
    gen.arrow(vs2, svis2, "SSH: VLM call", dash=True,
              waypoints=[(360, oy+84), (410, oy+84)])
    gen.down_arrow(sbase2, svis2); gen.down_arrow(svis2, ssh2)

    # Finalize
    for c in gen.cells_list():
        r.append(c)
    root.append(r)

    diag = ET.Element("diagram", {"id": "all", "name": "All Diagrams (single page)"})
    diag.append(root)
    mxfile.append(diag)
    return mxfile


def _write_element(f, elem, indent):
    tag = elem.tag
    if '}' in tag:
        tag = tag.split('}', 1)[1]
    attrs_list = []
    for k, v in sorted(elem.items()):
        attrs_list.append(f'{k}="{_esc_attr(v)}"')
    attrs = ' '.join(attrs_list)
    start = f'{"  " * indent}<{tag} {attrs}'.rstrip() + '>'

    text = elem.text or ''
    children = list(elem)

    if not children and not text.strip():
        f.write(f'{"  " * indent}<{tag} {attrs}/>\n')
    elif not children:
        f.write(f'{start}{_esc_text(text)}</{tag}>\n')
    else:
        f.write(f'{start}\n')
        if text.strip():
            f.write(f'{"  " * (indent+1)}{_esc_text(text.strip())}\n')
        for child in children:
            _write_element(f, child, indent + 1)
        f.write(f'{"  " * indent}</{tag}>\n')


def _esc_attr(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _esc_text(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def main():
    mxfile = build_all()
    output_path = "docs/architecture/diagrams_single.drawio"
    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        _write_element(f, mxfile, indent=0)
    size = os.path.getsize(output_path)
    print(f"Wrote {output_path} ({size} bytes, single page, 6 sections)")


if __name__ == "__main__":
    main()
