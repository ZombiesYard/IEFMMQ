# SimTutor Architecture Diagrams — Mermaid Format

Import into draw.io via: **Arrange → Insert → Advanced → Mermaid**

All six diagrams below. Paste each individually into draw.io.

---

## 1. System Overview

```mermaid
graph LR
  subgraph HW["Hardware / Sim"]
    Q3["Quest 3 HMD<br/>(fixed platform)"]
    DCS["DCS F/A-18C<br/>cold-start mission"]
    BIOS["DCS-BIOS<br/>UDP 239.255.50.10:5010"]
    PACK["Pack Config<br/>packs/fa18c_startup/"]
    VFACTS["Vision Facts<br/>13 facts, sticky"]
  end

  subgraph ST["SimTutor Process (live_dcs.py)"]
    L1["LiveDcsTutorLoop<br/>live_dcs.py"]
    L2["build_help_prompt_result()<br/>adapters/prompting.py"]
    L3["EvidencePacket<br/>core/evidence_packet.py"]
    L4["plan_harness_action()<br/>core/harness_validation.py"]
    L5["OverlayActionExecutor<br/>adapters/action_executor.py"]
    L6["experiment-export<br/>core/experiment_export.py"]
  end

  subgraph RO["Remote & Output"]
    R1["OpenAICompatModel<br/>Qwen3-8B — text LLM"]
    R2["VisionFactExtractor<br/>Qwen3.5-27B — VLM"]
    R3["vLLM Server<br/>Remote GPU"]
    R4["simtutor-base<br/>simtutor-vision (LoRA)"]
    R5["JSONL Event Log<br/>logs/*.jsonl"]
    R6["CSV Artifacts<br/>trial_summary, step_coding..."]
    R7["Analysis Output<br/>study_summary, figures"]
  end

  OVC["SimTutorConfig.lua<br/>DCS overlay config"]

  Q3 --> DCS --> BIOS
  BIOS -->|"telemetry"| L1
  DCS -->|"VR mirror"| L1
  L1 --> L2 --> L3 --> L4 --> L5
  L5 -->|"write config"| OVC
  L3 -->|"evidence for prompt"| R1
  L3 -->|"VLM facts"| R2
  R1 -->|"SSH tunnel"| R3
  R2 -->|"SSH tunnel"| R4
  L5 --> L6
  L6 -->|"freeze trial"| R5
  R5 --> R6 --> R7
  PACK -->|"steps/gates"| L3
```

---

## 2. Live Help Cycle

```mermaid
graph TD
  S1["1. Participant presses X1<br/>adapters/windows_global_help_trigger.py"]
  S2["2. Telemetry snapshot<br/>adapters/dcs_bios/receiver.py"]
  S3["3. Enrich derived vars<br/>adapters/telemetry_pipeline.py"]
  S4["4. Trigger vision capture<br/>adapters/vision_capture_trigger.py"]
  S5["5. Deterministic inference<br/>adapters/step_inference.py"]
  S6["6. Evaluate pack gates<br/>adapters/pack_gates.py"]
  S7["7. Build EvidencePacket<br/>core/evidence_packet.py"]
  S8["8. VLM extraction (if visual step)<br/>adapters/vision_fact_extractor.py"]
  S9["9. RAG knowledge retrieval<br/>adapters/knowledge_local.py"]
  S10["10. Build LLM prompt<br/>adapters/prompting.py"]
  S11["11. Call text LLM<br/>adapters/openai_compat_model.py"]
  S12["12. Map response<br/>adapters/response_mapping.py"]
  S13["13. Harness validate & plan<br/>core/harness_validation.py"]
  S14["14. Dispatch overlay<br/>adapters/action_executor.py"]
  S15["15. Log to JSONL<br/>core/event_store.py"]

  S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10 --> S11 --> S12 --> S13 --> S14 --> S15
```

---

## 3. VLM / LLM Separation

```mermaid
graph TB
  REQ["Help Request Received (X1)"]
  GATE{"requires_visual_<br/>confirmation?<br/>(from pack step config)"}

  subgraph VLM_PATH["VLM Path — visual-priority steps only"]
    VE["VisionFactExtractor<br/>adapters/vision_fact_extractor.py"]
    VF["Composite Panel Screenshots<br/>pre_trigger_frame + trigger_frame"]
    VO["13 VisionFact Objects<br/>seen / not_seen / uncertain"]
    VS["Vision Fact Summary<br/>→ EvidencePacket.vision_evidence"]
  end

  subgraph LLM_PATH["LLM Path — every help request"]
    LE["OpenAICompatModel<br/>adapters/openai_compat_model.py"]
    LP["Text Prompt<br/>Evidence + Candidates + RAG + Specs"]
    LO["HarnessDecision JSON<br/>step, diagnosis, targets, evidence_refs"]
    LR["TutorResponse<br/>→ overlay + text guidance to cockpit"]
  end

  VM["simtutor-vision (LoRA)<br/>Qwen3.5-27B on remote GPU"]
  LM["simtutor-base (text)<br/>Qwen3-8B on remote GPU"]

  REQ --> GATE
  GATE -->|"Yes (S08, S15, S18, S19)"| VE
  GATE -->|"No"| LE
  VE --> VF --> VO --> VS
  VS --> LM
  LE --> LP --> LO --> LR
  LR --> LM
  VS -.->|"facts injected into evidence"| LP
```

---

## 4. Harness Engineering Objects

```mermaid
graph TB
  subgraph EVIDENCE["Evidence Sources (5 types)"]
    EV1["TelemetryEvidence<br/>source_status, confidence"]
    EV2["TelemetryWindowDigest<br/>windowed var analysis"]
    EV3["VisionEvidence<br/>13 facts, anchors"]
    EV4["GateEvidence<br/>blocked/satisfied gates"]
    EV5["RecentActionEvidence<br/>target_ids, deltas"]
  end

  INF["infer_step_id()<br/>adapters/step_inference.py"]
  DET["DeterministicCandidate<br/>step_id, missing_conditions"]

  BEP["build_evidence_packet()<br/>core/evidence_packet.py"]
  EP["EvidencePacket<br/>+ conflicts tuple (4 types)"]

  BSC["build_step_candidates()<br/>max 8, ordered by confidence"]
  SC["StepCandidate[]<br/>source, confidence, evidence_refs"]

  SHS["StepHarnessSpec[]<br/>core/step_harness.py"]
  LDC["HarnessDecision (LLM output)<br/>JSON Schema contract"]

  SAP["State-Action Planner<br/>S08/S09/S12/S18/S19 rules"]
  PHA["plan_harness_action()<br/>core/harness_validation.py"]

  VLD["validate_final_evidence_consistency()"]
  HAP["HarnessActionPlan<br/>targets, guidance, text_only, repaired"]
  ECR["EvidenceConsistencyResult<br/>accepted, rejected, repair"]

  OUT["→ Public TutorResponse<br/>safe for cockpit display"]

  EV1 --> BEP
  EV2 --> BEP
  EV3 --> BEP
  EV4 --> BEP
  EV5 --> BEP
  INF --> DET --> EP
  BEP --> EP
  EP --> BSC --> SC
  EP --> LDC
  SC --> LDC
  SHS --> LDC
  LDC --> PHA
  SAP --> PHA
  PHA --> HAP
  PHA --> VLD --> ECR
  HAP --> OUT
  ECR --> OUT
```

---

## 5. Experiment Export and Analysis Pipeline

```mermaid
graph LR
  JLOG["Raw JSONL Event Log<br/>one .jsonl per trial"]
  CLI["CLI Metadata<br/>--participant-id --condition<br/>--study-id --model-name..."]
  PKI["Pack & Taxonomy<br/>pack.yaml, taxonomy.yaml<br/>ui_map.yaml, bios_to_ui.yaml"]

  EXP["experiment-export<br/>core/experiment_export.py<br/><br/>Repeat per participant/trial"]

  QG["quality_gate.passed?<br/>must be true"]

  subgraph PER_TRIAL["Per-Trial Outputs"]
    O1["trial_summary.csv"]
    O2["step_coding.csv"]
    O3["help_cycles.csv"]
    O4["action_timeline.csv"]
    O5["quality_gate.json"]
    O6["session.json"]
    O7["raw_events.jsonl (copy)"]
  end

  ANA["experiment-analyze<br/>All trials combined"]

  subgraph AGGREGATE["Aggregate Outputs"]
    A1["study_summary.csv"]
    A2["condition_summary.csv"]
    A3["step_accuracy_by_condition.csv"]
    A4["help_quality_summary.csv"]
    A5["fig_*.png (optional)"]
  end

  JLOG --> EXP
  CLI --> EXP
  PKI -->|"step coding rules"| EXP
  EXP --> O1
  EXP --> O2
  EXP --> O3
  EXP --> O4
  EXP --> O5
  EXP --> O6
  EXP --> O7
  QG --> EXP
  EXP --> ANA
  ANA --> A1
  ANA --> A2
  ANA --> A3
  ANA --> A4
  ANA --> A5
```

---

## 6. Deployment Topology

```mermaid
graph LR
  subgraph WIN["Windows Simulator Host"]
    DCS["DCS World F/A-18C"]
    BIOS["DCS-BIOS<br/>(UDP:5010)"]
    Q3["Quest 3<br/>(Oculus Link)"]
    VS["Vision Sidecar<br/>capture_vision_sidecar.py"]
    LD["SimTutor live-dcs"]
    GUI["GUI Launcher<br/>simtutor_launcher.py"]
    SG["Saved Games\\DCS\\<br/>SimTutorConfig.lua"]
    LOGF["logs/*.jsonl<br/>Runtime event log"]
    ART["artifacts/experiments/<br/>CSV per-trial exports"]
    EX2["Experiment Export"]
  end

  subgraph REMOTE["Remote GPU Server (cloud-247)"]
    VLLM["vLLM Server<br/>(localhost:8000)"]
    SBASE["simtutor-base<br/>Qwen3-8B (text LLM)"]
    SVIS["simtutor-vision<br/>Qwen3.5-27B + LoRA"]
    SSH["SSH Tunnel Endpoint"]
  end

  DCS --> BIOS --> Q3
  DCS -->|"VR mirror"| VS
  BIOS -->|"telemetry"| LD
  VS -->|"frame trigger"| LD
  LD -->|"write config"| SG
  SG --> LOGF --> ART
  LOGF --> EX2

  LD -.->|"SSH: text LLM call"| SBASE
  VS -.->|"SSH: VLM call"| SVIS
  SBASE --> VLLM
  SVIS --> VLLM
  VLLM --> SSH
```

---

## Import Instructions

For each diagram above:

1. Copy the Mermaid code block (including the ```mermaid ... ``` fences)
2. In draw.io: **Arrange → Insert → Advanced → Mermaid**
3. Paste and click **Insert**
4. The diagram renders as native draw.io shapes, fully editable

Alternatively, open the companion `diagrams.drawio` file which contains all six diagrams pre-built on separate tabs and ready to edit.
