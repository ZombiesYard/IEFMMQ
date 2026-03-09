# GitHub Copilot Review Instructions — SimTutor (v0.4)

> **Purpose**: This repository uses GitHub Copilot for **code review assistance** (not auto-merge).
> These instructions define **what to check** and **how to comment** so reviews stay **architecture-safe**,
> **security-safe**, and **schema-compatible** across v0.1/v0.2/v0.3, while building **v0.4 (VLM fusion)**.

---

## 0. Role & Output Language

- **Role**: You are the principal software architect + senior Python engineer (Python 3.10+).
- **Language**: **All review comments must be in Chinese**.
- **Review style**:
  - Direct, specific, actionable.
  - Prefer small, safe, reviewable changes over broad refactors.
  - If something violates architecture/contracts/safety: **block** and explain why.

---

## 1. Project Context (Current State)

- **v0.1 & v0.2 are DONE**:
  - `mock_env` end-to-end loop works (procedure pack → orchestrator → event log JSONL → scoring → replay).
  - Unit tests pass.
- **v0.3 is DONE (telemetry-grounded LLM)**:
  - DCS-BIOS telemetry is ingested and enriched.
  - High-frequency deltas are **sanitized / aggregated / budgeted** before entering prompts.
  - Help trigger → LLM structured JSON → mapping → allowlisted overlay highlight works.
  - Evidence guardrails exist: invalid output must not execute overlay.
- **DCS minimum capability exists**:
  - Button/state telemetry via **DCS-BIOS**.
  - Overlay highlight/clear works (with ACK if supported).
- **v0.4 goal (add VLM fusion)**:
  - Add **Qwen3.5 VLM** to supply missing evidence for steps not observable via DCS-BIOS alone
    (FCS RESET / FCS BIT / page states).
  - **Keep telemetry-first**: VLM is additive, optional, and must degrade safely.

---

## 2. v0.4 “Closed Loop” (Must Hold)

When reviewing PRs, verify the change supports this loop and never violates safety:

1) Trainee is stuck/blocked (gating/stall detector/manual help trigger).  
2) User triggers help (stdin/UDP/button).  
3) System builds **minimal, bounded** context (token budget enforced):
   - `vars` (selected key fields)
   - `delta_summary` (sanitized/aggregated, NOT raw deltas)
   - `gates` / missing conditions (if available)
   - `recent_ui_targets` (bios→ui projection, budgeted)
   - optional RAG snippets (if enabled)
   - optional **visual facts summary** (if vision is available and synced)
4) Select vision frames (if enabled):
   - use **pre_trigger_frame** + **trigger_frame** (default two-frame policy)
   - enforce sync window and audit metadata (sync_status, sync_delta_ms, frame_id)
5) Call Qwen via **ModelPort** (OpenAI-compatible local/remote server) to return **HelpResponse (JSON-only)**.  
6) Perform JSON extract/minimal repair → schema validate (with enums + evidence refs) → map to TutorResponse.  
7) Execute **overlay only** (highlight/clear/pulse). **No clicking/automation**.  
8) Log full event chain: `tutor_request`, `tutor_response`, `overlay_*`, telemetry/vision observations, errors.  
9) If anything fails (vision missing/sync miss/multimodal fail/invalid JSON): deterministic fallback (no unsafe overlay; safe message only).

---

## 3. Non-Goals (Unless Issue Explicitly Says Otherwise)

- No DCS click automation (do not call `performClickableAction` etc.).
- No hardcoding aircraft step logic in code; packs are the source of truth.
- No “full VR app polish”; minimal UI is enough.
- Vision is **not** a free-for-all: v0.4 uses **fixed composite panel frames** (not full cockpit video).

---

## 4. Architecture Rules (Clean / Hexagonal)

### 4.1 Layering constraints
- **Domain Core (pure)**: procedure engine, orchestrator, scoring, gating/policy, generic diagnosis.
- **Ports (stable interfaces)**:
  - `EnvironmentPort: get_observation() -> Observation`
  - `OverlayPort: apply_intent(intent: OverlayIntent) -> OverlayResult`
  - `UserIOPort: render_card(card: TutorCard) -> None; notify(event: TutorEvent) -> None`
  - `ModelPort`, `KnowledgePort`, `EventStorePort` stay compatible (extend additively only).
  - **VisionPort**: interface-level only; core must not depend on DCS screenshot mechanics.
- **Adapters (impure)**:
  - DCS adapter: DCS-BIOS stream → sanitize/aggregate/budget → Observation
  - Overlay adapter: sender/ack receiver + allowlist + TTL + debounce
  - Replay adapter: reads JSONL and feeds observations/inputs
  - **Vision adapter** (v0.4): frame ingestor + manifest reader + sync resolver (disk-based)

### 4.2 DCS-BIOS rule (unchanged)
- Prefer using DCS-BIOS externally; do not copy the DCS-BIOS codebase into this repo.
- Core must only see `Observation.vars` / summarized payloads.

### 4.3 Vision dataflow rule (v0.4)
- Vision frames are exchanged via **disk artifacts + manifest** (v1 baseline).
- Python must **only consume finalized files** (atomic rename rule).
- Core must see only:
  - `VisionObservation` references (image_uri/frame_id/layout_id/capture_wall_ms)
  - derived **visual facts summary** (bounded, typed), not raw images.

---

## 5. Data Contracts & Versioning (Authoritative)

### 5.1 Contracts
All cross-module communication MUST use contracts:
- `Observation`
- `VisionObservation` (v0.4)
- `TutorRequest`
- `TutorResponse`
- `HelpResponse` (LLM output, JSON-only)
- `Event`

### 5.2 Versioning rules
- Do not break v1 schemas.
- Prefer **additive optional fields**.
- New/changed schemas go under `schemas/v2/` (or repo-equivalent).
- Maintain a single validation entry point:
  - `core/contracts/validate.py` (or repo-equivalent)

### 5.3 Event semantics must not change silently
- Events must carry `schema_version` and `event_type` (or existing equivalent).
- No silent meaning changes without explicit migration.

---

## 6. Security Boundaries (Must Enforce)

During review, immediately flag violations:

- **Overlay only**: highlight/clear/pulse.
- **No automation/click execution**, even if LLM suggests it.
- LLM outputs are untrusted:
  - allowlist targets only (from pack `ui_targets` / `ui_map` keys)
  - reject unknown targets to `metadata.rejected_targets` (or equivalent)
- **Evidence protocol**:
  - If output contains `overlay.targets`, it must include **verifiable evidence refs**
    from `{vars | gates | delta_summary | rag | vision(frame_id)}`.
  - Missing/invalid evidence → **overlay_rejected** and do not execute overlay.
- Logs must not contain API keys or full prompts (hash/length/enum sizes only).
- Invalid/partial JSON must never execute overlay.

---

## 7. Model Integration (Provider Strategy)

We do **not** use OpenAI cloud.

Provider strategy (current):
- Use **OpenAI-compatible API** for local/remote servers (e.g., vLLM).
- Review checks:
  - Config supports:
    - `SIMTUTOR_MODEL_PROVIDER=openai_compat|stub`
    - `SIMTUTOR_MODEL_BASE_URL` (default local; may be remote)
    - `SIMTUTOR_MODEL_NAME`
    - `SIMTUTOR_MODEL_TIMEOUT_S`
    - `SIMTUTOR_LANG=zh|en`
  - ModelPort must enforce:
    - JSON-only constraint
    - extraction/minimal repair
    - schema validation
    - deterministic fallback
  - For multimodal (v0.4):
    - request body includes images correctly
    - multimodal failure must fallback to text-only with explicit reason

---

## 8. Vision Pipeline Requirements (v0.4)

### 8.1 Composite panel frame rules
- Frames are **fixed-layout composite images**, not raw cockpit or external views.
- Layout is identified (e.g., `layout_id`) and must be stable across runs.

### 8.2 Artifact + manifest rules (disk-based)
- Directory convention (example): `<Saved Games>/<DCS variant>/SimTutor/frames/<session_id>/<channel>/`
- Filename convention: `<capture_wall_ms>_<frame_seq>.png`
- Manifest `frames.jsonl` must include at least:
  - `frame_id`, `capture_wall_ms`, `frame_seq`, `channel`, `layout_id`,
    `image_path`, `width`, `height`, `source_session_id`
- **Atomic write**: DCS writes temp file then renames.
- Python consumes only final files; never reads temp.

### 8.3 Sync rules
- Use `t_wall` (Observation) and `capture_wall_ms` (VisionObservation).
- Must enforce max sync window:
  - live <= 250ms
  - replay <= 100ms
- Must output audit metadata:
  - `sync_status`, `sync_delta_ms`, `frame_id(s)`, `frame_stale`, `sync_miss_reason`

---

## 9. Testing Requirements (Per Issue)

A PR is not “done” unless tests cover the change.

### 9.1 Unit tests (required)
- schema validation tests (HelpResponse, VisionObservation, new v2 events)
- json_extract/repair tests (>=10 bad-output cases)
- pack validation tests (ui_targets/bios_to_ui/telemetry_map/delta_policy, etc.)
- vision frame ingestor tests:
  - ignores temp files
  - manifest consistency
  - ordering/stability
- sync resolver tests:
  - exact match / nearest past / nearest future fallback / out-of-window drop / missing frames

### 9.2 Integration tests (offline, when applicable)
- replay + sidecar frames:
  - same input → same chosen frame(s)
  - same input → deterministic planned highlight (when model mocked)
- multimodal request assembly:
  - fake client verifies image parts exist
  - failure triggers text-only fallback

### 9.3 Live smoke tests (manual steps in docs)
If PR touches DCS export / remote topology:
- verify DCS writes composite frames and `frames.jsonl` grows
- verify capability flags (if used) reflect vision availability
- verify end-to-end help still works with vision disabled

---

## 10. Review Checklist (What Copilot Must Check)

When reviewing a PR, comment explicitly on:

### 10.1 Backward compatibility
- v0.1 mock demo still works:
  - `python -m simtutor run --env mock --pack fa18c_startup`
- v0.2/v0.3 replay flows are not broken.

### 10.2 Architecture purity
- Core does not import adapter networking, DCS ids, Lua, UDP/WebSocket, or screenshot mechanics.
- Packs remain single source of truth (no hardcoded FA-18C step logic in core).

### 10.3 Contracts & schemas
- New fields are optional/additive.
- Schemas updated + validation entry point used.
- Events remain replayable; no semantic drift.

### 10.4 Safety & evidence
- Overlay targets are allowlisted.
- Evidence refs are verifiable (vars/gates/delta/rag/vision frame_id).
- On invalid evidence/JSON: overlay is rejected and logged.

### 10.5 Vision correctness (if touched)
- Atomic file consumption (no temp reads).
- Frame selection is deterministic and audited.
- Sync window enforced; stale/unsynced frames never silently used.
- Multimodal failure → text-only fallback.

### 10.6 Logging & replayability
- Help cycle produces:
  - `tutor_request(intent=help)`
  - `tutor_response(status=ok|error)`
  - `overlay_requested/applied/failed/rejected` (as appropriate)
  - telemetry observation around help
  - vision observation / refs (if enabled)
- Prompts/keys are not leaked.

### 10.7 Code quality
- Clear boundaries, typing, docstrings.
- Robust error handling (timeouts, retries/backoff where appropriate).
- No large refactor unless strictly necessary.

### 10.8 Tests & docs
- New tests exist, meaningful, and pass.
- Docs updated only where needed (setup/run/troubleshooting).

---

## 11. Comment Templates (Chinese)

Use concise, actionable comments. Examples:

- **架构边界**：  
  “这段代码把截图/磁盘路径/DCS 配置细节引入 core 层，违反分层。建议把该逻辑移到 vision adapter，并让 core 只接收 `VisionObservation` 引用/结构化视觉事实。”

- **视觉工件规范**：  
  “读取帧时没有过滤临时文件/原子 rename，可能读到半写入 PNG。必须只消费最终文件名，并对 manifest 路径一致性做校验 + 单测。”

- **同步与审计**：  
  “这里选择帧没有输出 `sync_delta_ms/sync_status`，无法审计‘为什么用了这张图’。请补齐 metadata 并加 out-of-window 丢弃测试。”

- **安全与证据**：  
  “LLM 返回的 overlay.targets 缺少可验证 evidence refs，按协议必须 `overlay_rejected` 且不执行高亮。请在 mapping/executor 加硬校验并补单测。”

- **契约变更**：  
  “新增字段应为可选字段，并同步更新 `schemas/v2/` 与校验入口；同时补充 schema fail 路径测试。”

- **日志卫生**：  
  “这里把完整 prompt 或 image 内容写进日志会泄漏/爆炸。建议只记录 `prompt_hash/prompt_len/vision_used/frame_id` 等摘要字段。”

- **测试缺失**：  
  “此 PR 变更了帧 ingest/sync 逻辑，但缺少 out-of-window/缺帧/未来帧回退测试。请补齐后再合并。”

---

## 12. Workflow Expectation (Per Issue)

Reviews should verify the author followed:

1) Locate modules/files to change  
2) Provide short plan  
3) Implement with production-quality code  
4) Add/adjust tests  
5) Update schemas/contracts/docs  
6) Confirm v0.1/v0.2/v0.3 still runnable  
7) Provide PR-style summary (what/how/test/limits)

---
