# ST-001 Help Flow Spec Freeze (v0.3)

This document defines the minimal executable v0.3 loop: `Help trigger -> Qwen -> Overlay highlight`.  
Its purpose is to freeze cross-module behavior and data boundaries for aligned implementation and testing.

## 1. Scope and Principles

- Applicable runtime modes:
  - `mock_env`
  - `DCS live adapter`
  - `DCS replay adapter`
- Core remains simulator-agnostic and must not depend on DCS/UDP/Lua details.
- LLM is optional; the system must still run without a model (deterministic fallback).
- Strict safety boundary: only overlay actions are allowed (`highlight/clear/pulse`); no auto-click or action injection.

## 2. Sequence (DoD Main Flow)

1. The system detects the learner is stuck/stalled/faulted (via gating, stall detector, or manual trigger).
2. The learner triggers Help (stdin/UDP/button are all valid).
3. The orchestrator builds minimal context (see Section 3).
4. If ModelPort is available, call Qwen and request `HelpResponse` (JSON-only).
5. Run `json_extract -> minimal repair -> schema validate`.
6. Map `HelpResponse` to internal `TutorResponse` (see Section 4).
7. If overlay targets are valid and in allowlist, send overlay.
8. Record the full event chain (see Section 6).
9. If any step fails, enter deterministic fallback (see Section 5).

## 3. Help Input Context (Minimal Set)

For `TutorRequest.intent=help`, context passed to model/rule engine may only include:

- `vars`
  - Current key state variables (stable keys projected from Observation/Telemetry).
  - Exclude unrelated large fields and raw noisy payload.
- `recent_deltas`
  - Key variable changes in the latest N seconds, including bios -> ui projection.
  - Used to represent what the learner recently did or did not do.
- `candidate_steps`
  - Candidate `step_id` list provided by the procedure pack.
  - LLM must not create step IDs outside the pack.
- `deterministic_step_hint`
  - Locally inferred fallback hint `{inferred_step_id, missing_conditions, recent_ui_targets}` from pack steps + vars + recent ui targets.
  - Used both as prompt guidance and deterministic fallback when model output is unavailable/invalid.
- `overlay_target_allowlist`
  - Derived only from `ui_map.yaml` and optional `pack.ui_targets`.
  - Any target not in allowlist must be dropped.
- `rag_topk` (optional)
  - Include short snippets only when a Knowledge index exists.
  - Missing RAG must not block the main flow.

## 4. Output Contract: HelpResponse -> TutorResponse

### 4.1 HelpResponse (LLM Structured Output, JSON-only)

Current v0.3 schema fields (implemented in `core/llm_schema.py`):

- `diagnosis` (object, required)
  - `step_id`: must be in `candidate_steps`
  - `error_category`: must be in taxonomy categories/trial flags
- `next` (object, required)
  - `step_id`: must be in `candidate_steps`
- `overlay` (object, required)
  - `targets`: non-empty unique array, each target must be in `overlay_target_allowlist`
- `explanations` (array, required)
  - non-empty strings for learner-facing/internal guidance
- `confidence` (number, required)
  - range `[0.0, 1.0]`

Notes:

- This schema is runtime-generated (not a static JSON artifact) because enum sets are injected from `pack.yaml`, `ui_map.yaml`, and `taxonomy.yaml`.
- Validation pipeline remains `json_extract -> minimal repair -> schema validate`.

### 4.2 Mapping to TutorResponse (`core/types.py`)

- `TutorResponse.status`:
  - `ok`: HelpResponse is valid and executable
  - `error`: model unavailable, invalid output, validation failure, or policy rejection
- `TutorResponse.message`:
  - Prefer first entry of `HelpResponse.explanations`
  - Use fallback text when explanations are unavailable
- `TutorResponse.explanations` <- `HelpResponse.explanations`
- `TutorResponse.actions`:
  - Only overlay actions are allowed (`highlight/clear/pulse`)
  - Mapping is done by `adapters/response_mapping.py` and must align with `OverlayIntent.to_action()`
  - Deduplicate `overlay.targets` and keep at most one primary highlight target by default
  - Unknown/unmappable targets are dropped and recorded under `TutorResponse.metadata.rejected_targets`
  - `click/execute` actions are forbidden
  - Runtime execution is enforced by `adapters/action_executor.py`: allowlist check, `max_targets=1` default, and system-controlled TTL/pulse (LLM can only choose target)
- `TutorResponse.metadata`:
  - Record `provider=qwen|fallback`
  - Record validation/repair result plus `diagnosis`, `next`, and `confidence`
  - Record `rejected_actions` from policy stage (if any)

## 5. Failure and Degradation (Deterministic Fallback)

Trigger conditions (any one):

- Qwen unreachable/timeout/auth failure
- Non-JSON output or output cannot be minimally repaired
- JSON schema validation failure (including enum/required fields)
- `diagnosis.step_id`, `next.step_id`, or any `overlay.targets[*]` not in candidate set/allowlist
- Output contains unsafe actions (click/auto-execution)

Fallback behavior (deterministic):

- Produce `TutorResponse.status=error` with `metadata.provider=fallback`
- Return a safe deterministic text hint (for example: "You are likely stuck at S03; please satisfy vars.apu_ready==true and retry Help.")
- By default, do not send overlay; only send a single safe highlight if local rule can prove a unique safe target
- Log full failure reason without sensitive data

## 6. Event Logging Requirements (Per Help Cycle)

At least the following `Event.kind` entries are required:

- `observation` (or equivalent telemetry event envelope)
- `tutor_request` (`intent=help`)
- `tutor_response` (`status=ok|error`, `provider=qwen|fallback`)
- `overlay_requested` (if overlay was requested)
- `overlay_applied` (on successful execution)
- `overlay_failed` (on failure/timeout/rejection)

Additional constraints:

- Events must be replayable to reproduce step trajectory and help decision chain.
- Use existing version field mechanism (currently `Event.version`).
- Never log API keys; only prompt hash/summary is allowed, not full prompt text.

## 7. Safety Boundary (Hard Constraint)

- Allowed:
  - overlay `highlight|clear|pulse`
- Forbidden:
  - any auto-click, auto-execution, or control injection (including `performClickableAction`)
- Enforced order:
  - validate allowlist first, then send overlay
  - if validation fails, reject execution and log `rejected_actions`

## 8. DoD (Definition of Done)

- Help main sequence, minimal input set, output mapping, failure fallback, event types, and safety boundary are all frozen.
- This document can be used directly as implementation and testing baseline.
- v0.1/v0.2 compatibility is preserved: no change to existing required contract fields.

