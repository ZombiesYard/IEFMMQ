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
7. If overlay intent is valid and target is in allowlist, send overlay.
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
- `overlay_target_allowlist`
  - Derived only from `ui_map.yaml` and optional `pack.ui_targets`.
  - Any target not in allowlist must be dropped.
- `rag_topk` (optional)
  - Include short snippets only when a Knowledge index exists.
  - Missing RAG must not block the main flow.

## 4. Output Contract: HelpResponse -> TutorResponse

### 4.1 HelpResponse (LLM Structured Output, JSON-only)

Recommended minimal fields (to be formalized in schema later):

- `status`: `ok | needs_clarification | unsafe | error`
- `message`: Short learner-facing guidance text
- `next_step_id`: Candidate step (must be in `candidate_steps`)
- `overlay`: Optional object
  - `intent`: `highlight | clear`
  - `target`: must be in `overlay_target_allowlist`
- `rationale`: Optional internal explanation
- `rejected_actions`: Optional list of unsafe actions rejected by system

### 4.2 Mapping to TutorResponse (`core/types.py`)

- `TutorResponse.status`:
  - `ok`: HelpResponse is valid and executable
  - `error`: model unavailable, invalid output, validation failure, or policy rejection
- `TutorResponse.message` <- `HelpResponse.message` (or fallback message)
- `TutorResponse.actions`:
  - Only overlay actions are allowed (`highlight/clear`)
  - `click/execute` actions are forbidden
- `TutorResponse.metadata`:
  - Record `provider=qwen|fallback`
  - Record validation/repair result and `rejected_actions` (if any)

## 5. Failure and Degradation (Deterministic Fallback)

Trigger conditions (any one):

- Qwen unreachable/timeout/auth failure
- Non-JSON output or output cannot be minimally repaired
- JSON schema validation failure (including enum/required fields)
- `next_step_id` or `overlay.target` not in candidate set/allowlist
- Output contains unsafe actions (click/auto-execution)

Fallback behavior (deterministic):

- Produce `TutorResponse.status=error` with `metadata.provider=fallback`
- Return a safe text hint (for example: "Please verify current step preconditions and retry Help.")
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

