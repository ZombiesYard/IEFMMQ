# Help Flow (Help Trigger -> Qwen -> Highlight)

MVP tutoring path for a learner-initiated help/hint that queries Qwen and responds with a highlight-only overlay. The goal is traceable guidance with strict safety boundaries and deterministic fallback.

## End-to-end flow
1. **Trigger**: Learner presses help/hint (UI or voice). System records `TutorRequest` with `intent="help_request"` and links it to the most recent `Observation` (procedure state, telemetry).
2. **Context assembly** (minimum set below) is bundled into a prompt for Qwen alongside guardrails (no clicks, highlight-only, JSON contract).
3. **LLM call**: Qwen returns a `HelpResponse` JSON payload (contract below). Raw text is rejected.
4. **Validation & mapping**:
   - Enforce schema and allowlists (overlay intent must be `highlight`; target must be in UI map allowlist).
   - Map to `TutorResponse` (`core/types.py`) with `status`=`ok|error|fallback`.
   - Translate `overlay.target` to an `OverlayIntent` via `OverlayPlanner` (UI map -> `element_id`).
5. **Delivery**:
   - Emit `TutorResponse` to the learner (message + optional explanations).
   - Send overlay intent to simulator (e.g., DCS overlay sender) expecting an ACK when enabled.
6. **Logging**: Append required events (see Logging) with `related_id` links to stitch request/response/overlay actions.
7. **Fallback**: If Qwen unavailable/invalid, emit deterministic guidance and (if safe) a highlight chosen from the allowlist; never issue clicks or executable actions.

## Minimum LLM input context
Always include this structured context in the prompt (JSON or tabular in the system prompt):
- `vars`: Latest derived telemetry booleans/numerics from `vars.*` (e.g., `battery_on`, `apu_ready`) sourced from `packs/.../telemetry_map.yaml`.
- `recent_deltas`: Last N (e.g., 3-5) state changes with `name`, `old`, `new`, `timestamp`, `source` (telemetry vs. gating) to show motion.
- `inferred_step`: Best guess of learner position (active/pending step id, phase, short_explanation, gating status/why blocked).
- `pack_step_summaries`: Current + next step snippets from the pack (`official_step`, `short_explanation`, `tutor_prompts`) for grounding.
- `overlay_target_allowlist`: Abstract UI targets permitted for highlight (from UI map; include `target` + `dcs_id`).
- `rag_snippets` (optional): Top-N retrieval snippets with `text` + `source` (doc path/page) when available.
- Session metadata: `session_id`, `pack_id`, `contract_version`, `llm_prompt_version`.

## LLM output contract (HelpResponse -> TutorResponse)
LLM must return JSON matching this contract:
```json
{
  "status": "ok | fallback | error",
  "message": "short learner-facing text",
  "step_id": "S01",
  "overlay": { "intent": "highlight", "target": "battery_switch" },
  "explanations": ["why this step is needed"],
  "metadata": { "model": "qwen", "prompt_id": "help_v1" }
}
```
- `status`: `ok` for valid help, `fallback` when LLM is unsure but suggests safe default, `error` to signal drop to deterministic path.
- `message`: 1-2 sentences; no commands to click/run code.
- `step_id`: Echo inferred step for traceability; used to link to pack summaries.
- `overlay`: Optional; only `intent="highlight"` allowed. `target` must exist in `overlay_target_allowlist`. No other intents/actions accepted.
- `explanations`: Optional bullets (max 3) to justify guidance.
- `metadata`: Carries model name, version, safety flags, token usage.

**Mapping to `TutorResponse` (`core/types.py`):**
- `status` -> `TutorResponse.status`
- `message` -> `TutorResponse.message`
- `explanations` -> `TutorResponse.explanations`
- `metadata` -> merge into `TutorResponse.metadata` with `{model: "qwen", llm: true, prompt_id, safety: {...}}`
- `overlay` -> convert with `OverlayPlanner.plan(target, intent="highlight")`, then `OverlayIntent.to_action()` appended to `TutorResponse.actions`.

## Fallback behavior (LLM unavailable/invalid)
- **Triggers**: LLM timeout, transport error, non-JSON reply, schema/allowlist violation, empty `message`, or missing/unknown `overlay.target`.
- **Response**: Emit `TutorResponse(status="fallback")` with deterministic text pulled from `pack_step_summaries` for `inferred_step` (or generic "Review current checklist step" if unknown). Include no overlay when target cannot be validated; otherwise highlight the pack-defined target for the inferred step.
- **No unsafe actions**: Never issue clicks/exec; only text and safe highlight from allowlist. Defer to user action for any execution.
- **Record**: Mark `metadata.safety.fallback_reason` and emit an `overlay_failed` event when an overlay was skipped due to validation.

## Logging (required kinds)
- `observation`: Raw or derived simulator state (already produced by env adapters).
- `tutor_request`: Captures help trigger intent, session id, and links to the `Observation` via `related_id`.
- `tutor_response`: Stores `TutorResponse` (status/message/actions/metadata) linked to the request via `in_reply_to` and `related_id`.
- `overlay_requested` / `overlay_applied` / `overlay_failed`: Emitted by overlay sender/ack receiver with `related_id` pointing to the `TutorResponse` that contained the overlay action.
- Optional `ui.highlight_requested` / `ui.highlight_ack` when UI layer replays overlay intents; keep consistent timestamps for metrics.

Each event uses contract version `v1`, ISO timestamps, and `session_id` for stitching.

## Safety boundary
- Highlight-only: allowed intents are `highlight` (and `clear` if UI needs to remove prior highlight). No automatic clicks, keypresses, or simulator commands.
- Target allowlist enforced via UI map; reject any overlay targets not present.
- Deterministic fallback must prefer text-only over risky actions when validation fails.
- Tutor messaging must avoid hallucinated procedures, external URLs, or instructions to disable safety systems.
