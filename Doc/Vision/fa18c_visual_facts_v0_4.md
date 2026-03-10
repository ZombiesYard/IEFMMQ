# F/A-18C v0.4 Visual Facts

This note documents the v0.4 visual-fact extraction layer introduced for the
composite left-stack panel flow.

## Scope

The extractor does not produce tutor answers. It converts synchronized
`pre_trigger_frame` plus `trigger_frame` inputs into structured facts that can be
recorded, replayed, audited, and reused by deterministic step inference.

Current fact ids:

- `fcs_page_visible`
- `bit_page_visible`
- `fcs_reset_seen`
- `fcs_bit_interaction_seen`
- `fcs_bit_result_visible`
- `takeoff_trim_seen`
- `ins_alignment_page_visible`
- `ins_go`

Each fact carries:

- `source_frame_id`
- `confidence`
- `expires_after_ms`
- `evidence_note`

## Sticky Rules

- Non-sticky facts: `fcs_page_visible`, `bit_page_visible`, `ins_alignment_page_visible`
- Sticky facts: `fcs_reset_seen`, `fcs_bit_interaction_seen`, `fcs_bit_result_visible`, `takeoff_trim_seen`, `ins_go`
- Sticky `seen` evidence is retained until TTL expiry. Later `not_seen` or `uncertain`
  observations do not clear an unexpired sticky `seen`.

## Step Bindings

- `S08`: `fcs_page_visible` and `bit_page_visible`
- `S15`: `fcs_reset_seen` and `fcs_page_visible`
- `S17`: `takeoff_trim_seen`
- `S18`: `fcs_bit_interaction_seen` and `fcs_bit_result_visible`

`ins_alignment_page_visible` and `ins_go` are kept for prompt support and audit
trails, but they do not directly complete a step in deterministic inference.

## Degradation

- If no synchronized frame is available, the help flow stays text-only.
- If the multimodal request fails, the help flow stays text-only and records
  `vision_fact_status=extractor_failed`.
- `overlay.evidence` remains limited to `var/gate/rag/delta`; visual facts are
  prompt/inference inputs and audit records only in this issue.
