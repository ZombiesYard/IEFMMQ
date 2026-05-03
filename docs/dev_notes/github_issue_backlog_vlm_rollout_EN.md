# GitHub Issue Backlog for SimTutor VLM Rollout

This file collects the English issue drafts that should be created on GitHub for the current SimTutor VLM rollout work.

Repository:

- `https://github.com/ZombiesYard/IEFMMQ`
- current branch in local worktree: `vlm-fineTune-Gemma`

Important note:

- The local `gh` CLI is installed, but the current GitHub token is invalid.
- As a result, the issue texts are prepared here first and should be pushed to GitHub after re-authentication.

---

## Issue 1

### Title

Align the runtime VLM contract with the Run-003 / Run-005 13-fact fine-tuned adapter

### Body

## Summary

The current runtime VLM contract in SimTutor is no longer aligned with the fine-tuned adapters trained on the Run-003 / Run-005 13-fact ontology.

At the moment:

- the fine-tuned Qwen and Gemma adapters were trained on the new 13-fact contract,
- but the runtime visual-fact prompt, schema, and downstream bindings still rely on an older fact set and older response expectations.

This is not only a wording mismatch. It is now a contract mismatch.

## Why this matters

If we directly plug the current fine-tuned adapters into the current runtime VLM path:

- fact IDs will not match cleanly,
- runtime schema expectations may conflict with the training target,
- downstream step inference will still consume old fact names,
- the deployed system may underperform even if the adapter itself is good.

## Scope

Align the runtime visual-fact extraction path with the Run-003 / Run-005 fine-tuning contract.

This includes at least:

1. runtime visual prompt
2. runtime fact response schema
3. runtime `vision_facts.yaml`
4. `core/vision_facts.py`
5. downstream step bindings / step inference / rule wording that still reference old fact names
6. tests covering the aligned contract

## Important design note

The training target does **not** require the model to emit `source_frame_id`, `frame_id`, or `confidence`.

The runtime should therefore reconsider whether:

- `source_frame_id` should remain a model-emitted field, or
- it should instead be attached by the tool/runtime layer as metadata.

## Acceptance criteria

- runtime VLM prompt uses the same 13-fact ontology as the fine-tuned adapters
- runtime response schema does not conflict with the fine-tuned target format
- downstream logic no longer depends on deprecated old fact IDs
- updated tests pass

Related local notes:

- `docs/dev_notes/issue_align_runtime_vlm_contract_with_run005_adapter.md`

---

## Issue 2

### Title

Add stable multi-target overlay tests using fake LLM JSON output

### Body

## Summary

We need a focused test path for multi-target overlay/highlight behavior using fake LLM output.

The specific case to validate is the FCS BIT action that should highlight:

- `fcs_bit_switch`
- `right_mdi_pb5`

in the same response.

## Why this matters

The codebase already contains multiple hints that multi-target overlay is partially supported, but we still need a clean and stable test path that verifies the full chain:

- fake LLM JSON output
- response parsing
- overlay target mapping
- allowlist enforcement
- action execution / overlay sender path

## Scope

Implement tests that use a fake model / fake LLM response returning multiple overlay targets.

The tests should verify that:

1. multiple overlay targets can be represented in JSON,
2. both `fcs_bit_switch` and `right_mdi_pb5` survive parsing,
3. both targets pass allowlist validation,
4. both targets reach the overlay/action executor path correctly,
5. no automatic click / execution behavior is introduced.

## Acceptance criteria

- automated tests exist for multi-target overlay
- `fcs_bit_switch` and `right_mdi_pb5` are both covered
- single-target behavior is not broken
- no unsafe action path is introduced

---

## Issue 3

### Title

Fix startup guidance order when all DDIs and the AMPCD are still dark

### Body

## Summary

When all three displays are still dark, the current help guidance can highlight the AMPCD power path first.

This is not strictly wrong from an isolated action perspective, but it is confusing for the user because the AMPCD will not visibly come alive until the DDI-related power path is already in place.

## Expected behavior

When all displays are dark:

- the system should guide the user through the DDI power-up path first,
- and only then guide the AMPCD-related control.

## Why this matters

Even when an action is technically valid, the help system should avoid guidance that looks visually ineffective or confusing to the user.

The user experience here matters because this is a tutoring system, not just a planner.

## Acceptance criteria

- when all displays are dark, help guidance no longer starts with the AMPCD path
- the recommended action order is more intuitive to the user
- updated tests cover this state

---

## Issue 4

### Title

Investigate why the bleed air knob step is being skipped

### Body

## Summary

The procedure appears to skip the bleed air knob step even though this step exists in the reference procedure/manual.

## Questions to answer

1. Is the step missing from the current pack definition?
2. Is it present but being inferred away by gating or deterministic step inference?
3. Is it being treated as already satisfied due to telemetry assumptions?
4. Is there a mismatch between the procedure model and the live/replay observability model?

## Scope

Trace the step through:

- pack / registry definitions
- gating rules
- deterministic step inference
- live and replay help behavior

## Acceptance criteria

- root cause is identified
- the step is either restored correctly or explicitly documented as intentionally omitted
- tests are updated accordingly

---

## Issue 5

### Title

Clean remote home-directory cache usage and move all model-related caches under `/scratch/yz50`

### Body

## Summary

The remote account home directory has a very small quota and was accidentally used earlier for model/cache-related files.

This caused quota warnings and should be cleaned up in a controlled way.

## Constraints

- do not touch other users' data
- stay within `/scratch/yz50` for future cache/model/temp usage
- do not delete anything without explicit confirmation

## Scope

1. identify model/cache/temp files still living in home
2. identify which ones are safe to migrate or remove
3. ensure current and future Hugging Face / pip / uv / temp caches point to `/scratch/yz50`
4. document the final environment setup

## Acceptance criteria

- remaining problematic cache usage in home is identified
- future runs no longer write meaningful cache/model artifacts to home
- cleanup instructions are documented

---

## Issue 6

### Title

Deploy a fine-tuned VLM on the H100 host and prepare SimTutor runtime rollout

### Body

## Summary

The current runtime still uses an unmodified remote Aliyun-hosted VLM.

The next rollout step is to deploy a fine-tuned VLM on the H100 host and prepare the runtime path to use the fine-tuned adapter instead of the unfine-tuned cloud model.

## Scope

This issue should cover:

1. selecting the rollout candidate backbone
   - likely fine-tuned Qwen first
   - optionally Gemma as comparison / backup
2. preparing the remote model-serving setup on the H100 machine
3. ensuring the runtime can point to the deployed endpoint cleanly
4. documenting required configuration and fallback behavior

## Important note

The current best practical adapter is the Qwen3.5-9B Run-003 + Run-005x2 LoRA.

## Acceptance criteria

- a fine-tuned VLM can be served from the H100 host
- runtime configuration can target that service
- fallback / failure behavior is documented

---

## Issue 7

### Title

Add explicit model-adapter compatibility guardrails for Qwen and Gemma VLM rollout

### Body

## Summary

We need explicit guardrails so that incompatible LoRA adapters are not accidentally paired with the wrong base models.

Example:

- a LoRA fine-tuned on `Qwen/Qwen3.5-9B-Base` must **not** be used on `Qwen3.5-27B`

## Why this matters

This is not just a tuning preference. It is a structural compatibility issue:

- layer counts differ
- hidden dimensions differ
- module shapes differ

So the system should fail clearly instead of allowing a misleading misconfiguration.

## Scope

1. document supported base-model / adapter pairings
2. add validation where appropriate in loading or configuration code
3. surface a clear error when an incompatible pairing is attempted

## Acceptance criteria

- incompatible adapter/base combinations are rejected explicitly
- supported combinations are documented

---

## Issue 8

### Title

Add systematic timing and token accounting for training and runtime generation

### Body

## Summary

We should record timing and token usage more systematically for both training-related generation tasks and runtime generation.

This is lower priority than the rollout and contract-alignment tasks, but still useful for reporting and practical operations.

## Possible scope

- generation latency
- prompt token count
- completion token count
- benchmark/runtime aggregation
- optional training-side reporting where applicable

## Acceptance criteria

- at least the main generation timing and token fields are captured consistently
- the implementation does not leak sensitive prompt contents
- documentation explains where the metrics are recorded

