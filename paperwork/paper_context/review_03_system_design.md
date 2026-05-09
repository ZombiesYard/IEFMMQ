# Review: 03_system_design.tex -- Strict Academic Reviewer Findings

**Chapter**: `paper/sections/03_system_design.tex` (462 lines, 57 Evidence comments)
**Review date**: 2026-05-09
**Reviewer stance**: Strict. This review treats the chapter as a Methods/System-Design chapter and flags every deviation from the source-of-truth documents.

---

## CRITICAL (3 issues)

### CRIT-1: Unsubstantiated evaluative closing sentence (lines 461--462)

**Location**: Closing paragraph of the chapter, lines 460--462.
**Issue**: The sentence reads: "Rather, the implemented system provides a **significantly more robust** technical baseline on which the planned VR deployment and human-subject evaluation can be built." The phrase "significantly more robust" is a triple violation:

1. **"Robust"** is a loaded evaluative term. No quantitative robustness metric is defined or cited anywhere in the chapter or claims registry. What does "robust" mean -- test coverage? runtime uptime? number of handled edge cases? None of these are measured.
2. **"Significantly"** implies a statistical or systematic comparison. No comparative baseline is provided. The original proposal system was never built, so no measured comparison is possible. The word "significantly" crosses into pseudo-statistical language.
3. **"Technical baseline"** framed as "more robust" is a forward-looking value judgment, not a descriptive system-design statement.

This is the chapter's final rhetorical landing point and carries disproportionate weight.

**Fix**: Replace with a neutral, descriptive sentence that does not evaluate quality. Example: "Rather, the implemented system provides a concrete technical foundation -- with running code, automated tests, and benchmark artifacts -- on which the planned VR deployment and human-subject evaluation can be built." Remove "significantly more robust."

**Cross-reference**: `paper_context/terminology.md` -- no definition of "robust" exists. `paper_context/claims_registry.md` -- C-PROPOSAL-01 and C-PROPOSAL-02 are marked `TODO`; no verified claim supports a "robustness" assertion.

---

### CRIT-2: Chapter identity conflict with paper_outline.md (entire file)

**Location**: Entire chapter; file `03_system_design.tex` and its title `\chapter{System Design}`.
**Issue**: The `paper_context/paper_outline.md` specifies the recommended chapter structure as:

1. Introduction
2. Related Work
3. **System Context and Background**
4. **Methods**
5. Current Technical Results
6. Planned Evaluation
7. Discussion
8. Conclusion

The `CLAUDE.md` workspace file defines a different structure where Chapter 3 is "System Design" (this file) and Chapter 4 is "Methodology." The outline places system-architecture description and method description in two separate chapters (3: Context, 4: Methods). The current chapter merges:

- System context (DCS platform, procedure description, design goals -- outline Chapter 3 material)
- Architecture and methods (ports-and-adapters layers, data flow, telemetry pipeline, procedure engine, event model, VLM pipeline, delta-from-proposal -- outline Chapter 4 material)

The resulting chapter is approximately 4000 words of mixed content whose boundary with the planned `04_methodology.tex` is unclear. If `04_methodology.tex` will cover the VLM adaptation pipeline, dataset curation, and benchmark protocol (as `paper_outline.md` sections 4.4--4.5 specify), then the VLM subsection in this chapter (Section 3.8, lines 349--395) overlaps with that scope.

**Fix**: Resolve the authority conflict between `CLAUDE.md` and `paper_outline.md`. Decide one outline and stick to it. If the `CLAUDE.md` 8-chapter structure (System Design / Methodology / Implementation / Evaluation) is adopted, then `paper_outline.md` must be updated. If `paper_outline.md` is the master, then this chapter should either be renamed to "System Context and Background" and stripped of methods-level detail, or split into two chapters. In either case, the boundary between this chapter and the next (04) must be explicitly documented in the chapter roadmap (lines 10--18) and in the next chapter's introduction.

---

### CRIT-3: Unresolved TODO:VERIFY on a limitation claim (line 346)

**Location**: Line 346, within Section 3.7 "Event Logging and Replay," subsection "Current Limitations" (line 344).
**Issue**: The text reads: "The replay infrastructure has not yet been extended to replay full multimodal sessions with synchronised vision frames." The immediately following line (346) is a `% TODO:VERIFY -- scope of vision-aware replay in replay_eval/fa18c_startup_v04/` comment. This means the limitation statement itself is unverified -- it is not known whether the replay infrastructure has or has not been extended to multimodal sessions. The author is making an assertion about a system limitation while simultaneously flagging that the assertion needs verification.

**Fix**: Before finalizing the draft, verify the actual scope of vision-aware replay by inspecting `replay_eval/fa18c_startup_v04/`. If the limitation is confirmed, remove the TODO:VERIFY comment. If vision-aware replay IS partially supported, rewrite the limitation to be precise about what IS and is NOT supported. In the current draft, either place the TODO:VERIFY on a separate line with a clear resolution plan, or delete the unverified claim and re-verify.

**Cross-reference**: `paper_context/claims_registry.md` -- C-RUNTIME-01 is `likely` status, not `verified`. Unverified limitation claims compound the risk.

---

## MAJOR (7 issues)

### MAJ-1: "Robust" used again before closing sentence (line 409)

**Location**: Line 409, within Section 3.9 "Differences from the Original Proposal."
**Issue**: "building a **robust** telemetry-grounded runtime proved a substantial prerequisite." Same problem as CRIT-1. "Robust" is undefined and unmeasured. This sentence also appears in the delta-from-proposal section, which is meant to be a factual accounting of differences, not an evaluation of the result's quality.

**Fix**: Replace "robust" with a factual description, e.g., "building a telemetry-grounded runtime with delta sanitisation, fallback handling, and automated tests proved a substantial prerequisite."

**Cross-reference**: `paper_context/terminology.md` -- no entry for "robust."

---

### MAJ-2: C-RUNTIME-01 (`likely`) presented as fully established (lines 319--347)

**Location**: Entire Section 3.7 "Event Logging and Replay" (lines 319--347).
**Issue**: The claims registry (C-RUNTIME-01) marks the replay/logging/benchmark infrastructure as `likely`, not `verified`. The usage note in `claims_registry.md` states: "`likely` items may appear in methods or discussion, but tone should be conservative and limitations should be noted." The chapter presents the event model (3.7.1), JSONL store (3.7.2), replay validation (3.7.3), and scoring (3.7.4) as established system facts without hedging. The "Current Limitations" paragraph (lines 344--347) is the only concession, but it carries an unverified TODO marker (see CRIT-3). Furthermore, the claim "These event logs must support offline replay for debugging, automated scoring against a procedure pack, and post-hoc analysis of learner behaviour" (lines 46--48, in G4) is stated as a design goal, but the chapter text implies these capabilities are fully realized.

**Fix**: Add hedging language. For example: "The event logging and replay infrastructure has been implemented at the code level with a JSONL event store, replay validation for step ordering, and a scoring engine. It has been exercised with recorded telemetry traces but has not yet been validated with end-to-end human-in-the-loop sessions." Update C-RUNTIME-01 status to `verified` only if adequate evidence is marshalled, or downgrade the language in the chapter.

**Cross-reference**: `paper_context/claims_registry.md` -- C-RUNTIME-01 is `likely` with risk "medium."

---

### MAJ-3: Summary table presents Gemma-4-31B as equal peer to Qwen3.5-9B (line 451)

**Location**: Summary table, line 451, "Model adaptation" row.
**Issue**: The table row reads: "Model adaptation & Not addressed & LoRA fine-tuning pipeline (Qwen3.5-9B, Gemma-4-31B)." This presents both backbones as co-equal contributions. The claims registry distinguishes:

- C-VLM-04 (Qwen holdout run002): `verified`
- C-VLM-05 (Qwen holdout run004): `verified`
- C-VLM-06 (Gemma): `likely`, with note "whether it becomes the primary experimental line for the thesis needs human decision"

By listing both models in a summary table without confidence qualifiers, the chapter implies both have equivalent evidential status. This matters because the Gemma benchmark results are categorized as `likely` (medium risk), and the claims registry notes that its inclusion as primary evidence requires human decision.

**Fix**: Either (a) add a footnote or parenthetical noting the Qwen results are the primary verified line while Gemma is supplementary, or (b) list only Qwen in the table with a note that Gemma results are reported separately. Do not present `likely` claims with the same confidence as `verified` claims in summary artifacts.

**Cross-reference**: `paper_context/claims_registry.md` -- C-VLM-04 (verified), C-VLM-05 (verified), C-VLM-06 (likely, medium risk).

---

### MAJ-4: "Complete" and "major" contribution claims (lines 385, 416)

**Location**: Lines 385--388 and 416--417.
**Issue**:

- Line 385--388: "This pipeline is a **complete**, self-contained contribution."
- Line 416: "The VLM pipeline evolved into a **complete** adaptation and benchmarking loop that now constitutes a **major** technical contribution."

"Complete" is a strong claim implying no gaps remain. The pipeline artifacts exist (capture -> prelabel -> review -> export -> train -> benchmark), but "complete" should be reserved for the conclusion or discussion chapters where the full pipeline scope is laid out and evaluated. "Major" is an evaluative claim about the contribution's significance that belongs in Discussion or Conclusion, not in a system design chapter. The role of a system design chapter is to describe what was built, not to rank its importance.

**Fix**: Replace with descriptive language: "This pipeline forms a self-contained contribution spanning data capture through benchmark evaluation." For line 416: "The VLM pipeline evolved into an adaptation and benchmarking loop that constitutes a substantial technical contribution." Or move evaluative language to the Discussion chapter.

---

### MAJ-5: Unsupported architectural purity claim (line 82)

**Location**: Line 82, within Section 3.2 "Overall Architecture."
**Issue**: "The core depends on nothing outside itself except standard Python libraries and the abstract interfaces defined in the ports layer." This is a strong, falsifiable claim about implementation purity. The evidence comment (lines 80--81) lists modules in `core/` but does not cite any dependency analysis (e.g., output of `pipdeptree`, `import-linter` config, or a manual audit). Architectural purity claims are common in system-design chapters but should be either:

- Verified by automated tooling (import linter, module dependency graph), or
- Hedged: "The core is designed to depend on nothing outside itself..."

**Fix**: Either provide an evidence path to a dependency audit (e.g., CI lint rule, import-linter output), or soften the claim to design intent: "By design, the core depends only on standard Python libraries and the abstract interfaces defined in the ports layer."

**Cross-reference**: `paper_context/repo_inventory.md` Section 2.1 -- describes `core/` as containing "simulator/model-provider-decoupled core logic" but does not assert zero external dependencies.

---

### MAJ-6: "Safety boundary" introduced without definition or evidence (line 453)

**Location**: Summary table, line 453, "Error handling" row.
**Issue**: The table cell reads: "Structured taxonomy, deterministic fallback, **safety boundary**." The term "safety boundary" appears nowhere else in the chapter. The system design chapter describes action restrictions (line 207--208: "Auto-click and control-injection actions are forbidden by design") and deterministic fallback (lines 263--266), but these are specific mechanisms, not a "boundary." The term "safety boundary" evokes a comprehensive safety architecture (e.g., a safety perimeter with defined ingress/egress points, formal threat model, invariants). No such architecture is described.

If "safety boundary" merely means "certain dangerous actions are prevented," then use the more precise terminology already in the chapter: "action restrictions and deterministic fallback." If it means something broader, it must be defined.

**Fix**: Replace "safety boundary" with the concrete mechanisms described in the chapter: "structured taxonomy, deterministic fallback, and action restrictions." Or, if a safety architecture is intended, define it explicitly earlier in the chapter or in Methodology.

**Cross-reference**: `paper_context/claims_registry.md` -- C-ARCH-03 mentions "safety handling" in the evidence column, but the term "safety boundary" does not appear in any source-of-truth document.

---

### MAJ-7: "Stall-detection heuristic" mentioned but never explained (line 177)

**Location**: Line 177, within Section 3.4 "Data Flow," subsection "Tutor Request."
**Issue**: "When the learner triggers help---via a hotkey, a UDP message, or a **stall-detection heuristic**---a tutor request is created." This is the only mention of stall detection in the entire chapter. No further description, no evidence comment, no definition of what constitutes a "stall," and no reference to the heuristic's implementation. The evidence comment on line 179 references only `core/types.py:TutorRequest`, which defines the data type, not the stall-detection logic. If stall detection is implemented, it deserves at least a sentence of description. If it is planned but not implemented, it should be marked as such.

**Fix**: Either (a) add a brief description of the stall-detection heuristic with evidence reference (e.g., timeout threshold, inactivity detection window), or (b) remove the reference if it is not yet implemented, or (c) mark it as planned with "a stall-detection heuristic (planned)."

---

## MINOR (10 issues)

### MIN-1: "Visual observations" vs defined term "visual facts" (line 31)

**Location**: Line 31--33, within G1 description.
**Issue**: The text reads: "cite evidence from telemetry, documentation, or **visual observations**." The defined paper term per `paper_context/terminology.md` is "visual fact" (for the intermediate representation extracted from screenshots). "Visual observations" could be confused with the code-level `VisionObservation` type, which is the raw data container, not the extracted fact. Terminology.md explicitly distinguishes: "visual facts: used for intermediate representations."

**Fix**: Replace "visual observations" with "visual facts."

**Cross-reference**: `paper_context/terminology.md` -- "vision observation" is the raw container; "vision fact" / "visual fact" is the structured proposition.

---

### MIN-2: "Vision-fact" hyphenation inconsistency (lines 78, 199)

**Location**: Lines 78 ("vision-fact ontology and aggregation rules") and 199 ("vision-fact observation schema").
**Issue**: The terminology.md uses unhyphenated forms: "vision fact ontology" and "vision fact." The chapter introduces hyphenated compounds ("vision-fact") not present in the terminology document. This is a minor consistency issue but the terminology document is the single source of truth for naming.

**Fix**: Either update terminology.md to accept hyphenated compounds, or replace all "vision-fact" with "vision fact" (adjective use) consistently throughout the chapter.

---

### MIN-3: G1 and G2 lack evidence comments (lines 29--39)

**Location**: Lines 29--33 (G1) and lines 35--39 (G2).
**Issue**: G3 through G6 each carry `% Evidence:` comments (lines 43, 48--49, 55--56, 61). G1 and G2 do not. Since G1 and G2 describe the core tutoring functionality (observing learner actions, providing guidance, state-machine tracking), evidence exists in the repository for these claims. The asymmetry is a drafting inconsistency.

**Fix**: Add evidence comments for G1 and G2. Example: `% Evidence: core/procedure.py, core/gating.py, core/step_hint.py, help_flow_en.md`.

---

### MIN-4: Port interface claim lacks evidence reference (line 147)

**Location**: Line 147--148.
**Issue**: "The port interfaces are intentionally minimal: a telemetry port requires `start`, `poll`, and `stop` methods; an overlay port requires only the ability to send highlight, clear, and pulse commands." This is a precise factual claim about the port interface surface. No evidence comment references `ports/telemetry.py` or the overlay port definition to verify these method signatures.

**Fix**: Add an evidence comment referencing the specific port definitions, e.g., `% Evidence: ports/telemetry.py (TelemetryPort), adapters/dcs_adapter.py (overlay methods)`.

---

### MIN-5: "Safe" text hint / target not defined (lines 264, 265)

**Location**: Lines 264--265, within Section 3.5 "Procedure Engine," subsection "Deterministic Step Inference."
**Issue**: "produce a **safe** text hint" and "unless a local rule can prove a unique **safe** target." The word "safe" is used without definition. What property makes a hint or target "safe"? Is it that the hint cannot reference a wrong cockpit element? That it cannot cause the learner to perform a dangerous action? That it is guaranteed to be contextually correct? The term is ambiguous.

**Fix**: Define "safe" in this context, or replace with a more precise term. For example: "produce a text hint restricted to the currently active step" and "unless a local rule can prove a unique, valid target." Or define "safe" earlier in the section.

---

### MIN-6: Figure placeholders with TODO comments (lines 118--119, 166--167)

**Location**: Lines 118--119 (architecture diagram) and lines 166--167 (data flow diagram).
**Issue**: Two figures in a system design chapter are missing. The architecture diagram and data flow diagram are central to a reader's understanding of the system design. The TODOs are appropriately marked and not hidden, which is good practice, but their absence weakens the chapter for any reader.

**Fix**: No textual fix needed for the draft, but these diagrams should be prioritized before any formal submission or advisor review. The review notes this as a minor issue only because the TODOs are transparent.

---

### MIN-7: "25 steps (S01--S25) spanning six phases" without verification path (line 24)

**Location**: Line 24.
**Issue**: The claim that the F/A-18C cold-start procedure comprises "25 steps (S01--S25) spanning six phases" is specific, numeric, and verifiable via the procedure pack. The evidence comment for the procedure pack structure appears later (lines 277--278) but is not linked to this specific numeric claim in the introduction. If the pack defines a different step count or phase count, this sentence would be wrong.

**Fix**: Verify the step count and phase count against `packs/fa18c_startup/step_registry.yaml` and `packs/fa18c_startup/pack.yaml`. Add a brief evidence reference near line 24.

---

### MIN-8: Model name precision in summary table (line 451)

**Location**: Line 451, "Model adaptation" row.
**Issue**: "Qwen3.5-9B" and "Gemma-4-31B" are informal shorthands. The terminology.md uses `Qwen/Qwen3.5-9B-Base` and `google/gemma-4-31B` as the exact references (line 37 of terminology.md). In a formal summary table, model names should match the terminology document.

**Fix**: Use full model identifiers, or define the shorthand at first use: "Qwen3.5-9B (Qwen/Qwen3.5-9B-Base) and Gemma-4-31B (google/gemma-4-31B)."

---

### MIN-9: Unmeasured "fast" claim (line 391)

**Location**: Line 391, within Section 3.8 "VLM Adaptation."
**Issue**: "Rule-based gating evaluates telemetry variables deterministically; it is **fast** and auditable but limited to telemetry-representable conditions." "Fast" is a quantitative claim without measurement. "Auditable" is defensible for deterministic evaluation but would benefit from clarification.

**Fix**: Drop "fast" and keep "deterministic and auditable," or add an evidence reference if timing measurements exist.

---

### MIN-10: "Aggregation rules" term not defined (line 78)

**Location**: Line 78--79.
**Issue**: "the vision-fact ontology and **aggregation rules**" -- The term "aggregation rules" is not defined in the chapter, the terminology document, or the vision facts section (3.8). The VLM section discusses "sticky and TTL semantics" (line 373) and "merge the vision-fact observation" (line 378), which are the actual aggregation behaviors. The term "aggregation rules" appears only once and is left as an undefined label.

**Fix**: Either define "aggregation rules" explicitly (e.g., "the rules governing how consecutive visual fact observations are merged, including sticky persistence and time-to-live expiry") or replace with the more specific terminology used later: "sticky persistence and TTL semantics."

---

## CROSS-CUTTING OBSERVATIONS

### OBS-1: Evidence comment coverage is uneven

The chapter has 57 evidence comments, which is thorough. However, some key claims (architectural purity at line 82, port interface minimality at line 147, stall detection at line 177, G1/G2 at lines 29--39) lack direct evidence comments. The pattern suggests evidence comments were added where code paths are explicit but omitted where claims are about design properties.

### OBS-2: VLM optionality is correctly maintained -- PASS

The review instructions explicitly asked to check whether the VLM is "correctly presented as OPTIONAL, not mandatory." The chapter passes this check. VLM optionality is stated at the design-goal level (G5, lines 51--56), the architecture level (line 150--153: "can include the VLM component, omit it entirely, or replace it"), the runtime level (line 379: "If the vision adapter is unavailable... the help cycle proceeds with telemetry and knowledge only"), and in the section title (line 349: "Optional VLM Adaptation"). No corrective action needed.

### OBS-3: VR and human-subject results are NOT implied as complete -- PASS

The review instructions explicitly asked to check for any implication that VR or human-subject results are complete. The chapter passes this check. VR is described as "planned" (lines 316--317, 410--411), the human-subject evaluation is "postponed" (lines 432--433), and the table row (line 455) states "Postponed; infrastructure prepared." The delta-from-proposal section (3.9) accurately and transparently marks all postponed items. No corrective action needed.

### OBS-4: Delta-from-proposal section correctly marks postponed items -- PASS

The review instructions explicitly asked to check that the delta-from-proposal section correctly marks postponed items. The section passes this check for its structural content. The individual subsections (lines 406--433) correctly mark VR and human-subject evaluation as planned/postponed with evidence references to `system_delta_from_proposal.md`. The table (lines 438--458) is well-structured and accurately distinguishes original proposal from implemented system. The only defect is the evaluative closing language (CRIT-1) and the use of "robust" within this section (MAJ-1).

### OBS-5: Chapter length and claim density

At 462 lines, this is a substantial chapter covering design goals, architecture, data flow, telemetry pipeline, procedure engine, adapters, event logging, VLM pipeline, AND delta-from-proposal -- nine distinct topics. The density means some topics (stall detection, safety boundary, aggregation rules) receive single-sentence treatment. If the chapter is kept as "System Design," consider whether some content (e.g., the detailed data-flow walkthrough in Section 3.4) should move to Chapter 4 (Methodology) or Chapter 5 (Implementation) to keep this chapter focused on architectural decisions and design rationale rather than implementation detail.

---

## SUMMARY TABLE

| # | Severity | Lines | Topic | Root Cause |
|---|----------|-------|-------|------------|
| CRIT-1 | Critical | 461--462 | "significantly more robust" | Unsubstantiated evaluative language; vague term |
| CRIT-2 | Critical | Entire file | Chapter identity vs paper_outline.md | Structural incoherence between source-of-truth documents |
| CRIT-3 | Critical | 346 | TODO:VERIFY on limitation claim | Unverified assertion about system limitation |
| MAJ-1 | Major | 409 | "robust" in delta section | Vague evaluative language |
| MAJ-2 | Major | 319--347 | C-RUNTIME-01 (likely) as established fact | Claims registry status mismatch; missing hedging |
| MAJ-3 | Major | 451 | Gemma as equal peer to Qwen | Likely vs verified claim distinction erased |
| MAJ-4 | Major | 385, 416 | "complete"/"major" contribution | Evaluative language in design chapter |
| MAJ-5 | Major | 82 | Architectural purity claim | Unverified implementation assertion |
| MAJ-6 | Major | 453 | "safety boundary" | Undefined term introduced only in summary table |
| MAJ-7 | Major | 177 | Stall-detection heuristic | Mentioned but never explained or evidenced |
| MIN-1 | Minor | 31 | "visual observations" | Terminology mismatch with terminology.md |
| MIN-2 | Minor | 78, 199 | "vision-fact" hyphenation | Inconsistent with terminology.md |
| MIN-3 | Minor | 29--39 | G1, G2: missing evidence comments | Drafting inconsistency |
| MIN-4 | Minor | 147 | Port interface claim | Missing evidence reference |
| MIN-5 | Minor | 264--265 | "safe" text hint/target | Undefined qualifier |
| MIN-6 | Minor | 118, 166 | Missing figure diagrams | Incomplete drafting (TODOs are transparent) |
| MIN-7 | Minor | 24 | Step/phase count unverified | Unreferenced numeric claim |
| MIN-8 | Minor | 451 | Model name precision | Informal shorthands vs terminology.md |
| MIN-9 | Minor | 391 | "fast" | Unmeasured quantitative claim |
| MIN-10 | Minor | 78 | "aggregation rules" | Undefined term |

**Total**: 3 critical, 7 major, 10 minor = 20 findings.

**Overall assessment**: The chapter is well-organized and evidence-dense (57 evidence comments). The optional VLM presentation, the VR/human-subject boundary, and the delta-from-proposal section are correctly handled. The primary weaknesses are (a) evaluative language ("robust," "significantly," "complete," "major," "safe") in a chapter that should be descriptive, (b) structural tension with the paper outline, and (c) three critical items that should be resolved before any draft is shared with advisors.
