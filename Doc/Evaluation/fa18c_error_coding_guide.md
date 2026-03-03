# F/A-18C Cold Start - Error Coding Guide

This guide defines the coding taxonomy and scoring rules for F/A-18C cold-start
trials (S01-S25). It is designed for reproducible human rating from video/logs and
for alignment with the thesis metrics.

Canonical step sequence: `fa18c_startup_master.md`.

---

## 1. Scope and Goals

The guide measures procedural performance for sequence-sensitive training:

- Sequence correctness
- State-aware correctness (precondition satisfaction)
- Recovery ability when stuck
- Assistance dependence

Out of scope: stick-and-rudder quality and tactical mission performance.

---

## 2. Error Taxonomy

Each error is attached to a `StepID` unless explicitly marked as trial-level.

### 2.1 Omission (`OM`)

Required canonical step never completed by trial end.

### 2.2 Commission (`CO`)

Non-required operation that is outside canonical flow and not simulator-required.

### 2.3 Order Error (`OR`)

Canonical step executed in dependency-breaking order.

### 2.4 Parameter/Setting Error (`PA`)

Step attempted with wrong setting/value (wrong switch state, wrong numeric target, wrong mode).

### 2.5 State Violation (`SV`)

Step executed while required preconditions are unsatisfied.

### 2.6 Dead-End / Stuck (`DE`, trial-level)

A trial-level flag. Set `DE=yes` if any stuck episode meets the formal criterion:

- No valid canonical step advancement for >= 90 s, or
- >= 3 ineffective repeated interactions within 60 s with no gate advancement,

and the episode is not recovered before timeout/end.

### 2.7 Help Request / External Intervention (`HR`, trial-level, optionally step-linked)

`HR` records assistance use outside planned condition allowances.

- For `video_notes`: only pre-provided material is allowed.
- For LLM conditions: platform tutor help is planned; unplanned outside help remains `HR`.
- Experimenter direct instruction that changes participant behavior is always recorded.

---

## 3. Co-Coding and Disambiguation Rules

To reduce rater drift, use these deterministic rules:

1. `OR` and `SV` may co-exist on the same step when both order and state are violated.
2. Repeated same-category errors on one step are capped at one count per category per step.
3. If a step is never completed, code `OM=1`; do not additionally code `OR` for that missing step.
4. If an extra action both is non-canonical and breaks state, code `CO=1` and `SV=1`.
5. When unsure between `PA` and `SV`, prioritize `SV` if a precondition is explicitly violated; otherwise `PA`.

---

## 4. Criticality

Criticality is **not guessed by raters**. It must come from the study-controlled
step table (`packs/.../scoring.yaml` or exported step criticality table version).

- `Critical=yes`: error can block safe taxi-ready outcome or safety baseline.
- `Critical=no`: error is recoverable/minor for taxi-ready validity.

All scoring must cite the exact criticality table version used in the experiment run.

---

## 5. Scoring Rules

### 5.1 Base Weights

- `OM`: 2
- `CO`: 1
- `OR`: 2
- `PA`: 1
- `SV`: 2

### 5.2 Critical Multiplier

If `Critical=yes`, multiply step subtotal by `1.5`.

### 5.3 Rounding Rule

Use deterministic rounding: **ceiling to nearest integer**.

Example:

- Step subtotal = 3, critical step -> `3 * 1.5 = 4.5` -> score `5`.

### 5.4 Step Score Formula

`StepErrorScore = ceil((2*OM + 1*CO + 2*OR + 1*PA + 2*SV) * (1.5 if Critical else 1.0))`

---

## 6. Trial-Level Outputs

For each trial compute:

1. `TotalErrorScore`
2. `CriticalErrorScore`
3. `NonCriticalErrorScore`
4. Category counts: `Count_OM`, `Count_CO`, `Count_OR`, `Count_PA`, `Count_SV`
5. `Completed` (yes/no), `TaskTime_sec`
6. `DeadEnd_DE` (yes/no), `HelpRequest_HR` (yes/no)
7. Stuck metrics:
   - `StuckEpisodeCount`
   - `TotalStuckDuration_sec`
   - `MaxSingleStuckDuration_sec`
   - `TimeToFirstHelp_sec` (if any)

These outputs directly support the thesis dependent variables.

---

## 7. Coding Procedure

1. Prepare artifacts:
   - Trial recording/log
   - Canonical step table (S01-S25)
   - Criticality table version
   - This guide version

2. Perform step coding:
   - Mark completion timestamp per step.
   - Assign error categories by rule.
   - Record notes only for ambiguous cases.

3. Mark trial-level events:
   - `DE`, `HR`, stuck metrics, completion, time.

4. Inter-rater reliability:
   - Double-code at least 20% of trials.
   - Report Cohen's kappa per category (`OM/CO/OR/PA/SV`) and for `DE`.
   - If kappa < 0.70 for any category, refine instructions and re-code affected subset.

---

## 8. Relation to Study Conditions

Same taxonomy and scoring are applied to all conditions:

1. `video_notes`
2. `llm_ungrounded`
3. `grounded_rag_llm`

Only the assistance channel differs by condition; rating logic does not.

---

## 9. Versioning

This guide is versioned with the repository. Any scoring-impacting change must:

- increment the guide version tag,
- list changed rules,
- and specify whether previously coded trials need re-scoring.
