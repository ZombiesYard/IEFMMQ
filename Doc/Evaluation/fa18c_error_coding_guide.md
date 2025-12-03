# F/A-18C Cold Start – Error Coding Guide

This document defines the error taxonomy and scoring rules for the F/A-18C
cold-start task used in the study. It is meant to be used by human raters
when watching recordings or inspecting logs of participants’ runs.

The canonical step sequence is defined in `fa18c_startup_master.md`.

---

## 1. Goals

The coding scheme focuses on **procedural** performance:

- Can the learner follow the correct **sequence** of steps?
- Do they **skip**, **add**, or **mis-order** steps?
- Do they perform steps under the **wrong system state**?
- Do they recover from mistakes or get stuck in **dead ends**?

The scheme does **not** attempt to capture fine-grained stick-and-rudder
skills, only the cockpit procedure from cold-and-dark to taxi-ready.

---

## 2. Error Taxonomy

Each observed deviation from the canonical procedure is coded into one or
more of the following categories. Errors are always coded with respect to
a **target step** (StepID) and the corresponding **phase**.

For each category we give a short definition and one or more examples
from the F/A-18C cold-start task.

### 2.1 Omission (OM)

**Definition:** A required step in the canonical procedure is not performed
at all by the end of the trial.

- **Code:** `OM`
- **Example (S02 – Fire test):**
  - The learner never runs FIRE TEST A or B before proceeding to engine start.
- **Example (S14 – OBOGS ON):**
  - The learner never turns on OBOGS and FLOW, even though both engines
    are running and they taxi away.

**Notes:**

- If a step is performed only **partially** or in an obviously wrong way,
  code additional categories (e.g., `PA` Parameter error) on the same step.

---

### 2.2 Commission (CO)

**Definition:** The learner performs a **non-required** or clearly
unnecessary step that is not part of the canonical checklist and is not
implicitly required by the simulator.

- **Code:** `CO`
- **Example:**
  - The learner toggles random lighting or weapon switches that are not
    needed for startup and not mentioned in the syllabus.
  - The learner changes radar modes or weapon profiles during startup.

**Notes:**

- Commission errors are often less critical than omissions but may
  increase workload and confusion.
- If a commission step leads to a **state violation** (e.g., turning off
  a necessary system), code `CO` and `SV` together.

---

### 2.3 Order Error (OR)

**Definition:** A step that belongs to the canonical procedure is executed
in the wrong **relative order** compared to other steps.

- **Code:** `OR`
- **Example (S12 – INS mode vs. S11 – left engine to IDLE):**
  - The learner sets INS to GND/CV (S12) **before** the left engine is
    stabilized above 60% RPM (S11).
- **Example (S08 – display power vs. S03 – APU READY):**
  - The learner powers DDIs/HUD/MPCD (S08) **before** turning the battery
    on and starting the APU (S01–S03).

**Notes:**

- Only code an order error if the mis-order is **procedurally meaningful**
  (i.e., violates a dependency or could cause confusion).
- Harmless re-orderings within a phase that do not violate dependencies
  may be ignored or coded as comments only.

---

### 2.4 Parameter / Setting Error (PA)

**Definition:** The correct step is executed, but with wrong parameter
values, positions, or magnitude.

- **Code:** `PA`
- **Example (S21 – BINGO fuel):**
  - The learner sets BINGO fuel to a clearly unreasonable value
    (e.g., near-zero or maximum tank capacity) instead of the mission’s
    recommended threshold.
- **Example (S23 – Radar altimeter bug):**
  - The learner sets the radar altimeter bug to 5000 ft instead of
    200 ft (airfield) or 40 ft (carrier).

**Notes:**

- For some steps, the DCS manual defines acceptable ranges (e.g., engine
  TEMP, nozzle position). If a learner continues with values clearly
  outside the green range, code a `PA` in addition to a possible
  `SV` (state violation).

---

### 2.5 State Violation (SV)

**Definition:** The learner executes a step while required **preconditions**
about the aircraft state are not satisfied.

- **Code:** `SV`
- **Example (S05 – Right throttle to IDLE at ≥25% RPM):**
  - The learner advances the right throttle from OFF to IDLE when RPM is
    below 20% (e.g., ~10%), violating the “≥25% RPM” rule.
- **Example (S10 – Left engine crank before right engine stable):**
  - The learner moves ENG CRANK to LEFT while the right engine has not
    yet stabilized inside the nominal RPM/TEMP/FF/nozzle/oil ranges.
- **Example (INS alignment):**
  - The learner switches INS knob to GND/CV before the left engine is
    stabilized above 60% RPM as specified.

**Notes:**

- State violations often co-occur with order errors (`OR`), but they are
  coded separately because they explicitly reason about **system state**.
- In many cases, state violations are considered **critical** (see Section 3).

---

### 2.6 Dead-End / Stuck (DE)

**Definition:** The learner reaches a state from which they **cannot
complete** the procedure without external help, or they repeatedly loop
without making progress.

- **Code:** `DE` (trial-level, not per-step)
- **Example:**
  - The learner shuts down critical systems unintentionally and cannot
    identify how to recover.
  - The learner keeps cycling the same incorrect switches for more than
    a predefined time window (e.g., 2 minutes) without advancing in the
    sequence.

**Notes:**

- Dead-ends are coded at the **trial level**, not per step.
- A trial with `DE` is typically classified as **failed** unless the
  protocol explicitly allows instructor intervention and re-try.

---

### 2.7 Help Request / External Intervention (HR)

**Definition:** The learner explicitly asks for help outside the assigned
condition (e.g., looking up external guides or asking the experimenter),
or the experimenter intervenes to correct a mistake.

- **Code:** `HR` (trial-level) or attached to specific steps.
- **Example:**
  - Participant pauses the VR session and watches YouTube or a different
    checklist that is not part of the condition.
  - Experimenter tells the participant which switch to flip to recover.

**Notes:**

- For conditions where external help is allowed (e.g. video+notes), the
  definition of `HR` should be adapted accordingly (e.g., “unplanned help”).
- The presence of `HR` is logged for analysis (e.g., frequency of
  unplanned assistance) but does not always make a trial invalid.

---

## 3. Step Criticality

Each canonical step in `fa18c_startup_master.md` is assigned a **criticality**
flag, indicating whether an error on that step is considered **critical** for
task success.

- `Critical = Yes`:
  - Omitting the step or violating its preconditions directly compromises
    safety or makes the aircraft not taxi-ready.
- `Critical = No`:
  - Errors are undesirable but do not necessarily prevent safe taxi or
    may be corrected easily later.

**Examples (suggested classification):**

- **Critical steps (examples):**
  - S01 (Battery / generators ON)
  - S03 (APU ON / APU READY)
  - S04–S05–S10–S11 (engine start sequence)
  - S12 (INS mode set correctly)
  - S14 (OBOGS ON)
  - S15–S17–S18 (FCS checks and takeoff trim)
  - S22–S24–S25 (core attitude / altimeter instruments)

- **Non-critical or lower-criticality steps (examples):**
  - S07 (lights test)
  - S09 (mission-specific comms)
  - S21 (BINGO fuel, if the scenario does not test fuel management)
  - Cosmetic parts of the “four down” test if they are not essential for
    your specific training goals.

The exact criticality labels should be finalized with the domain expert
(e.g., instructor) and recorded in the step table.

---

## 4. Scoring Rules

### 4.1 Per-Step Error Scoring

Each step can have zero or multiple error codes attached. We assign
**weights** to error categories:

- Omission (`OM`): **2 points**
- Commission (`CO`): **1 point**
- Order (`OR`): **2 points**
- Parameter (`PA`): **1 point**
- State violation (`SV`): **2 points**

For **critical steps**, all error weights are multiplied by **1.5** (round
up to nearest integer if needed).

**Example scoring for one step:**

- Step S12, critical = Yes:
  - One `OR` and one `SV` → base score = 2 (OR) + 2 (SV) = 4  
    Critical multiplier 1.5 → 4 × 1.5 = 6 → recorded as 6 error points.

If multiple instances of the **same** error category occur on the same
step, the rater may:

- Either count only the **first occurrence**, or
- Count repeated occurrences up to a maximum per step (e.g., max 3 points
  per category per step), as defined in the study protocol.

For simplicity, we recommend:

> **Count at most 1 occurrence per error category per step.**

---

### 4.2 Trial-Level Scores

For each trial, compute:

1. **Total Error Score**  
   Sum of all per-step error points across all steps.

2. **Error Counts by Category**  
   The total count of `OM`, `CO`, `OR`, `PA`, `SV` across the trial.

3. **Critical vs. Non-Critical Errors**  
   - `CriticalErrorScore`: Sum of points on steps with `Critical = Yes`.
   - `NonCriticalErrorScore`: Sum of points on steps with `Critical = No`.

4. **Dead-End / Help Flags**  
   - `DeadEnd (DE)`: Boolean.
   - `HelpRequest (HR)`: Boolean, plus free-text notes.

5. **Completion & Time**  
   - `Completed`: Yes/No (taxi-ready state reached).
   - `TaskTime`: Time from first cockpit interaction to end condition.

These scores can be used as dependent variables in the later analysis
(e.g., comparing groups on total error score, critical error score,
completion rate, and time).

---

## 5. Coding Procedure

1. **Data Source:**
   - Use VR video recordings and/or DCS track/log files.
   - Ensure raters have access to the canonical sequence (`fa18c_startup_master.md`)
     and this coding guide.

2. **Step-by-Step Coding:**
   - For each trial, follow the video/log and mark when each canonical step
     (S01–S25) is executed.
   - For each step, record:
     - Whether the step was performed (`Performed: Yes/No`).
     - Any error categories (`OM`, `CO`, `OR`, `PA`, `SV`).
     - Free-text comments if necessary.

3. **Inter-Rater Agreement:**
   - For a subset of trials (e.g., 20–30%), have two raters code independently.
   - Compute agreement metrics (e.g., percent agreement, Cohen’s kappa) for
     presence/absence of each error category per step.
   - Discuss and refine ambiguous definitions if agreement is low.

4. **Consistency Over Time:**
   - Keep this document version-controlled (e.g., Git) and update it only
     with tracked changes.
   - Note any major revisions so analyses can refer to the correct version.

---

## 6. Relation to Study Conditions

The same error taxonomy and scoring rules are applied across all three
conditions:

1. **Video + notes (baseline)**  
2. **Ungrounded RAG+LLM (text-only tutor)**  
3. **Grounded RAG+LLM (state-aware VR tutor)**

This allows a consistent comparison of how each condition affects:

- Total and critical error scores,
- Specific error types (e.g., fewer `OR` and `SV` errors in the grounded condition),
- Dead-end frequency and help requests,
- Completion rates and task times.

