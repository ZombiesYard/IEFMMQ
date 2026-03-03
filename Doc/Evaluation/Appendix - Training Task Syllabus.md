# Task Syllabus: F/A-18C Cold Start to Taxi-Ready

This syllabus defines the experimental task for the thesis study:
a full F/A-18C cold start in DCS World (VR) from cold-and-dark to taxi-ready.

The canonical sequence is defined in `fa18c_startup_master.md` (S01-S25),
and the authoritative study scope is grounded in `Doc/Evaluation/index.json`
(entries corresponding to the thesis evaluation set).

---

## 1. Scenario Definition

- **Platform:** DCS World, F/A-18C Lot 20, VR mode (Meta Quest 3).
- **Starting condition:** Aircraft in cold-and-dark state (engines off, electrical buses unpowered, parking brake set).
- **Task objective:** Reach taxi-ready state by completing canonical steps S01-S25.
- **End condition (taxi-ready):**
  - Both engines started and stabilized in normal range.
  - Electrical and bleed-air configuration valid for ground operation.
  - INS alignment started in correct mode (GND for airfield, CV for carrier).
  - FCS checks completed and takeoff trim set.
  - Core cockpit instruments configured (BINGO, standby baro, radar alt bug, standby attitude).

Learners must follow the canonical sequence. Any assistance behavior is coded by
condition-specific rules (Section 4) and the error coding guide.

---

## 2. Phase Structure

For analysis and tutoring, the 25 canonical steps are grouped into six phases.

### Phase P1 - Power-Up & Safety (S01-S03)

**Goal:** transition from unpowered aircraft to safe powered state.

- S01: Battery ON, generators verified ON.
- S02: Fire detection test A/B.
- S03: APU ON and APU READY.

**Completion conditions:**

- Battery switch ON state confirmed.
- Generator switches in expected normal state.
- Fire test channels A and B completed.
- APU READY indication present.

### Phase P2 - Right Engine Start & Basic Checks (S04-S07)

**Goal:** start and stabilize right engine, validate immediate warnings and bleed logic.

- S04: ENG CRANK RIGHT.
- S05: Right throttle OFF -> IDLE at valid RPM threshold.
- S06: Bleed air knob cycle after right engine reaches required RPM.
- S07: Caution/warning/advisory lights test.

### Phase P3 - Displays & Communications (S08-S09)

**Goal:** bring core displays online and set baseline comm configuration.

- S08: DDIs/MPCD/HUD power and required pages.
- S09: COMM1/COMM2 baseline setup.

### Phase P4 - Left Engine Start & Core Systems (S10-S14)

**Goal:** start left engine and activate navigation/survivability core systems.

- S10: ENG CRANK LEFT after right engine verification.
- S11: Left throttle OFF -> IDLE at valid RPM threshold.
- S12: INS knob to GND/CV after left engine reaches threshold.
- S13: Radar knob to OPR.
- S14: OBOGS and FLOW ON.

### Phase P5 - FCS & Flight Controls (S15-S19)

**Goal:** complete FCS checks and surface/configuration checks before taxi.

- S15: FCS RESET.
- S16: Flaps AUTO.
- S17: Takeoff trim.
- S18: FCS BIT.
- S19: Four-down related checks and required final positions.

### Phase P6 - Pre-Taxi Instruments & Final Checks (S20-S25)

**Goal:** finalize pre-taxi instruments and complete readiness items.

- S20: Parking brake release when all prior requirements are satisfied.
- S21: BINGO fuel set.
- S22: Standby barometric altimeter set.
- S23: Radar altimeter bug set.
- S24: Standby attitude uncage.
- S25: Attitude source AUTO.

---

## 3. Success, Failure, and Stuck Criteria

### 3.1 Trial Success

A trial is **successful** only if all conditions below are met:

1. **Completion:** canonical end condition reached (taxi-ready).
2. **Safety correctness:** no unresolved critical error at end of S25.
3. **Time bound:** completed within **20 minutes (1200 s)** from first cockpit action.

### 3.2 Trial Failure

A trial is **failed** if any of the following occurs:

- Hard timeout: exceeds 20 minutes before taxi-ready.
- Persistent critical error at trial end.
- Dead-end/stuck state not recovered within allowed recovery window.

### 3.3 Operational Stuck Definition

A **stuck episode** is recorded when either condition holds:

1. No canonical step progress for **>= 90 s** after the last valid step completion.
2. Repeated ineffective interaction loop: **>= 3** interactions on non-progressing controls within a 60 s window, with no gate advancement.

For each trial, record:

- `StuckEpisodeCount`
- `TotalStuckDuration_sec`
- `MaxSingleStuckDuration_sec`
- `RecoveredFromStuck` (yes/no)

These fields are mandatory for thesis analysis on flow interruption and tutor utility.

---

## 4. Condition Protocol (A/B/C)

The experiment uses three conditions with the same task and scoring rubric:

1. **A: video_notes (baseline)**
   - Participant may use only pre-provided video/notes.
   - Any additional unplanned external source is coded as `HR`.

2. **B: llm_ungrounded (text-only LLM)**
   - Tutor text is allowed but not state-grounded.
   - No cockpit overlay target guarantee.

3. **C: grounded_rag_llm (state-aware tutor)**
   - Tutor must be conditioned on current state and retrieved snippets.
   - Output is constrained to one actionable next step + one self-check + short why.
   - Overlay target must be from allowlist and pass guardrails.

To keep cross-condition comparability, all groups are evaluated with the same
error taxonomy, same timeout, and same stuck criteria.

---

## 5. Evaluation Outputs Required by This Syllabus

Each trial must produce:

- Step trajectory with timestamps (S01-S25)
- Trial-level completion/time fields
- Per-category error counts (`OM`, `CO`, `OR`, `PA`, `SV`)
- `DE`/`HR` and stuck metrics
- NASA Raw-TLX result
- Optional retention/transfer linkage ID for delayed tests (2-4 weeks)

This ensures direct alignment with thesis objectives: completion rate, error rate,
time, dead-end/stuck behavior, workload, and delayed retention/transfer analysis.
