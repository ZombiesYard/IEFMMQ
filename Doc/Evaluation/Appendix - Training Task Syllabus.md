# Task Syllabus: F/A-18C Cold Start to Taxi-Ready

This syllabus specifies the procedural training task used in the study:
a cold start of the F/A-18C from a “cold and dark” parking position to a
taxi-ready configuration in DCS World (VR).

The canonical step sequence and detailed step wording are defined in
`fa18c_startup_master.md`. This syllabus focuses on task scope, phases,
and success criteria for experimental evaluation.

---

## 1. Scenario Definition

- **Platform:** DCS World, F/A-18C Lot 20, VR mode (Meta Quest 3).
- **Starting condition:** Aircraft parked at an airfield ramp or carrier deck in a
  cold-and-dark state (engines off, no electrical power, parking brake set).
- **End condition (“taxi-ready”):**
  - Both engines running and stabilized in the green range.
  - Electrical and bleed air systems configured for normal operation.
  - Core avionics and INS alignment initialized for taxi and departure.
  - Flight control system (FCS) checks completed, takeoff trim set.
  - Key instruments configured (altimeters, attitude references, BINGO fuel).
  - Parking brake released only when the instructor/system permits taxi.

Learners are instructed to follow the standard F/A-18C cold-start procedure
as defined in the DCS manual, with minor simplifications to ensure consistent
measurement across participants.

---

## 2. Phase Structure

For analysis and guidance, the 25 steps in `fa18c_startup_master.md` are grouped
into six phases. Each phase has a clear functional goal and observable completion
conditions (DCS state + cockpit configuration).

### Phase P1 – Power-Up & Safety (S01–S03)

**Goal:** Bring the aircraft from cold and dark to a safe, powered state with
basic safety systems verified.

- **Content (reference steps):**
  - S01: Battery ON, generators verified ON.
  - S02: Fire detection tests A and B completed.
  - S03: APU ON and APU READY.

**Completion conditions:**

- BATTERY switch ON; both generator switches in NORM/ON as per DCS default.
- Fire warning test successfully run for both channels (A/B).
- APU READY light illuminated.

**Typical errors to monitor:**

- Skipping a fire test.
- Advancing to engine crank with no APU READY.
- Forgetting to power the aircraft before manipulating other systems.

---

### Phase P2 – Right Engine Start & Basic Checks (S04–S07)

**Goal:** Start and stabilize the right engine and verify basic bleed air and
warning systems.

- **Content:**
  - S04: ENG CRANK to RIGHT.
  - S05: Right throttle from OFF to IDLE at ≥25% RPM.
  - S06: Bleed air knob 360° rotation once RPM >60%.
  - S07: Caution/warning/advisory lights test.

**Completion conditions:**

- Right engine RPM stabilized in the nominal range (≈63–70% with normal TEMP,
  fuel flow, nozzle position, and oil pressure).
- Bleed air knob cycle completed without abnormal indications.
- Light test confirms all annunciators illuminate.

**Typical errors:**

- Moving throttle too early or too late.
- Forgetting to run the lights test.
- Misinterpreting engine parameters and continuing with abnormal values.

---

### Phase P3 – Displays & Communications (S08–S09)

**Goal:** Bring up the primary displays and configure basic communications.

- **Content:**
  - S08: Power on DDIs, MPCD, HUD; select FCS page (left DDI) and BIT page (right DDI).
  - S09: Configure COMM1/COMM2 as required.

**Completion conditions:**

- Both DDIs, the MPCD, and the HUD powered and showing correct pages.
- FCS page visible for later FCS checks; BIT page available for system tests.
- Radios configured to pre-defined training frequencies (or at minimum,
  COMM panels and UFC appropriately set as per experiment instructions).

**Typical errors:**

- Forgetting to bring up FCS/BIT pages, making later checks harder.
- Skipping radio configuration entirely.

---

### Phase P4 – Left Engine Start & Core Systems (S10–S14)

**Goal:** Start the left engine and bring core navigation and life-support
systems online.

- **Content:**
  - S10: Verify right engine parameters; ENG CRANK to LEFT.
  - S11: Left throttle from OFF to IDLE at ≥25% RPM.
  - S12: INS knob to GND or CV once RPM >60%.
  - S13: Radar knob to OPR.
  - S14: OBOGS control and FLOW switches to ON.

**Completion conditions:**

- Both engines running and stabilized within nominal RPM and parameter ranges.
- INS alignment initiated in the correct mode (GND for airfield, CV for carrier).
- Radar powered to OPR.
- OBOGS and FLOW switches ON.

**Typical errors:**

- Starting the left engine without verifying right engine health.
- Setting INS to the wrong mode or forgetting to start alignment.
- Forgetting OBOGS, especially for new learners.

---

### Phase P5 – FCS & Flight Controls (S15–S19)

**Goal:** Verify the flight control system and complete major control-surface
and systems checks before taxi.

- **Content:**
  - S15: FCS RESET, monitor FCS page.
  - S16: Flaps to AUTO.
  - S17: Takeoff trim.
  - S18: FCS BIT test (switch + OSB).
  - S19: “Four down” checks: refueling probe, speed brake, launch bar, arrestor hook, pitot heat, flaps to HALF.

**Completion conditions:**

- No critical FCS faults displayed after RESET and BIT.
- Flaps set to the prescribed position (AUTO or HALF) per the training
  scenario’s final configuration.
- “Four down” items checked and returned to the correct positions.

**Typical errors:**

- Skipping FCS RESET or BIT.
- Leaving flaps in the wrong position for taxi/takeoff.
- Forgetting one of the “four down” items, especially pitot heat or launch bar.

---

### Phase P6 – Pre-Taxi Instruments & Final Checks (S20–S25)

**Goal:** Configure key instruments and final references; release the aircraft
for taxi when cleared.

- **Content:**
  - S20: Release parking brake (only when ready).
  - S21: Set BINGO fuel on IFEI.
  - S22: Set standby barometric altimeter to field elevation.
  - S23: Set radar altimeter bug (e.g., 200 ft airfield / 40 ft carrier).
  - S24: Uncage standby attitude indicator.
  - S25: Set attitude source to AUTO.

**Completion conditions:**

- Parking brake released only after all previous phases are complete and the
  instructor/system indicates taxi clearance.
- BINGO fuel set to the training mission’s reference value.
- Standby baro altimeter and radar altimeter correctly set.
- Standby ADI uncaged and attitude source in AUTO.

**Typical errors:**

- Releasing the parking brake prematurely, before systems are ready.
- Forgetting to set altimeters or BINGO fuel.
- Leaving attitude instruments caged or in the wrong source mode.

---

## 3. Success and Failure Criteria

For the experiment, a trial is considered **successful** if:

1. The learner reaches the “taxi-ready” end condition:
   - Both engines running and stabilized.
   - INS alignment started in the correct mode.
   - Radar and OBOGS correctly configured.
   - FCS checks completed with acceptable status.
   - Key instruments configured (altimeters, attitude, BINGO fuel).
2. No **critical** safety or configuration errors remain at the end of Phase P6
   (as defined in the error taxonomy and scoring sheet).
3. The learner completes the procedure within a predefined time budget
   (e.g., X minutes; to be specified in the experimental protocol).

A trial is treated as **failed** if:

- The learner becomes stuck in a dead end and cannot progress without external help.
- A critical configuration or safety error persists (e.g., wrong INS mode, major FCS fault, or incorrect flap setting) after the allowed correction phase.
- The time limit is exceeded without reaching the taxi-ready state.

---

## 4. Use in the Study

- The **video+notes** group will receive a static version of this syllabus plus
  a conventional checklist.
- The **ungrounded LLM** group will receive text-only assistance that may refer
  to these phases and steps but is not constrained by DCS state.
- The **grounded RAG+LLM tutor** is explicitly conditioned on:
  - The current phase and step from `fa18c_startup_master.md`.
  - Minimal DCS state flags (e.g., engine RPM ranges, switch positions).
  - Retrieved manual/checklist snippets that correspond to the current step(s).

All performance metrics (errors, time, dead ends) will be coded with respect
to the step IDs and phases defined in `fa18c_startup_master.md`.
