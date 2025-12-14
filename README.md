# F/A-18C Cold Start Training & Evaluation Pack

## Overview
- Purpose: Standardized cold-start-to-taxi training and evaluation in DCS World (F/A-18C Lot 20, VR/Meta Quest 3).
- Core assets: 25-step canonical startup table (S01–S25), phase-based syllabus, error taxonomy + scoring rules, NASA Raw-TLX form, knowledge quiz, and scoring sheet templates.
- Key locations: `Doc/Evaluation` holds the text assets; `Doc/week1-2 Pre/week1 Pre.pdf` contains a supplemental PDF.

## Core Documents
- `Doc/Evaluation/fa18c_startup_master.md`: Canonical cold-start table, 25 steps across 6 phases (P1–P6), with official wording, learner-friendly wording, and cockpit area.
- `Doc/Evaluation/Appendix - Training Task Syllabus.md`: Task syllabus with scenario, start/end conditions, phase goals, completion conditions, and common errors.
- `Doc/Evaluation/fa18c_error_coding_guide.md`: Error taxonomy and scoring rules, including criticality guidance.
- `Doc/Evaluation/fa18c_scoring_sheet_template.md`: Trial-level and step-level scoring templates.
- `Doc/Evaluation/fa18c_coldstart_quiz.md`: Knowledge quiz (single/multiple choice with answers).
- `Doc/Evaluation/fa18c_nasatlx_vr.md`: NASA Raw-TLX (0–20 scale, 6 dimensions, with prompts).

## Phase-at-a-Glance (S01–S25)
- P1 Power-up & Safety: Battery/generators, fire tests A/B, APU ON/READY.
- P2 Right Engine Start: ENG CRANK RIGHT, throttle to IDLE at ≥25% RPM, bleed-air knob 360° at >60% RPM, caution/warning/advisory light test.
- P3 Displays & Comms: Power DDIs/MPCD/HUD; left DDI FCS, right DDI BIT; set COMM1/COMM2.
- P4 Left Engine & Core Systems: Verify right engine parameters; ENG CRANK LEFT; left throttle at ≥25% RPM; INS to GND/CV once >60% RPM; radar OPR; OBOGS + FLOW ON.
- P5 FCS & Flight Controls: FCS RESET; flaps AUTO; takeoff trim; FCS BIT; “four-down” check (probe, speed brake, launch bar, hook, pitot heat; flaps HALF).
- P6 Pre-Taxi Instruments: Release parking brake when ready; set BINGO fuel; standby baro altimeter; radar alt bug; uncage standby attitude indicator; attitude source AUTO.

## Error Coding & Scoring (from `fa18c_error_coding_guide.md`)
- Categories: OM (omission), CO (commission), OR (order), PA (parameter), SV (state violation); trial-level flags DE (dead-end), HR (help request).
- Base weights: OM 2, CO 1, OR 2, PA 1, SV 2; critical steps ×1.5.
- Step score formula: `(2*OM + 1*CO + 2*OR + 1*PA + 2*SV) * (Critical ? 1.5 : 1)`; recommend max 1 occurrence per category per step.

## Scoring Templates (`fa18c_scoring_sheet_template.md`)
- Trial-level: ParticipantID, Condition (e.g., video_notes / llm_ungrounded / grounded_rag_llm), TrialID, Completed, DE/HR, TaskTime, TotalErrorScore, Critical/NonCritical scores, counts per category, comments.
- Step-level: StepID (S01–S25), Phase, Critical, Performed, Error_OM/CO/OR/PA/SV (0/1), StepErrorScore, notes.

## Assessment Tools
- Knowledge quiz (`fa18c_coldstart_quiz.md`): covers goals, throttle timing, INS mode, FCS prep, final instruments; includes answer key.
- NASA Raw-TLX (`fa18c_nasatlx_vr.md`): 6 dimensions (Mental, Physical, Temporal, Performance, Effort, Frustration), 0–20 scale with anchors.
