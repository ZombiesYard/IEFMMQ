# F/A-18C Cold Start – Master Step List

This file defines the canonical F/A-18C cold-start sequence used in the study.
It is derived from the DCS F/A-18C manual / training mission description and
is organized into phases for analysis and training.

Each row is an *instructional* step (what we track in the experiment), not
necessarily a one-to-one mapping to all real-world checklist items.

Columns:

- **StepID**: Stable identifier used in scoring sheets and logs.
- **Phase**: Coarse-grained phase of the startup procedure.
- **OfficialStep**: Wording close to the DCS manual / in-game tutorial.
- **ShortExplanation**: Simplified phrasing for learners and the tutor UI.
- **CockpitArea**: Primary cockpit region where the action happens.

---

## Master Step Table

| StepID | Phase | OfficialStep (DCS-style wording) | ShortExplanation (for learners/tutor) | CockpitArea |
|--------|-------|-----------------------------------|----------------------------------------|-------------|
| S01 | P1 – Power-up & safety | Set the BATTERY switch to ON and confirm both Left and Right Generators are ON. | Turn on aircraft battery and verify both generators are on to power the jet. | Right console |
| S02 | P1 – Power-up & safety | Move and hold the fire detection switch to FIRE TEST A and wait for all the audio caution messages to play. Once complete, wait 10 seconds and then do the same for FIRE TEST B. Between running FIRE TEST A and FIRE TEST B, you can reset the battery switch to rewind the fire test tape. | Run fire detection tests A and B and listen for all caution tones to confirm the fire warning system works. | Left console |
| S03 | P1 – Power-up & safety | APU switch to ON and wait for green APU READY light. | Start the APU and wait until the APU READY light comes on. | Left console |
| S04 | P2 – Right engine start | Move the ENG CRANK switch to the right to start the right engine. | Set the engine crank switch to RIGHT to begin right engine start. | Left console |
| S05 | P2 – Right engine start | Move the right throttle from OFF to IDLE when the right engine is above 25% rpm (as shown on IFEI). | When right engine RPM passes 25%, advance the right throttle from OFF to IDLE. | Throttles / IFEI |
| S06 | P2 – Right engine start | Once right engine RPM is over 60%, rotate the BLEED AIR knob 360° clockwise, from NORM to NORM. | After the right engine stabilizes above 60% RPM, rotate the BLEED AIR knob through a full clockwise turn to confirm operation. | Right console |
| S07 | P2 – Right engine start | Test the CAUTION, WARNING and ADVISORY lights test. | Run the caution/warning/advisory lights test and verify all annunciators illuminate correctly. | Right console |
| S08 | P3 – Displays & comms | Turn on the power to both DDIs, MPCD and HUD. Select the FCS page on the left DDI and the BIT page on the right DDI. | Power up both DDIs, MPCD and HUD, then show FCS on the left DDI and BIT on the right DDI. | Center instrument panel |
| S09 | P3 – Displays & comms | Set COMM 1 and COMM 2 radios as required for the mission. | Set COMM1 and COMM2 radios to the preset frequencies used in the training mission. | UFC / comm panels |
| S10 | P4 – Left engine & core systems | Move the ENG CRANK switch to the left after confirming that the right engine has an rpm between 63 and 70%, a TEMP between 190° and 590°, Fuel Flow between 420 and 900 PPH, a nozzle position between 73 and 84%, and an OIL pressure between 45 and 110 psi. | Verify right engine parameters are in normal ranges, then set the engine crank switch to LEFT for left engine start. | Left console / IFEI |
| S11 | P4 – Left engine & core systems | Move the left throttle from OFF to IDLE when left engine has reached at least 25% rpm. | When left engine RPM passes 25%, advance the left throttle from OFF to IDLE. | Throttles / IFEI |
| S12 | P4 – Left engine & core systems | Once the left engine has an RPM greater than 60%, rotate the INS knob to GND (ground) or CV (carrier), depending on your parking location. | After the left engine stabilizes above 60% RPM, set the INS knob to GND (airfield) or CV (carrier) to start alignment. | Right console |
| S13 | P4 – Left engine & core systems | Set the Radar knob to OPR (operate). | Turn the radar power knob to OPR to bring the radar online. | Right console |
| S14 | P4 – Left engine & core systems | Set the OBOGS control switch and FLOW switch to ON. | Turn OBOGS and OBOGS FLOW to ON to enable onboard oxygen generation. | Left console |
| S15 | P5 – FCS & flight controls | Press the FCS RESET button and monitor FCS DDI page. | Press FCS RESET and watch the FCS page for any faults. | Left console / left DDI |
| S16 | P5 – FCS & flight controls | Set the Flap switch to AUTO. | Set the flap switch to AUTO for normal takeoff configuration. | Left quarter panel |
| S17 | P5 – FCS & flight controls | Press Takeoff Trim button. | Press the TAKEOFF TRIM button to set stabilator trim for takeoff. | Left console |
| S18 | P5 – FCS & flight controls | While holding up the FCS BIT switch on the right wall, press the FCS OSB on the BIT / FCS page at the same time. | Run the FCS BIT: hold the FCS BIT switch and press the FCS option on the BIT/FCS page simultaneously. | Right console / right DDI |
| S19 | P5 – FCS & flight controls | Four down test. Cycle/test the refueling probe, speed brake, launch bar, arrestor hook, pitot heat, and set flaps to HALF. | Perform the “four down” checks: cycle refueling probe, speed brake, launch bar, arrestor hook, pitot heat, and set flaps to HALF. | Multiple (consoles, throttles, quarter panels) |
| S20 | P6 – Pre-taxi instruments & final checks | Left mouse click on the hand brake to release it. | Release the parking brake when ready to taxi. | Left console (parking brake) |
| S21 | P6 – Pre-taxi instruments & final checks | Set your BINGO fuel level (minimum fuel to return home) by pressing the up and down arrows on the IFEI. | Set the BINGO fuel on the IFEI to the mission’s minimum return fuel. | IFEI |
| S22 | P6 – Pre-taxi instruments & final checks | Set the Standby Barometric Altimeter to airfield elevation. | Adjust the standby barometric altimeter to the current airfield elevation/QNH. | Right instrument panel |
| S23 | P6 – Pre-taxi instruments & final checks | Set the Radar Altimeter to 200 feet for an airfield takeoff or 40 feet from the carrier. | Set the radar altimeter bug (low altitude warning) to 200 ft (airfield) or 40 ft (carrier). | Right quarter panel |
| S24 | P6 – Pre-taxi instruments & final checks | Uncage the standby Attitude Indicator. | Uncage the standby attitude indicator so it can display correct pitch/roll. | Right instrument panel |
| S25 | P6 – Pre-taxi instruments & final checks | Set the Attitude Source to AUTO. | Set the attitude source selector to AUTO to use the normal attitude reference. | Center instrument panel |
