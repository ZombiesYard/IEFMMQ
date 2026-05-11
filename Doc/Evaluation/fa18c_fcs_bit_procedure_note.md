<!--
AUTO-GENERATED FILE. DO NOT EDIT.
generator: tools/regenerate_eval_docs.py
source_index: Doc/Evaluation/index.json
policy_id: fa18c_cold_start_whitelist_v1
policy_version: v2
version_stamp: 2bf9cb39647cc52b
source_chunks:
- fa18c_fcs_bit_procedure_note/fa18c_fcs_bit_procedure_note_0:1-24
-->

# FCS BIT Procedure — Clarification

This document supplements the DCS FA-18C Early Access Guide with the correct
FCS BIT procedure for S18 during F/A-18C cold start.

## Correct S18 FCS BIT Procedure

1. Ensure the right DDI is powered on and shows the BIT/FCS page.
2. Hold up the FCS BIT switch (right wall panel, key [Y]).
3. While holding the FCS BIT switch, press the FCS OSB (PB5) on the right DDI.
4. Once the DDI displays "IN TEST", **release both the FCS BIT switch and the OSB**.
   You do NOT need to keep holding either control.
5. The BIT test will now run automatically. Wait for all four channels
   (MC1, MC2, FCSA, FCSB) to show GO.
6. When all four channels display GO, the FCS BIT is complete and S18 is done.

## Common Mistakes

- **Mistake**: Continuing to hold the FCS BIT switch and/or OSB after IN TEST appears.
  **Correction**: Release both controls once IN TEST is visible. The test runs on its own.
- **Mistake**: Expecting an instant result. The BIT test takes approximately 1–2 minutes
  to complete after IN TEST appears.
- **Mistake**: Confusing intermediate PBIT GO with final GO. Wait until MC1, MC2, FCSA,
  and FCSB all show their individual GO indications; PBIT GO alone is not sufficient.
