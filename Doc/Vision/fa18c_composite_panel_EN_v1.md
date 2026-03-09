# F/A-18C Native Viewport Layout v1

`layout_id`: `fa18c_composite_panel_v1`

This document freezes the current v0.4 visual-input contract: only the three DCS-native display exports remain in scope. `warning_panel`, `ufc`, `ifei`, and `standby_hud` are no longer part of the visual contract.

## Fixed Coverage

- `left_ddi`
- `ampcd`
- `right_ddi`

## Explicitly Removed From Visual Input

- `MASTER CAUTION` and the left/right warning-caution-advisory blocks
- All `IFEI` numeric content and BINGO readouts
- UFC scratchpad, option windows, and COMM1/COMM2 displays
- Standby instruments and any HUD crop
- Outside scenery, the main camera view, large cockpit background areas, and unrelated consoles

These signals should now come from DCS-BIOS instead of image recognition.

## Region Naming Rules

- All future VLM prompts, manifests, documents, and debug output may only use these 3 `region_id` values
- Alias drift is forbidden; for example, do not rename `ampcd` to `mpcd`
- The region order is fixed: `left_ddi -> ampcd -> right_ddi`

## Region Definitions

The canonical canvas is fixed to `880x1440`, matching the cropped left export strip used in the current ultrawide deployment.

| region_id | English Name | Position and Size (x,y,w,h) | Native DCS viewport | Primary Content |
| --- | --- | --- | --- | --- |
| `left_ddi` | Left DDI | `216,24,448,448` | `LEFT_MFCD` | Left DDI pages such as FCS/HSI |
| `ampcd` | AMPCD | `216,496,448,448` | `CENTER_MFCD` | Center AMPCD page text and symbology |
| `right_ddi` | Right DDI | `216,968,448,448` | `RIGHT_MFCD` | Right DDI pages such as BIT/FCS |

## Step Priority Boundary

Prefer native viewports first:

- `S08`
- `S15`
- `S18`

Still prefer DCS-BIOS first:

- `S01`
- `S03`
- `S04`
- `S05`
- `S06`
- `S07`
- `S09`
- `S10`
- `S11`
- `S12`
- `S13`
- `S21`

Currently outside the first native-viewport scope, requiring manual confirmation or remaining out of layout:

- `S02`
- `S14`
- `S16`
- `S17`
- `S19`
- `S20`
- `S22`
- `S23`
- `S24`
- `S25`

## Sample Asset

- Asset path: `Doc/Vision/assets/fa18c_composite_panel_v1.svg`
- The sample image only validates the three `region_id` values, their order, and border treatment; it does not represent the main game camera layout

## Current Deployment Layout

The frozen deployment layout is the current `3440x1440` ultrawide single-monitor setup:

- DCS monitor setup uses `ultrawide-left-stack`
- The left `880px` strip stacks the three native display exports vertically
- The right side keeps a full `2560x1440` `16:9` playable main view
- The VLM side must only consume the left `880x1440` export strip and must not ingest the right-side main camera
- On a `16:9` single screen, `single-monitor` now uses the same left-stack geometry family and only scales the export strip by screen height, so downstream code does not need a second single-screen layout model

Install command:

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode ultrawide-left-stack --main-width 3440 --main-height 1440
```

After installation:

- Select `SimTutor_FA18C_CompositePanel_v1` in DCS Options
- Set the resolution to `3440x1440`
- For a `16:9` single-screen setup, use:

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode single-monitor --main-width 1920 --main-height 1080
```

- `single-monitor` and `ultrawide-left-stack` now share the same left-stack three-viewport family; `extended-right` remains debug-only
