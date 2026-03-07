# F/A-18C Composite Panel Layout v1

`layout_id`: `fa18c_composite_panel_v1`

This document freezes the first v0.4 visual input layout. The goal is to help Qwen3.5 recognize display pages, text, and annunciator states more reliably, not to recreate a photoreal cockpit screenshot.

## Fixed Coverage

- `left_ddi`
- `ampcd`
- `right_ddi`
- `warning_panel`
- `ufc`
- `ifei`
- `standby_hud`

## Explicitly Excluded

- Outside scenery
- Large cockpit background areas
- Unrelated consoles
- Extra cropped regions that are not required by the current pack

## Region Naming Rules

- All future VLM prompts, manifests, documents, and debug output may only use these 7 `region_id` values
- Alias drift is forbidden; for example, do not rename `ampcd` to `mpcd`
- The region order is fixed: `left_ddi -> ampcd -> right_ddi -> warning_panel -> ufc -> ifei -> standby_hud`

## Region Definitions

| region_id | English Name | Position and Size (x,y,w,h) | Primary Content |
| --- | --- | --- | --- |
| `left_ddi` | Left DDI | `32,32,768,768` | Left DDI pages such as FCS/HSI/ENG |
| `ampcd` | AMPCD | `896,32,768,768` | Central AMPCD page text and symbols |
| `right_ddi` | Right DDI | `1760,32,768,768` | Right DDI pages such as BIT/FCS |
| `warning_panel` | Warning Panel | `32,832,848,248` | `MASTER CAUTION`, caution/warning/advisory lamps, and fire-related indicators |
| `ufc` | UFC | `912,832,736,248` | UFC display and COMM-related readouts |
| `ifei` | IFEI | `1680,832,848,248` | Engine/fuel display and BINGO-related readouts |
| `standby_hud` | Standby / HUD | `912,1112,736,296` | Standby instruments and any small HUD crop required by the pack |

## Step Priority Boundary

Prefer the composite panel first:

- `S02`
- `S07`
- `S08`
- `S09`
- `S15`
- `S18`
- `S21`
- `S22`
- `S23`
- `S24`
- `S25`

Still prefer DCS-BIOS first:

- `S01`
- `S03`
- `S04`
- `S06`
- `S10`
- `S12`
- `S13`

Currently outside the first composite-panel priority scope, requiring manual confirmation or remaining out of layout:

- `S05`
- `S11`
- `S14`
- `S16`
- `S17`
- `S19`
- `S20`

## Sample Asset

- Asset path: `Doc/Vision/assets/fa18c_composite_panel_v1.svg`
- The sample image is generated from the frozen layout and is only used to validate region naming, crop stability, and human readability
