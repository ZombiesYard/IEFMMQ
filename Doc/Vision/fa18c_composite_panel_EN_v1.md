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

## Three-Screen Native Viewport PoC

- The first PoC only validates `left_ddi`, `ampcd`, and `right_ddi`
- These three regions are exported through DCS native monitor viewports: `LEFT_MFCD`, `CENTER_MFCD`, and `RIGHT_MFCD`
- The composite export canvas is fixed at `2560x1440` and must live on a desktop region extended to the right of the main screen
- Install command:

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode extended-right --main-width 1920 --main-height 1080
```

- After installation, select `SimTutor_FA18C_CompositePanel_v1` in DCS Options
- The recommended total DCS resolution is `main_width + 2560` by `max(main_height, 1440)`
- This PoC does not yet export `warning_panel`, `ufc`, `ifei`, or `standby_hud` as native viewports
- If you only have one screen, use single-monitor mode instead:

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode single-monitor --main-width 1920 --main-height 1080
```

- Single-monitor mode places the three MFCD/AMPCD exports on the top band and keeps the main camera on the lower band
- For `3440x1440` and similar 21:9 screens, the ultrawide mode is usually better:

```bash
python -m tools.install_dcs_monitor_setup --dcs-variant DCS --mode ultrawide-left-stack --main-width 3440 --main-height 1440
```

- This mode stacks the three exported screens vertically in the extra left strip and keeps a full `2560x1440` `16:9` main view on the right
