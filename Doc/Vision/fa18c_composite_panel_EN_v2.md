# F/A-18C Normalized Native Viewport Layout v2

`layout_id`: `fa18c_composite_panel_v2`

This document freezes the v2 visual-input contract for v0.4. v2 no longer treats pixel coordinates as the source of truth. Instead, the layout is defined by a normalized left export strip plus three normalized regions inside that strip.

## Fixed Visual Scope

- `left_ddi`
- `ampcd`
- `right_ddi`

All other evidence remains DCS-BIOS-first, including:

- `MASTER CAUTION` and the left/right warning-caution-advisory blocks
- `IFEI` numeric values and BINGO
- UFC scratchpad, option displays, and COMM1/COMM2 windows
- standby / HUD

## v2 Geometry Rules

- First cut the left export strip from the full screen using normalized strip geometry
- Then cut the three viewport regions from inside that strip using normalized region geometry
- Downstream VLM code must not hard-code full-screen pixel crops for `left_ddi / ampcd / right_ddi`

The frozen `strip_norm` is:

- `anchor: left`
- `x_norm: 0.0`
- `y_norm: 0.0`
- `height_norm: 1.0`
- `target_aspect_ratio: 880 / 1440`
- `min_main_view_width_px: 640`

## Region Order

The order is fixed:

- `left_ddi`
- `ampcd`
- `right_ddi`

Alias drift is forbidden. For example, do not rename `ampcd` to `mpcd`.

## Reference Assets

- Strip preview: `Doc/Vision/assets/fa18c_composite_panel_v2.svg`
- Full-screen diagram: `Doc/Vision/assets/fa18c_composite_panel_fullscreen_v2.svg`

Notes:

- Both SVGs use English labels only
- The strip preview only shows the internal export-strip structure
- The full-screen diagram only shows the normalized relationship between the left export strip and the right main simulator view; it is not tied to a specific monitor resolution

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

## Monitor Setup Behavior

- `single-monitor` and `ultrawide-left-stack` share the same normalized left-stack solver
- `extended-right` remains a debug mode, but its three viewports are still solved from the same normalized region geometry
- This keeps capture, cropping, and VLM references on one geometry model
