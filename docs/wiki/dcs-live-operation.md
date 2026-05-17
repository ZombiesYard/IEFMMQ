# DCS Live Operation

This page covers simulator-facing setup. DCS itself is a commercial external dependency and is not redistributed by this repository.

## Install Hook Files

Linux/WSL command for installing into a Windows Saved Games directory:

```bash
python -m tools.install_dcs_hook \
  --dcs-variant DCS \
  --saved-games-dir "<saved-games-dir>"
```

PowerShell equivalent on the simulator host:

```powershell
python -m tools.install_dcs_hook `
  --dcs-variant DCS `
  --saved-games-dir "<saved-games-dir>"
```

## Install Composite Panel Monitor Setup

```bash
python -m tools.install_dcs_hook \
  --dcs-variant DCS \
  --install-composite-panel \
  --monitor-mode extended-right \
  --saved-games-dir "<saved-games-dir>"
```

Re-run this installer after enabling multi-target overlay, for example when using
`--max-overlay-targets 2`. The generated `SimTutorConfig.lua` must contain
`overlay.hilite_ids = {9101, 9102, 9103, 9104}` or more IDs; legacy configs with only
`overlay.hilite_id = 9101` expose a single DCS highlight slot and can make the
last target replace the first visible highlight.

Single-monitor example:

```bash
python -m tools.install_dcs_monitor_setup \
  --dcs-variant DCS \
  --mode single-monitor \
  --main-width 1920 \
  --main-height 1080 \
  --saved-games-dir "<saved-games-dir>"
```

Ultrawide example:

```bash
python -m tools.install_dcs_monitor_setup \
  --dcs-variant DCS \
  --mode ultrawide-left-stack \
  --main-width 3440 \
  --main-height 1440 \
  --saved-games-dir "<saved-games-dir>"
```

VR note:

- The generated monitor setup now includes a `VR_MIRROR` block constrained to the main viewport rectangle.
- It also enables `VR_allow_MFD_out_of_HMD = true` so native MFD/DDI exports can remain on the desktop canvas while VR is active.
- In DCS itself, enable `Options -> VR -> VR Mirror Options -> Use DCS System Resolution`.
- Keep the DCS resolution set to the full recommended canvas size shown in the generated monitor setup comment.

## Run Live Loop

```bash
python live_dcs.py \
  --stdin-help \
  --help-udp-port 7792 \
  --vision-saved-games-dir "<saved-games-dir>" \
  --vision-session-id sess-live \
  --output logs/live_dcs.jsonl
```

If `--output` already exists, the live loop opens a unique timestamped or numeric-suffixed JSONL path instead of overwriting it. The final resolved path is printed to the console and recorded in the startup `system` event metadata as `resolved_output_path`.

## Run Vision Sidecar

PowerShell example on the simulator host:

```powershell
python .\tools\capture_vision_sidecar.py `
  --saved-games-dir "<saved-games-dir>" `
  --session-id sess-live
```

By default the sidecar is help-triggered. Add `--capture-fps 1` or `--capture-fps 2` only when a continuous low-fps stream is needed.

## Record Replay Material

```bash
python -m simtutor record-vlm \
  --output logs/dcs_bios_raw.jsonl \
  --session-id sess-record \
  --vision-saved-games-dir "<saved-games-dir>" \
  --max-frames 2000
```

## Extract One Failed Help Cycle

For a live failure reported as `request_id=<id>`, extract the relevant request, response, evidence, harness trace, vision facts, and final overlay targets into a compact fixture:

```bash
python -m simtutor extract-live-fixture \
  --input logs/live_dcs.jsonl \
  --request-id "<request-id>" \
  --output-dir artifacts/live_fixtures
```

See [Replay and CLI reference](replay-and-cli-reference.md#extract-a-live-help-fixture) for the fixture shape and regression workflow.
