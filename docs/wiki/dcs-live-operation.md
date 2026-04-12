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

## Run Live Loop

```bash
python live_dcs.py \
  --stdin-help \
  --help-udp-port 7792 \
  --vision-saved-games-dir "<saved-games-dir>" \
  --vision-session-id sess-live \
  --output logs/live_dcs.jsonl
```

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
