# SimTutor Core

SimTutor Core is a simulator-agnostic tutoring backend for F/A-18C cold-start tutoring. The repository keeps domain logic separate from simulator integration, supports deterministic mock runs, DCS-BIOS replay, and live DCS operation, and writes JSONL event logs that can be replayed, validated, and scored offline.

The current repository state covers:

- Clean Architecture / Hexagonal boundaries across `core/`, `ports/`, and `adapters/`
- Pack-driven FA-18C startup procedure coverage in `packs/fa18c_startup/`
- Mock, replay, and live DCS help flows
- JSON Schemas under `simtutor/schemas/v1/` and `simtutor/schemas/v2/`
- Reserved vision contracts without forcing a VLM pipeline into the core

## TL;DR

All examples below assume bash and a repository-local virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Build the grounding index:

```bash
python -m tools.index_docs --output Doc/Evaluation/index.json
```

Run one deterministic mock scenario:

```bash
python -m simtutor run \
  --pack packs/fa18c_startup/pack.yaml \
  --scenario mock_scenarios/correct_process.json \
  --output logs/run_demo.jsonl
```

Replay and validate the generated log:

```bash
python -m simtutor replay logs/run_demo.jsonl --pack packs/fa18c_startup/pack.yaml
python -m simtutor score logs/run_demo.jsonl \
  --pack packs/fa18c_startup/pack.yaml \
  --taxonomy packs/fa18c_startup/taxonomy.yaml
```

Replay DCS-BIOS frames through the tutor loop without sending overlay commands:

```bash
python -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --auto-help-once \
  --stdin-help \
  --dry-run-overlay
```

Replay with a local vLLM OpenAI-compatible endpoint:

```bash
export SIMTUTOR_MODEL_PROVIDER=openai_compat
export SIMTUTOR_MODEL_BASE_URL=http://127.0.0.1:8000
export SIMTUTOR_MODEL_NAME=Qwen3-8B-Instruct
export SIMTUTOR_MODEL_TIMEOUT_S=20
export SIMTUTOR_MODEL_API_KEY=dummy
export SIMTUTOR_LANG=zh

python -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --model-provider openai_compat \
  --auto-help-once \
  --stdin-help
```

Run the live DCS loop:

```bash
python live_dcs.py \
  --stdin-help \
  --help-udp-port 7792 \
  --output logs/live_dcs.jsonl
```

Install the DCS hook files into Saved Games:

```bash
python -m tools.install_dcs_hook --dcs-variant DCS
```

Install the monitor setup for the F/A-18C three-screen viewport PoC:

```bash
python -m tools.install_dcs_monitor_setup \
  --dcs-variant DCS \
  --mode extended-right \
  --main-width 1920 \
  --main-height 1080
```

Install the single-monitor variant that places left/center/right MFCDs on the top band of one screen:

```bash
python -m tools.install_dcs_monitor_setup \
  --dcs-variant DCS \
  --mode single-monitor \
  --main-width 1920 \
  --main-height 1080
```

Install the ultrawide variant that stacks the three exported screens in the extra left strip and keeps a full 16:9 main view on the right:

```bash
python -m tools.install_dcs_monitor_setup \
  --dcs-variant DCS \
  --mode ultrawide-left-stack \
  --main-width 3440 \
  --main-height 1440
```

## Repository Layout

- `core/`: pure domain logic, procedure engine, gating, scoring, overlay planning
- `ports/`: stable interfaces for environment, model, knowledge, event store, vision
- `adapters/`: mock adapters, DCS adapters, DCS-BIOS adapters, event-store adapters
- `simtutor/`: CLI entrypoints, runtime config, schema registry
- `simtutor/schemas/v1/`: stable v1 contracts
- `simtutor/schemas/v2/`: transport and forward-compatible v2 contracts
- `packs/fa18c_startup/`: pack, taxonomy, UI mapping, telemetry mapping, BIOS mapping
- `tools/`: indexing, install, diagnostics, telemetry capture, replay-input generation
- `DCS/Scripts/`: DCS-side Lua scripts for telemetry/export/highlight integration
- `Doc/Evaluation/`: authoritative knowledge source material and generated `index.json`
- `mock_scenarios/`: deterministic mock observations for tests and demos
- `tests/`: unit and integration coverage

## Setup

### Requirements

- Python 3.10+
- A virtual environment
- Optional: DCS + DCS-BIOS for live integration
- Optional: local vLLM OpenAI-compatible endpoint for LLM-backed help

### Install

Create and activate a local virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Alternative with Poetry:

```bash
poetry install
poetry run python -m simtutor --help
```

## Runtime Configuration

There is no installed console script at the moment. Use module/script entrypoints directly:

- `python -m simtutor`
- `python live_dcs.py`
- `python -m tools.<tool_name>`

### Environment Variables

| Variable | Purpose | Typical value / behavior |
| --- | --- | --- |
| `SIMTUTOR_MODEL_PROVIDER` | Model provider used by `model-config` and CLI defaults | `stub`, `openai_compat`, or `ollama` |
| `SIMTUTOR_MODEL_NAME` | Model identifier | Defaults by provider; for OpenAI-compatible mode the CLI default is `Qwen3-8B-Instruct` |
| `SIMTUTOR_MODEL_BASE_URL` | Base URL for the model API | Required for `openai_compat`; default for `ollama` is `http://127.0.0.1:11434` |
| `SIMTUTOR_MODEL_TIMEOUT_S` | Model timeout in seconds | Positive float; CLI default is `20` |
| `SIMTUTOR_MODEL_API_KEY` | API key for provider access | Required for `openai_compat`; can be a dummy token for a local vLLM endpoint if that endpoint ignores auth |
| `SIMTUTOR_LANG` | Tutor language | `zh` or `en`; CLI default is `zh` |
| `SIMTUTOR_COLD_START_PRODUCTION` | Default switch for cold-start production mode | Boolean-like string such as `1`, `0`, `true`, `false` |
| `SIMTUTOR_LOG_RAW_LLM_TEXT` | Default switch for raw LLM text logging | Off by default; enable only for debugging |
| `SIMTUTOR_FA18C_CLICKABLEDATA_PATH` | Optional path for clickable-data validation tests | Used by `tests/test_pack_ui_targets_valid.py` |
| `SIMTUTOR_PROMPT_TRIM_PRINT` | Debug print switch for prompt trimming | Developer-only diagnostic toggle |

### Recommended OpenAI-Compatible Configuration

```bash
export SIMTUTOR_MODEL_PROVIDER=openai_compat
export SIMTUTOR_MODEL_BASE_URL=http://127.0.0.1:8000
export SIMTUTOR_MODEL_NAME=Qwen3-8B-Instruct
export SIMTUTOR_MODEL_TIMEOUT_S=20
export SIMTUTOR_MODEL_API_KEY=dummy
export SIMTUTOR_LANG=zh
```

To validate the environment without printing secrets:

```bash
python -m simtutor model-config
```

## Main Workflows

### 1. Mock / Deterministic Flow

1. Build the grounding index if it does not exist yet.
2. Run `python -m simtutor run`.
3. Replay the resulting log.
4. Score the run.

Example:

```bash
python -m tools.index_docs --output Doc/Evaluation/index.json
python -m simtutor run \
  --pack packs/fa18c_startup/pack.yaml \
  --scenario mock_scenarios/correct_process.json \
  --output logs/run_demo.jsonl
python -m simtutor replay logs/run_demo.jsonl --pack packs/fa18c_startup/pack.yaml
python -m simtutor score logs/run_demo.jsonl \
  --pack packs/fa18c_startup/pack.yaml \
  --taxonomy packs/fa18c_startup/taxonomy.yaml
```

### 2. Replay DCS-BIOS Offline

This is the safest way to iterate on the help loop because overlay is dry-run by default.

```bash
python -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --auto-help-once \
  --stdin-help \
  --dry-run-overlay
```

### 3. Live DCS

The live loop listens for DCS-BIOS UDP frames, performs help cycles, and can emit overlay commands through the action executor.

```bash
python live_dcs.py \
  --host 0.0.0.0 \
  --port 7790 \
  --stdin-help \
  --help-udp-port 7792 \
  --output logs/live_dcs.jsonl
```

## CLI Reference

### `python -m simtutor`

Top-level subcommands:

- `validate`
- `run`
- `replay`
- `score`
- `batch`
- `model-config`
- `replay-bios`

#### `simtutor validate`

Syntax:

```bash
python -m simtutor validate [--schema SCHEMA_NAME] FILE [FILE ...]
```

Parameters:

| Argument | Required | Description |
| --- | --- | --- |
| `files` | Yes | One or more JSONL files |
| `--schema` | No | One of `dcs_bios_frame`, `dcs_caps`, `dcs_hello`, `dcs_observation`, `dcs_overlay_ack`, `dcs_overlay_command`, `event`, `observation`, `telemetry_frame`, `tutor_request`, `tutor_response`, `vision_observation`; default `event` |

#### `simtutor run`

Syntax:

```bash
python -m simtutor run --pack PACK --scenario SCENARIO [--output OUTPUT]
```

Parameters:

| Argument | Required | Description |
| --- | --- | --- |
| `--pack` | Yes | Path to `pack.yaml` |
| `--scenario` | Yes | Path to a mock scenario JSON file |
| `--output` | No | Output log path; defaults to `logs/run_<timestamp>.jsonl` |

#### `simtutor replay`

Syntax:

```bash
python -m simtutor replay [FILE] [--pack PACK] [--telemetry TELEMETRY ...]
```

Parameters:

| Argument | Required | Description |
| --- | --- | --- |
| `file` | Conditionally | Event log JSONL for replay validation; required unless `--telemetry` is used |
| `--pack` | Conditionally | Pack path required for event-log replay |
| `--telemetry` | No | One or more telemetry JSONL logs for sequence / wall-clock validation |

#### `simtutor score`

Syntax:

```bash
python -m simtutor score FILE --pack PACK --taxonomy TAXONOMY
```

Parameters:

| Argument | Required | Description |
| --- | --- | --- |
| `file` | Yes | Event log JSONL |
| `--pack` | Yes | Pack path |
| `--taxonomy` | Yes | Taxonomy path |

#### `simtutor batch`

Syntax:

```bash
python -m simtutor batch --pack PACK [--taxonomy TAXONOMY] [--scenarios SCENARIOS ...] [--output-dir OUTPUT_DIR]
```

Parameters:

| Argument | Required | Description |
| --- | --- | --- |
| `--pack` | Yes | Pack path |
| `--taxonomy` | No | Taxonomy path; defaults to `packs/fa18c_startup/taxonomy.yaml` |
| `--scenarios` | No | Scenario JSON files; defaults to `mock_scenarios/*.json` |
| `--output-dir` | No | Output directory for logs and `results.csv`; default `logs` |

#### `simtutor model-config`

Syntax:

```bash
python -m simtutor model-config
```

This validates environment-based provider configuration and prints non-sensitive startup info.

#### `simtutor replay-bios`

Syntax:

```bash
python -m simtutor replay-bios --input INPUT [options]
```

Parameters:

| Argument | Required | Description |
| --- | --- | --- |
| `--input` | Yes | Path to `dcs_bios_raw.jsonl` |
| `--speed` | No | Replay speed, `1.0` realtime and `0` max speed; default `1.0` |
| `--pack` | No | Pack path; default `packs/fa18c_startup/pack.yaml` |
| `--ui-map` | No | UI map path; default `packs/fa18c_startup/ui_map.yaml` |
| `--telemetry-map` | No | Telemetry map path; default `packs/fa18c_startup/telemetry_map.yaml` |
| `--bios-to-ui` | No | BIOS-to-UI path; default `packs/fa18c_startup/bios_to_ui.yaml` |
| `--knowledge-index` | No | Grounding index path; default `Doc/Evaluation/index.json` |
| `--rag-top-k` | No | Number of grounding snippets injected; default `5` |
| `--cold-start-production` | No | Force-enable cold-start production mode |
| `--no-cold-start-production` | No | Force-disable cold-start production mode even if the env default is enabled |
| `--knowledge-source-policy` | No | Path to `knowledge_source_policy.yaml`; also enables policy filtering outside production mode |
| `--output` | No | Output JSONL path; default `logs/replay_bios_<timestamp>.jsonl` |
| `--session-id` | No | Optional session id for emitted events |
| `--cooldown-s` | No | Help cache cooldown in seconds; default `4.0` |
| `--max-frames` | No | Maximum frames to process; `0` means unlimited |
| `--duration` | No | Run duration in seconds; `0` means unlimited |
| `--auto-help-once` | No | Trigger one help cycle after the first frame |
| `--auto-help-every-n-frames` | No | Trigger help every N frames |
| `--stdin-help` | No | Enable stdin help trigger |
| `--help-udp-host` | No | UDP host for the help trigger listener; default `127.0.0.1` |
| `--help-udp-port` | No | UDP port for help trigger; `0` disables it |
| `--help-udp-timeout` | No | UDP trigger timeout in seconds; default `0.2` |
| `--dry-run-overlay` | No | Keep overlay in dry-run mode; default enabled for replay safety |
| `--no-dry-run-overlay` | No | Send overlay commands instead of dry-run planning |
| `--model-provider` | No | `stub`, `openai_compat`, or `ollama`; default `stub` |
| `--model-name` | No | Model name; default from env or `Qwen3-8B-Instruct` |
| `--model-base-url` | No | Provider base URL |
| `--model-timeout-s` | No | Model timeout in seconds; default env or `20` |
| `--model-api-key` | No | Provider API key |
| `--stub-mode` | No | Stub model mode; default `A` |
| `--lang` | No | `zh` or `en`; default env or `zh` |
| `--scenario-profile` | No | `airfield` or `carrier`; default `airfield` |
| `--log-raw-llm-text` | No | Log raw model text into `tutor_response.metadata` |
| `--no-log-raw-llm-text` | No | Force-disable raw LLM text logging even if the env flag is on |

### `python live_dcs.py`

Syntax:

```bash
python live_dcs.py [options]
```

Parameters:

| Argument | Required | Description |
| --- | --- | --- |
| `--host` | No | DCS-BIOS UDP bind host; default `0.0.0.0` |
| `--port` | No | DCS-BIOS UDP bind port; default `7790` |
| `--timeout` | No | Receiver socket timeout in seconds; default `0.2` |
| `--merge-full-state` | No | Merge BIOS deltas into a full-state cache; enabled by default |
| `--no-merge-full-state` | No | Disable full-state merging and use delta-only BIOS payloads |
| `--pack` | No | Pack path; default `packs/fa18c_startup/pack.yaml` |
| `--ui-map` | No | UI map path; default `packs/fa18c_startup/ui_map.yaml` |
| `--telemetry-map` | No | Telemetry map path; default `packs/fa18c_startup/telemetry_map.yaml` |
| `--bios-to-ui` | No | BIOS-to-UI path; default `packs/fa18c_startup/bios_to_ui.yaml` |
| `--knowledge-index` | No | Grounding index path; default `Doc/Evaluation/index.json` |
| `--rag-top-k` | No | Number of grounding snippets injected; default `5` |
| `--cold-start-production` | No | Force-enable cold-start production mode |
| `--no-cold-start-production` | No | Force-disable cold-start production mode even if env default is enabled |
| `--knowledge-source-policy` | No | Path to `knowledge_source_policy.yaml`; also enables policy filtering outside production mode |
| `--output` | No | Event log JSONL output path; default `logs/live_dcs_<timestamp>.jsonl` |
| `--session-id` | No | Optional event session id |
| `--cooldown-s` | No | Same-state help reuse cooldown; default `4.0` |
| `--max-frames` | No | Maximum frames to process; `0` means unlimited |
| `--duration` | No | Run duration in seconds; `0` means unlimited |
| `--auto-help-once` | No | Trigger one help cycle after the first frame |
| `--auto-help-every-n-frames` | No | Trigger help every N frames |
| `--stdin-help` | No | Enable stdin help trigger |
| `--help-udp-host` | No | UDP host for help trigger listener; default `127.0.0.1` |
| `--help-udp-port` | No | UDP port for the help trigger listener; `0` disables it |
| `--help-udp-timeout` | No | UDP trigger timeout in seconds; default `0.2` |
| `--dry-run-overlay` | No | Do not send UDP overlay commands; print planned actions only |
| `--replay-bios` | No | Replay BIOS JSONL instead of listening on UDP |
| `--speed` | No | Replay speed multiplier for `--replay-bios`; default `1.0` |
| `--model-provider` | No | `stub`, `openai_compat`, or `ollama`; default `stub` |
| `--model-name` | No | Model name; default env or `Qwen3-8B-Instruct` |
| `--model-base-url` | No | Provider base URL |
| `--model-timeout-s` | No | Model timeout in seconds; default env or `20` |
| `--model-api-key` | No | Provider API key |
| `--stub-mode` | No | Stub model mode; default `A` |
| `--lang` | No | `zh` or `en`; default env or `zh` |
| `--scenario-profile` | No | `airfield` or `carrier`; default `airfield` |
| `--log-raw-llm-text` | No | Log raw model text into `tutor_response.metadata` |
| `--no-log-raw-llm-text` | No | Force-disable raw LLM text logging even if the env flag is on |

## Supporting Utilities

### `python -m tools.index_docs`

Builds the grounding index from Markdown/PDF sources.

```bash
python -m tools.index_docs --output Doc/Evaluation/index.json
```

| Argument | Required | Description |
| --- | --- | --- |
| `--input` | No | Files or directories to index; default `Doc/Evaluation` |
| `--output` | Yes | Output index JSON path |

### `python -m tools.install_dcs_hook`

Copies DCS-side scripts into Saved Games and optionally patches `Export.lua`.

```bash
python -m tools.install_dcs_hook --dcs-variant DCS
```

| Argument | Required | Description |
| --- | --- | --- |
| `--source-root` | No | Repository root containing `DCS/` and `adapters/` |
| `--saved-games` | No | Explicit Saved Games path; otherwise `<home>/Saved Games/<variant>` |
| `--dcs-variant` | No | DCS variant folder, for example `DCS` or `DCS.openbeta`; default `DCS` |
| `--no-export` | No | Copy files only and do not patch `Export.lua` |

### `python -m tools.record_dcs_telemetry`

Records DCS telemetry frames into JSONL.

```bash
python -m tools.record_dcs_telemetry --output logs/dcs_telemetry.jsonl --duration 30
```

| Argument | Required | Description |
| --- | --- | --- |
| `--output` | Yes | Output JSONL path |
| `--host` | No | UDP bind host; default `0.0.0.0` |
| `--port` | No | UDP bind port; default `7780` |
| `--session-id` | No | Optional session id |
| `--no-handshake` | No | Disable DCS capabilities handshake on start |
| `--caps-host` | No | Handshake host; default `127.0.0.1` |
| `--caps-port` | No | Handshake port; default `7793` |
| `--caps-timeout` | No | Handshake timeout seconds; default `1.0` |
| `--duration` | No | Seconds to record; `0` requires `--max-frames` |
| `--max-frames` | No | Maximum frames to record; `0` requires `--duration` |
| `--print` | No | Print each frame payload to stdout |

### `python -m tools.listen_dcs_bios_raw`

Decodes raw DCS-BIOS export frames.

```bash
python -m tools.listen_dcs_bios_raw --aircraft FA-18C_hornet --once
```

| Argument | Required | Description |
| --- | --- | --- |
| `--host` | No | DCS-BIOS export host; default `239.255.50.10` |
| `--port` | No | DCS-BIOS export port; default `5010` |
| `--aircraft` | Yes | Aircraft name, for example `FA-18C_hornet` |
| `--control-dir` | No | Control-reference directory; default `DCS/Scripts/DCS-BIOS/doc/json` |
| `--once` | No | Read one frame and exit |
| `--wait` | No | Wait time for `--once`; default `5.0` seconds |
| `--min-keys` | No | Minimum BIOS keys before exiting `--once`; default `200` |
| `--stable-frames` | No | Exit after N frames with no key growth; default `5` |
| `--duration` | No | Continuous-mode duration; default `0` means until Ctrl+C |
| `--output` | No | Write decoded frame(s) to JSON or JSONL |

### `python tools/build_coldstart_state_matrix.py`

Builds replayable state-matrix inputs for offline help regression.

```bash
python tools/build_coldstart_state_matrix.py --output-dir artifacts/regression/coldstart_state_matrix
```

| Argument | Required | Description |
| --- | --- | --- |
| `--output-dir` | No | Output directory for `matrix.json` and replay inputs |
| `--pack` | No | Pack path |
| `--telemetry-map` | No | Telemetry map path |
| `--bios-to-ui` | No | BIOS-to-UI path |
| `--scenario-profile` | No | `airfield`, `carrier`, or `all` |

## DCS Integration Notes

### Overlay Boundary

The repository is intentionally limited to overlay actions:

- `highlight`
- `clear`
- `pulse`

It does not auto-click cockpit controls and does not execute unsafe cockpit actions on behalf of the user.

### DCS-Side Files

Relevant DCS-side scripts live under:

- `DCS/Scripts/Hooks/SimTutorHighlight.lua`
- `DCS/Scripts/SimTutor/SimTutor.lua`
- `DCS/Scripts/SimTutor/SimTutor Function.lua`
- `DCS/Scripts/SimTutor/SimTutorDcsBiosHub.lua`

### Typical Live Bring-Up

1. Install the hook files with `python -m tools.install_dcs_hook`.
2. If you are testing the viewport PoC, install the monitor setup with `python -m tools.install_dcs_monitor_setup --mode <extended-right|single-monitor> --main-width <screen_width> --main-height <screen_height>`.
3. For `extended-right`, ensure a right-side extended desktop region exists with at least `2560x1440` pixels.
4. In DCS Options, select `SimTutor_FA18C_CompositePanel_v1` as the monitor setup and set the total resolution to the tool's printed recommended resolution.
5. Ensure DCS or DCS.openbeta loads the copied Lua files.
6. Start `python live_dcs.py`.
7. Trigger help with Enter on stdin or send `help` to the configured UDP help port.

## Output Artifacts

Typical generated outputs:

- `logs/run_<timestamp>.jsonl`
- `logs/replay_bios_<timestamp>.jsonl`
- `logs/live_dcs_<timestamp>.jsonl`
- `logs/dcs_telemetry.jsonl`
- `logs/results.csv`
- `Doc/Evaluation/index.json`
- `artifacts/regression/coldstart_state_matrix/`

Event logs are designed to be replayed and validated with:

```bash
python -m simtutor validate logs/example.jsonl --schema event
python -m simtutor replay logs/example.jsonl --pack packs/fa18c_startup/pack.yaml
```

## Development Notes

- The authoritative machine-executable procedure data lives in `packs/fa18c_startup/`.
- Grounding source material is indexed from `Doc/Evaluation/`.
- Replay and live flows intentionally summarize and budget DCS-BIOS deltas before prompting.
- `vision_observation` exists as a reserved forward-compatible contract, but there is no production VLM pipeline in this repository yet.

## License

This repository is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full text.
